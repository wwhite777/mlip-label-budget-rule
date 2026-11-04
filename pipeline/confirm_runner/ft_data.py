"""Guarded FT targets; no evaluation label is needed for test-energy conversion."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, YAML, ProtocolError, checked, load_yaml, sha256, dump, resolve
from .labels import LabelAccessError, require_protocol
from .inputs import metadata_csv, load_part, campaign_assignments
from .ft_launcher import draw_indices, ft_units

METADATA={'formula','n_sites'}
TRAINING={'uncorrected_energy_from_cse'}


class FTAccess:
    def __init__(self, doc, unit, ids, *, dry_run=False):
        self.doc,self.unit,self.ids,self.dry_run=doc,unit,ids,dry_run
        self.draw=set(ids['train']);self.all_ids=self.draw|set(ids['test'])
        self.trace=[]

    def assert_ids(self, ids, *, target=False):
        wanted=list(ids)
        if not wanted or len(wanted)!=len(set(wanted)) or not set(wanted)<=(self.draw if target else self.all_ids):
            raise LabelAccessError('FT target outside drawn training IDs' if target else 'FT unknown or duplicate IDs')

    def selected_rows(self,path,ids,columns):
        target=bool(set(columns)&TRAINING)
        if not set(columns)<=METADATA|TRAINING: raise LabelAccessError('FT forbidden summary label column')
        self.assert_ids(ids,target=target)
        if Path(path).resolve()!=resolve(self.doc['population']['summary']['path']).resolve():
            raise LabelAccessError('FT summary path changed')
        all_ids=pd.read_csv(path,usecols=['material_id'],dtype=str,keep_default_na=False).material_id
        wanted=set(ids); lines=set((np.flatnonzero(all_ids.isin(wanted))+1).tolist())
        frame=pd.read_csv(path,usecols=['material_id']+list(columns),keep_default_na=False,
                          skiprows=lambda i:i!=0 and i not in lines)
        if frame.material_id.duplicated().any() or set(frame.material_id)!=wanted: raise ProtocolError('FT selected-row coverage')
        self.trace.append(dict(path=str(path),columns=columns,ids=list(ids),training_target=target))
        return frame.set_index('material_id').loc[ids]

    def load_selected(self, common, path, column, ids):
        target=column=='computed_structure_entry'
        self.assert_ids(ids,target=target)
        spec=self.doc['population']['relaxed_structures_ft_targets' if target else 'initial_structures']
        if column not in ('computed_structure_entry','initial_structure') or Path(path).resolve()!=resolve(spec['path']).resolve():
            raise LabelAccessError('FT structure source changed')
        wanted=set(ids);out={}
        for _,_,obj,mid in common.column_items(path,column,wanted):
            if mid in out: raise ProtocolError('duplicate FT structure')
            out[mid]=obj
            if len(out)==len(wanted): break
        if set(out)!=wanted: raise ProtocolError('FT structure coverage')
        self.trace.append(dict(path=str(path),column=column,ids=list(ids),training_target=target))
        return out


def authorize_manifest(path, *, dry_run=False, yaml_path=YAML):
    doc=load_yaml(yaml_path);m=json.loads(Path(path).read_text())
    # Metadata in manifest is untrusted until the local protocol or exact PILOT fixture is verified.
    if m.get('dry_run') is not dry_run or m.get('yaml_sha256')!=sha256(yaml_path): raise ProtocolError('FT manifest/protocol mismatch')
    if not dry_run:
        require_protocol(yaml_path=yaml_path)
        stage=m.get('stage')
        if stage not in ('FT-P','FT-S') or m.get('stage_root') is None or m.get('unit',{}).get('design')!=stage[-1]:
            raise ProtocolError('FT manifest requires its fixed execution stage and stage root')
        from .stages import require_previous
        require_previous(m['stage_root'],stage,yaml_path=yaml_path)
    ids={name:checked(spec).read_text().split() for name,spec in m['ids'].items()}
    if set(ids)!={'train','fit','valid','test'}: raise ProtocolError('FT ID roster')
    if set(ids['fit'])&set(ids['valid']) or set(ids['fit'])|set(ids['valid'])!=set(ids['train']) or set(ids['train'])&set(ids['test']):
        raise ProtocolError('FT split identity')
    unit=m['unit']
    if dry_run:
        expected={name:checked(spec).read_text().split() for name,spec in doc['validation_inputs']['ft_smoke_ids'].items()}
        if ids!=expected or len(set(ids['train']+ids['test']))!=8 or m['part']!='PILOT': raise ProtocolError('FT smoke fixture mismatch')
        partition=metadata_csv(checked(doc['population']['partition']),['material_id','part'], table='partition').set_index('material_id')
        if set(partition.loc[list(set(ids['train']+ids['test'])),'part'])!={'PILOT'}: raise LabelAccessError('FT smoke must be PILOT only')
        expected_unit=dict(design='P',n=6,fold=0,draw=0,checkpoint=unit['checkpoint'],descriptive=True,seed=0)
        if unit!=expected_unit or unit['checkpoint'] not in doc['families']['F4']['checkpoints']: raise ProtocolError('FT smoke unit drift')
    else:
        if unit not in ft_units(doc) or m['part']!='CONFIRM': raise ProtocolError('FT unit outside frozen grid')
        part=load_part(doc);train,tests,_=campaign_assignments(doc,part)
        if ids!=draw_indices(doc,part,unit,train,tests): raise ProtocolError('FT IDs differ from frozen draw')
    return FTAccess(doc,unit,ids,dry_run=dry_run)


def convert_totals(doc, ids, raw, counts, *, summary=None, corrections=None, refs_path=None):
    """Equivalent MP2020-once conversion using label-free pinned fixed-hull values."""
    from pymatgen.core import Composition
    # finetune's reference reader is passed by the driver to avoid module-name ambiguity.
    import finetune_common as common
    summary=summary or resolve(doc['population']['summary']['path'])
    corrections=corrections or resolve(doc['inputs']['auxiliary']['mp2020_corrections_initial']['path'])
    refs_path=refs_path or resolve(doc['inputs']['ft_elemental_references']['path'])
    s=metadata_csv(summary,['material_id','formula','n_sites']).set_index('material_id').loc[ids]
    counts=np.asarray(counts,dtype=np.int64)
    if not np.array_equal(counts,s.n_sites.astype(float).astype(int)): raise ProtocolError('FT atom counts drift')
    with np.load(corrections,allow_pickle=False) as z:
        if str(z['structures'])!='initial' or str(z['pymatgen_version'])!=doc['inputs']['pymatgen_version']: raise ProtocolError('FT correction convention')
        corr=pd.Series(z['correction'],index=z['material_id'].astype(str)).loc[ids].to_numpy()
    refs=common.elemental_refs(refs_path)
    raw=np.asarray(raw,dtype=float)
    formation=np.array([(e+c-sum(refs[str(el)]*amount for el,amount in Composition(formula).items()))/n
                       for formula,e,c,n in zip(s.formula,raw,corr,counts)])
    fixed=metadata_csv(resolve(doc['inputs']['reference_inputs']['mace_medium']['path']),
                       ['material_id','e_form_pred_sp','e_hull_pred_sp'], table='reference').set_index('material_id').loc[ids]
    hmp=fixed.e_form_pred_sp.astype(float).to_numpy()-fixed.e_hull_pred_sp.astype(float).to_numpy()
    out=pd.DataFrame(dict(material_id=ids,n_atoms=counts,energy_total_eV=raw,mp2020_correction_eV=corr,
                         e_form_pred_sp=formation,e_hull_pred_sp=formation-hmp))
    if not np.isfinite(out.drop(columns='material_id').to_numpy()).all(): raise ProtocolError('nonfinite FT conversion')
    return out
