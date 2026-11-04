"""finetune run/backends unchanged; guarded input preparation for the frozen FT unit."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .common import HERE, ROOT, ProtocolError, checked, dump, load_yaml, resolve
from .ft_data import authorize_manifest, convert_totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--e0',choices=['foundation'],required=True)
    p.add_argument('--train-geometry',choices=['relaxed'],required=True)
    a=p.parse_args()
    if not a.dry_run:
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible.isdigit() or int(visible)<=0: raise ProtocolError('one positive physical GPU index required; GPU 0 forbidden')
    access=authorize_manifest(a.manifest,dry_run=a.dry_run)
    doc=access.doc;unit=access.unit
    for spec in doc['implementation']['imports'].values(): checked(spec)
    src=ROOT/'src/finetune'
    # arms and finetune have different pilot_data modules. Imports above use only metadata;
    # now bind finetune's own helper before loading its unchanged drivers.
    sys.modules.pop('pilot_data',None)
    sys.path.insert(0,str(src))
    common=importlib.import_module('finetune_common')
    family='mace' if unit['checkpoint']=='mace_medium' else 'orb'
    backend=importlib.import_module('finetune_'+family)
    backend_class=backend.MaceBackend if family=='mace' else backend.OrbBackend
    recipe=doc['ft_recipe'];manifest=json.loads(a.manifest.read_text())
    args=common.parser(__doc__).parse_args([
        '--checkpoint',str(checked(doc['inputs']['checkpoint_inputs'][unit['checkpoint']])),
        '--train-ids',manifest['ids']['train']['path'],'--valid-ids',manifest['ids']['valid']['path'],
        '--test-ids',manifest['ids']['test']['path'],'--output',str(a.output),'--seed',str(unit['seed']),
        '--variant','full','--e0','foundation','--train-geometry','relaxed',
        '--device','cpu' if a.dry_run else 'cuda','--threads','1',
        '--epochs',str(recipe['epochs_max']),'--patience',str(recipe['patience']),
        '--batch-size',str(recipe['batch_size']),'--lr',str(recipe['full_lr']),
        '--cse',str(resolve(doc['population']['relaxed_structures_ft_targets']['path'])),
        '--summary',str(resolve(doc['population']['summary']['path'])),
        '--initial',str(resolve(doc['population']['initial_structures']['path'])),
        '--corrections',str(resolve(doc['inputs']['auxiliary']['mp2020_corrections_initial']['path'])),
        '--elemental-refs',str(resolve(doc['inputs']['ft_elemental_references']['path']))
    ]+(['--dry-run'] if a.dry_run else []))
    from .ft_prepare import prepare
    common.prepare=lambda args:prepare(args,common,access)
    common.score_totals=lambda ids,raw,counts,summary,corrections,refs_path:convert_totals(
        doc,ids,raw,counts,summary=summary,corrections=corrections,refs_path=refs_path)
    common.run(args,backend_class)
    rp=a.output/'run_record.json';run_record=json.loads(rp.read_text())
    run_record['scoring_convention']='raw INITIAL total + one pinned MP2020 correction; rounded elemental refs; fixed MP hull from pinned MACE reference formation minus hull; no evaluation labels opened'
    run_record['v2.3_unit']=unit
    run_record['ft_label_access']=dict(part='PILOT' if a.dry_run else 'CONFIRM',
        allowed_training_ids=access.ids['train'],training_target_ids=access.ids['train'],
        test_label_columns_opened=[],summary_label_columns_opened=['uncorrected_energy_from_cse'],
        trace=access.trace,manifest_path=str(a.manifest),manifest_sha256=common.sha256(a.manifest))
    dump(rp,run_record)
    return 0
if __name__=='__main__':raise SystemExit(main())
