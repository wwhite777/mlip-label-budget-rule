"""Synthetic test-energy files through production ingestion, paired bootstrap and judge."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from confirm_runner.common import HERE,load_yaml,dump,sha256
from confirm_runner.ft_launcher import ft_units,unit_id,stage_unit
from confirm_runner.ft_ingest import collect_ft_scores
from confirm_runner.statistics import infer,judge
from synthetic_check import fixture


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root','--output',dest='output',type=Path,required=True)
    args=parser.parse_args()
    doc=load_yaml();inp,train,tests=fixture(doc);part=inp.part
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    root=out/'energies';units=ft_units(doc)
    # Write only one replicate of energy arrays. Other replicates use hard links to identical
    # synthetic scores with distinct ID manifests, exercising replicate aggregation without disk duplication.
    rng=np.random.default_rng(314159)
    cache={}
    for u in units:
        test=np.flatnonzero(part.fold==u['fold']) if u['design']=='P' else tests['S-a']
        # No training label is needed: test-energy fixtures carry identity only.
        mp=stage_unit(root,u,dict(train=['SYNTHETIC_TRAIN'],fit=['SYNTHETIC_TRAIN'],valid=[],test=part.ids[test].tolist()))
        m=json.loads(mp.read_text());m['synthetic_only']=True;dump(mp,m)
        dest=root/'units'/unit_id(u);attempt=dest/'synthetic_attempt';attempt.mkdir(parents=True,exist_ok=True)
        key=(u['design'],u['checkpoint'],u['n'],u['fold'])
        path=attempt/'test_energies.csv'
        if key in cache: path.hardlink_to(cache[key])
        else:
            # Large negative improvement: both F4 checkpoints should DOES_NOT_EXCEED at positive m.
            values=rng.normal(size=len(test))
            pd.DataFrame(dict(material_id=part.ids[test],energy_total_eV=values*2,
                e_hull_pred_sp=values)).to_csv(path,index=False)
            cache[key]=path
        run_record=dict(status='complete',dry_run=False,synthetic_only=True,e0_mode='foundation',variant='full',train_geometry='relaxed',
                     output_sha256={'test_energies.csv':sha256(path)})
        dump(attempt/'run_record.json',run_record)
        dump(dest/'DONE.json',dict(status='COMPLETE',exit_code=0,attempt=attempt.name,manifest_sha256=sha256(mp),
            output_sha256={**run_record['output_sha256'],'run_record.json':sha256(attempt/'run_record.json')},
            driver_sha256=doc['implementation']['imports']['frozen_ft_driver']['sha256'],unit=u))
    readings=[];counts={}
    for design in ['P','S']:
        idx=np.arange(part.n) if design=='P' else tests['S-a']
        scores=collect_ft_scores(doc,part,design,idx,root,list(range(5)),synthetic=True,tests=tests,include_descriptive=True)
        scores.update({f'R0_{b}':-inp.hulls[idx,j][None,:] for j,b in enumerate(doc['backbones']['order'])})
        for variant in (['P'] if design=='P' else ['S-a','S-b']):
            mask=np.ones(len(idx),bool) if variant!='S-b' else np.isin(idx,tests['S-b'])
            dest=out/variant;dest.mkdir(exist_ok=True)
            y=inp.labels.lookup(part.ids[idx[mask]]).stable.to_numpy()
            rows,_=infer(doc,{k:v[:,mask] for k,v in scores.items()},y,part.chemsys[idx[mask]],[],list(range(5)),variant,200,dest,True,workers=4)
            counts[variant]=len([x for x in rows if x['family']=='F4']);readings+=rows
    assert counts=={'P':4,'S-a':4,'S-b':4}
    decision=judge(doc,readings,synthetic=True,complete=False)
    assert not decision['scientific_reading_authorized'] and decision['status']=='INCOMPLETE'
    dump(out/'readings.json',readings)
    report=dict(status='PASS',synthetic_only=True,real_label_columns_opened=[],designs=['P','S-a','S-b'],
        F4_contrasts_per_design=4,judge=decision,energy_units=len(units),replicates=list(range(5)),B=200,
        scores_saved_replicates=[0],scores_other_replicates='hard links to replicate 0 synthetic arrays; no additional physical copies')
    dump(out/'run.json',report);print(json.dumps(report))
if __name__=='__main__':main()
