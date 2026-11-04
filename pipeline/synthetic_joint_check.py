"""Planted all-family scores exercise active F1-F4 and the success conjunction."""
import json
from pathlib import Path
import numpy as np
from confirm_runner.common import HERE,load_yaml,dump
from confirm_runner.statistics import infer,judge
from confirm_runner.ft_ingest import collect_ft_scores
from synthetic_check import fixture

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root','--output',dest='output',type=Path,required=True)
    parser.add_argument('--ft-root',type=Path,required=True)
    args=parser.parse_args()
    doc=load_yaml();inp,train,tests=fixture(doc);part=inp.part;reps=list(range(5))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    y=inp.labels.lookup(part.ids).stable.to_numpy()
    base=np.random.default_rng(77).random(part.n)
    scores={f'R0_{bb}':base[None,:] for bb in doc['backbones']['order']}
    scores['E0_mean']=base[None,:]
    for n in [300,1000,3000,10000]:
        for bb in doc['backbones']['order']:
            scores[f'A5f_hist_gbm_{bb}@{n}']=np.tile(y if n==10000 else base,(5,1))
            scores[f'C1_ridge_{bb}@{n}']=np.tile(base,(5,1))
        scores[f'A4_hist_gbm_ensemble@{n}']=np.tile(base,(5,1))
    scores.update(collect_ft_scores(doc,part,'P',np.arange(part.n),args.ft_root,reps,synthetic=True,tests=tests))
    rows,_=infer(doc,scores,y,part.chemsys,[300,1000,3000,10000],reps,'P',200,out,True,workers=4)
    counts={f:sum(r['family']==f for r in rows) for f in ['F1','F2','F3','F4']}
    assert counts==dict(F1=6,F2=3,F3=5,F4=4)
    decision=judge(doc,rows,synthetic=True,complete=False)
    assert decision['success'] and not decision['scientific_reading_authorized']
    # A4, inner-selected A5, and FT may ADDS but cannot satisfy the high-budget clause.
    wrong=[dict(r,arm='A4') if r['family']=='F1' else r for r in rows]
    assert not judge(doc,wrong,synthetic=True)['success']
    for r in rows:
        if r['family']=='F2' and r['n']==300:r['reading']='INCONCLUSIVE'
    assert not judge(doc,rows,synthetic=True)['success']
    report=dict(status='PASS',synthetic_only=True,all_families=counts,planted_success=decision,
        A4_cannot_satisfy_success=True,both_low_budgets_required=True,
        real_label_columns_opened=[],FT_source='synthetic test-energy ingestion',
        descriptive_FT_n10000_replicates=1,descriptive_FT_has_no_F4_reading=True)
    dump(out/'run.json',report);print(json.dumps(report))
if __name__=='__main__':main()
