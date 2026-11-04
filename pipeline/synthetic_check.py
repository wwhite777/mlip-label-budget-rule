"""Reduced CPU matrix on a small generated population; no real labels or GPU."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from confirm_runner.common import HERE,load_yaml,dump
from confirm_runner import runner
from confirm_runner.inputs import campaign_assignments
from confirm_runner.labels import read_labels
from analysis.data import build_part


def fixture(doc,N=5000):
    rng=np.random.default_rng(271828)
    # Rounds 1/2 train, 3/4/5 test; duplicate systems across campaign rounds exercise S-a overlap.
    ids=np.array([f'wbm-{i%5+1}-{i:06d}' for i in range(N)])
    systems=np.array([f"{'T' if i%5>=2 and (i//5)%2 else 'S'}{i//5:05d}" for i in range(N)])
    folds=np.array([i//5%5 for i in range(N)])
    part=build_part('CONFIRM',ids,systems,folds,['H']*N,dict.fromkeys(ids,'CONFIRM'))
    # build_part canonicalizes rows; arrays below are defined in that order.
    comp=rng.dirichlet(np.ones(6),N);hulls=rng.normal(size=(N,5))
    refs={bb:dict(r_sp=dict(e_form=hulls[:,i]-2,e_hull=hulls[:,i])) for i,bb in enumerate(doc['backbones']['order'])}
    inp=SimpleNamespace(part=part,comp=comp,hulls=hulls,hmp=np.full(N,-2.),refs=refs,
        feats={bb:rng.normal(size=(N,6)).astype('float32') for bb in doc['backbones']['order']},
        coverage=dict(part='SYNTHETIC_GENERATED',structures=N,systems=len(part.systems),labels_opened=False,source='generated; no CONFIRM data input'))
    inp.labels=read_labels(doc,part,synthetic=True)
    rd=np.array([int(m.split('-')[1]) for m in part.ids])
    train=np.flatnonzero(rd<=2);sa=np.flatnonzero(rd>=3)
    sb=sa[~np.isin(part.chemsys[sa],part.chemsys[train])]
    assert len(sb)>0 and not set(part.chemsys[sb])&set(part.chemsys[train])
    tests={'S-a':sa,'S-b':sb}
    return inp,train,tests


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root','--output',dest='output',type=Path,required=True)
    args=parser.parse_args()
    doc=load_yaml();inp,train,tests=fixture(doc)
    runner.campaign_assignments=lambda *a,**k:(train,tests,{})
    out=args.output
    report=runner.execute(doc,inp,out,workers=7,reduced=True,synthetic=True)
    assert report['all_requested_cpu_units_complete']
    assert report['replicates']==[0,1] and report['B']==200 and 'A5f' in report['arms_executed']
    assert report['judge']['status']=='INCOMPLETE' and not report['judge']['scientific_reading_authorized']
    dump(out/'fixture.json',dict(source='generated features, references and independent synthetic labels',N=inp.part.n,
        budgets=[300,1000],replicates=[0,1],B=200,real_label_columns_opened=[],
        limitation='This small fixture exercises execution and the overlapping S-a and system-disjoint S-b score subsets; full real population coverage is checked separately without labels.'))
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],judge=report['judge'],arms=report['arms_executed'])))
if __name__=='__main__':main()
