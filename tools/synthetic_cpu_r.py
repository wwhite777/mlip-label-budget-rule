"""Exercise actual CPU-R heads on generated data and synthetic labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'src')]
from synthetic_check import fixture
from confirm_runner.common import load_yaml
from confirm_runner import relaxed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,required=True);p.add_argument('--workers',type=int,default=7)
    a=p.parse_args()
    if not 1<=a.workers<=10:p.error('workers must be 1..10')
    doc=load_yaml(ROOT/'pipeline/protocol_v2.4.yaml')
    inp,train,tests=fixture(doc,N=30000)
    rng=np.random.default_rng(20261104)
    inp.relaxed=rng.normal(size=inp.hulls.shape);inp.relaxed[::97,0]=np.nan
    inp.synthetic_campaign=(train,tests)
    relaxed.load_inputs=lambda *args,**kwargs:inp
    relaxed.load_relaxed=lambda *args,**kwargs:{'source':'generated relaxed predictions, including missing values'}
    relaxed.reduce_synthetic_population=lambda doc,inputs:inputs
    sys.argv=[sys.argv[0],'--synthetic-labels','--reduced','--workers',str(a.workers),'--out-root',str(a.out_root)]
    relaxed.main()
    result=json.loads((a.out_root/'run.json').read_text())
    assert result['execution_complete'] and result['completed_units']==264
    assert result['judge']['status']=='INCOMPLETE' and not result['judge']['scientific_reading_authorized']
    assert not result['real_label_columns_opened']
    print('PASS: CPU-R generated-input matrix, missing values, both designs, no scientific authorization')

if __name__=='__main__':main()
