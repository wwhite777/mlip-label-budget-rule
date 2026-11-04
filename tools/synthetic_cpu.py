"""Run the frozen reduced matrix on generated inputs and independent synthetic labels."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'src')]
from synthetic_check import fixture
from confirm_runner.common import load_yaml
from confirm_runner import runner

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-root',type=Path,required=True);p.add_argument('--workers',type=int,default=7);a=p.parse_args()
    if not 1<=a.workers<=7:p.error('use 1..7 one-thread workers')
    doc=load_yaml();inp,train,tests=fixture(doc)
    runner.campaign_assignments=lambda *args,**kwargs:(train,tests,{})
    runner.load_inputs=lambda *args,**kwargs:inp
    sys.argv=[sys.argv[0],'--synthetic-labels','--reduced','--workers',str(a.workers),'--out-root',str(a.out_root)]
    runner.main()
    report=json.loads((a.out_root/'run.json').read_text())
    assert report['all_requested_cpu_units_complete']
    assert report['judge']['status']=='INCOMPLETE' and not report['judge']['scientific_reading_authorized']
    print('PASS: reduced synthetic matrix; primary judgment INCOMPLETE')
if __name__=='__main__':main()
