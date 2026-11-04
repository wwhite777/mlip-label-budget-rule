"""Compare arrays, metrics and readings between two stage directories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left',type=Path);p.add_argument('right',type=Path);a=p.parse_args()
    files=arrays=0
    for left in sorted(a.left.rglob('*.npz')):
        right=a.right/left.relative_to(a.left)
        with np.load(left,allow_pickle=False) as x,np.load(right,allow_pickle=False) as y:
            assert set(x.files)==set(y.files),left.name
            for key in x.files:
                assert x[key].dtype==y[key].dtype,(left.name,key)
                if np.issubdtype(x[key].dtype,np.number):assert np.array_equal(x[key],y[key],equal_nan=True),(left.name,key)
                else:assert np.array_equal(x[key],y[key]),(left.name,key)
                arrays+=1
        files+=1
    for left in sorted(a.left.rglob('*metrics.csv')):
        right=a.right/left.relative_to(a.left)
        x=list(csv.DictReader(left.open()));y=list(csv.DictReader(right.open()))
        key=lambda row:json.dumps(row,sort_keys=True)
        assert sorted(x,key=key)==sorted(y,key=key),left.name;files+=1
    for name in ('readings.json','judge.json'):
        for left in sorted(a.left.rglob(name)):
            assert json.loads(left.read_text())==json.loads((a.right/left.relative_to(a.left)).read_text()),name;files+=1
    assert files,'no scientific outputs found'
    print(json.dumps(dict(pass_all=True,files=files,arrays=arrays)))
if __name__=='__main__':main()
