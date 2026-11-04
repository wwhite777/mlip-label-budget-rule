"""Check the exact inputs required by CPU fitting or the PILOT FT smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from confirm_runner.common import load_yaml,checked

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['cpu','ft-smoke'],required=True);a=p.parse_args()
    doc=load_yaml();specs=[]
    specs.extend(doc['population'][k] for k in ['summary','partition'])
    if a.kind=='cpu':
        specs += list(doc['inputs']['feature_inputs'].values())+list(doc['inputs']['reference_inputs'].values())
        specs += [doc['inputs']['element_basis'],doc['designs']['P']['folds']]
        specs += [doc['designs']['S'][d]['assignment'] for d in ('S-a','S-b')]
    else:
        specs += [doc['population']['initial_structures'],doc['population']['relaxed_structures_ft_targets'],doc['inputs']['auxiliary']['mp2020_corrections_initial'],doc['inputs']['ft_elemental_references'],doc['inputs']['reference_inputs']['mace_medium']]
        specs += [doc['inputs']['checkpoint_inputs'][bb] for bb in ('mace_medium','orb')]
        specs += list(doc['validation_inputs']['ft_smoke_ids'].values())
    for spec in specs:checked(spec);print('Verified',spec['path'])
    print(json.dumps(dict(pass_all=True,kind=a.kind,files=len(specs))))
if __name__=='__main__':main()
