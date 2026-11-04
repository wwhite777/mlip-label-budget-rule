"""Planted fixtures for the primary success rule."""
from pathlib import Path
from copy import deepcopy
import json,sys
ROOT=Path(__file__).resolve().parents[1]
import argparse
p=argparse.ArgumentParser();p.add_argument('--out-root',type=Path,required=True);args=p.parse_args()
out=args.out_root/'judge.json'
sys.path.insert(0,str(ROOT/'pipeline'))
from confirm_runner.common import load_yaml,ProtocolError
from confirm_runner.statistics import contrast_specs,judge
doc=load_yaml();rows=[]
for family,specs in contrast_specs(doc,[300,1000,3000,10000]).items():
 if family not in ('F1','F2'):continue
 for name,arm,bb,n,_,_ in specs:
  rows.append(dict(design='P',family=family,n=n,reading='DOES_NOT_EXCEED' if family=='F2' else 'INCONCLUSIVE',arm=arm,model='hist_gbm' if family=='F1' else 'maximum',checkpoint=bb,contrast=name,family_complete=True,synthetic_only=False))
fixtures={}
def add(name,rr,expected,**kw):
 result=judge(doc,rr,**kw);assert result['status']==expected,(name,result);fixtures[name]=result
r=deepcopy(rows);r[0]['reading']='ADDS';add('success',r,'SUCCESS',complete=True)
r=deepcopy(rows);r[5]['reading']='ADDS';add('energy_stack_alone',r,'RULE_NOT_MET',complete=True)
r=deepcopy(rows);r[0]['reading']='ADDS';r[6]['reading']='INCONCLUSIVE';add('missing_low_reading',r,'RULE_NOT_MET',complete=True)
r=deepcopy(rows);r[0]['reading']='ADDS';r[0]['design']='S-b';add('secondary_cannot_substitute',r,'INCOMPLETE')
r=deepcopy(rows);r[0]['reading']='ADDS';r.pop(1);add('missing_primary_member',r,'INCOMPLETE')
r=deepcopy(rows);r[0]['reading']='ADDS';r.append(deepcopy(r[0]));add('duplicate_member',r,'INCOMPLETE')
r=deepcopy(rows);r[0]['reading']='ADDS';r[0]['checkpoint']='wrong';add('wrong_identity',r,'INCOMPLETE')
r=deepcopy(rows);r[0]['reading']='ADDS';r[0]['synthetic_only']=True;add('synthetic_row_in_real',r,'INCOMPLETE')
r=deepcopy(rows);r[0]['reading']='ADDS';add('synthetic_never_authorizes',r,'SUCCESS',synthetic=True);assert fixtures['synthetic_never_authorizes']['scientific_reading_authorized'] is False
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(fixtures,indent=2)+'\n');print('PASS',len(fixtures),'judge fixtures')
