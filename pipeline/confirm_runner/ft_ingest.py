"""Read hash-verified FT test-energy files, align exact populations, emit FT scores."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ProtocolError, checked, sha256
from .ft_launcher import unit_id, validate_done
from .ft_data import authorize_manifest


def read_unit(root, unit, expected_ids, *, synthetic=False):
    dest=Path(root)/'units'/unit_id(unit)
    mp=Path(root)/'ids'/unit_id(unit)/'manifest.json'
    if not validate_done(dest,mp): raise ProtocolError('FT incomplete/invalid DONE or output hash: '+unit_id(unit))
    manifest=json.loads(mp.read_text())
    if bool(manifest.get('synthetic_only')) is not synthetic: raise ProtocolError('FT synthetic/real source mismatch')
    if manifest['unit']!=unit: raise ProtocolError('FT unit identity mismatch')
    if not synthetic: authorize_manifest(mp,dry_run=False)
    ids=checked(manifest['ids']['test']).read_text().split()
    if ids!=list(expected_ids): raise ProtocolError('FT test IDs differ from expected population/order')
    done=json.loads((dest/'DONE.json').read_text())
    run_record=json.loads((dest/done['attempt']/'run_record.json').read_text())
    if run_record.get('dry_run') or run_record.get('e0_mode')!='foundation' or run_record.get('variant')!='full' or run_record.get('train_geometry')!='relaxed':
        raise ProtocolError('FT recipe mismatch or CPU smoke used as real unit')
    frame=pd.read_csv(dest/done['attempt']/'test_energies.csv',usecols=['material_id','e_hull_pred_sp'])
    if frame.material_id.duplicated().any() or frame.material_id.tolist()!=list(expected_ids): raise ProtocolError('FT energy coverage/order mismatch')
    score=-frame.e_hull_pred_sp.to_numpy(dtype=float)
    if not np.isfinite(score).all(): raise ProtocolError('nonfinite FT energies')
    return score


def collect_ft_scores(doc, part, design, eval_idx, root, reps, *, synthetic=False, tests=None, include_descriptive=True):
    pos=np.full(part.n,-1,dtype=int);pos[eval_idx]=np.arange(len(eval_idx));scores={}
    budgets=list(doc['families']['F4']['budgets'])
    if design=='P' and include_descriptive: budgets+=[10000]
    for n in budgets:
        selected_reps=[0] if n==10000 else list(reps)
        for bb in doc['families']['F4']['checkpoints']:
            values=np.full((len(selected_reps),len(eval_idx)),np.nan)
            for ri,r in enumerate(selected_reps):
                for f in (range(doc['designs']['P']['n_folds']) if design=='P' else [0]):
                    test=np.flatnonzero(part.fold==f) if design=='P' else tests['S-a']
                    unit=dict(design=design,n=n,fold=f,draw=r,checkpoint=bb,descriptive=n==10000,seed=1000*(f if design=='P' else 5)+r)
                    score=read_unit(root,unit,part.ids[test],synthetic=synthetic)
                    if (pos[test]<0).any() or not np.isnan(values[ri,pos[test]]).all(): raise ProtocolError('FT duplicate or invalid held-out population')
                    values[ri,pos[test]]=score
            if not np.isfinite(values).all(): raise ProtocolError('FT incomplete pooled score matrix')
            scores[f'FT_full_{bb}@{n}']=values
    return scores
