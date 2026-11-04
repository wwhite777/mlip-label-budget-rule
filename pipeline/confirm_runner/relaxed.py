"""CPU-R appended stage. Real entry is gated before inputs or CONFIRM labels."""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
import json
import multiprocessing as mp
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import HERE, ROOT, ProtocolError, dump, checked, sha256, load_yaml
from .inputs import load_inputs, campaign_assignments
from .labels import read_labels as frozen_read_labels, Labels, LabelAccessError, FORMATION, HULL
from .config import V2Config
from . import runner as frozen
from . import relaxed_statistics as stats
from .stages import finish_stage
from analysis.data import build_part

YAML24=HERE/'protocol_v2.4.yaml'
FIXED=dict(learning_rate=.05,max_leaf_nodes=31)
G=DEST=SIGNATURE=None


def read_labels(doc,part,*,synthetic=False,yaml_path=YAML24,root=ROOT):
    if synthetic:
        return frozen_read_labels(doc,part,synthetic=True)
    from integrity import verify_protocol
    verify_protocol(yaml_path=yaml_path, root=root)
    if part.part!='CONFIRM':
        raise LabelAccessError('non-CONFIRM population cannot authorize real labels')
    path=checked(doc['population']['summary'],root)
    ids=pd.read_csv(path,usecols=['material_id'],dtype=str,keep_default_na=False).material_id.to_numpy()
    selected=pd.Index(part.ids).get_indexer(ids)>=0
    if len(ids[selected])!=part.n or set(ids[selected])!=set(part.ids):
        raise LabelAccessError('summary label IDs do not cover population exactly')
    lines=set((np.flatnonzero(selected)+1).tolist())
    frame=pd.read_csv(path,usecols=['material_id',FORMATION,HULL],dtype=str,keep_default_na=False,
                      skiprows=lambda i:i!=0 and i not in lines)
    if frame.material_id.duplicated().any():
        raise LabelAccessError('duplicate label IDs')
    frame=frame.set_index('material_id').loc[part.ids]
    for column in (FORMATION,HULL): frame[column]=pd.to_numeric(frame[column],errors='raise')
    frame['stable']=(frame[HULL]<=doc['labels']['stability']['threshold_eV_atom']).astype(np.int8)
    return Labels(frame,part,dict(source='REAL_POST_FREEZE_CPU_R',label_columns_opened=[FORMATION,HULL],
                  real_label_rows_parsed=part.n,protocol_sha256=sha256(yaml_path)))


def ranked_score(hulls):
    score=-np.asarray(hulls,dtype=np.float64)
    if np.isinf(score).any():
        raise ProtocolError('infinite relaxed value')
    finite=np.isfinite(score)
    if not finite.any():
        return np.zeros_like(score)
    floor=np.nextafter(score[finite].min(),-np.inf)
    if not np.isfinite(floor):
        raise ProtocolError('missing rank floor underflow')
    return np.where(finite,score,floor)


def load_relaxed(doc,inp):
    cols=[];report={}
    for b in doc['backbones']['order']:
        specs=doc['amendment_A2']['relaxed_inputs']
        path=checked(specs[b+'.csv.gz']);meta=checked(specs[b+'.json'])
        record=json.loads(meta.read_text())
        if record['output_sha256']!=sha256(path):
            raise ProtocolError('relaxed provenance hash mismatch')
        f=pd.read_csv(path,usecols=['material_id','e_hull_pred_is2re'],dtype={'material_id':str})
        if f.material_id.duplicated().any():
            raise ProtocolError('duplicate relaxed IDs')
        h=f.set_index('material_id').reindex(inp.part.ids).e_hull_pred_is2re.to_numpy(dtype=np.float64)
        if np.isinf(h).any():
            raise ProtocolError('infinite relaxed input')
        cols.append(h);report[b]=dict(rows=len(h),missing=int(np.isnan(h).sum()),sha256=sha256(path))
    inp.relaxed=np.column_stack(cols)
    return report


def reference_scores(inp,idx,doc):
    bbs=doc['backbones']['order']
    return {**{f'R0_{b}':ranked_score(inp.hulls[idx,j])[None,:] for j,b in enumerate(bbs)},
            **{f'R0R_{b}':ranked_score(inp.relaxed[idx,j])[None,:] for j,b in enumerate(bbs)},
            'ENS':ranked_score(inp.hulls[idx].mean(axis=1))[None,:],
            'ENS-R':ranked_score(inp.relaxed[idx].mean(axis=1))[None,:]}


def prepare_context(inp,doc,design,dest,train=None,tests=None,signature='fixture'):
    global G,DEST,SIGNATURE
    G,DEST,SIGNATURE=inp,Path(dest),signature
    frozen.setup_context(inp,doc,design,DEST,train,tests)


def jobs_for(doc,budgets,reps,design,*,descriptive=True):
    methods=[('A5fR',b) for b in doc['backbones']['order']]+[('A4R','ensemble')]
    if descriptive:
        methods += [('A5fRS',b) for b in doc['backbones']['order']]
    return [(a,b,n,f,r) for n in budgets for a,b in methods for r in reps
            for f in (range(doc['designs']['P']['n_folds']) if design=='P' else (0,))]


def run_unit(job):
    arm,bb,n,f,r=job
    draw,tr,va,test=frozen.split(n,f,r)
    path=DEST/'units'/f'{arm}__{bb}__n{n}__f{f}__r{r}.npz'
    identity=dict(arm=arm,checkpoint=bb,n=n,fold=f,draw=r,context_signature=SIGNATURE)
    if path.exists() or path.with_suffix('.json').exists():
        info=json.loads(path.with_suffix('.json').read_text())
        if any(info.get(k)!=v for k,v in identity.items()) or info.get('npz_sha256')!=sha256(path):
            raise ProtocolError('stored CPU-R unit identity/hash drift')
        validate_unit(path,job)
        return info
    started=time.monotonic()
    if arm=='A4R':
        X=np.column_stack([G.relaxed,G.comp])
    elif arm in ('A5fR','A5fRS'):
        X=np.column_stack([G.feats[bb].astype(np.float64),G.relaxed] if arm=='A5fR'
                          else [G.feats[bb].astype(np.float64),G.hulls,G.relaxed])
    else:
        raise ProtocolError('unknown CPU-R arm')
    G.labels.active_draw=set(G.part.ids[draw])
    view=lambda idx:G.labels.lookup(G.part.ids[idx],allowed_ids=G.part.ids[draw]).stable.to_numpy()
    try:
        with threadpool_limits(1):
            if arm=='A4R':
                result,flags=frozen.ARMS.binary_select(['hist_gbm'],{'hist_gbm':X},view,tr,va,draw,test)
                score=result['scores']['hist_gbm'];chosen=result['chosen']['hist_gbm'];records=result['records']
            else:
                y=view(draw);flags=[];chosen=FIXED;records=[]
                if np.unique(y).size==1:
                    score=np.full(len(test),float(y[0]));flags=['draw_single_class_constant_score']
                else:
                    score=frozen.PILOT.models.make('hist_gbm',FIXED,G.cfg.y,'cpu').fit(X[draw],y).score(X[test])
    finally:
        G.labels.active_draw=None
    if not np.isfinite(score).all():
        raise ProtocolError('nonfinite CPU-R score')
    np.savez_compressed(path,test_idx=test,material_id=G.part.ids[test],score=score,draw_idx=draw,
                        draw_material_id=G.part.ids[draw],inner_train_idx=tr,inner_train_material_id=G.part.ids[tr],
                        inner_val_idx=va,inner_val_material_id=G.part.ids[va])
    info=dict(identity,model='hist_gbm',chosen=chosen,records=records,flags=flags,
              fit_label_count=len(draw),input_dim=X.shape[1],inner_selection_used=arm=='A4R',
              fit_partition=G.part.part,label_source=G.labels.trace['source'],seconds=time.monotonic()-started,
              npz_sha256=sha256(path))
    dump(path.with_suffix('.json'),info)
    return info


def validate_unit(path,job):
    _,_,n,f,r=job;draw,tr,va,test=frozen.split(n,f,r)
    with np.load(path,allow_pickle=False) as z:
        for key,expected in [('test_idx',test),('draw_idx',draw),('inner_train_idx',tr),('inner_val_idx',va),
                             ('material_id',G.part.ids[test]),('draw_material_id',G.part.ids[draw]),
                             ('inner_train_material_id',G.part.ids[tr]),('inner_val_material_id',G.part.ids[va])]:
            if not np.array_equal(z[key],expected):
                raise ProtocolError('stored draw/split/row drift: '+key)
        score=z['score'].copy()
    if score.shape!=(len(test),) or not np.isfinite(score).all():
        raise ProtocolError('incomplete stored score')
    return test,score


def collect(doc,budgets,reps,design):
    idx=np.arange(G.part.n) if design=='P' else frozen.TESTS['S-a']
    pos=np.full(G.part.n,-1,dtype=int);pos[idx]=np.arange(len(idx))
    scores=reference_scores(G,idx,doc)
    for a,b,n,f,r in jobs_for(doc,budgets,reps,design):
        name=stats.fitted_name(a,b,n)
        if name not in scores:
            scores[name]=np.full((len(reps),len(idx)),np.nan)
        path=DEST/'units'/f'{a}__{b}__n{n}__f{f}__r{r}.npz'
        test,score=validate_unit(path,(a,b,n,f,r))
        if (pos[test]<0).any() or not np.isnan(scores[name][reps.index(r),pos[test]]).all():
            raise ProtocolError('overlapping or invalid held-out scores')
        scores[name][reps.index(r),pos[test]]=score
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ProtocolError('incomplete pooled CPU-R predictions')
    return idx,scores


def metric_rows(inp,variant,idx,scores):
    y=inp.labels.lookup(inp.part.ids[idx]).stable.to_numpy();rows=[]
    from analysis.metrics import ap,auroc
    for name,values in scores.items():
        for r,score in enumerate(values):
            rows.append(dict(design=variant,method=name,replicate=-1 if name in ('ENS','ENS-R') or name.startswith(('R0_','R0R_')) else r,
                N=len(y),positives=int(y.sum()),prevalence=float(y.mean()),AP=ap(y,score),
                AUROC=auroc(y,score) if len(np.unique(y))==2 else None,
                precision_top5=frozen.precision(y,score,inp.part.ids[idx],inp.part.fold[idx] if variant=='P' else None)))
    return rows


def execute(doc,inp,out,*,synthetic,workers=10,reduced=False,designs=('P','S'),signature='fixture'):
    if not 1<=workers<=10:
        raise ProtocolError('at most ten single-thread compute workers')
    if reduced and not synthetic:
        raise ProtocolError('reduced execution is synthetic only')
    a=doc['amendment_A2'];budgets=[300,10000] if reduced else a['budgets']
    reps=[0,1] if reduced else a['replicates'];B=200 if reduced else a['inference']['B']
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];metrics=[];completed=0
    for design in designs:
        if design=='S':
            if hasattr(inp,'synthetic_campaign'):
                train,tests=inp.synthetic_campaign
            else:
                train,tests,_=campaign_assignments(doc,inp.part)
        else:
            train=tests=None
        dest=out/design;prepare_context(inp,doc,design,dest,train,tests,signature)
        jobs=jobs_for(doc,budgets,reps,design)
        for n in budgets:
            for r in reps:
                for f in (range(5) if design=='P' else (0,)):
                    frozen.split(n,f,r)
        infos=[]
        with mp.get_context('fork').Pool(workers) as pool:
            for info in pool.imap_unordered(run_unit,jobs,chunksize=1):
                infos.append(info)
                if len(infos)%20==0 or len(infos)==len(jobs):
                    print(f'CPU-R fits {design}: {len(infos)}/{len(jobs)}',flush=True)
        completed+=len(infos);dump(dest/'units.json',sorted(infos,key=lambda x:(x['arm'],x['checkpoint'],x['n'],x['fold'],x['draw'])))
        idx,scores=collect(doc,budgets,reps,design)
        for variant in (('P',) if design=='P' else ('S-a','S-b')):
            mask=np.ones(len(idx),dtype=bool) if variant in ('P','S-a') else np.isin(idx,tests['S-b'])
            vidx=idx[mask];vs={k:v[:,mask] for k,v in scores.items()}
            metrics.extend(metric_rows(inp,variant,vidx,vs))
            y=inp.labels.lookup(inp.part.ids[vidx]).stable.to_numpy()
            rows.extend(stats.infer(doc,vs,y,inp.part.chemsys[vidx],budgets,reps,variant,B,dest,synthetic=synthetic,workers=workers))
        dump(out/'readings.json',rows);dump(out/'judge.json',stats.judge(doc,rows,synthetic=synthetic))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    report=dict(status='PASS',stage='CPU-R',synthetic_only=synthetic,reduced=reduced,
                execution_complete=True,all_requested_cpu_units_complete=True,completed_units=completed,
                designs=list(designs),budgets=budgets,replicates=reps,B=B,N=inp.part.n,
                real_label_columns_opened=inp.labels.trace['label_columns_opened'],label_trace=inp.labels.trace,
                judge=stats.judge(doc,rows,synthetic=synthetic),original_v231_outputs_changed=False,
                implementation_binding_sha256=sha256(HERE/'implementation_v24.json'),
                stage_code_sha256=sha256(Path(__file__)),statistics_code_sha256=sha256(Path(stats.__file__)),
                environment=doc['amendment_A2']['environment'])
    dump(out/'run.json',report)
    return report


def authorize_real(doc,stage_root,*,yaml_path=YAML24,root=ROOT):
    from integrity import verify_protocol
    from .stages import require_previous
    result=verify_protocol(yaml_path=yaml_path,root=root)
    require_previous(stage_root,'CPU-R',yaml_path=HERE/'protocol.yaml',root=root)
    return result


def validate_output_root(out_root,stage_root):
    if stage_root is not None and Path(out_root).resolve().is_relative_to(Path(stage_root).resolve()):
        raise ProtocolError('CPU-R out-root must be separate from the frozen stage root')


def reduce_synthetic_population(doc,inp):
    # Exact frozen full-population metadata is read first, without labels.
    train,tests,_=campaign_assignments(doc,inp.part)
    keep=np.arange(inp.part.n)[::5]
    ids=inp.part.ids[keep]
    part=build_part('CONFIRM',ids,inp.part.chemsys[keep],inp.part.fold[keep],inp.part.formula[keep],inp.part.part_of)
    def sub(idx): return part.index_of(inp.part.ids[np.intersect1d(keep,idx)])
    campaign=(sub(train),{v:sub(idx) for v,idx in tests.items()})
    return SimpleNamespace(part=part,comp=inp.comp[keep],hulls=inp.hulls[keep],relaxed=inp.relaxed[keep],
        feats={b:x[keep] for b,x in inp.feats.items()},synthetic_campaign=campaign,coverage=inp.coverage,
        reduction=dict(rule='every fifth canonical row; metadata and inputs only',full_N=inp.part.n,N=part.n,
                       training_n=len(campaign[0]),test_n={v:len(x) for v,x in campaign[1].items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--yaml',type=Path,default=YAML24)
    p.add_argument('--stage',choices=['CPU-R'],default='CPU-R')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--stage-root',type=Path)
    p.add_argument('--synthetic-labels',action='store_true')
    p.add_argument('--reduced',action='store_true')
    p.add_argument('--workers',type=int,default=10)
    args=p.parse_args()
    if not 1<=args.workers<=10: p.error('workers must be 1..10, one compute thread each')
    if args.out_root.name!='CPU-R': p.error('--out-root must end in CPU-R, in a new extension output root')
    validate_output_root(args.out_root,args.stage_root)
    if args.reduced and not args.synthetic_labels: p.error('--reduced requires synthetic labels')
    from extension_gate import validate_document
    doc=load_yaml(args.yaml)
    from importlib.metadata import version
    if any(version(k)!=v for k,v in doc['amendment_A2']['environment']['packages'].items()):
        raise ProtocolError('light environment package version drift')
    if not args.synthetic_labels:
        if args.stage_root is None: p.error('real CPU-R requires predecessor --stage-root')
        authorize_real(doc,args.stage_root,yaml_path=args.yaml)
    else:
        validate_document(doc,yaml_path=args.yaml,verify_files=False)
    inp=load_inputs(doc);coverage=load_relaxed(doc,inp)
    if args.reduced: inp=reduce_synthetic_population(doc,inp)
    inp.labels=read_labels(doc,inp.part,synthetic=args.synthetic_labels,yaml_path=args.yaml)
    dump(args.out_root/'input_coverage.json',dict(single_point=inp.coverage,relaxed=coverage,
         reduction=getattr(inp,'reduction',None),real_labels_opened=not args.synthetic_labels))
    report=execute(doc,inp,args.out_root,synthetic=args.synthetic_labels,workers=args.workers,reduced=args.reduced,
                   signature=sha256(args.yaml)+':'+('SYNTHETIC-20261022' if args.synthetic_labels else 'PUBLIC-REPRODUCTION'))
    from extension_gate import finish_extension
    marker=finish_extension(args.out_root,args.stage_root,synthetic=args.synthetic_labels,yaml_path=args.yaml)
    print(marker['manifest_line'],flush=True)
    print(report['status'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
