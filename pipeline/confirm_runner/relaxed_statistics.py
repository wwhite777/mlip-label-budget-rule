"""F5/F6 extension; unchanged metric, cluster RNG, tails and Holm machinery."""
from pathlib import Path
import multiprocessing as mp
import time
import numpy as np

from .common import ProtocolError, dump
from . import statistics as frozen
from analysis.metrics import SortedScore, ap
from analysis.inference import cluster_weights, p_up, p_down, holm, interval, interval_finite


def references(doc):
    return doc['amendment_A2']['comparator']['scores']


def fitted_name(arm, bb, n):
    return f'{arm}_hist_gbm_{bb}@{n}'


def f6_names(doc, n):
    return [fitted_name('A5fR',b,n) for b in doc['backbones']['order']] + [fitted_name('A4R','ensemble',n), 'ENS-R']


def add_maxima(doc, values, budgets):
    out = dict(values)
    out['LF_max'] = np.max([values[k] for k in references(doc)],axis=0)
    for n in doc['amendment_A2']['families']['F6']['budgets']:
        if n in budgets:
            out[f'F6_max@{n}'] = np.max([values[k] for k in f6_names(doc,n)],axis=0)
    return out


def contrast_specs(doc, budgets):
    a=doc['amendment_A2']; result={}
    if a['families']['F5']['n'] in budgets:
        n=a['families']['F5']['n']
        result['F5']=[(f'A5fR_{b}@{n}','A5fR',b,n,fitted_name('A5fR',b,n)) for b in doc['backbones']['order']]
        result['F5'].append((f'A4R@{n}','A4R','ensemble',n,fitted_name('A4R','ensemble',n)))
    ns=[n for n in a['families']['F6']['budgets'] if n in budgets]
    if ns:
        result['F6']=[(f'max@{n}','MAX','ensemble',n,f'F6_max@{n}') for n in ns]
    return result


def c05_confirmatory(doc):
    a=doc['amendment_A2']['C05']
    return (set(a['verdicts'])==set(doc['backbones']['order']) and
            all(v=='PASS' for v in a['verdicts'].values()) and a['A4R_verdict']=='PASS')


def spread(doc, points, M_r, left):
    if M_r is None:
        return None
    if left.startswith('F6_max@'):
        left=max(f6_names(doc,int(left.split('@')[1])),key=lambda k:points[k])
    right=max(references(doc),key=lambda k:points[k])
    fixed=doc['amendment_A2']['comparator']['fixed_down_reference']
    l,r,d=np.broadcast_arrays(M_r[left],M_r[right],M_r[fixed])
    sd=lambda x:float(np.std(x,ddof=1)) if len(x)>1 else 0.
    return dict(arm=left,reference=right,SD_M_r=sd(l),reference_SD_M_r=sd(r),SD_difference_r=sd(l-r),
                down_reference=fixed,down_reference_SD_M_r=sd(d),SD_down_difference_r=sd(l-d),
                definition='sample SD across draws; maxima use fixed point-estimate winners')


def read_families(doc, points, stars, budgets, design, *, synthetic=False, M_r=None):
    a=doc['amendment_A2']; inf=a['inference']; m=inf['margin_AP']
    fixed=a['comparator']['fixed_down_reference']; rows=[]
    if not synthetic and M_r is None:
        raise ProtocolError('draw-level M_r required for real F5/F6 readings')
    for family,specs in contrast_specs(doc,budgets).items():
        size=a['families'][family]['size']
        if len(specs)!=size and not synthetic:
            raise ProtocolError('partial family cannot receive real readings')
        ups=[]; downs=[]; components=[]; diffs=[]; dd=[]
        for _,_,_,_,left in specs:
            required=[left,'LF_max',fixed,*references(doc)]
            if any(k not in stars or k not in points for k in required):
                raise ProtocolError('incomplete twelve-reference/contender coverage')
            comp={k:p_up(np.asarray(stars[left])-stars[k],m) for k in references(doc)}
            diff=np.asarray(stars[left])-stars['LF_max']; down=np.asarray(stars[left])-stars[fixed]
            if any(not np.isfinite(stars[k]).all() for k in required):
                raise ProtocolError('nonfinite bootstrap contrast')
            ups.append(max(comp.values()) if family=='F5' else p_up(diff,m))
            downs.append(p_down(down,m)); components.append(comp if family=='F5' else {})
            diffs.append(diff); dd.append(down)
        names=[s[0] for s in specs]+[f'UNRUN_{i}' for i in range(size-len(specs))]
        au=holm(ups+[1.]*(size-len(specs)),names)[:len(specs)]
        ad=holm(downs+[1.]*(size-len(specs)),names)[:len(specs)]
        for i,(name,arm,bb,n,left) in enumerate(specs):
            up,down=au[i]<=inf['alpha'],ad[i]<=inf['alpha']
            if up and down:
                raise ProtocolError('both directional readings')
            reading=('ADDS' if family=='F5' else 'DESCRIPTIVE_ADDS') if up else 'DOES_NOT_EXCEED' if down else 'INCONCLUSIVE'
            if not c05_confirmatory(doc):
                reading='DESCRIPTIVE_C05'
            rows.append(dict(design=design,family=family,contrast=name,arm=arm,checkpoint=bb,n=n,
                model='maximum' if arm=='MAX' else 'hist_gbm',estimate=float(points[left]-points['LF_max']),
                estimand_reference='LF_max',estimate_role='descriptive oracle estimate',margin=m,
                interval=interval(diffs[i],inf['coverage']),down_reference=fixed,
                down_estimate=float(points[left]-points[fixed]),down_interval=interval(dd[i],inf['coverage']),
                p_up_components=components[i],p_up=ups[i],p_down=downs[i],p_up_holm=float(au[i]),p_down_holm=float(ad[i]),
                up_role='confirmatory, exposure-informed' if family=='F5' and c05_confirmatory(doc) else 'descriptive only',
                family_role='confirmatory, exposure-informed' if c05_confirmatory(doc) else 'descriptive C05',
                family_size=size,family_complete=len(specs)==size,reading=reading,synthetic_only=synthetic,
                draw_spread=spread(doc,points,M_r,left),
                percentile_band_qualification='Oracle/contender-maximum bands are descriptive nominal percentile summaries; not calibrated confidence intervals.',
                inference_scope='conditional on fitted models and frozen draws'))
    return rows


def primary_complete(doc, rows):
    a=doc['amendment_A2'];specs=contrast_specs(doc,a['budgets'])
    for family in ('F5','F6'):
        actual=[x for x in rows if x['design']=='P' and x['family']==family]
        expected={(s[0],s[1],s[2],s[3]) for s in specs[family]}
        found={(x['contrast'],x['arm'],x['checkpoint'],x['n']) for x in actual}
        expected_model='maximum' if family=='F6' else 'hist_gbm'
        if found!=expected or len(actual)!=len(expected) or any(not x['family_complete'] or x.get('model')!=expected_model for x in actual):
            return False
    return True


def judge(doc, rows, *, synthetic=False):
    a=doc['amendment_A2'];rule=a['extension_rule'];complete=primary_complete(doc,rows)
    relevant=[x for x in rows if x['design']==rule['design']]
    lows={n:[x for x in relevant if x['family']=='F6' and x['n']==n] for n in rule['low_budgets_required']}
    low={str(n):len(v)==1 and v[0]['reading']==rule['low_required_reading'] for n,v in lows.items()}
    highs=[x['contrast'] for x in relevant if x['family']=='F5' and x['arm'] in rule['high_arms'] and x['reading']==rule['high_required_reading']]
    eligible=complete and c05_confirmatory(doc) and (synthetic or all(x.get('synthetic_only') is False for x in relevant))
    success=eligible and all(low.values()) and len(highs)>=rule['high_min_contrasts']
    return dict(success=bool(success),status=('SUCCESS' if success else 'RULE_NOT_MET') if eligible else 'DESCRIPTIVE_C05' if not c05_confirmatory(doc) else 'INCOMPLETE',
                eligible=eligible,complete=complete,low_budgets=low,qualifying_high_contrasts=highs,
                synthetic_only=synthetic,scientific_reading_authorized=bool(eligible and not synthetic),
                rule='separate extension only; original v2.3.1 rule and readings unchanged')


def infer(doc, scores, y, systems, budgets, reps, design, B, out, *, synthetic, workers=1):
    started=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a=doc['amendment_A2'];inf=a['inference']; refs=references(doc)
    if not synthetic and (list(reps)!=a['replicates'] or B!=inf['B'] or list(budgets)!=a['budgets']):
        raise ProtocolError('real F5/F6 requires full budgets, draws and B')
    if not 1<=workers<=10:
        raise ProtocolError('at most ten single-thread workers')
    if not set(refs)<=set(scores):
        raise ProtocolError('all twelve references required')
    sorted_scores={};point_rows={}
    for name,score in scores.items():
        score=np.atleast_2d(score)
        expected=1 if name in refs else len(reps)
        if score.shape!=(expected,len(y)) or not np.isfinite(score).all():
            raise ProtocolError('incomplete/nonfinite score matrix: '+name)
        sorted_scores[name]=[SortedScore(s,y) for s in score]
        point_rows[name]=[ap(y,s) for s in score]
    points=add_maxima(doc,{k:float(np.mean(v)) for k,v in point_rows.items()},budgets)
    stars={k:np.empty(B) for k in scores};auc={k:np.empty(B) for k in scores}
    _,si=np.unique(systems,return_inverse=True);S=len(np.unique(systems))
    rng=np.random.default_rng(inf['seed'])  # restart for each design, unchanged b-order
    frozen.BOOT_SCORES=sorted_scores
    jobs=((b,cluster_weights(rng,si,S)) for b in range(B))
    pool=mp.get_context('fork').Pool(workers) if workers>1 else None
    try:
        results=pool.imap_unordered(frozen._bootstrap_one,jobs,chunksize=1) if pool else map(frozen._bootstrap_one,jobs)
        for completed,(b,values) in enumerate(results,1):
            for name,(value,av) in values.items():
                stars[name][b]=value;auc[name][b]=av
            if completed%50==0:
                print(f'CPU-R bootstrap {design}: {completed}/{B}',flush=True)
        if pool: pool.close()
    except BaseException:
        if pool: pool.terminate()
        raise
    finally:
        if pool: pool.join()
        frozen.BOOT_SCORES=None
    stars=add_maxima(doc,stars,budgets)
    mr={k:v*len(reps) if k in refs else v for k,v in point_rows.items()}
    rows=read_families(doc,points,stars,budgets,design,synthetic=synthetic,M_r=mr)
    np.savez_compressed(out/f'bootstrap_{design}.npz',**stars)
    np.savez_compressed(out/f'bootstrap_AUROC_{design}.npz',**auc)
    descriptive=[]
    for name in scores:
        ai,missing=interval_finite(auc[name],inf['coverage'])
        descriptive.append(dict(method=name,AP=points[name],M_r=point_rows[name],
            SD_M_r=float(np.std(point_rows[name],ddof=1)) if len(point_rows[name])>1 else 0.,
            AP_interval=interval(stars[name],inf['coverage']),AUROC_interval=ai if np.isfinite(ai).all() else None,
            undefined_AUROC_draws=missing,readings=False))
    dump(out/f'descriptive_intervals_{design}.json',descriptive)
    dump(out/f'inference_{design}.json',dict(design=design,B=B,seed=inf['seed'],systems=S,points=points,M_r=mr,
        replicate_ids=list(reps),readings=rows,synthetic_only=synthetic,seconds=time.monotonic()-started,
        family_completion={f:'COMPLETE' if sum(r['family']==f for r in rows)==a['families'][f]['size'] else 'INCOMPLETE' for f in ('F5','F6')}))
    return rows
