"""Conditional paired inference: F1 IUT up, F1/F2 fixed-ORB down ()."""
from __future__ import annotations

import numpy as np
import multiprocessing as mp
import resource
import time

from .common import ProtocolError, dump
from analysis.metrics import SortedScore, ap
from analysis.inference import cluster_weights, holm, p_up, p_down, interval, interval_finite

BOOT_SCORES = None


def _bootstrap_one(job):
    b, weights = job
    return b, {name: (float(np.mean([s.ap(weights) for s in rows])),
                      float(np.mean([s.auroc(weights) for s in rows]))) for name, rows in BOOT_SCORES.items()}


def contrast_specs(doc, budgets, ft_budgets=()):
    out = {}
    bbs = doc['backbones']['order']
    if doc['families']['F1']['n'] in budgets:
        n = doc['families']['F1']['n']
        out['F1'] = [(f'A5f_{bb}@{n}', 'A5f', bb, n, f'A5f_hist_gbm_{bb}@{n}', 'R0_star') for bb in bbs]
        out['F1'] += [(f'A4_hist_gbm@{n}', 'A4', 'ensemble', n, f'A4_hist_gbm_ensemble@{n}', 'R0_star')]
    f2 = [n for n in doc['families']['F2']['budgets'] if n in budgets]
    if f2:
        out['F2'] = [(f'max@{n}', 'MAX', 'ensemble', n, f'F2_max@{n}', 'R0_star') for n in f2]
    if doc['families']['F3']['n'] in budgets:
        n = doc['families']['F3']['n']
        out['F3'] = [(f'C1_{bb}@{n}', 'C1', bb, n, f'C1_ridge_{bb}@{n}', f'R0_{bb}') for bb in bbs]
    if ft_budgets:
        out['F4'] = [(f'FT_{bb}@{n}', 'FT', bb, n, f'FT_full_{bb}@{n}', f'R0_{bb}')
                     for n in doc['families']['F4']['budgets'] if n in ft_budgets
                     for bb in doc['families']['F4']['checkpoints']]
    return out


def add_maxima(doc, values, budgets):
    # values are already M=mean_r(AP_r) or its bootstrap analogue, never score means.
    result = dict(values)
    result['R0_star'] = np.max([values[f'R0_{b}'] for b in doc['backbones']['order']], axis=0)
    for n in doc['families']['F2']['budgets']:
        if n not in budgets:
            continue
        names = f2_names(doc, n)
        missing = [name for name in names if name not in values]
        if missing:
            raise ProtocolError('F2 missing contender: ' + ', '.join(missing))
        result[f'F2_max@{n}'] = np.max([values[k] for k in names], axis=0)
    return result


def c05_confirmatory(doc):
    verdicts = doc['execution']['C05_A5f_verdicts']
    return set(verdicts) == set(doc['backbones']['order']) and all(v == 'PASS' for v in verdicts.values())


def draw_spread(doc, points, M_r, left, right, n):
    """Spread of the fixed point-estimate winners across draws; no winner switching."""
    if M_r is None:
        return dict(SD_M_r=None, reference_SD_M_r=None, SD_difference_r=None,
                    definition='not supplied (scalar synthetic fixture only)')
    if left.startswith('F2_max@'):
        candidates = f2_names(doc, n)
        left = max(candidates, key=lambda k: points[k])
    if right == 'R0_star':
        right = max((f'R0_{b}' for b in doc['backbones']['order']), key=lambda k: points[k])
    l, r = np.broadcast_arrays(np.asarray(M_r[left]), np.asarray(M_r[right]))
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.
    return dict(SD_M_r=sd(l), reference_SD_M_r=sd(r), SD_difference_r=sd(l-r),
                arm=left, reference=right, definition='sample SD across frozen draw M_r; fixed point-estimate winners')


def f2_names(doc, n):
    names = []
    for method in doc['families']['F2']['contenders']:
        if method.startswith('A5f_'):
            name = 'A5f_hist_gbm_' + method[4:] + f'@{n}'
        elif method.startswith('C1_'):
            name = 'C1_ridge_' + method[3:] + f'@{n}'
        elif method == 'A4_hist_gbm':
            name = f'A4_hist_gbm_ensemble@{n}'
        elif method == 'E0_mean':
            name = 'E0_mean'
        else:
            raise ProtocolError(f'unknown F2 contender {method}')
        names.append(name)
    return names


def read_families(doc, points, stars, budgets, design, *, synthetic=False, M_r=None, allow_incomplete_f3=False):
    rows = []
    fixed = doc['inference']['fixed_down_reference']
    ft_budgets = sorted({int(k.split('@')[1]) for k in points if k.startswith('FT_full_') and int(k.split('@')[1]) in doc['families']['F4']['budgets']})
    for family, specs in contrast_specs(doc, budgets, ft_budgets).items():
        names = [s[0] for s in specs]
        if any(s[4] not in stars or s[5] not in stars for s in specs):
            if family=='F3' and allow_incomplete_f3:
                continue  # Separate incomplete family; never required by P success.
            raise ProtocolError(f'{family}: incomplete contrast score coverage')
        size, m = doc['families'][family]['size'], doc['margin']['AP']
        if len(specs) != size and not synthetic:
            raise ProtocolError(f'{family}: partial family cannot receive real readings')
        if family=='F1' and any(f'R0_{b}' not in stars for b in doc['backbones']['order']):
            raise ProtocolError('F1: all five IUT references required')
        if not synthetic and M_r is None:
            raise ProtocolError('draw-level M_r required beside real readings')
        diffs, downs, pu, pd, components = [], [], [], [], []
        for _, _, _, _, left, right in specs:
            down_ref = fixed if family in ('F1', 'F2') else right
            if down_ref not in stars:
                raise ProtocolError('fixed down reference missing')
            diff = np.asarray(stars[left]) - np.asarray(stars[right])
            down = np.asarray(stars[left]) - np.asarray(stars[down_ref])
            comp = {f'R0_{bb}': p_up(np.asarray(stars[left])-np.asarray(stars[f'R0_{bb}']), m)
                    for bb in doc['backbones']['order']} if family == 'F1' else {}
            arrays = [diff, down] + ([np.asarray(stars[left])-np.asarray(stars[k]) for k in comp])
            if any(not np.isfinite(x).all() for x in arrays):
                raise ProtocolError('nonfinite bootstrap contrast; no reading')
            diffs.append(diff); downs.append(down); components.append(comp)
            pu.append(max(comp.values()) if comp else p_up(diff, m))
            pd.append(p_down(down, m))
        # Full predeclared family even in reduced synthetic checks.
        missing = size-len(specs)
        labels = names + [f'UNRUN_{i}' for i in range(missing)]
        au = holm(pu + [1.]*missing, labels)[:len(specs)]
        ad = holm(pd + [1.]*missing, labels)[:len(specs)]
        for i, (name, arm, bb, n, left, right) in enumerate(specs):
            up, down = au[i] <= doc['inference']['alpha'], ad[i] <= doc['inference']['alpha']
            if up and down:
                raise ProtocolError('both directional readings')
            c05 = c05_confirmatory(doc) or family not in ('F1', 'F2')
            reading = ('DESCRIPTIVE_ADDS' if family == 'F2' else 'ADDS') if up else 'DOES_NOT_EXCEED' if down else 'INCONCLUSIVE'
            if not c05:
                reading = 'DESCRIPTIVE_C05'
            down_ref = fixed if family in ('F1', 'F2') else right
            rows.append(dict(design=design, family=family, contrast=name, arm=arm, checkpoint=bb, n=n,
                model='full' if arm == 'FT' else 'ridge' if arm == 'C1' else 'maximum' if arm == 'MAX' else 'hist_gbm',
                estimate=float(points[left]-points[right]), estimand_reference=right, margin=m,
                estimate_role='descriptive oracle estimate' if right == 'R0_star' else 'matched-reference estimate',
                percentile_band_qualification=('Oracle and contender-maximum percentile bands are descriptive, nominal percentile summaries; '
                    'their coverage can be poor at ties and they must not be presented as calibrated 90 % confidence intervals.')
                    if family in ('F1', 'F2') else None,
                interval=interval(diffs[i], doc['inference']['coverage']),
                down_reference=down_ref, down_estimate=float(points[left]-points[down_ref]),
                down_interval=interval(downs[i], doc['inference']['coverage']),
                p_up_components=components[i], p_up=pu[i], p_down=pd[i],
                p_up_holm=float(au[i]), p_down_holm=float(ad[i]),
                up_role='descriptive only' if family == 'F2' else 'confirmatory' if c05 else 'descriptive C05',
                family_role='confirmatory, exposure-informed' if family in ('F1','F2') and c05 else 'confirmatory' if c05 else 'descriptive C05',
                family_size=size, reading=reading, synthetic_only=synthetic, family_complete=len(specs)==size,
                draw_spread=draw_spread(doc, points, M_r, left, right, n),
                inference_scope='for the five frozen draws, conditional on fitted models'))
    return rows


def primary_complete(doc, rows):
    """Exact P F1/F2 membership, not only a row count or caller-supplied flag."""
    specs = contrast_specs(doc, [doc['families']['F1']['n'], *doc['families']['F2']['budgets']])
    for family in ('F1', 'F2'):
        subset = [x for x in rows if x['design'] == doc['success']['design'] and x['family'] == family]
        expected = {(s[0], s[1], s[2], s[3]) for s in specs[family]}
        actual = {(x['contrast'], x['arm'], x.get('checkpoint'), x['n']) for x in subset}
        if len(subset) != len(expected) or actual != expected or any(not x.get('family_complete', False) for x in subset):
            return False
    return True


def judge(doc, rows, *, synthetic=False, complete=False):
    """option (a): only complete P F1/F2 determine success."""
    rule = doc['success']
    cpu_complete = primary_complete(doc, rows)
    if complete and not synthetic and not cpu_complete:
        raise ProtocolError('real judgment requires complete design-P F1/F2')
    relevant = [x for x in rows if x['design'] == rule['design']]
    lows = {n: [x for x in relevant if x['family'] == rule['low_family'] and x['n'] == n] for n in rule['low_budgets_required']}
    low = all(len(v) == 1 and v[0]['reading'] == rule['low_required_reading'] for v in lows.values())
    high = [x for x in relevant if x['family'] == rule['high_family'] and x['arm'] in rule['high_arms']
            and x.get('model') in rule['high_models'] and x['n'] == doc['families'][rule['high_family']]['n']
            and x['reading'] == rule['high_required_reading']]
    if not rule['conjunction'] or rule['low_budget_operator'] != 'all':
        raise ProtocolError('unsupported success rule: conjunction required')
    primary_rows=[x for x in relevant if x['family'] in ('F1','F2')]
    real_rows=all(x.get('synthetic_only') is False for x in primary_rows)
    eligible = cpu_complete and c05_confirmatory(doc) and (synthetic or real_rows)
    success = eligible and low and len(high) >= rule['high_min_contrasts']
    status = 'DESCRIPTIVE_C05' if not c05_confirmatory(doc) else ('SUCCESS' if success else 'RULE_NOT_MET') if eligible else 'INCOMPLETE'
    return dict(success=bool(success), status=status,
                low_budgets={str(n): bool(len(v)==1 and v[0]['reading']==rule['low_required_reading']) for n,v in lows.items()},
                qualifying_high_contrasts=[x['contrast'] for x in high], eligible=eligible,
                complete=cpu_complete, C05_confirmatory=c05_confirmatory(doc), synthetic_only=synthetic,
                scientific_reading_authorized=bool(eligible and not synthetic),
                completion_scope='design-P F1/F2 only; F3/F4/S/mechanism reported separately')


def infer(doc, scores, y, systems, budgets, replicates, design, B, out, synthetic, workers=1, allow_incomplete_f3=False):
    """scores maps a stable method ID to score rows, one per replicate (R0/ENS once)."""
    started = time.monotonic()
    cpu_started = time.process_time()
    child_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    if not synthetic and (list(replicates) != doc['replicates']['ids'] or B != doc['inference']['B']):
        raise ProtocolError('partial replicates/bootstrap cannot receive real readings')
    _, sys_index = np.unique(systems, return_inverse=True)
    S = len(np.unique(systems))
    sorted_scores, point_rows = {}, {}
    for name, score in scores.items():
        score = np.atleast_2d(score)
        expected = 1 if name.startswith('R0_') or name.startswith('E0_') or (name.startswith('FT_full_') and name.endswith('@10000')) else len(replicates)
        if score.shape != (expected, len(y)) or not np.isfinite(score).all():
            raise ProtocolError(f'incomplete/nonfinite score matrix {name}: {score.shape}')
        sorted_scores[name] = [SortedScore(s, y) for s in score]
        point_rows[name] = [ap(y, s) for s in score]
    points = add_maxima(doc, {k: float(np.mean(v)) for k, v in point_rows.items()}, budgets)
    stars = {k: np.empty(B) for k in sorted_scores}
    auc_stars = {k: np.empty(B) for k in sorted_scores}
    rng = np.random.default_rng(doc['inference']['seed'])  # restart separately per design
    global BOOT_SCORES
    BOOT_SCORES = sorted_scores
    # Parent RNG consumes draws in exactly b order, regardless of completion order.
    jobs = ((b, cluster_weights(rng, sys_index, S)) for b in range(B))
    if not 1 <= workers <= 15:
        raise ProtocolError('bootstrap workers must be 1..15, one compute thread each')
    pool = mp.get_context('fork').Pool(workers) if workers > 1 else None
    try:
        results = pool.imap_unordered(_bootstrap_one, jobs, chunksize=1) if pool else map(_bootstrap_one, jobs)
        for completed, (b, values) in enumerate(results, 1):
            for name, (value, auc) in values.items():
                stars[name][b] = value
                auc_stars[name][b] = auc
            if completed % 20 == 0:
                print(f'bootstrap {design}: {completed}/{B}, {time.monotonic()-started:.1f}s', flush=True)
        if pool:
            pool.close()
    except BaseException:
        if pool:
            pool.terminate()
        raise
    finally:
        if pool:
            pool.join()
        BOOT_SCORES = None
    stars = add_maxima(doc, stars, budgets)
    M_r = {k: v*len(replicates) if k.startswith(('R0_','E0_')) else v for k,v in point_rows.items()}
    rows = read_families(doc, points, stars, budgets, design, synthetic=synthetic, M_r=M_r, allow_incomplete_f3=allow_incomplete_f3)
    np.savez_compressed(out / f'bootstrap_{design}.npz', **stars)
    np.savez_compressed(out / f'bootstrap_AUROC_{design}.npz', **auc_stars)
    descriptive = []
    for name in scores:
        ap_ci = interval(stars[name], doc['inference']['coverage'])
        auc_ci, undefined = interval_finite(auc_stars[name], doc['inference']['coverage'])
        descriptive.append(dict(method=name, AP=points[name], AP_interval=ap_ci,
             AUROC_interval=auc_ci if np.isfinite(auc_ci).all() else None,
             undefined_AUROC_draws=undefined, readings=False))
        if name.startswith('R0_'):
            continue
        matched = [b for b in doc['backbones']['order'] if name.split('@')[0].endswith('_' + b)]
        for bb in matched or doc['backbones']['order']:
            reference = f'R0_{bb}'
            auc_ci, undefined = interval_finite(auc_stars[name]-auc_stars[reference], doc['inference']['coverage'])
            descriptive.append(dict(method=name, reference=reference, contrast='matched_reference' if matched else 'each_reference',
                 AP_difference=float(points[name]-points[reference]),
                 AP_interval=interval(stars[name]-stars[reference],doc['inference']['coverage']),
                 AUROC_interval=auc_ci if np.isfinite(auc_ci).all() else None,
                 undefined_AUROC_draws=undefined, readings=False))
    dump(out / f'descriptive_intervals_{design}.json', descriptive)
    dump(out / f'inference_{design}.json', dict(design=design, B=B, seed=doc['inference']['seed'], systems=S,
         points=points, M_r=M_r,
         replicate_ids=list(replicates), deterministic_references='same metric in each replicate',
         readings=rows, synthetic_only=synthetic,
         family_completion={f:('COMPLETE' if sum(r['family']==f for r in rows)==doc['families'][f]['size'] else 'INCOMPLETE') for f in ('F1','F2','F3','F4')},
         bootstrap_workers=workers, bootstrap_seconds=time.monotonic()-started,
         bootstrap_cpu_seconds=(time.process_time()-cpu_started + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
             + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-child_started.ru_utime-child_started.ru_stime)))
    return rows, time.monotonic()-started
