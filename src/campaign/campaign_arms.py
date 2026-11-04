"""campaign: campaign adapter around byte-identical arms-a3 arm/selection code.

One fit per arm/checkpoint/n/draw scores S-a; S-b is the preassigned subset.
Only the sampler, fixed test population, and reporting differ from arms-a3.
"""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_arms as a
import pilot_arms as p
from analysis.draws import inner_split

OUT = Path(__file__).resolve().parent
BUDGETS = (10000, 3000, 1000, 300)
ARMS = ('C1', 'A1', 'A4', 'A5')
KEEP = {'C1': ('ridge',), 'A1': ('best', 'hist_gbm'),
        'A4': ('best', 'logistic', 'hist_gbm'), 'A5': ('hist_gbm',)}
TRAIN = TEST = None
ASSIGNMENTS = {}


def campaign_draw(part, train, n, r):
    """Sorted systems -> seeded permutation; canonical ids within each system.

    Concatenate whole train-pool systems until >=n, truncate the last system's
    canonical id list. The seed is exactly 500000 + 1000*n + r.
    """
    train = np.asarray(train, dtype=np.int64)
    if n < 1 or n > len(train):
        raise ValueError('budget exceeds train pool or is nonpositive')
    canonical = train[np.argsort(part.ids[train], kind='mergesort')]
    systems = np.unique(part.chemsys[canonical])
    order = np.random.default_rng(500000 + 1000*n + r).permutation(systems)
    chunks, count = [], 0
    for system in order:
        members = canonical[part.chemsys[canonical] == system]
        chunks.append(members)
        count += len(members)
        if count >= n:
            break
    return np.concatenate(chunks)[:n]


def validate_assignments(frames, part, expected=True):
    """Validate only ids, part, systems, rounds, assignments before label use."""
    for variant, frame in frames.items():
        if frame.material_id.duplicated().any() or set(frame.material_id) != set(part.ids):
            raise ValueError('campaign must cover unique PILOT ids exactly once')
        p.assert_pilot_ids(frame.material_id, part.part_of)
        if set(frame.part) != {'PILOT'} or set(frame.variant) != {variant}:
            raise ValueError('wrong campaign part/variant')
        idx = part.index_of(frame.material_id.to_numpy())
        if not np.array_equal(frame.chemsys, part.chemsys[idx]):
            raise ValueError('campaign chemical-system mismatch')
        rounds = frame.material_id.map(lambda mid: int(mid.split('-')[1]))
        if not np.array_equal(rounds, frame['round']) or not set(rounds) <= {1,2,3,4,5}:
            raise ValueError('campaign round mismatch')
        train = frame.assignment == 'train'
        test = frame.assignment == 'test'
        if not np.array_equal(train, rounds.isin([1,2])):
            raise ValueError('training must be all rounds 1/2')
        if set(frame.loc[train, 'material_id']) & set(frame.loc[test, 'material_id']):
            raise ValueError('train/test id overlap')
        shared = frame.chemsys.isin(set(frame.loc[train, 'chemsys']))
        want_test = ~train if variant == 'S-a' else ~train & ~shared
        if not np.array_equal(test, want_test):
            raise ValueError('incorrect campaign test assignment')
        if variant == 'S-b' and set(frame.loc[train,'chemsys']) & set(frame.loc[test,'chemsys']):
            raise ValueError('S-b systems overlap')
        if expected and (train.sum(), test.sum()) != (19877, 22922 if variant == 'S-a' else 8735):
            raise ValueError('unexpected campaign counts')
    train_a = set(frames['S-a'].loc[frames['S-a'].assignment == 'train','material_id'])
    train_b = set(frames['S-b'].loc[frames['S-b'].assignment == 'train','material_id'])
    if train_a != train_b:
        raise ValueError('variants must share the train pool')


def setup():
    global TRAIN, TEST, ASSIGNMENTS
    # Load and validate label-free campaigns before the arms PILOT-only reader.
    cfg, universe, partition = p.load_metadata()
    part = p.load_pilot_part(cfg, universe, partition)
    ASSIGNMENTS = {v: pd.read_csv(p.pilot / f'campaign_pilot_{v.lower()}.csv.gz',
                         keep_default_na=False) for v in ('S-a', 'S-b')}
    validate_assignments(ASSIGNMENTS, part)
    p.G = p.load_inputs()
    frame = ASSIGNMENTS['S-a']
    TRAIN = p.G.part.index_of(frame.loc[frame.assignment == 'train','material_id'].to_numpy())
    TEST = p.G.part.index_of(frame.loc[frame.assignment == 'test','material_id'].to_numpy())
    p.split = split


def split(n, f, r):
    if f != 0:
        raise ValueError('one campaign split; frozen inner rule uses fold=0')
    draw = campaign_draw(p.G.part, TRAIN, n, r)
    tr, va = inner_split(p.G.cfg, p.G.part, draw, 0, r)
    if set(draw) & set(TEST):
        raise AssertionError('train/test id overlap')
    return draw, tr, va, TEST


def precision(y, scores, ids):
    k = int(np.ceil(0.05 * len(y)))
    order = np.lexsort((np.asarray(ids).astype(str), -np.asarray(scores)))
    return float(np.asarray(y)[order[:k]].sum() / k)


def evaluate(arm, model, bb, n, r, score, test, source, chosen, flags):
    rows = []
    for variant, frame in ASSIGNMENTS.items():
        wanted = set(frame.loc[frame.assignment == 'test','material_id'])
        mask = np.array([mid in wanted for mid in p.G.part.ids[test]])
        idx = test[mask]
        if len(idx) != len(wanted):
            raise AssertionError('test population not fully covered')
        y = p.label_frame(idx).stable.to_numpy()  # evaluation only, outside selector
        rows.append(dict(variant=variant, arm=arm, model=model, checkpoint=bb,
            n=n, draw=r, AP=p.metrics.ap(y, score[mask]),
            precision_top5pct=precision(y, score[mask], p.G.part.ids[idx]),
            n_test=len(idx), selected_params=json.dumps(chosen, sort_keys=True),
            flags=json.dumps(flags), unit=source))
    return rows


def baselines():
    rows, rounds = [], []
    arrays = [('R0','zero_shot',bb,-p.G.hulls[TEST,i]) for i,bb in enumerate(p.CHECKPOINTS)]
    arrays.append(('E0','mean','ensemble',-p.G.hulls[TEST].mean(axis=1)))
    (OUT/'baselines').mkdir(exist_ok=True)
    for arm, model, bb, score in arrays:
        path = OUT/'baselines'/f'{arm}__{bb}.npz'
        np.savez_compressed(path, material_id=p.G.part.ids[TEST], score=score)
        rows += evaluate(arm,model,bb,0,-1,score,TEST,str(path.relative_to(OUT)),{},[])
        if arm == 'R0':
            for variant, frame in ASSIGNMENTS.items():
                for rd in (3,4,5):
                    wanted = set(frame.loc[(frame.assignment == 'test') & (frame['round'] == rd),'material_id'])
                    mask = np.array([mid in wanted for mid in p.G.part.ids[TEST]])
                    y = p.label_frame(TEST[mask]).stable.to_numpy()
                    rounds.append(dict(variant=variant,round=rd,checkpoint=bb,n_test=len(y),
                                       positives=int(y.sum()),prevalence=float(y.mean()),
                                       AP=p.metrics.ap(y,score[mask])))
    pd.DataFrame(rounds).to_csv(OUT/'r0_per_round.csv',index=False)
    prevalence = []
    for variant, frame in ASSIGNMENTS.items():
        for rd in ('all',3,4,5):
            mask = frame.assignment == 'test'
            if rd != 'all':
                mask &= frame['round'] == rd
            ids = frame.loc[mask,'material_id'].to_numpy()
            y = p.G.labels.lookup(ids).stable.to_numpy()
            prevalence.append(dict(variant=variant,round=rd,n_test=len(y),positives=int(y.sum()),
                                   prevalence=float(y.mean())))
    pd.DataFrame(prevalence).to_csv(OUT/'test_prevalence.csv',index=False)
    return rows


def unit(args):
    arm, bb, n, r = args
    info = a.unit(arm, bb, n, 0, r)  # byte-identical arms arm inputs/grids/selection
    prefix = OUT/'units'/f'{arm}__{bb}__n{n}__f0__r{r}'
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as z:
        saved = {key:z[key] for key in z.files}
    # Make each fitting unit independently traceable without an index join.
    for key, source in [('draw_material_id','draw_idx'),('inner_train_material_id','inner_train_idx'),
                        ('inner_val_material_id','inner_val_idx')]:
        saved[key] = p.G.part.ids[saved[source]]
    np.savez_compressed(prefix.with_suffix('.npz'),**saved)
    rows = []
    for model, score in zip(saved['model'],saved['score']):
        model = str(model)
        if model not in KEEP[arm]:
            continue
        kind = info['best'] if model == 'best' else model
        rows += evaluate(arm,model,bb,n,r,score,saved['test_idx'],
                         str(prefix.with_suffix('.npz').relative_to(OUT)),
                         dict(model=kind,params=info['chosen'][kind]),info['flags'])
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.workers <= 15:
        parser.error('1..15 one-thread workers plus parent; <=16 CPU threads')
    clock = json.loads((OUT/'task_clock.json').read_text())
    setup()
    p.dump(OUT/'run_manifest.json',dict(p.G.manifest,task='campaign development PILOT only',
        workers=args.workers,inner_fold=0,inner_seed='r+7 (v1.2 fold=0)',
        sampler='sorted train systems, default_rng(500000+1000*n+r) permutation; canonical ids within systems; truncate last to exact n',
        budget_order=BUDGETS,draws=[0,1,2],arms=KEEP,clock=clock,
        precision='ceil(0.05*N_test), score descending / material_id ascending ties, one test set per variant',
        reuse='byte-identical arms-a3 run_arms.unit and binary_select; fit once, evaluate both variants',
        train_n=len(TRAIN),test_sa_n=len(TEST),tie_order=['logistic','hist_gbm']))
    rows = baselines()
    def save():
        pd.DataFrame(rows).sort_values(['variant','arm','model','checkpoint','n','draw']).to_csv(
            OUT/'campaign_arms_ap.csv',index=False)
    save()
    timings = []
    for n in BUDGETS:
        if time.time() >= clock['fit_deadline_epoch']:
            break
        jobs = [(arm,bb,n,r) for arm in ARMS for bb in (('ensemble',) if arm == 'A4' else p.CHECKPOINTS)
                for r in range(3)]
        done = 0
        t0 = time.monotonic()
        pool = mp.get_context('fork').Pool(args.workers)
        pending = pool.imap_unordered(unit,jobs,chunksize=1)
        try:
            while done < len(jobs) and time.time() < clock['fit_deadline_epoch']:
                try:
                    new, info = pending.next(timeout=1)
                except mp.TimeoutError:
                    continue
                rows.extend(new)
                done += 1
                save()
                print(f"n={n} {done}/{len(jobs)} {info['arm']}/{info['checkpoint']}/r{info['draw']} {info['seconds']:.1f}s",flush=True)
            pool.close() if done == len(jobs) else pool.terminate()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
        timings.append(dict(n=n,completed_units=done,planned_units=len(jobs),seconds=time.monotonic()-t0))
        p.dump(OUT/'timings.json',timings)
    complete = len(timings) == 4 and all(t['completed_units'] == t['planned_units'] for t in timings)
    p.dump(OUT/'arms_run.json',dict(required_complete=complete,exit_code=0 if complete else 3,
                                  finished_epoch=time.time()))
    return 0 if complete else 3


if __name__ == '__main__':
    raise SystemExit(main())
