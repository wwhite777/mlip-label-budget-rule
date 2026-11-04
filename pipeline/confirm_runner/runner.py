"""CONFIRM execution adapter. Real mode is refused without a verified freeze.

Usage: python -m confirm_runner.runner --synthetic-labels --reduced --workers 12
"""
from __future__ import annotations

import argparse
import json
import importlib.util
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

from .common import ROOT, HERE, YAML, ProtocolError, dump, load_yaml, sha256, checked
from .config import V2Config
from .inputs import load_inputs, campaign_assignments
from .labels import read_labels, FORMATION, HULL
from .statistics import infer, judge
from .stages import finish_stage, require_previous
from analysis.draws import draw_labels, inner_split, outer_train_test
from analysis.metrics import ap, auroc

G = ARMS = PILOT = CAMPAIGN = A5F = None
DESIGN = OUT = TRAIN = TESTS = None
SPLITS = {}
MEMORY_UNITS = {}


def source_imports():
    """Import source functions unchanged; do not call their PILOT-only entrypoints."""
    global ARMS, PILOT, CAMPAIGN, A5F
    if ARMS is None:
        sys.path[:0] = [str(ROOT / 'src/arms'), str(ROOT / 'src/pilot')]
        import run_arms
        import pilot_arms
        ARMS, PILOT = run_arms, pilot_arms
        spec = importlib.util.spec_from_file_location('frozen_campaign_source', ROOT / 'src/campaign/campaign_arms.py')
        CAMPAIGN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(CAMPAIGN)
        spec = importlib.util.spec_from_file_location('frozen_a5f_source', ROOT / 'src/fixed_head/a5f.py')
        A5F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A5F)
        A5F.p = PILOT
        A5F.FIXED = dict(learning_rate=.05, max_leaf_nodes=31)
    return ARMS, PILOT, CAMPAIGN


def split(n, f, r):
    key = (n, f, r)
    if key in SPLITS:
        return SPLITS[key]
    if DESIGN == 'P':
        draw = draw_labels(G.cfg, G.part, f, r, n)
        _, test = outer_train_test(G.part, f)
    else:
        if f != 0:
            raise ProtocolError('campaign inner fold must be zero')
        draw = CAMPAIGN.campaign_draw(G.part, TRAIN, n, r)
        test = TESTS['S-a']
    tr, va = inner_split(G.cfg, G.part, draw, f, r)
    if len(draw) != n or len(np.unique(draw)) != n or set(draw) & set(test):
        raise ProtocolError('invalid n-label draw or test overlap')
    if set(G.part.chemsys[tr]) & set(G.part.chemsys[va]):
        raise ProtocolError('inner systems overlap')
    SPLITS[key] = (draw, tr, va, test)
    return SPLITS[key]


def label_frame(idx, draw=None):
    return G.labels.lookup(G.part.ids[idx], allowed_ids=None if draw is None else G.part.ids[draw])


def setup_context(inp, doc, design, out, train=None, tests=None):
    global G, DESIGN, OUT, TRAIN, TESTS, SPLITS, MEMORY_UNITS
    source_imports()
    G, DESIGN, OUT, TRAIN, TESTS = inp, design, Path(out), train, tests
    SPLITS = {}
    MEMORY_UNITS = {}
    G.cfg = V2Config(doc)
    G.protocol_doc = doc
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'units').mkdir(exist_ok=True)
    PILOT.G, PILOT.OUT = G, OUT
    PILOT.split, PILOT.label_frame = split, label_frame


def precision(y, score, ids, fold=None):
    if fold is None:
        groups = [np.arange(len(y))]
    else:
        groups = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    hits = count = 0
    for idx in groups:
        k = int(np.ceil(.05 * len(idx)))
        order = np.lexsort((ids[idx], -score[idx]))[:k]
        hits += np.asarray(y)[idx][order].sum()
        count += k
    return float(hits / count)


def metric_row(design, arm, model, bb, n, r, idx, score, fold=-1, scope='pooled', round_id='all'):
    y = label_frame(idx).stable.to_numpy()
    return dict(design=design, arm=arm, model=model, checkpoint=bb, n=n, replicate=r, fold=fold,
        scope=scope, round=round_id, N=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
        AP=ap(y, score), AUROC=auroc(y, score) if len(np.unique(y)) == 2 else None,
        precision_top5=precision(y, score, G.part.ids[idx], G.part.fold[idx] if design == 'P' and scope == 'pooled' else None))


def run_unit(job):
    arm, bb, n, f, r = job
    draw, tr, va, test = split(n, f, r)
    t0 = time.process_time()
    path = OUT / 'units' / f'{arm}__{bb}__n{n}__f{f}__r{r}.npz'
    compact = getattr(G, 'compact_synthetic', False) and r != 0
    captured = {}
    original_save = np.savez_compressed
    def capture(filename, **arrays):
        if compact and str(filename) == str(path):
            captured.update(arrays)
        else:
            original_save(filename, **arrays)
    G.labels.active_draw = set(G.part.ids[draw])
    np.savez_compressed = capture
    try:
        info = A5F.unit(bb,n,f,r) if arm == 'A5f' else ARMS.unit(*job)
    finally:
        np.savez_compressed = original_save
        G.labels.active_draw = None
    info['cpu_seconds'] = time.process_time() - t0
    info['design'] = DESIGN
    if compact:
        saved = captured
        if not saved:
            raise ProtocolError('synthetic score capture failed')
    else:
        with np.load(path, allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
    for key, source in [('draw_material_id', 'draw_idx'), ('inner_train_material_id', 'inner_train_idx'), ('inner_val_material_id', 'inner_val_idx')]:
        saved[key] = G.part.ids[saved[source]]
    if not compact:
        np.savez_compressed(path, **saved)
    dump(path.with_suffix('.json'), info)
    rows = []
    for model, score in zip(saved['model'], saved['score']):
        if DESIGN == 'P':
            rows.append(metric_row('P', arm, str(model), bb, n, r, test, score, f, 'unit'))
        else:
            for variant, idx in TESTS.items():
                mask = np.isin(test, idx)
                rows.append(metric_row(variant, arm, str(model), bb, n, r, test[mask], score[mask], 0, 'unit'))
    if compact:
        info['_score_payload'] = saved
    return info, rows


def primary_job(doc, job):
    arm, _, n, _, _ = job
    return ((n == doc['families']['F1']['n'] and arm in ('A5f','A4')) or
            (n in doc['families']['F2']['budgets'] and arm in ('A5f','A4','C1')))


def run_unit_recording_optional_failure(job):
    """Optional CPU-P failures are explicit; required F1/F2 failures still stop."""
    started=time.monotonic();cpu=time.process_time()
    try:
        return run_unit(job)
    except Exception as exc:
        if DESIGN != 'P' or primary_job(G.protocol_doc, job):
            raise
        arm,bb,n,f,r=job
        info=dict(arm=arm,checkpoint=bb,n=n,fold=f,draw=r,design=DESIGN,status='INCOMPLETE',
                  error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-started,
                  cpu_seconds=time.process_time()-cpu,primary_success_required=False)
        dump(OUT/'units'/f'{arm}__{bb}__n{n}__f{f}__r{r}.failure.json',info)
        return info, []


def noise_unit(arm):
    if arm != 'A5f':
        return ARMS.noise_unit(arm)
    bb='mace_medium'
    draw,tr,va,_=split(3000,0,0)
    params=A5F.FIXED
    ytr=label_frame(tr,draw).stable.to_numpy()
    yva=label_frame(va,draw).stable.to_numpy()
    def evaluate(features):
        X=np.column_stack([features.astype(np.float64),G.hulls])
        if len(np.unique(ytr))==1:
            score=np.full(len(va),float(ytr[0]))
        else:
            score=PILOT.models.make('hist_gbm',params,G.cfg.y,'cpu').fit(X[tr],ytr).score(X[va])
        return ap(yva,score)
    original=evaluate(G.feats[bb])
    noisy=[evaluate(PILOT.perturb_features(G.feats[bb],seed)) for seed in range(5)]
    return [dict(arm=arm,model='hist_gbm',checkpoint=bb,n=3000,fold=0,draw=0,params=params,
        original_inner_ap=original,noise_floor=max(abs(v-original) for v in noisy),perturbed_inner_ap=noisy,
        perturbed_input='encoder features; fixed hull inputs unchanged',relative_gaussian_scale=1e-7,
        seeds=list(range(5)),selected_params_fixed=True,inner_selection_used=False)]


def jobs_for(doc, design, budgets, reps):
    # Matrix and model menus are in YAML; the unchanged unit emits those menus.
    return [(spec['arm'], bb, n, f, r) for n in budgets for r in reps
            for spec in doc['run_matrix']['named_cpu']['fitted']
            for bb in (doc['backbones']['order'] if 'checkpoint_ref' in spec else [spec['checkpoint']])
            for f in (range(doc['designs']['P']['n_folds']) if design == 'P' else [0])]


def collect_scores(doc, design, budgets, reps, out):
    eval_idx = np.arange(G.part.n) if design == 'P' else TESTS['S-a']
    position = np.full(G.part.n, -1, dtype=int)
    position[eval_idx] = np.arange(len(eval_idx))
    scores = {f'R0_{bb}': -G.hulls[eval_idx,j][None,:] for j,bb in enumerate(doc['backbones']['order'])}
    scores['E0_mean'] = -G.hulls[eval_idx].mean(axis=1)[None,:]
    if design == 'P':
        scores['E0_median'] = -np.median(G.hulls[eval_idx],axis=1)[None,:]
    required_keys=set(scores);unavailable=[]
    for job in jobs_for(doc,design,budgets,reps):
        arm,bb,n,f,r=job
        path=out/'units'/f'{arm}__{bb}__n{n}__f{f}__r{r}.npz'
        optional=design=='P' and not primary_job(doc,job)
        try:
            if path.with_suffix('.failure.json').is_file():
                raise ProtocolError('unit failed; see failure record')
            if path.name in MEMORY_UNITS:
                z=MEMORY_UNITS[path.name]
            elif path.is_file():
                with np.load(path,allow_pickle=False) as file:
                    z={k:file[k] for k in file.files}
            else:
                raise ProtocolError(f'missing unit; no partial reading: {path.name}')
            idx=z['test_idx']
            if not np.array_equal(z['material_id'],G.part.ids[idx]):
                raise ProtocolError('score ID alignment mismatch')
            if not np.array_equal(idx,split(n,f,r)[3]):
                raise ProtocolError('score test population mismatch')
            entries=[];ridx=reps.index(r);pos=position[idx]
            if (pos<0).any():raise ProtocolError('invalid held-out positions')
            for model,score in zip(z['model'],z['score']):
                key=f'{arm}_{model}_{bb}@{n}'
                if np.asarray(score).shape!=(len(idx),) or not np.isfinite(score).all():
                    raise ProtocolError('incomplete/nonfinite unit scores')
                if key in scores and not np.isnan(scores[key][ridx,pos]).all():
                    raise ProtocolError('duplicate held-out scores')
                entries.append((key,score))
            # No partially validated unit is inserted into an inference matrix.
            for key,score in entries:
                if key not in scores:scores[key]=np.full((len(reps),len(eval_idx)),np.nan)
                scores[key][ridx,pos]=score
                if not optional:required_keys.add(key)
        except Exception as exc:
            if not optional:raise
            unavailable.append(dict(unit=path.name,status='INCOMPLETE',error=f'{type(exc).__name__}: {exc}'))
    for name in list(scores):
        if not np.isfinite(scores[name]).all():
            if name in required_keys:raise ProtocolError('incomplete held-out score population')
            unavailable.append(dict(method=name,status='INCOMPLETE',error='partial optional pooled score matrix; no reading'))
            del scores[name]
    dump(out/'optional_incomplete.json',unavailable)
    retained={k:v[:1] for k,v in scores.items()} if getattr(G,'compact_synthetic',False) else scores
    np.savez_compressed(out/'pooled_scores.npz',material_id=G.part.ids[eval_idx],
        replicates=reps[:1] if getattr(G,'compact_synthetic',False) else reps,**retained)
    return eval_idx,scores


def report_scores(doc, variant, idx, scores, reps, rows, rounds=False):
    for key, arr in scores.items():
        if key.startswith('R0_'):
            arm, model, bb, n = 'R0', 'zero_shot', key[3:], 0
        elif key.startswith('E0_'):
            arm, model, bb, n = 'E0', key[3:], 'ensemble', 0
        else:
            method, n = key.split('@')
            arm = method.split('_')[0]
            bb = next((b for b in doc['backbones']['order'] if method.endswith('_' + b)), 'ensemble')
            model = method[len(arm)+1:-(len(bb)+1)]
            n = int(n)
        for ri, score in enumerate(arr):
            r = -1 if key.startswith(('R0_','E0_')) else 0 if key.startswith('FT_') and n == 10000 else reps[ri]
            rows.append(metric_row(variant, arm, model, bb, n, r, idx, score))
            if rounds:
                rd = np.array([int(m.split('-')[1]) for m in G.part.ids[idx]])
                for round_id in doc['designs']['S']['test_rounds']:
                    mask = rd == round_id
                    rows.append(metric_row(variant, arm, model, bb, n, r, idx[mask], score[mask], scope='round', round_id=round_id))


def execute(doc, inp, out, *, workers=12, reduced=False, B=None, synthetic=True, designs=('P', 'S'), ft_root=None, stage_root=None, stage=None):
    started = time.monotonic()
    if reduced and not synthetic:
        raise ProtocolError('reduced mode is synthetic only')
    if not synthetic:
        if stage not in ('CPU-P', 'CPU-S') or stage_root is None:
            raise ProtocolError('real CPU execution requires a single fixed stage and stage root')
        if ft_root is not None or tuple(designs) != (stage[-1],):
            raise ProtocolError('CPU stage cannot ingest FT or reorder designs')
        require_previous(stage_root, stage)
        if Path(out).resolve() != Path(stage_root).resolve()/stage:
            raise ProtocolError('stage output must be stage-root/stage')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reps = [0, 1] if reduced else doc['replicates']['ids']
    B = B if B is not None else (200 if reduced else doc['inference']['B'])
    if reduced and not synthetic:
        raise ProtocolError('reduced mode is synthetic only')
    train, tests, _ = campaign_assignments(doc, inp.part) if 'S' in designs else (None, None, None)
    inp.compact_synthetic = synthetic
    dump(out / 'input_coverage.json', inp.coverage)
    dump(out / 'label_access.json', inp.labels.trace)
    source_hashes = {name: sha256(ROOT / spec['path']) for name, spec in doc['implementation']['imports'].items()
}
    dump(out / 'implementation_sha256.json', source_hashes)
    readings, timings, metrics, failed_cpu_units = [], [], [], []
    for design in designs:
        budgets = [300, 1000] if reduced else doc['run_matrix']['named_cpu']['budgets_by_fit_design'][design]
        dest = out / design
        setup_context(inp, doc, design, dest, train, tests)
        jobs = jobs_for(doc, design, budgets, reps)
        # Cache exact source sampler outputs once per shared draw, before forking.
        for n in budgets:
            for r in reps:
                for f in (range(5) if design == 'P' else [0]):
                    split(n, f, r)
        infos, per_unit = [], []
        t0 = time.monotonic()
        with mp.get_context('fork').Pool(workers) as pool:
            for info, rows in pool.imap_unordered(run_unit_recording_optional_failure, jobs, chunksize=1):
                if '_score_payload' in info:
                    key=f"{info['arm']}__{info['checkpoint']}__n{info['n']}__f{info['fold']}__r{info['draw']}.npz"
                    MEMORY_UNITS[key]=info.pop('_score_payload')
                infos.append(info)
                per_unit.extend(rows)
                if len(infos) % 10 == 0 or len(infos) == len(jobs):
                    print(f'{design} fits {len(infos)}/{len(jobs)}, {time.monotonic()-t0:.1f}s', flush=True)
                dump(dest / 'fit_progress.json', dict(completed=len(infos), planned=len(jobs)))
        fitting_wall = time.monotonic() - t0
        dump(dest / 'units.json', infos)
        failed_cpu_units.extend(x for x in infos if x.get('status')=='INCOMPLETE')
        dump(dest/'failed_cpu_units.json',[x for x in infos if x.get('status')=='INCOMPLETE'])
        pd.DataFrame(per_unit).to_csv(dest / 'per_unit_metrics.csv', index=False)
        if not reduced and 3000 in budgets:
            diagnostics=[]
            for spec in doc['run_matrix']['named_cpu']['fitted']:
                try:diagnostics.extend(noise_unit(spec['arm']))
                except Exception as exc:
                    diagnostics.append(dict(arm=spec['arm'],status='INCOMPLETE',descriptive_only=True,error=str(exc)))
            dump(dest/'noise_floor.json',diagnostics)
        # Distinct labels, label reuse across folds/replicates and per-learner budgets.
        reuse = []
        for n in budgets:
            draws = [split(n, f, r)[0] for r in reps for f in (range(5) if design == 'P' else [0])]
            all_draws = np.concatenate(draws)
            reuse.append(dict(n_per_learner=n, learners=len(draws), nominal_label_uses=len(all_draws),
                              unique_labels=len(np.unique(all_draws)), repeated_uses=len(all_draws)-len(np.unique(all_draws)),
                              arms_share_draws=True, S_variants_share_fits=design == 'S'))
        dump(dest / 'label_reuse.json', reuse)
        idx, scores = collect_scores(doc, design, budgets, reps, dest)
        if ft_root is not None:
            from .ft_ingest import collect_ft_scores
            scores.update(collect_ft_scores(doc, inp.part, design, idx, ft_root, reps, synthetic=synthetic, tests=tests))
        bootstrap_wall = bootstrap_cpu = 0
        for variant in (['P'] if design == 'P' else ['S-a', 'S-b']):
            mask = np.ones(len(idx), bool) if variant in ('P', 'S-a') else np.isin(idx, tests['S-b'])
            vidx = idx[mask]
            vscores = {k: v[:, mask] for k, v in scores.items()}
            report_scores(doc, variant, vidx, vscores, reps, metrics, rounds=design == 'S')
            y = label_frame(vidx).stable.to_numpy()
            rows, elapsed = infer(doc, vscores, y, inp.part.chemsys[vidx], budgets, reps, variant, B, dest, synthetic,
                                  workers=workers, allow_incomplete_f3=design=='P')
            readings.extend(rows)
            bootstrap_wall += elapsed
            import json
            bootstrap_cpu += json.loads((dest / f'inference_{variant}.json').read_text())['bootstrap_cpu_seconds']
        timings.append(dict(design=design, units=len(infos), workers=workers, fitting_wall_seconds=fitting_wall,
                            fitting_cpu_seconds=sum(x['cpu_seconds'] for x in infos), bootstrap_wall_seconds=bootstrap_wall,
                            bootstrap_cpu_seconds=bootstrap_cpu,
                            unit_timings=[{k: x[k] for k in ('arm', 'checkpoint', 'n', 'seconds', 'cpu_seconds')} for x in infos]))
        dump(out / 'timings.json', timings)
        if design == 'P':
            dump(out / 'readings.json', readings)
            decision = judge(doc, readings, synthetic=synthetic, complete=not reduced and not synthetic)
            dump(out / 'judge.json', decision)
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    dump(out / 'readings.json', readings)
    # The P judgment is already durable. Later stage failures cannot remove it.
    decision = judge(doc, readings, synthetic=synthetic)
    if 'P' in designs:
        dump(out / 'judge.json', decision)
    complete = decision['complete']
    report = dict(status='PASS', execution_complete=True, protocol_complete=False, primary_complete=complete, synthetic_only=synthetic,
                  all_requested_cpu_units_complete=not failed_cpu_units, failed_cpu_units=failed_cpu_units,
                  separate_reports=dict(F3_P='COMPLETE' if sum(x['design']=='P' and x['family']=='F3' for x in readings)==doc['families']['F3']['size'] else 'INCOMPLETE',
                    CPU_S='COMPLETE' if 'S' in designs and not reduced else 'INCOMPLETE',
                    F4='COMPLETE' if ft_root is not None else 'INCOMPLETE', mechanism='INCOMPLETE'),
                  designs=list(designs), budgets=[300, 1000] if reduced else doc['run_matrix']['named_cpu']['budgets_by_fit_design'],
                  replicates=reps, B=B, real_label_columns_opened=[],
                  scores_saved_replicates=[0] if synthetic else reps,
                  arms_executed=sorted({x['arm'] for x in doc['run_matrix']['named_cpu']['fitted']}),
                  implementation_sha256=source_hashes,
                  protocol_schema_sha256=sha256(HERE / 'protocol_schema.json'),
                  seconds=time.monotonic()-started, judge=decision, timings=timings)
    dump(out / 'run.json', report)
    if stage_root is not None:
        paths = [p for p in out.iterdir() if p.name not in ('judge.json', 'stage_manifest.json')]
        stage_marker = finish_stage(stage_root, stage, paths, synthetic=synthetic)
        print(stage_marker['manifest_line'], flush=True)
        if stage == 'CPU-P':
            judge_hash = finish_stage(stage_root, 'judge', [out/'judge.json'], synthetic=synthetic)
            print(judge_hash['manifest_line'], flush=True)
    return report


def execute_later_stage(doc, inp, out, stage, stage_root, ft_root, *, synthetic, workers=1):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not synthetic:
        require_previous(stage_root, stage)
    if stage == 'mechanism':
        capped = {}
        for previous in ('FT-P', 'FT-S'):
            manifest = Path(stage_root)/previous/'stage_manifest.json'
            if manifest.is_file():
                data = json.loads(manifest.read_text())
                if data.get('status') == 'INCOMPLETE_CAP':
                    capped[previous] = data['incomplete_cells']
        if capped:
            dump(out/'F4_incomplete.json', dict(status='INCOMPLETE', stages=capped, readings=[],
                 primary_judgment='retained in CPU-P/judge.json'))
        from .mechanism import execute as execute_mechanism
        try:
            execute_mechanism(doc, inp, out, workers=min(workers, 3))
        except Exception as exc:
            dump(out/'run.json', dict(status='INCOMPLETE', error=str(exc), descriptive_only=True))
            finish_stage(stage_root, stage, [out], synthetic=synthetic, status='INCOMPLETE')
            return dict(status='INCOMPLETE', primary_judgment='retained in CPU-P/judge.json')
    else:
        from .ft_ingest import collect_ft_scores
        design = stage[-1]
        train, tests, _ = campaign_assignments(doc, inp.part) if design == 'S' else (None, None, None)
        setup_context(inp, doc, design, out, train, tests)
        idx = np.arange(inp.part.n) if design == 'P' else tests['S-a']
        scores = {f'R0_{bb}': -inp.hulls[idx,j][None,:] for j,bb in enumerate(doc['backbones']['order'])}
        scores.update(collect_ft_scores(doc, inp.part, design, idx, ft_root, doc['replicates']['ids'], synthetic=synthetic, tests=tests))
        rows = []
        for variant in (['P'] if design == 'P' else ['S-a','S-b']):
            mask = np.ones(len(idx),bool) if variant in ('P','S-a') else np.isin(idx,tests['S-b'])
            vidx = idx[mask]
            result, _ = infer(doc, {k:v[:,mask] for k,v in scores.items()}, label_frame(vidx).stable.to_numpy(),
                inp.part.chemsys[vidx], [], doc['replicates']['ids'], variant, doc['inference']['B'], out, synthetic, workers=workers)
            rows.extend(result)
        dump(out/'readings.json', rows)
        dump(out/'run.json', dict(status='PASS', stage=stage, synthetic_only=synthetic, primary_judgment='retained in CPU-P/judge.json'))
    marker = finish_stage(stage_root, stage, [out], synthetic=synthetic)
    print(marker['manifest_line'], flush=True)
    return dict(status='PASS', stage=stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=YAML)
    parser.add_argument('--out-root', '--output', dest='output', type=Path, required=True)
    parser.add_argument('--stage', choices=['CPU-P','CPU-S','FT-P','FT-S','mechanism'])
    parser.add_argument('--stage-root', type=Path)
    parser.add_argument('--synthetic-labels', action='store_true')
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--ft-root', type=Path, help='Verified launcher output with complete F4 test energies')
    args = parser.parse_args()
    if not 1 <= args.workers <= 7:
        parser.error('1..7 one-thread workers plus parent; total <=8')
    doc = load_yaml(args.yaml)
    import json
    expected = json.loads((HERE / 'protocol_schema.json').read_text())
    if any(doc.get(key) != value for key, value in expected.items()):
        raise ProtocolError('YAML differs from specified v2.3 schema; no synthetic or real execution')
    import sklearn
    if sklearn.__version__ != doc['metrics']['sklearn_version']:
        raise ProtocolError('sklearn version differs from protocol')
    for name, spec in doc['implementation']['imports'].items():
        checked(spec)
    if not args.synthetic_labels:
        from integrity import verify_protocol
        verify_protocol(yaml_path=args.yaml)
    if not args.synthetic_labels:
        if args.stage is None or args.stage_root is None:
            raise ProtocolError('real execution requires --stage and --stage-root')
        require_previous(args.stage_root, args.stage, yaml_path=args.yaml)
    inp = load_inputs(doc)
    inp.labels = read_labels(doc, inp.part, synthetic=args.synthetic_labels, yaml_path=args.yaml)
    if args.stage in ('FT-P','FT-S','mechanism'):
        if args.stage != 'mechanism' and args.ft_root is None:
            parser.error('FT evaluation requires --ft-root')
        report = execute_later_stage(doc, inp, args.output, args.stage, args.stage_root, args.ft_root,
                                    synthetic=args.synthetic_labels, workers=args.workers)
    else:
        designs = (args.stage[-1],) if args.stage else ('P','S')
        report = execute(doc, inp, args.output, workers=args.workers, reduced=args.reduced,
                         synthetic=args.synthetic_labels, designs=designs, ft_root=args.ft_root,
                         stage=args.stage, stage_root=args.stage_root)
    print(report['status'])



if __name__ == '__main__':
    main()
