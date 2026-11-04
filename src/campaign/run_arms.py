"""arms-a3 development PILOT arms; regenerated features accepted in ."""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import pilot_arms as p
from analysis.selection import select_and_refit
from posthoc import s9_additions as s9
from posthoc.s7_additions import shortlist
from rereference import ALPHAS, fit_rereference, score_rereference

BUDGETS = (100, 300, 1000, 3000, 10000)
ORDER = ("C1", "A1", "A2", "A4", "A3", "A5")
COLUMNS = ["arm", "model", "checkpoint", "n", "draw", "AP", "precision_top5pct", "selected_params", "flags"]
DEADLINE = 1792652934  # 2026-10-22 07:08:54 UTC: this task start + 150 minutes
FIT_DEADLINE = DEADLINE - 180  # reserve three minutes for final verification and run_record
GATE = p.OUT.parent / "arms/previous/equivalence_gate_summary.json"


def accepted_gate():
    """Disclose the accepted arrays' retained failed equivalence check."""
    return dict(recorded_status='FAIL', rerun=False, interpretation='The explicit protocol hashes identify the accepted input arrays.')


def top5(y, scores, ids):
    """Frozen S9 convention: ceil(0.05*N_fold) per fold, pool hits / k."""
    ranks = np.argsort(np.argsort(np.asarray(ids).astype(str)))
    hits, k, _ = shortlist(np.asarray(y), scores, p.G.part.fold, ranks, 0.05)
    return hits / k


def binary_select(types, Xs, view, tr, va, draw, test):
    cfg = p.G.cfg
    ytr, yva = view(tr), view(va)
    flags = []
    # Frozen selectors reject single-class validation; the task explicitly overrides this case.
    if np.unique(yva).size == 2 and np.unique(ytr).size == 2:
        result = select_and_refit(types, types, {k: cfg.head_grid(k) for k in types}, Xs,
                                  view, tr, va, draw, test, cfg.y, "cpu")
    else:
        flags.append("inner_validation_no_positives" if not yva.any() else "inner_single_class")
        chosen = {k: cfg.head_grid(k)[0] for k in types}
        inner = dict.fromkeys(types, float(yva.mean()))
        scores = {}
        yall = view(draw)
        for kind in types:
            if np.unique(yall).size == 1:
                scores[kind] = np.full(len(test), float(yall[0]))
                flags.append("draw_single_class_constant_score")
            else:
                scores[kind] = p.models.make(kind, chosen[kind], cfg.y, "cpu").fit(
                    Xs[kind][draw], yall).score(Xs[kind][test])
        result = dict(best=types[0], scores=scores, chosen=chosen, inner_ap=inner,
                      records=[dict(type=k, grid_index=0, params=chosen[k], inner_val_ap=inner[k],
                                    fallback_first_grid=True) for k in types])
    return result, flags


def reg_select(X, target, to_score, view, tr, va, draw, test):
    cfg = p.G.cfg
    yva = view(va)
    if np.unique(yva).size == 2:
        return s9.select_and_refit_reg(cfg, X, target, to_score, view, tr, va, draw, test), []
    kinds = s9.TIE["regression"]
    chosen = {k: s9.reg_grid(cfg, k)[0] for k in kinds}
    scores = {k: to_score(s9.make_reg(k, chosen[k], cfg).fit(X[draw], target(draw)).predict(X[test]), test)
              for k in kinds}
    return dict(best=kinds[0], scores=scores, chosen=chosen, inner_ap=dict.fromkeys(kinds, float(yva.mean())),
                records=[dict(type=k, grid_index=0, params=chosen[k], inner_val_ap=float(yva.mean()),
                              fallback_first_grid=True) for k in kinds]), [
                    "inner_validation_no_positives" if not yva.any() else "inner_validation_all_positives"]


def unit(arm, bb, n, f, r):
    t0 = time.monotonic()
    draw, tr, va, test = p.split(n, f, r)
    view = lambda idx: p.label_frame(idx, draw).stable.to_numpy()
    target = lambda idx: p.label_frame(idx, draw)[p.HULL].to_numpy() - p.G.refs[bb]["r_sp"]["e_hull"][idx]
    flags = []
    with threadpool_limits(1):
        if arm == "C1":
            X = p.G.comp
            ef, eh = (p.G.refs[bb]["r_sp"][key] for key in ("e_form", "e_hull"))
            yva = view(va)
            vals = []
            for a in ALPHAS:
                mdl = fit_rereference(X[tr], p.label_frame(tr, draw)[p.FORMATION].to_numpy(), ef[tr], a)
                vals.append(p.metrics.ap(yva, score_rereference(mdl, X[va], eh[va])))
            if not yva.any():
                flags.append("inner_validation_no_positives")
            a = ALPHAS[int(np.argmax(vals))]
            mdl = fit_rereference(X[draw], p.label_frame(draw, draw)[p.FORMATION].to_numpy(), ef[draw], a)
            res = dict(best="ridge", scores={"ridge": score_rereference(mdl, X[test], eh[test])},
                       chosen={"ridge": {"alpha": a}}, inner_ap={"ridge": max(vals)},
                       records=[dict(type="ridge", grid_index=i, params={"alpha": a}, inner_val_ap=v)
                                for i, (a, v) in enumerate(zip(ALPHAS, vals))])
        elif arm in ("A2", "A3"):
            X = p.G.feats[bb].astype(np.float64) if arm == "A2" else p.G.comp
            to_score = lambda pred, idx: -(p.G.refs[bb]["r_sp"]["e_hull"][idx] + pred)
            res, flags = reg_select(X, target, to_score, view, tr, va, draw, test)
            if arm == "A3":
                # The frozen reg selector also fits ridge. Only the requested HistGBM arm is retained.
                res["scores"] = {"hist_gbm_reg": res["scores"]["hist_gbm_reg"]}
                res["best"] = "hist_gbm_reg"
        else:
            if arm == "A1":
                X = np.column_stack([p.G.feats[bb].astype(np.float64), p.G.refs[bb]["r_sp"]["e_hull"]])
                Xs = dict(logistic=X, hist_gbm=X)
            elif arm == "A4":
                Xs = dict(logistic=p.G.hulls, hist_gbm=np.column_stack([p.G.hulls, p.G.comp]))
            elif arm == "A5":
                Xs = dict(hist_gbm=np.column_stack([p.G.feats[bb].astype(np.float64), p.G.hulls]))
            else:
                raise ValueError(arm)
            res, flags = binary_select(list(Xs), Xs, view, tr, va, draw, test)
    methods = list(res["scores"])
    if arm in ("A1", "A2", "A4"):
        res["scores"]["best"] = res["scores"][res["best"]]
        methods.append("best")
    prefix = p.OUT / "units" / f"{arm}__{bb}__n{n}__f{f}__r{r}"
    prefix.parent.mkdir(exist_ok=True)
    score = np.stack([res["scores"][k] for k in methods])
    assert np.isfinite(score).all()
    np.savez_compressed(str(prefix) + ".npz", test_idx=test, material_id=p.G.part.ids[test],
                        model=np.array(methods), score=score, draw_idx=draw,
                        inner_train_idx=tr, inner_val_idx=va)
    info = dict(arm=arm, checkpoint=bb, n=n, fold=f, draw=r, best=res["best"], chosen=res["chosen"],
                inner_ap=res["inner_ap"], records=res["records"], flags=flags,
                n_inner_train=len(tr), n_inner_val=len(va), n_val_positives=int(view(va).sum()),
                seconds=time.monotonic()-t0)
    p.dump(str(prefix) + ".json", info)
    return info


def assemble(arm, bb, n, r):
    scores, infos = {}, []
    for f in range(5):
        prefix = p.OUT / "units" / f"{arm}__{bb}__n{n}__f{f}__r{r}"
        if not prefix.with_suffix(".npz").exists():
            return []
        with np.load(prefix.with_suffix(".npz")) as z:
            idx = z["test_idx"]
            np.testing.assert_array_equal(z["material_id"], p.G.part.ids[idx])
            for model, score in zip(z["model"], z["score"]):
                if str(model) not in scores:
                    scores[str(model)] = np.full(p.G.part.n, np.nan)
                assert np.isnan(scores[str(model)][idx]).all()
                scores[str(model)][idx] = score
        infos.append(json.loads(prefix.with_suffix(".json").read_text()))
    y = p.label_frame(np.arange(p.G.part.n)).stable.to_numpy()  # evaluation only
    rows = []
    for model, score in scores.items():
        assert np.isfinite(score).all()
        chosen = {str(info["fold"]): dict(model=info["best"] if model == "best" else model,
                                         params=info["chosen"][info["best"] if model == "best" else model]) for info in infos}
        flags = {str(info["fold"]): info["flags"] for info in infos if info["flags"]}
        rows.append(dict(arm=arm, model=model, checkpoint=bb, n=n, draw=r, AP=p.metrics.ap(y, score),
                         precision_top5pct=top5(y, score, p.G.part.ids), selected_params=json.dumps(chosen),
                         flags=json.dumps(flags)))
    return rows


def baselines():
    y = p.label_frame(np.arange(p.G.part.n)).stable.to_numpy()
    rows = []
    arrays = [("R0", "zero_shot", bb, -p.G.hulls[:, i]) for i, bb in enumerate(p.CHECKPOINTS)]
    arrays += [("E0", "mean", "ensemble", -p.G.hulls.mean(axis=1)),
               ("E0", "median", "ensemble", -np.median(p.G.hulls, axis=1))]
    for arm, model, bb, score in arrays:
        rows.append(dict(arm=arm, model=model, checkpoint=bb, n=0, draw=-1, AP=p.metrics.ap(y, score),
                         precision_top5pct=top5(y, score, p.G.part.ids), selected_params="{}", flags="{}"))
    return rows


def run_stage(arm, budgets, draws, workers, save_rows):
    t0 = time.monotonic()
    bbs = ("ensemble",) if arm == "A4" else p.CHECKPOINTS
    jobs = [(arm, bb, n, f, r) for n in budgets for bb in bbs for r in draws for f in range(5)]
    complete, rows, done = {}, [], 0
    pool = mp.get_context("fork").Pool(workers)
    pending = pool.imap_unordered(unit_args, jobs, chunksize=1)
    try:
        while done < len(jobs) and time.time() < FIT_DEADLINE:
            try:
                info = pending.next(timeout=min(1, max(0.01, FIT_DEADLINE-time.time())))
            except mp.TimeoutError:
                continue
            done += 1
            key = (info["checkpoint"], info["n"], info["draw"])
            complete.setdefault(key, set()).add(info["fold"])
            if len(complete[key]) == 5:
                new = assemble(arm, *key)
                rows.extend(new)
                save_rows(new)
            if done % 25 == 0 or done == len(jobs):
                print(f"{arm}: {done}/{len(jobs)} units, stage {time.monotonic()-t0:.1f}s", flush=True)
        if done < len(jobs):
            pool.terminate()
        else:
            pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return rows, dict(arm=arm, budgets=list(budgets), draws=list(draws), seconds=time.monotonic()-t0,
                      units=done, planned_units=len(jobs),
                      status="complete" if done == len(jobs) else "deadline")


def unit_args(args):
    return unit(*args)


def noise_unit(arm):
    """Chosen configurations, n3000/f0/r0, inner AP under five 1e-7 input perturbations.

    Encoder features only for A1/A2/A5, composition for C1/A3, learner inputs for A4.
    Reference hull values used by scoring/targets stay fixed. No gate or acceptance rule.
    """
    bb = "ensemble" if arm == "A4" else "mace_medium"
    prefix = p.OUT / "units" / f"{arm}__{bb}__n3000__f0__r0"
    if not prefix.with_suffix(".json").exists():
        return []
    info = json.loads(prefix.with_suffix(".json").read_text())
    draw, tr, va, _ = p.split(3000, 0, 0)
    ytr = p.label_frame(tr, draw).stable.to_numpy()
    yva = p.label_frame(va, draw).stable.to_numpy()
    hull = None if arm == "A4" else p.G.refs[bb]["r_sp"]["e_hull"]
    rows = []
    for kind, params in info["chosen"].items():
        if arm == "A3" and kind != "hist_gbm_reg":
            continue
        base = (p.G.comp if arm in ("C1", "A3") else
                (p.G.hulls if kind == "logistic" else np.column_stack([p.G.hulls, p.G.comp]))
                if arm == "A4" else p.G.feats[bb])
        # Match arm feature assembly dtype exactly; perturb the original float32 encoder first.
        def evaluate(X):
            if arm == "C1":
                mdl = fit_rereference(X[tr], p.label_frame(tr, draw)[p.FORMATION].to_numpy(),
                                      p.G.refs[bb]["r_sp"]["e_form"][tr], params["alpha"])
                score = score_rereference(mdl, X[va], hull[va])
            elif arm in ("A2", "A3"):
                X = X.astype(np.float64)
                target = p.label_frame(tr, draw)[p.HULL].to_numpy() - hull[tr]
                mdl = s9.make_reg(kind, params, p.G.cfg).fit(X[tr], target)
                score = -(hull[va] + mdl.predict(X[va]))
            else:
                if arm == "A1":
                    X = np.column_stack([X.astype(np.float64), hull])
                elif arm == "A5":
                    X = np.column_stack([X.astype(np.float64), p.G.hulls])
                mdl = p.models.make(kind, params, p.G.cfg.y, "cpu").fit(X[tr], ytr)
                score = mdl.score(X[va])
            return p.metrics.ap(yva, score)
        with threadpool_limits(1):
            original = evaluate(base)
            noisy = [evaluate(p.perturb_features(base, seed)) for seed in range(5)]
        rows.append(dict(arm=arm, model=kind, checkpoint=bb, n=3000, fold=0, draw=0,
                         params=params, original_inner_ap=original,
                         noise_floor=max(abs(a-original) for a in noisy),
                         perturbed_inner_ap=noisy,
                         perturbed_input="encoder features" if arm in ("A1", "A2", "A5") else
                         "composition fractions" if arm in ("C1", "A3") else "five hulls + composition (tree only)",
                         relative_gaussian_scale=1e-7, seeds=list(range(5)), selected_params_fixed=True))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.workers <= 16
    gate = accepted_gate()
    p.G = p.load_inputs()
    p.dump(p.OUT / "run_manifest.json", dict(p.G.manifest, feature_acceptance=gate,
            start_epoch=1792643934, hard_deadline_epoch=DEADLINE, fit_deadline_epoch=FIT_DEADLINE,
            workers=args.workers, budgets=BUDGETS, required_draws=list(range(3)),
            optional_draws=[3, 4], tie_order=s9.TIE,
            precision="S9: ceil(0.05*N_fold) per fold, pooled hits / total selected",
            hist_grid_policy="all four frozen points, in frozen order"))
    rows, timings = [], []
    noise = []
    t0 = time.monotonic()
    rows += baselines()
    timings.append(dict(arm="R0/E0", seconds=time.monotonic()-t0, status="complete"))
    pd.DataFrame(rows, columns=COLUMNS).to_csv(p.OUT / "pilot_arms_ap.csv", index=False)
    p.dump(p.OUT / "arm_timings.json", timings)
    def save_rows(new):
        rows.extend(new)
        pd.DataFrame(rows, columns=COLUMNS).sort_values(
            ["arm", "model", "checkpoint", "n", "draw"]).to_csv(p.OUT / "pilot_arms_ap.csv", index=False)
    for arm in ORDER:
        if time.time() >= FIT_DEADLINE:
            break
        _, timing = run_stage(arm, BUDGETS, range(3), args.workers, save_rows)
        timings.append(timing)
        p.dump(p.OUT / "arm_timings.json", timings)
        if time.time() < FIT_DEADLINE - 120:
            nt0 = time.monotonic()
            noise.extend(noise_unit(arm))
            p.dump(p.OUT / "noise_floor.json", noise)
            timing["noise_seconds"] = time.monotonic()-nt0
            p.dump(p.OUT / "arm_timings.json", timings)
    # Optional draws only after all required arms, preserving the required arm order.
    for arm in ORDER:
        if time.time() >= FIT_DEADLINE - 900 or len(timings) < 7 or any(t["status"] != "complete" for t in timings):
            break
        _, timing = run_stage(arm, BUDGETS[:3], (3, 4), args.workers, save_rows)
        timings.append(timing)
        p.dump(p.OUT / "arm_timings.json", timings)
    # Required-stage completeness is distinct from optional-draw completion.
    complete = all(any(t["arm"] == a and t["draws"] == [0, 1, 2] and t["status"] == "complete"
                       for t in timings if "draws" in t) for a in ORDER)
    p.dump(p.OUT / "arms_run.json", dict(exit_code=0 if complete else 3,
           required_complete=complete, finished_epoch=time.time()))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
