"""Additional post-hoc DESCRIPTIVE arms with fixed computational settings. No readings, no Holm, no margins; every output is labelled post-hoc.

fit       one (n, fold) unit over the five replicates and the four confirmatory checkpoints, on the frozen draws and inner
          splits (analysis.draws) and the frozen head grids:
            energy_only:{logistic,hist_gbm}  X = R-sp hull distance only
            feat_hull:{logistic,hist_gbm}    X = frozen features + h_MP(c) (fixed MP hull energy at the composition)
            reg:{ridge,hist_gbm_reg}         regress the DFT e_above_hull of the n drawn structures; score = -prediction
            delta:{ridge,hist_gbm_reg}       regress (DFT e_above_hull - R-sp hull distance); score = -(R-sp + prediction)
            residual:{logistic,hist_gbm}     the frozen energy-augmented arms, refitted as the REPRODUCTION GATE
          Selection inside each draw: every grid point fitted on the inner-training labels, scored by inner-validation AP
          of the arm's score against the binary label; best type by inner AP (tie order: linear before tree); every
          type refitted on all n labels. Continuous values are read only for drawn indices (asserted).
evaluate  assembles the units: reproduction gate (refitted residual arms == frozen S6 out-of-fold scores, max |d| <=
          1e-12, else exit 1), M = mean over replicates of pooled OOF AP, top-5 % shortlist precision, paired system
          bootstrap contrasts (B = 2000, seed 20261002, 90 % percentile), and the earlier-exposure exclusion sensitivity
          of the primary contrasts at n = 3,000 (systems of the earlier 18k subset removed).
Usage: PYTHONPATH=src python -m posthoc.s9_additions fit --n 3000 --fold 0 --out result/s9_additions
       PYTHONPATH=src python -m posthoc.s9_additions evaluate --out result/s9_additions
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S6 = ROOT / "result/s6_confirm"
CONF = ["mace_medium", "chgnet", "sevenn", "mattersim_5m"]
BUDGETS = [3000, 10000]
REPS = list(range(5))
B, SEED, COVERAGE, TOPQ = 2000, 20261002, 0.90, 0.05
RIDGE_ALPHAS = [100.0, 10.0, 1.0, 0.1]          # mirror of the logistic C grid {0.01, 0.1, 1, 10}
TIE = {"binary": ["logistic", "hist_gbm"], "regression": ["ridge", "hist_gbm_reg"]}
ARMS = {"energy_only": "binary", "feat_hull": "binary", "residual": "binary", "reg": "regression", "delta": "regression"}
OLD_SUBSET = ROOT / "cache/wbm/wbm_mace_feats_n18000_s0.npz"
OLD_SUBSET_SHA = "9f3b8f3a03a8b4d807420252cf88877ddb0df80d549e35c13ce67de99b0d0082"
WBM_SUMMARY = ROOT / "cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz"
TAG = "post-hoc descriptive (); no readings"
TAG49 = "post-hoc descriptive (); no readings"           # same-menu comparator, extended delta grid, within-fold


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ regression learners
class Ridge:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, X, t):
        from sklearn.linear_model import Ridge as R
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.m = R(alpha=self.alpha).fit(self.sc.transform(X), t)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X)).astype(np.float64)


class HistGBMReg:
    def __init__(self, learning_rate: float, max_leaf_nodes: int, max_iter: int):
        self.p = dict(learning_rate=float(learning_rate), max_leaf_nodes=int(max_leaf_nodes), max_iter=int(max_iter))

    def fit(self, X, t):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(**self.p, early_stopping=False, random_state=0).fit(X, t)
        return self

    def predict(self, X):
        return self.m.predict(X).astype(np.float64)


def reg_grid(cfg, kind: str) -> list:
    return [{"alpha": a} for a in RIDGE_ALPHAS] if kind == "ridge" else cfg.head_grid("hist_gbm")


def make_reg(kind: str, params: dict, cfg):
    if kind == "ridge":
        return Ridge(params["alpha"])
    return HistGBMReg(params["learning_rate"], params["max_leaf_nodes"], cfg.y["heads"]["hist_gbm"]["max_iter"])


def select_and_refit_reg(cfg, X, target, to_score, view, tr, va, draw, test) -> dict:
    """target(idx) -> continuous training target; to_score(pred, idx) -> score (higher = more likely stable)."""
    from analysis import metrics
    from analysis.selection import _check_val
    y_va = view(va)
    _check_val(y_va)
    view(tr)                                    # the inner-training labels belong to the draw (budget accounting)
    records, chosen, inner = [], {}, {}
    for kind in TIE["regression"]:
        best_ap, best = -np.inf, None
        for gi, params in enumerate(reg_grid(cfg, kind)):
            mdl = make_reg(kind, params, cfg).fit(X[tr], target(tr))
            a = metrics.ap(y_va, to_score(mdl.predict(X[va]), va))
            records.append({"type": kind, "grid_index": gi, "params": params, "inner_val_ap": float(a)})
            if a > best_ap:
                best_ap, best = a, params
        chosen[kind], inner[kind] = best, best_ap
    best_type = None
    for kind in TIE["regression"]:
        if best_type is None or inner[kind] > inner[best_type]:
            best_type = kind
    view(draw)
    scores = {k: to_score(make_reg(k, chosen[k], cfg).fit(X[draw], target(draw)).predict(X[test]), test)
              for k in TIE["regression"]}
    return {"best": best_type, "scores": scores, "records": records, "inner_ap": inner, "chosen": chosen}


# ------------------------------------------------------------------------------------------------ fit
def build_cfg(n: int):
    from analysis import stages as S
    from analysis.config import Config
    cfg = S.stage_config(Config.build("confirm", ["stability"], split_key="chemsys_primary"), "primary", [n])
    if cfg.part != "CONFIRM" or cfg.replicates_max != 5:
        raise SystemExit(f"unexpected config: part {cfg.part}, replicates {cfg.replicates_max}")
    return cfg


def h_mp(inp) -> np.ndarray:
    """Fixed MP hull formation energy at each composition (checkpoint-independent in exact arithmetic). Taken from
    mace_medium; the other checkpoints must agree within 1e-5 eV/atom (observed 2026-10-04: CHGNet and SevenNet
    <= 2e-15; MatterSim one structure at 1.0e-6, a rounding of that checkpoint's stored value)."""
    hs = [inp.refs[bb]["r_sp"]["e_form"] - inp.refs[bb]["r_sp"]["e_hull"] for bb in CONF]
    for bb, h in zip(CONF[1:], hs[1:]):
        dev = np.abs(h - hs[0])
        if not np.isfinite(dev).all() or dev.max() > 1e-5:
            raise SystemExit(f"h_MP differs between mace_medium and {bb} (max {np.nanmax(dev):.3e})")
        print(f"h_MP {bb} vs mace_medium: max |d| {dev.max():.1e}; {int((dev > 1e-7).sum())} structure(s) > 1e-7",
              flush=True)
    if not np.isfinite(hs[0]).all():
        raise SystemExit("h_MP not finite")
    return hs[0]


def fit_unit(n: int, f: int, out: Path, threads: int) -> Path:
    from threadpoolctl import threadpool_limits

    from analysis.data import LabelVault
    from analysis.draws import draw_labels, inner_split, outer_train_test
    from analysis.pipeline import load_inputs
    from analysis.selection import select_and_refit
    t0 = time.time()
    cfg = build_cfg(n)
    with threadpool_limits(threads):
        inp = load_inputs(cfg, lambda m: print(m, flush=True))
        part = inp.part
        vals = np.asarray(inp.values["stability"], dtype=np.float64)
        hmp = h_mp(inp)
        vault = LabelVault(inp.labels["stability"], "stability")
        _, test = outer_train_test(part, f)
        methods, scores, recs = [], {}, []
        for r in REPS:
            draw = draw_labels(cfg, part, f, r, n)
            tr, va = inner_split(cfg, part, draw, f, r)
            key = ("stability", n, f, r)
            view = vault.open(key, draw)
            cont = guarded(vals, draw)

            for bb in CONF:
                X = inp.feats[bb].astype(np.float64)
                rsp = np.asarray(inp.refs[bb]["r_sp"]["e_hull"], dtype=np.float64)
                binary = {"energy_only": rsp[:, None], "feat_hull": np.hstack([X, hmp[:, None]]),
                          "residual": np.hstack([inp.feats[bb].astype(np.float64), rsp[:, None]])}
                for arm, Xa in binary.items():
                    for m in TIE["binary"]:
                        rr = select_and_refit([m], [m], {m: cfg.head_grid(m)}, {m: Xa}, view, tr, va, draw, test,
                                              cfg.y, "cpu")
                        scores[(f"{arm}:{m}:{bb}", r)] = rr["scores"][m]
                        recs += [{"arm": arm, "backbone": bb, "replicate": r, **x} for x in rr["records"]]
                    best = max(TIE["binary"], key=lambda m: (max(x["inner_val_ap"] for x in recs if x["arm"] == arm and
                                                                 x["backbone"] == bb and x["replicate"] == r and
                                                                 x["type"] == m), -TIE["binary"].index(m)))
                    scores[(f"best_{arm}:{bb}", r)] = scores[(f"{arm}:{best}:{bb}", r)]
                for arm in ("reg", "delta"):
                    if arm == "reg":
                        res = select_and_refit_reg(cfg, X, cont, lambda p, idx: -p, view, tr, va, draw, test)
                    else:
                        res = select_and_refit_reg(cfg, X, lambda idx: cont(idx) - rsp[idx],
                                                   lambda p, idx, rsp=rsp: -(rsp[idx] + p), view, tr, va, draw, test)
                    for k, s in res["scores"].items():
                        scores[(f"{arm}:{k}:{bb}", r)] = s
                    scores[(f"best_{arm}:{bb}", r)] = res["scores"][res["best"]]
                    recs += [{"arm": arm, "backbone": bb, "replicate": r, **x} for x in res["records"]]
            vault.close(key)
        methods = sorted({k for k, _ in scores})
        arr = np.stack([np.stack([scores[(m, r)] for r in REPS]) for m in methods])   # (M, R, len(test))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"units/unit_n{n}_f{f}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, material_id=part.ids[test], test_idx=test, method=np.array(methods), score=arr)
    (out / f"units/unit_n{n}_f{f}_selection.json").write_text(json.dumps(recs, default=float))
    print(f"unit n={n} f={f}: {len(methods)} methods x {len(REPS)} replicates x {len(test)} test structures in "
          f"{time.time() - t0:.0f}s -> {dst}", flush=True)
    return dst


# ------------------------------------------------------------------------------------------------ evaluate
def assemble(out: Path, n: int, part) -> tuple:
    units = [np.load(out / f"units/unit_n{n}_f{f}.npz", allow_pickle=False) for f in range(5)]
    methods = [str(m) for m in units[0]["method"]]
    if any([str(m) for m in u["method"]] != methods for u in units):
        raise SystemExit(f"n={n}: method lists differ between units")
    oof = np.full((len(methods), len(REPS), len(part.ids)), np.nan)
    for u in units:
        idx = np.array([part.pos[m] for m in u["material_id"]])
        oof[:, :, idx] = u["score"]
    if not np.isfinite(oof).all():
        raise SystemExit(f"n={n}: out-of-fold scores incomplete")
    return methods, oof


def guarded(vals: np.ndarray, draw: np.ndarray):
    """Accessor for continuous labels restricted to the replicate's draw (the same DFT calculations as its n labels)."""
    drawn = set(np.asarray(draw).tolist())

    def cont(idx):
        if not set(np.asarray(idx).tolist()) <= drawn:
            raise SystemExit("continuous label requested outside the draw")
        v = vals[idx]
        if not np.isfinite(v).all():
            raise SystemExit("non-finite continuous label inside the draw")
        return v
    return cont


def reproduction_gate(out: Path, part, tol: float = 1e-12) -> list:
    from posthoc.s7_additions import load_scores, row
    gate = []
    for n in BUDGETS:
        methods, oof = assemble(out, n, part)
        s6 = load_scores(S6 / f"primary_stability_n{n}/final/scores_stability_n{n}.npz", part)
        for bb in CONF:
            for m in TIE["binary"]:
                d = float(np.max(np.abs(oof[methods.index(f"residual:{m}:{bb}")] - row(s6, f"residual:{m}:{bb}"))))
                gate.append({"n": n, "method": f"residual:{m}:{bb}", "max_abs_diff": d, "pass": d <= tol})
    return gate


def evaluate(out: Path) -> int:
    from analysis import inference as I
    from analysis import metrics as Mx
    from posthoc.s7_additions import Part, load_scores, row, shortlist
    t0 = time.time()
    part = Part()
    y = part.y["stability"]
    gate = reproduction_gate(out, part)                     # REPRODUCTION GATE first: fail before any bootstrap
    write_csv(out / "reproduction_gate.csv", gate)
    if not all(g["pass"] for g in gate):
        raise SystemExit(f"REPRODUCTION GATE FAILED: {[g for g in gate if not g['pass']][:4]}")
    print(f"reproduction gate: {len(gate)}/{len(gate)} pass", flush=True)
    res_rows, con_rows = [], []
    for n in BUDGETS:
        methods, oof = assemble(out, n, part)
        s6 = load_scores(S6 / f"primary_stability_n{n}/final/scores_stability_n{n}.npz", part)
        get = {m: oof[i] for i, m in enumerate(methods)}
        for bb in CONF:
            for m in (f"ref:r_sp:{bb}", f"best_head:{bb}", f"residual:hist_gbm:{bb}"):
                get[m] = row(s6, m)
        point = {m: float(np.mean([Mx.ap(y, v[r]) for r in REPS])) for m, v in get.items()}
        for m, v in sorted(get.items()):
            tp = k = 0
            for r in REPS:
                a, b, _ = shortlist(y, v[r], part.fold, part.id_rank, TOPQ)
                tp, k = tp + a, k + b
            res_rows.append({"n": n, "method": m, "M": point[m],
                             "M_r": json.dumps([round(Mx.ap(y, v[r]), 12) for r in REPS]),
                             "precision_top5pct": tp / k})
        pairs = []
        for bb in CONF:
            pairs += [(f"best_energy_only:{bb}", f"ref:r_sp:{bb}"), (f"best_feat_hull:{bb}", f"ref:r_sp:{bb}"),
                      (f"best_reg:{bb}", f"ref:r_sp:{bb}"), (f"best_delta:{bb}", f"ref:r_sp:{bb}"),
                      (f"residual:hist_gbm:{bb}", f"ref:r_sp:{bb}"), (f"best_feat_hull:{bb}", f"best_head:{bb}"),
                      (f"best_reg:{bb}", f"best_head:{bb}"), (f"best_delta:{bb}", f"residual:hist_gbm:{bb}"),
                      (f"residual:hist_gbm:{bb}", f"best_energy_only:{bb}")]
        need = sorted({m for p in pairs for m in p})
        star = I.bootstrap({m: [Mx.SortedScore(get[m][r], y) for r in REPS] for m in need}, part.sys_index, part.S,
                           B, SEED)
        for a, b in pairs:
            lo, hi = I.interval(star[a] - star[b], COVERAGE)
            con_rows.append({"n": n, "backbone": a.split(":")[-1], "a": a, "b": b, "estimate": point[a] - point[b],
                             "ci_lo": lo, "ci_hi": hi})
        print(f"n={n}: {len(get)} methods, {len(pairs)} contrasts ({time.time() - t0:.0f}s)", flush=True)
    write_csv(out / "arms_metrics.csv", res_rows)
    write_csv(out / "arms_contrasts.csv", con_rows)
    write_csv(out / "exposure_exclusion.csv", exposure_exclusion(part))
    return write_manifest(out, gate, t0)


def planned_contrasts(out: Path, tol: float = 1e-12) -> int:
    """ finding 5: the specification planned 'best arm − energy-augmented HistGBM' for every arm; evaluate
    reported delta − residual, residual − energy_only (reversed) and omitted feat_hull − residual and reg − residual. This
    step computes the planned pairs with the same paired bootstrap (seed 20261002: the weights of draw b do not depend on
    which methods are evaluated, so the intervals equal those of a joint run) and checks the overlapping pair delta −
    residual against arms_contrasts.csv (|d| <= 1e-12). Writes arms_contrasts_planned.csv (planned set + additional)."""
    from analysis import inference as I
    from analysis import metrics as Mx
    from posthoc.s7_additions import Part, load_scores, row
    t0 = time.time()
    part = Part()
    y = part.y["stability"]
    old = {(r["n"], r["a"], r["b"]): r for r in csv.DictReader(open(out / "arms_contrasts.csv"))}
    rows, checks = [], []
    arms = ["best_energy_only", "best_feat_hull", "best_reg", "best_delta"]
    for n in BUDGETS:
        methods, oof = assemble(out, n, part)
        s6 = load_scores(S6 / f"primary_stability_n{n}/final/scores_stability_n{n}.npz", part)
        get = {}
        for bb in CONF:
            for a in arms:
                get[f"{a}:{bb}"] = oof[methods.index(f"{a}:{bb}")]
            get[f"residual:hist_gbm:{bb}"] = row(s6, f"residual:hist_gbm:{bb}")
        point = {m: float(np.mean([Mx.ap(y, v[r]) for r in REPS])) for m, v in get.items()}
        star = I.bootstrap({m: [Mx.SortedScore(v[r], y) for r in REPS] for m, v in get.items()}, part.sys_index,
                           part.S, B, SEED)
        for bb in CONF:
            for a in arms:
                ka, kb = f"{a}:{bb}", f"residual:hist_gbm:{bb}"
                lo, hi = I.interval(star[ka] - star[kb], COVERAGE)
                rows.append({"n": n, "backbone": bb, "a": ka, "b": kb, "estimate": point[ka] - point[kb],
                             "ci_lo": lo, "ci_hi": hi, "plan": "planned ()"})
                o = old.get((str(n), ka, kb))
                if o is not None:
                    d = max(abs(float(o[k]) - v) for k, v in (("estimate", point[ka] - point[kb]), ("ci_lo", lo),
                                                                ("ci_hi", hi)))
                    checks.append(d)
        print(f"n={n}: planned contrasts done ({time.time() - t0:.0f}s)", flush=True)
    if not checks or max(checks) > tol:
        raise SystemExit(f"cross-check with arms_contrasts.csv failed: {checks}")
    planned_keys = {(r["n"], r["a"], r["b"]) for r in rows}
    for (n, a, b), r in old.items():
        if (int(n), a, b) in planned_keys:
            continue
        is_planned = b.startswith("ref:r_sp") or (a.startswith("best_feat_hull") and b.startswith("best_head"))
        rows.append({"n": int(n), "backbone": r["backbone"], "a": a, "b": b, "estimate": float(r["estimate"]),
                     "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                     "plan": "planned ()" if is_planned and not a.startswith("residual") else "additional"})
    write_csv(out / "arms_contrasts_planned.csv", rows)
    print(f"planned contrasts: {len(rows)} rows; cross-check {len(checks)} pairs, max |d| {max(checks):.1e}", flush=True)
    return 0


RIDGE_EXT = [1e5, 1e4, 1e3] + RIDGE_ALPHAS                    # (b): near-no-op options added for extended comparisons


def delta_ext_fit(out: Path, n: int, threads: int) -> Path:
    """(b): extended-grid ridge delta heads on the frozen draws / inner splits for all five folds at budget n.
    Reproduction check: the inner-validation APs of the four original alphas must equal the recorded ones (|d| <= 1e-12)."""
    from threadpoolctl import threadpool_limits

    from analysis import metrics
    from analysis.draws import draw_labels, inner_split, outer_train_test
    from analysis.pipeline import load_inputs
    cfg = build_cfg(n)
    rows, checks = [], []
    with threadpool_limits(threads):
        inp = load_inputs(cfg, lambda m: print(m, flush=True))
        part = inp.part
        vals = np.asarray(inp.values["stability"], dtype=np.float64)
        ybin = np.asarray(inp.labels["stability"])
        oof = {bb: np.full((len(REPS), part.n), np.nan) for bb in CONF}
        for f in range(5):
            recs = json.load(open(out / f"units/unit_n{n}_f{f}_selection.json"))
            _, test = outer_train_test(part, f)
            for r in REPS:
                draw = draw_labels(cfg, part, f, r, n)
                tr, va = inner_split(cfg, part, draw, f, r)
                cont = guarded(vals, draw)
                for bb in CONF:
                    X = inp.feats[bb].astype(np.float64)
                    rsp = np.asarray(inp.refs[bb]["r_sp"]["e_hull"], dtype=np.float64)
                    tgt = lambda idx: cont(idx) - rsp[idx]
                    best = (-np.inf, None)
                    for a in RIDGE_EXT:
                        m = Ridge(a).fit(X[tr], tgt(tr))
                        ap = metrics.ap(ybin[va], -(rsp[va] + m.predict(X[va])))
                        if a in RIDGE_ALPHAS:
                            rec = [x for x in recs if x["arm"] == "delta" and x["backbone"] == bb and x["replicate"] == r
                                   and x["type"] == "ridge" and float(x["params"]["alpha"]) == a]
                            checks.append(abs(rec[0]["inner_val_ap"] - ap))
                        if ap > best[0]:
                            best = (ap, a)
                    hgb = max(x["inner_val_ap"] for x in recs if x["arm"] == "delta" and x["backbone"] == bb
                              and x["replicate"] == r and x["type"] == "hist_gbm_reg")
                    ridge_score = -(rsp[test] + Ridge(best[1]).fit(X[draw], tgt(draw)).predict(X[test]))
                    rows.append({"n": n, "fold": f, "replicate": r, "backbone": bb, "ridge_alpha": best[1],
                                 "ridge_inner_ap": best[0], "hgb_inner_ap": hgb,
                                 "selected": "ridge" if best[0] >= hgb else "hist_gbm_reg"})
                    oof[bb][r, test] = ridge_score
                    rows[-1]["_score_key"] = (bb, r)
    if not checks or max(checks) > 1e-12:
        raise SystemExit(f"extended-grid reproduction check failed: max {max(checks) if checks else None}")
    dst = out / f"delta_ext/delta_ext_n{n}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, material_id=part.ids, method=np.array([f"delta_ext_ridge:{bb}" for bb in CONF]),
                        score=np.stack([oof[bb] for bb in CONF]))
    for r_ in rows:
        r_.pop("_score_key", None)
    write_csv(out / f"delta_ext/delta_ext_selection_n{n}.csv", rows, TAG49)
    print(f"delta_ext n={n}: {len(rows)} unit-replicates; reproduction check {len(checks)} inner APs, max |d| "
          f"{max(checks):.1e}; selected {dict((k, sum(1 for x in rows if x['selected'] == k)) for k in ('ridge', 'hist_gbm_reg'))}",
          flush=True)
    return dst


def extended_comparison(out: Path, b: int = None) -> int:
    """(a)-(c): same-menu comparator, extended delta, within-fold point estimates; one paired bootstrap per n."""
    import pandas as pd

    from analysis import inference as I
    from analysis import metrics as Mx
    from posthoc.s7_additions import Part, load_scores, row, shortlist
    B_ = b or B
    t0 = time.time()
    part = Part()
    y = part.y["stability"]
    con, met, wf = [], [], []
    for n in BUDGETS:
        methods, oof = assemble(out, n, part)
        d = S6 / f"primary_stability_n{n}/final"
        sc = load_scores(d / f"scores_stability_n{n}.npz", part)
        sel = pd.read_csv(d / "selection_choices.csv")
        ext = np.load(out / f"delta_ext/delta_ext_n{n}.npz", allow_pickle=False)
        ext_idx = np.array([part.pos[m] for m in ext["material_id"]])
        ext_sel = list(csv.DictReader(open(out / f"delta_ext/delta_ext_selection_n{n}.csv")))
        get = {}
        for bb in CONF:
            s = sel[(sel.menu == "heads") & (sel.backbone == bb) & (sel.chosen_for_type.astype(str) == "True")]
            two = np.full((len(REPS), len(y)), np.nan)
            picks = {"logistic": 0, "hist_gbm": 0}
            for rid, g in s.groupby("run_id"):
                best = None
                for t in ("logistic", "hist_gbm"):                      # tie order: logistic before HistGBM
                    a = g[g.type == t].inner_val_ap.max()
                    if best is None or a > best[1]:
                        best = (t, a)
                f = int(rid.split("__f")[1].split("__")[0]); r = int(rid.split("__r")[1])
                idx = np.flatnonzero(part.fold == f)
                two[r, idx] = row(sc, f"head:{best[0]}:{bb}")[r, idx]
                picks[best[0]] += 1
            if not np.isfinite(two).all():
                raise SystemExit(f"n={n} {bb}: same-menu head incomplete")
            get[f"head_2type:{bb}"] = two
            dx = np.full((len(REPS), len(y)), np.nan)
            ridge = np.full((len(REPS), len(y)), np.nan)
            ridge[:, ext_idx] = ext["score"][list(ext["method"]).index(f"delta_ext_ridge:{bb}")]
            hgb = oof[methods.index(f"delta:hist_gbm_reg:{bb}")]
            for x in ext_sel:
                if x["backbone"] != bb:
                    continue
                f, r = int(x["fold"]), int(x["replicate"])
                idx = np.flatnonzero(part.fold == f)
                dx[r, idx] = (ridge if x["selected"] == "ridge" else hgb)[r, idx]
            if not np.isfinite(dx).all():
                raise SystemExit(f"n={n} {bb}: extended delta incomplete")
            get[f"best_delta_ext:{bb}"] = dx
            for m in (f"best_feat_hull:{bb}", f"best_reg:{bb}", f"best_delta:{bb}", f"best_energy_only:{bb}",
                      f"energy_only:logistic:{bb}", f"energy_only:hist_gbm:{bb}"):
                get[m] = oof[methods.index(m)]
            for m in (f"ref:r_sp:{bb}", f"residual:hist_gbm:{bb}", f"best_head:{bb}"):
                get[m] = row(sc, m)
            met.append({"n": n, "backbone": bb, "method": f"head_2type:{bb}", "picks": json.dumps(picks)})
            alphas = [float(x["ridge_alpha"]) for x in ext_sel if x["backbone"] == bb and x["selected"] == "ridge"]
            met.append({"n": n, "backbone": bb, "method": f"best_delta_ext:{bb}",
                        "picks": json.dumps({"ridge": len(alphas), "hist_gbm_reg": sum(1 for x in ext_sel if x["backbone"] == bb
                                                                                      and x["selected"] == "hist_gbm_reg"),
                                             "ridge_alphas": sorted(set(alphas))})})
        point = {m: float(np.mean([Mx.ap(y, v[r]) for r in REPS])) for m, v in get.items()}
        for rr in met:
            if rr["n"] == n:
                v = get[rr["method"]]
                tp = k = 0
                for r in REPS:
                    a_, b2, _ = shortlist(y, v[r], part.fold, part.id_rank, TOPQ)
                    tp, k = tp + a_, k + b2
                rr["M"], rr["precision_top5pct"] = point[rr["method"]], tp / k
        pairs = []
        for bb in CONF:
            pairs += [(f"best_feat_hull:{bb}", f"head_2type:{bb}"), (f"best_reg:{bb}", f"head_2type:{bb}"),
                      (f"head_2type:{bb}", f"best_head:{bb}"), (f"best_delta_ext:{bb}", f"ref:r_sp:{bb}"),
                      (f"best_delta_ext:{bb}", f"residual:hist_gbm:{bb}")]
        need = sorted({m for pr in pairs for m in pr})
        star = I.bootstrap({m: [Mx.SortedScore(get[m][r], y) for r in REPS] for m in need}, part.sys_index, part.S, B_, SEED)
        for a_, b2 in pairs:
            lo, hi = I.interval(star[a_] - star[b2], COVERAGE)
            con.append({"n": n, "backbone": a_.split(":")[-1], "a": a_, "b": b2, "estimate": point[a_] - point[b2],
                        "ci_lo": lo, "ci_hi": hi})
        for bb in CONF:
            for m in (f"ref:r_sp:{bb}", f"energy_only:logistic:{bb}", f"energy_only:hist_gbm:{bb}",
                      f"best_energy_only:{bb}", f"residual:hist_gbm:{bb}", f"best_delta:{bb}", f"best_delta_ext:{bb}"):
                tot = val = 0.0
                for r in REPS:
                    for f in range(5):
                        idx = np.flatnonzero(part.fold == f)
                        val += len(idx) * Mx.ap(y[idx], get[m][r, idx])
                        tot += len(idx)
                wf.append({"n": n, "backbone": bb, "method": m, "within_fold_ap": val / tot, "pooled_M": point[m]})
        print(f"extended_comparison n={n}: {len(pairs)} contrasts ({time.time() - t0:.0f}s)", flush=True)
    write_csv(out / "extended_comparison_contrasts.csv", con, TAG49)
    write_csv(out / "extended_comparison_metrics.csv", met, TAG49)
    write_csv(out / "extended_comparison_within_fold.csv", wf, TAG49)
    return 0


def write_manifest(out: Path, gate: list, t0: float) -> int:
    man = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tag": TAG, "tag_d049": TAG49,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "code_sha256": sha(Path(__file__)),
           "outputs": {p.name: sha(p) for p in sorted(out.glob("*.csv"))},
           "units": {p.name: sha(p) for p in sorted((out / "units").glob("*.npz"))},
           "delta_ext": {p.name: sha(p) for p in sorted((out / "delta_ext").glob("*"))},
           "reproduction_gate": f"{len(gate)}/{len(gate)} pass (max |d| {max(g['max_abs_diff'] for g in gate):.1e})"}
    (out / "MANIFEST_s9.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"evaluate done in {time.time() - t0:.0f}s", flush=True)
    return 0


def old_subset_systems() -> set:
    import gzip
    if sha(OLD_SUBSET) != OLD_SUBSET_SHA:
        raise SystemExit("old-subset file hash differs from the provenance record")
    # the cached ids are an object array; the file's SHA-256 is verified above against the provenance record, so
    # loading it with pickle enabled reads only the bytes that were recorded (2026-10-04: allow_pickle=False failed here)
    mids = [str(m) for m in np.load(OLD_SUBSET, allow_pickle=True)["mids"]]
    form = {}
    with gzip.open(WBM_SUMMARY, "rt") as fh:
        rd = csv.DictReader(fh)
        col = "formula" if "formula" in rd.fieldnames else next(c for c in rd.fieldnames if "formula" in c.lower())
        for r in rd:
            form[r["material_id"]] = r[col]
    return {"-".join(sorted(set(re.findall(r"[A-Z][a-z]?", form[m])))) for m in mids}


def exposure_exclusion(part) -> list:
    """Primary contrasts at n = 3,000 on CONFIRM without the systems that appear in the earlier 18k subset."""
    import pandas as pd

    from analysis import inference as I
    from analysis import metrics as Mx
    from posthoc.s7_additions import LABELS, load_scores, row
    df = pd.read_csv(LABELS, dtype={"material_id": str, "chemsys": str}, keep_default_na=False)
    systems = np.array(sorted(set(df["chemsys"])))
    exposed = np.isin(systems, sorted(old_subset_systems()))
    if int(exposed.sum()) != 11159:
        raise SystemExit(f"exposed CONFIRM systems {int(exposed.sum())} != 11,159 (provenance record)")
    keep = ~exposed[part.sys_index]
    idx = np.flatnonzero(keep)
    sub_sys = np.unique(part.sys_index[keep])
    remap = np.full(part.S, 0)
    remap[sub_sys] = np.arange(len(sub_sys))                 # excluded structures are never gathered (pop below)
    rows = []
    for task in ("stability", "band_gap"):
        sc = load_scores(S6 / f"primary_{task}_n3000/final/scores_{task}_n3000.npz", part)
        y = part.y[task]
        pops = {}
        for bb in CONF:
            if task == "stability":
                fin_is = idx[np.isfinite(row(sc, f"ref:r_is2re:{bb}")[0, idx])]
                pops[("d_sp", bb)] = (f"best_head:{bb}", f"ref:r_sp:{bb}", idx)
                pops[("d_is", bb)] = (f"best_head:{bb}", f"ref:r_is2re:{bb}", fin_is)
            else:
                pops[("g", bb)] = (f"best_head:{bb}", "best_control", idx)
        specs, point = {}, {}
        for key, (a, b, pop) in pops.items():
            for side, m in (("a", a), ("b", b)):
                specs[(key, side)] = [Mx.SortedScore(row(sc, m)[r], y, pop) for r in REPS]
                point[(key, side)] = float(np.mean([Mx.ap(y[pop], row(sc, m)[r][pop]) for r in REPS]))
        star = I.bootstrap(specs, remap[part.sys_index], len(sub_sys), B, SEED)
        for key, (a, b, pop) in pops.items():
            lo, hi = I.interval(star[(key, "a")] - star[(key, "b")], COVERAGE)
            rows.append({"task": task, "contrast": key[0], "backbone": key[1],
                         "estimate": point[(key, "a")] - point[(key, "b")], "ci_lo": lo, "ci_hi": hi,
                         "n_structures": int(len(pop)), "n_systems": int(len(np.unique(part.sys_index[pop]))),
                         "excluded_systems": int(exposed.sum())})
    return rows


def write_csv(path: Path, rows: list, tag: str = TAG) -> None:
    if not rows:
        raise SystemExit(f"{path}: no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["note"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "note": tag})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "evaluate", "exposure", "planned", "delta_ext", "extended_comparison"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "fit":
        if a.n not in BUDGETS or a.fold not in range(5):
            raise SystemExit("fit needs --n in {3000, 10000} and --fold in 0..4")
        fit_unit(a.n, a.fold, out, a.threads)
        return 0
    if a.cmd == "delta_ext":
        if a.n not in BUDGETS:
            raise SystemExit("delta_ext needs --n in {3000, 10000}")
        delta_ext_fit(out, a.n, a.threads)
        return 0
    if a.cmd == "extended_comparison":
        extended_comparison(out)
        gate = [dict(r, max_abs_diff=float(r["max_abs_diff"]), **{"pass": r["pass"] == "True"})
                for r in csv.DictReader(open(out / "reproduction_gate.csv"))]
        return write_manifest(out, gate, time.time())
    if a.cmd == "planned":
        planned_contrasts(out)
        gate = [dict(r, max_abs_diff=float(r["max_abs_diff"]), **{"pass": r["pass"] == "True"})
                for r in csv.DictReader(open(out / "reproduction_gate.csv"))]
        return write_manifest(out, gate, time.time())
    if a.cmd == "exposure":                                   # resume after the bootstraps: exposure table + manifest
        from posthoc.s7_additions import Part
        t0 = time.time()
        part = Part()
        gate = [dict(r, max_abs_diff=float(r["max_abs_diff"]), **{"pass": r["pass"] == "True"})
                for r in csv.DictReader(open(out / "reproduction_gate.csv"))]
        if not gate or not all(g["pass"] for g in gate):
            raise SystemExit("reproduction gate file missing or not all pass")
        for f in ("arms_metrics.csv", "arms_contrasts.csv"):
            if not (out / f).is_file():
                raise SystemExit(f"{f} missing: run evaluate")
        write_csv(out / "exposure_exclusion.csv", exposure_exclusion(part))
        return write_manifest(out, gate, t0)
    return evaluate(out)


if __name__ == "__main__":
    sys.exit(main())
