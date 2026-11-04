"""Additional post-hoc DESCRIPTIVE additions from existing CONFIRM outputs . CPU only, no refitting.

Items: (a) energy-augmented budget curves; (b) shortlist precision/recall/hits at 1 % and 5 % of each fold's test set;
(c) replicate-level primary contrasts (mean asserted equal to the frozen bundle); (d) within-fold AP diagnostic;
(e) seed-0 reduced-menu (no CGCNN) control for the fold-seed comparison; (h) zero-probability ties; (i) covered /
out-of-coverage prevalence. Every output is post-hoc descriptive: no readings, no Holm, no margins.
Usage: PYTHONPATH=src python -m posthoc.s7_additions --out result/s7_additions [--items a,b,c,d,e,h,i]
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import inference as I
from analysis import metrics as Mx

ROOT = Path(__file__).resolve().parents[2]
S6 = ROOT / "result/s6_confirm"
LABELS = ROOT / "records//phaseA_export/confirm_labels_and_systems.csv.gz"
BUNDLE = S6 / "final/bundle.json"
YAML = ROOT / "research/protocol_legacy.yaml"
CONF = ["mace_medium", "chgnet", "sevenn", "mattersim_5m"]
EXTRA = {"orb": "exploratory", "mace_small": "secondary"}
BUDGETS = [300, 1000, 3000, 10000]
B, SEED, COVERAGE = 2000, 20261002, 0.90
TAG = "post-hoc descriptive (); no readings"
READ: dict = {}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def note(p: Path) -> Path:
    p = Path(p)
    key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    READ[key] = sha(p)
    return p


def write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["note"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "note": TAG})


# ------------------------------------------------------------------------------------------------ inputs
class Part:
    def __init__(self):
        df = pd.read_csv(note(LABELS), dtype={"material_id": str, "chemsys": str}, keep_default_na=False)
        self.ids = df["material_id"].to_numpy()
        self.fold = df["fold"].to_numpy(dtype=np.int64)
        self.y = {"stability": df["y_stability"].to_numpy(dtype=np.int8),
                  "band_gap": df["y_band_gap"].to_numpy(dtype=np.int8)}
        systems = np.array(sorted(set(df["chemsys"])))
        self.sys_index = np.searchsorted(systems, df["chemsys"].to_numpy())
        self.S = len(systems)
        self.pos = {m: i for i, m in enumerate(self.ids)}
        self.id_rank = np.argsort(np.argsort(self.ids, kind="mergesort"), kind="mergesort")  # material_id order


def load_scores(path: Path, part: Part, fold_col: np.ndarray | None = None) -> dict:
    z = np.load(note(path), allow_pickle=False)
    mid = z["material_id"]
    idx = np.array([part.pos[m] for m in mid])
    if len(idx) != len(part.ids) or len(set(idx.tolist())) != len(idx):
        raise SystemExit(f"{path}: material_id set differs from the CONFIRM label table")
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))          # file row of each label-table row
    fold = z["fold"][inv]
    if fold_col is None and not np.array_equal(fold, part.fold):
        raise SystemExit(f"{path}: folds differ from the primary split")
    return {"method": [str(m) for m in z["method"]], "rep": z["replicate"].tolist(), "score": z["score"][:, :, inv],
            "fold": fold}


def row(sc: dict, method: str) -> np.ndarray:
    return sc["score"][sc["method"].index(method)]          # (R, N)


# ------------------------------------------------------------------------------------------------ (a)
def item_a(out: Path, part: Part) -> dict:
    rows = []
    for n in BUDGETS:
        d = S6 / f"primary_stability_n{n}/final"
        met = {r["method"]: r for r in csv.DictReader(open(note(d / f"metrics_stability_n{n}.csv"))) if r["population"] == "all"}
        con = {r["contrast"]: r for r in csv.DictReader(open(note(d / f"contrasts_stability_n{n}.csv"))) if r["population"] == "all"}
        for bb in CONF + list(EXTRA):
            r = {"n": n, "backbone": bb, "role": "confirmatory" if bb in CONF else EXTRA[bb]}
            for k, m in (("M_best_head", f"best_head:{bb}"), ("M_energy_aug_hist_gbm", f"residual:hist_gbm:{bb}"),
                         ("M_energy_aug_logistic", f"residual:logistic:{bb}"), ("M_r_sp", f"ref:r_sp:{bb}"),
                         ("M_r_is2re", f"ref:r_is2re:{bb}")):
                r[k] = met[m]["M"] if m in met else ""
            for k, c in (("hist_gbm_minus_r_sp", f"residual_hist_gbm_vs_r_sp:{bb}"),
                         ("logistic_minus_r_sp", f"residual_logistic_vs_r_sp:{bb}"),
                         ("hist_gbm_minus_best_head", f"residual_hist_gbm_vs_best_head:{bb}")):
                if c in con:
                    r[k], r[k + "_lo"], r[k + "_hi"] = con[c]["estimate"], con[c]["ci_lo"], con[c]["ci_hi"]
                else:
                    r[k] = r[k + "_lo"] = r[k + "_hi"] = ""
            r["prevalence"] = met[f"best_head:{bb}"]["prevalence"]
            rows.append(r)
    write(out / "a_energy_augmented_budget/energy_augmented_budget.csv", rows)
    return {"rows": len(rows)}


# ------------------------------------------------------------------------------------------------ (b)
def shortlist(y, s, fold, id_rank, q):
    """Per fold: k = ceil(q * N_f); top-k by score desc, ties by material_id asc; NaN scores ranked last."""
    tp = k_tot = pos = 0
    s = np.where(np.isfinite(s), s, -np.inf)
    for f in np.unique(fold):
        m = np.flatnonzero(fold == f)
        k = int(np.ceil(q * len(m)))
        order = np.lexsort((id_rank[m], -s[m]))[:k]
        tp += int(y[m][order].sum())
        k_tot += k
        pos += int(y[m].sum())
    return tp, k_tot, pos


def item_b(out: Path, part: Part) -> dict:
    rows = []
    for n in (3000, 10000):
        for task in ("stability", "band_gap"):
            sc = load_scores(S6 / f"primary_{task}_n{n}/final/scores_{task}_n{n}.npz", part)
            y = part.y[task]
            if task == "stability":
                meths = [(bb, m) for bb in CONF for m in (f"best_head:{bb}", f"residual:hist_gbm:{bb}", f"ref:r_sp:{bb}",
                                                         f"ref:r_is2re:{bb}")]
            else:
                meths = [(bb, f"best_head:{bb}") for bb in CONF] + [("controls", "best_control")] + \
                        [("controls", m) for m in sc["method"] if m.startswith("control:")]
            for bb, m in meths:
                S = row(sc, m)
                n_nan = int((~np.isfinite(S[0])).sum())
                for q in (0.01, 0.05):
                    per = [shortlist(y, S[r], part.fold, part.id_rank, q) for r in range(S.shape[0])]
                    prec = [tp / k for tp, k, _ in per]
                    rec = [tp / p for tp, _, p in per]
                    rows.append({"task": task, "n": n, "backbone": bb, "method": m, "shortlist_fraction": q,
                                 "k_total": per[0][1], "positives_total": per[0][2], "prevalence": float(y.mean()),
                                 "precision_mean": float(np.mean(prec)), "precision_min": float(np.min(prec)),
                                 "precision_max": float(np.max(prec)), "recall_mean": float(np.mean(rec)),
                                 "hits_mean": float(np.mean([tp for tp, _, _ in per])),
                                 "hits_per_replicate": json.dumps([tp for tp, _, _ in per]), "missing_scores_ranked_last": n_nan})
    write(out / "b_shortlist/shortlist.csv", rows)
    return {"rows": len(rows)}


# ------------------------------------------------------------------------------------------------ (c)
def item_c(out: Path, part: Part) -> dict:
    bundle = json.load(open(note(BUNDLE)))
    st = load_scores(S6 / "primary_stability_n3000/final/scores_stability_n3000.npz", part)
    bg = load_scores(S6 / "primary_band_gap_n3000/final/scores_band_gap_n3000.npz", part)
    defs = []
    for bb in CONF:
        defs += [("E1:d_sp:" + bb, st, "stability", f"best_head:{bb}", f"ref:r_sp:{bb}"),
                 ("E1:d_is:" + bb, st, "stability", f"best_head:{bb}", f"ref:r_is2re:{bb}"),
                 ("E2:g:" + bb, bg, "band_gap", f"best_head:{bb}", "best_control"),
                 ("C4:q:" + bb, st, "stability", f"head:logistic:{bb}", f"head:mlp_ensemble:{bb}")]
    rows, worst = [], 0.0
    for cid, sc, task, a, b in defs:
        A, Bm, y = row(sc, a), row(sc, b), part.y[task]
        pop = np.flatnonzero(np.isfinite(Bm[0]) & np.isfinite(A[0]))
        d = [Mx.ap(y[pop], A[r][pop]) - Mx.ap(y[pop], Bm[r][pop]) for r in range(A.shape[0])]
        est = bundle["contrasts"][cid]["estimate"]
        diff = abs(float(np.mean(d)) - est)
        worst = max(worst, diff)
        if diff > 1e-12:
            raise SystemExit(f"{cid}: replicate mean {np.mean(d)!r} != bundle estimate {est!r} (|diff| {diff:.3e})")
        rows.append({"contrast": cid, "population_n": len(pop), "bundle_estimate": est, "replicate_mean": float(np.mean(d)),
                     **{f"r{r}": d[r] for r in range(len(d))}, "min": float(np.min(d)), "max": float(np.max(d)),
                     "sd": float(np.std(d, ddof=1)), "bundle_interval": json.dumps([bundle["contrasts"][cid]["interval"]["low"],
                                                                                  bundle["contrasts"][cid]["interval"]["high"]])})
    write(out / "c_replicate_contrasts/replicate_contrasts.csv", rows)
    return {"rows": len(rows), "max_abs_diff_vs_bundle": worst}


# ------------------------------------------------------------------------------------------------ (d)
def item_d(out: Path, part: Part) -> dict:
    rows, crow = [], []
    for task in ("stability", "band_gap"):
        sc = load_scores(S6 / f"primary_{task}_n3000/final/scores_{task}_n3000.npz", part)
        y = part.y[task]
        want = ([f"best_head:{bb}" for bb in CONF] + [f"head:{t}:{bb}" for bb in CONF for t in ("logistic", "mlp_ensemble", "hist_gbm")]
                + ["best_control"] + [m for m in sc["method"] if m.startswith("control:")]
                + ([f"ref:r_sp:{bb}" for bb in CONF] if task == "stability" else []))
        meths = [m for m in want if m in sc["method"]]
        wf = {}
        for m in meths:
            S = row(sc, m)
            for r in range(S.shape[0]):
                for f in np.unique(part.fold):
                    idx = np.flatnonzero((part.fold == f) & np.isfinite(S[r]))
                    v = Mx.ap(y[idx], S[r][idx])
                    wf[(m, r, int(f))] = (v, len(idx))
                    rows.append({"task": task, "method": m, "replicate": r, "fold": int(f), "n_test": len(idx),
                                 "prevalence": float(y[idx].mean()), "ap": v})
        def wmean(m):
            vals = [(v, n) for (mm, r, f), (v, n) in wf.items() if mm == m]
            return float(sum(v * n for v, n in vals) / sum(n for _, n in vals))
        for bb in CONF:
            pairs = ([("d_sp", f"best_head:{bb}", f"ref:r_sp:{bb}"), ("q", f"head:logistic:{bb}", f"head:mlp_ensemble:{bb}")]
                     if task == "stability" else [("g", f"best_head:{bb}", "best_control")])
            for name, a, b in pairs:
                if a in sc["method"] and b in sc["method"]:
                    crow.append({"task": task, "contrast": f"{name}:{bb}", "within_fold_weighted_mean_ap_a": wmean(a),
                                 "within_fold_weighted_mean_ap_b": wmean(b), "within_fold_difference": wmean(a) - wmean(b),
                                 "estimand": "test-size-weighted mean of within-fold AP over folds and replicates (differs from the frozen pooled AP)"})
    write(out / "d_within_fold/within_fold_ap.csv", rows)
    write(out / "d_within_fold/within_fold_contrasts.csv", crow)
    return {"rows": len(rows), "contrasts": len(crow)}


# ------------------------------------------------------------------------------------------------ (e)
def tie_order(y_text: str) -> list:
    import yaml
    y = yaml.safe_load(y_text)
    return list(y["controls"]["tie_order"])


def select_controls(sel: pd.DataFrame, allowed: list, order: list) -> dict:
    """Per run_id: among the types' chosen grid points (chosen_for_type), the highest inner AP; ties -> tie_order."""
    out = {}
    c = sel[(sel["menu"] == "controls") & (sel["chosen_for_type"].astype(str) == "True") & sel["type"].isin(allowed)]
    for run_id, g in c.groupby("run_id"):
        best = max(g["inner_val_ap"].astype(float))
        tied = [t for t in order if t in set(g.loc[g["inner_val_ap"].astype(float) == best, "type"])]
        out[run_id] = tied[0]
    return out


def item_e(out: Path, part: Part) -> dict:
    d = S6 / "primary_band_gap_n3000/final"
    sel = pd.read_csv(note(d / "selection_choices.csv"))
    order = tie_order(note(YAML).read_text())
    full = select_controls(sel, order, order)
    truth = {r: t for r, t in sel[(sel["menu"] == "controls") & (sel["selected_best"].astype(str) == "True")][["run_id", "type"]].values}
    mism = {k: (full[k], truth.get(k)) for k in full if full[k] != truth.get(k)}
    if mism or set(full) != set(truth):
        raise SystemExit(f"(e) the reconstructed full-menu rule does not reproduce the recorded selection: {mism}")
    reduced = select_controls(sel, [t for t in order if t != "cgcnn"], order)
    sc = load_scores(d / "scores_band_gap_n3000.npz", part)
    y = part.y["band_gap"]
    R = sc["score"].shape[1]
    red = np.full((R, len(y)), np.nan)
    for run_id, typ in reduced.items():
        f, r = int(run_id.split("__f")[1].split("__")[0]), int(run_id.split("__r")[1])
        idx = np.flatnonzero(part.fold == f)
        red[r, idx] = row(sc, f"control:{typ}")[r, idx]
    if not np.isfinite(red).all():
        raise SystemExit("(e) reduced-menu control scores incomplete")
    specs = {"reduced_control": [Mx.SortedScore(red[r], y) for r in range(R)]}
    for bb in CONF:
        specs[f"best_head:{bb}"] = [Mx.SortedScore(row(sc, f"best_head:{bb}")[r], y) for r in range(R)]
    point = {k: float(np.mean([Mx.ap(y, (red if k == "reduced_control" else row(sc, k))[r]) for r in range(R)])) for k in specs}
    t0 = time.time()
    star = I.bootstrap(specs, part.sys_index, part.S, B, SEED)
    rows = []
    sens = {}
    for i in (0, 1):
        p = S6 / f"fold_sensitivity_{i}_band_gap/final/contrasts_band_gap_n3000.csv"
        for r in csv.DictReader(open(note(p))):
            if r["population"] == "all" and r["contrast"].startswith("g:"):
                sens[(i, r["contrast"].split(":")[1])] = (r["estimate"], r["ci_lo"], r["ci_hi"])
    bundle = json.load(open(note(BUNDLE)))
    for bb in CONF:
        g = point[f"best_head:{bb}"] - point["reduced_control"]
        lo, hi = I.interval(star[f"best_head:{bb}"] - star["reduced_control"], COVERAGE)
        rows.append({"backbone": bb, "seed0_full_menu_g": bundle["contrasts"][f"E2:g:{bb}"]["estimate"],
                     "seed0_reduced_menu_g": g, "seed0_reduced_lo": lo, "seed0_reduced_hi": hi,
                     "M_reduced_control": point["reduced_control"],
                     "seed1_g": sens.get((0, bb), ("",) * 3)[0], "seed1_lo": sens.get((0, bb), ("",) * 3)[1],
                     "seed1_hi": sens.get((0, bb), ("",) * 3)[2], "seed2_g": sens.get((1, bb), ("",) * 3)[0],
                     "seed2_lo": sens.get((1, bb), ("",) * 3)[1], "seed2_hi": sens.get((1, bb), ("",) * 3)[2]})
    counts = pd.Series(list(reduced.values())).value_counts().to_dict()
    write(out / "e_reduced_menu_seed0/reduced_menu_seed0.csv", rows)
    (out / "e_reduced_menu_seed0/selection_reduced.json").write_text(json.dumps(
        {"reduced_menu_selection": reduced, "counts": counts, "full_menu_rule_reproduced": True,
         "bootstrap": {"B": B, "seed": SEED, "seconds": round(time.time() - t0, 1)}}, indent=1) + "\n")
    return {"rows": len(rows), "reduced_counts": counts}


# ------------------------------------------------------------------------------------------------ (h)
def item_h(out: Path, part: Part) -> dict:
    rows = []
    for p in sorted(S6.glob("*/final/scores_*.npz")):
        z = np.load(note(p), allow_pickle=False)
        mid = z["material_id"]
        task = "stability" if "stability" in p.name else "band_gap"
        yy = part.y[task][np.array([part.pos[m] for m in mid])]
        for i, m in enumerate(z["method"]):
            m = str(m)
            if not (m.startswith(("head:mlp_ensemble", "best_head", "best_control", "residual"))):
                continue
            S = z["score"][i]
            zeros = [int((S[r] == 0.0).sum()) for r in range(S.shape[0])]
            if sum(zeros):
                pos0 = [int(((S[r] == 0.0) & (yy == 1)).sum()) for r in range(S.shape[0])]
                rows.append({"file": str(p.relative_to(ROOT)), "method": m, "zero_scores_per_replicate": json.dumps(zeros),
                             "positives_among_zero": json.dumps(pos0), "n": S.shape[1]})
    if not rows:
        rows = [{"file": "all score files", "method": "all MLP-derived rows", "zero_scores_per_replicate": "0",
                 "positives_among_zero": "0", "n": ""}]
    write(out / "h_sigmoid_zero_ties/zero_ties.csv", rows)
    return {"rows_with_zeros": sum(1 for r in rows if r["zero_scores_per_replicate"] != "0")}


# ------------------------------------------------------------------------------------------------ (i)
def item_i(out: Path, part: Part) -> dict:
    rows = []
    reps = {"mace_medium": "mace_medium", "chgnet": "chgnet", "sevenn": "sevenn", "mattersim_5m": "mattersim_5m",
            "composition": "composition"}
    for rep in reps:
        z = np.load(note(S6 / f"coverage_chemsys_primary/final/coverage_chemsys_primary__{rep}.npz"), allow_pickle=False)
        idx = np.array([part.pos[m] for m in z["material_id"]])
        ooc = np.zeros(len(part.ids), dtype=bool)
        ooc[idx] = z["out_of_coverage"]
        for task in ("stability", "band_gap"):
            y = part.y[task]
            rows.append({"representation": rep, "task": task, "covered_n": int((~ooc).sum()),
                         "covered_positives": int(y[~ooc].sum()), "covered_prevalence": float(y[~ooc].mean()),
                         "ooc_n": int(ooc.sum()), "ooc_positives": int(y[ooc].sum()), "ooc_prevalence": float(y[ooc].mean())})
    # consistency with the stored coverage_link prevalences (both populations); must compare every representation x task
    checked = set()
    for task in ("stability", "band_gap"):
        for r in csv.DictReader(open(note(S6 / f"coverage_link/final/coverage_link_{task}_n3000.csv"))):
            mine = [x for x in rows if x["representation"] == r["representation"] and x["task"] == task]
            if not mine:
                raise SystemExit(f"(i) coverage_link representation {r['representation']} not recomputed")
            for k_st, k_me in (("prevalence_out_of_coverage", "ooc_prevalence"), ("prevalence_covered", "covered_prevalence"),
                               ("n_out_of_coverage", "ooc_n"), ("n_positive_covered", "covered_positives")):
                if abs(float(r[k_st]) - float(mine[0][k_me])) > 1e-12:
                    raise SystemExit(f"(i) {k_st} mismatch {task} {r['representation']}: {r[k_st]} vs {mine[0][k_me]}")
            checked.add((r["representation"], task))
    if len(checked) != len(rows):
        raise SystemExit(f"(i) only {len(checked)} of {len(rows)} representation x task pairs checked against coverage_link")
    write(out / "i_coverage_prevalence/coverage_prevalence.csv", rows)
    return {"rows": len(rows)}


ITEMS = {"a": item_a, "b": item_b, "c": item_c, "d": item_d, "e": item_e, "h": item_h, "i": item_i}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="result/s7_additions")
    ap.add_argument("--items", default="a,b,c,d,e,h,i")
    a = ap.parse_args()
    out = ROOT / a.out
    part = Part()
    res = {}
    for k in a.items.split(","):
        t0 = time.time()
        res[k] = ITEMS[k](out, part)
        res[k]["seconds"] = round(time.time() - t0, 1)
        print(f"[{k}] {res[k]}", flush=True)
    outs = {str(p.relative_to(ROOT)): sha(p) for p in sorted(out.rglob("*.csv")) + sorted(out.rglob("*.json"))
            if p.name != "MANIFEST_s7.json"}
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "MANIFEST_s7.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                      "git": git, "items": res, "inputs": READ, "outputs": outs,
                                                      "note": TAG}, indent=1) + "\n")
    print("done", json.dumps({k: v.get("rows") for k, v in res.items()}))


if __name__ == "__main__":
    main()
