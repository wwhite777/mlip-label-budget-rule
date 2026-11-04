"""probe: CPU-only, PILOT-only pooled-feature mechanism probe.

Run with python3; all products are confined to this script's directory.
The singleton HistGBM grid is fixed before observing decoder results. Only ridge
requires selection, using the other four frozen chemical-system folds as inner CV.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.config import Config, PREREG, sha256_file
from analysis.data import load_part, assert_in_part
from analysis.draws import outer_train_test
from analysis.features import composition_features, correction_strata
from analysis.metrics import ap, topk_hits, topk_k
from posthoc.s9_additions import h_mp

CHECKPOINTS = ["mace_medium", "chgnet", "sevenn", "mattersim_5m", "orb"]
INPUTS = ["features", "composition", "features+composition"]
TARGETS = ["e_k", "h", "f", "y", "error_k"]
ALPHAS = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
# A singleton grid, with no validation- or test-dependent early stopping.
HIST_GRID = [dict(learning_rate=0.1, max_leaf_nodes=31, max_iter=200,
                  min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=False, random_state=0)]
LABELS = ["e_form_per_atom_mp2020_corrected", "e_above_hull_mp2020_corrected_ppd_mp"]
ATOMIC_ORDER = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
                "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
                "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
                "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def pilot_guard(ids, part_of):
    """Deliberately assert before opening any label columns, including in tests."""
    bad = [str(m) for m in ids if part_of.get(str(m)) != "PILOT"]
    assert not bad, f"PILOT-only label guard: non-PILOT/CONFIRM id(s): {bad[:3]}"
    assert_in_part(ids, "PILOT", part_of, "probe label rows")


def load_pilot_labels(path, part):
    pilot_guard(part.ids, part.part_of)
    ids = pd.read_csv(path, usecols=["material_id"], dtype=str, keep_default_na=False)["material_id"].to_numpy()
    selected = pd.Index(part.ids).get_indexer(ids) >= 0
    selected_ids = ids[selected]
    pilot_guard(selected_ids, part.part_of)
    assert len(selected_ids) == part.n and set(selected_ids) == set(part.ids)
    keep = set((np.flatnonzero(selected) + 1).tolist())
    # As in analysis.data: excluded rows are skipped BEFORE label values are parsed.
    df = pd.read_csv(path, usecols=["material_id"] + LABELS, dtype=str,
                     keep_default_na=False, skiprows=lambda i: i != 0 and i not in keep)
    pilot_guard(df["material_id"].to_numpy(), part.part_of)
    assert len(df) == part.n and not df.material_id.duplicated().any()
    arr = df.set_index("material_id").loc[part.ids, LABELS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    assert np.isfinite(arr).all()
    return arr, {"part": "PILOT", "rows_parsed": len(arr), "columns": LABELS,
                 "selection": "material_id-only first pass; skiprows before parsing label columns",
                 "confirm_label_rows_parsed": 0, "guard_asserts_before_label_read": True}


def reference_path(k):
    name = "zero_label_sp_chgnet_initial_nomp2020.csv.gz" if k == "chgnet" else f"zero_label_sp_{k}_initial.csv.gz"
    return ROOT / "result/reference" / name


def prepare():
    t0 = time.time()
    (OUT / "prepared").mkdir(exist_ok=True)
    cfg = Config.build("pilot", ["stability"])
    # Config.build only constructs the configuration; it does not run freeze/git.
    assert cfg.part == "PILOT" and cfg.n_folds == 5 and cfg.split_seed == 0
    part = load_part(cfg)  # calls the existing load_universe and partition guards
    pilot_guard(part.ids, part.part_of)
    summary = ROOT / cfg.y["population"]["summary_file"]["path"]
    labels, label_trace = load_pilot_labels(summary, part)
    f, y = labels.T
    # Reconstruct the exact documented 89-element basis without pymatgen (not installed).
    basis_path = ROOT / cfg.y["auxiliary_inputs"]["element_basis_source"]["path"]
    assert sha256_file(basis_path) == cfg.y["auxiliary_inputs"]["element_basis_source"]["sha256"]
    formulas = pd.read_csv(basis_path, usecols=["formula"], dtype=str, keep_default_na=False)["formula"]
    present = set(re.findall(r"[A-Z][a-z]?", " ".join(formulas)))
    elements = [e for e in ATOMIC_ORDER if e in present]
    assert len(elements) == 89 and set(elements) == present
    comp = composition_features(part.formula, elements)
    assert np.allclose(comp.sum(axis=1), 1)
    strata, strata_info = correction_strata(cfg, part)
    refs, identities, baselines, input_paths = {}, [], [], [PREREG, summary, cfg.split_path,
        ROOT / cfg.y["population"]["partition"]["path"], basis_path, ROOT / strata_info["path"]]
    for k in CHECKPOINTS:
        path = reference_path(k)
        input_paths.append(path)
        df = pd.read_csv(path, usecols=["material_id", "e_form_pred_sp", "e_hull_pred_sp"], dtype={"material_id": str})
        assert not df.material_id.duplicated().any()
        df = df.set_index("material_id").loc[part.ids]
        e, hull = df[["e_form_pred_sp", "e_hull_pred_sp"]].to_numpy(dtype=float).T
        assert np.isfinite(e).all() and np.isfinite(hull).all()
        refs[k] = {"r_sp": {"e_form": e, "e_hull": hull}}
        delta = (hull - y) - (e - f)
        identities.append(dict(checkpoint=k, n=part.n, max_abs_eV=float(np.max(np.abs(delta))),
                               mean_abs_eV=float(np.mean(np.abs(delta)))))
        baselines.append(stability_row(k, "reference", "zero_shot", "-e_hull_pred_sp", y <= 0, -hull, "pooled_oof"))
    canonical_h = h_mp(SimpleNamespace(refs=refs))
    h_deviations = {k: float(np.max(np.abs(refs[k]["r_sp"]["e_form"] - refs[k]["r_sp"]["e_hull"] - canonical_h)))
                    for k in CHECKPOINTS}
    assert max(h_deviations.values()) <= 1e-5
    for k in CHECKPOINTS:
        path = ROOT / f"cache/features/wbm_full_{k}_initial.npz"
        input_paths.append(path)
        with np.load(path, allow_pickle=False) as z:
            assert str(z["structures_source"]) == "initial"
            mids = z["mids"].astype(str)
            assert not pd.Index(mids).duplicated().any()
            pos = pd.Index(mids).get_indexer(part.ids)
            assert np.all(pos >= 0)
            assert not (set(z["failed_ids"].astype(str)) & set(part.ids))
            X = np.asarray(z["feats"], dtype=np.float64)[pos]
        assert np.isfinite(X).all()
        r = refs[k]["r_sp"]
        h = r["e_form"] - r["e_hull"]  # preserve each checkpoint's stored rounding exactly
        T = np.column_stack([r["e_form"], h, f, y, f - r["e_form"]])
        np.save(OUT / f"prepared/{k}_features.npy", X)
        np.save(OUT / f"prepared/{k}_targets.npy", T)
        np.save(OUT / f"prepared/{k}_hull.npy", r["e_hull"])
        print(f"prepared {k}: {X.shape}; identity {identities[CHECKPOINTS.index(k)]['max_abs_eV']:.3e} eV; "
              f"reference AP {baselines[CHECKPOINTS.index(k)]['ap']:.9f}", flush=True)
    for name, arr in [("composition", comp), ("ids", part.ids), ("fold", part.fold),
                      ("chemsys", part.chemsys), ("num_elements", np.count_nonzero(comp, axis=1))]:
        np.save(OUT / f"prepared/{name}.npy", arr)
    np.savez(OUT / "prepared/strata.npz", **strata)
    pd.DataFrame(identities).to_csv(OUT / "identity_check.csv", index=False)
    pd.DataFrame(baselines).to_csv(OUT / "reference_ap.csv", index=False)
    for k, expected in [("mace_medium", 0.613756), ("chgnet", 0.604982)]:
        got = next(r["ap"] for r in baselines if r["checkpoint"] == k)
        assert abs(got - expected) <= 0.0000005, (k, got, expected)
    source_paths = [Path(__file__), ROOT / "src/posthoc/s9_additions.py"]
    source_paths += [ROOT / f"src/analysis/{s}.py" for s in ["data", "draws", "features", "config", "metrics"]]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "n": part.n,
                "n_systems": len(part.systems), "fold_sizes": np.bincount(part.fold).tolist(),
                "label_trace": label_trace, "elements": elements, "strata": strata_info,
                "h_max_deviation_from_mace_eV": h_deviations,
                "inputs_sha256": {str(p): sha256_file(p) for p in input_paths},
                "source_sha256": {str(p): sha256_file(p) for p in source_paths},
                "ridge_alphas": ALPHAS.tolist(), "hist_grid": HIST_GRID,
                "inner_cv": "leave one of the four remaining frozen chemical-system folds out; pooled inner MAE",
                "hist_selection": "singleton fixed grid; no inner selection needed; all outer training rows used",
                "max_threads": 12, "workers": 3, "threads_per_worker": 4,
                "units": "targets eV/atom; MAE and RMSE reported meV/atom",
                "prepare_seconds": time.time() - t0}
    dump_json(OUT / "input_manifest.json", manifest)


def ridge_path(Xtr, Ttr, Xva, alphas=ALPHAS):
    """Exact centered/scaled ridge path; shared factorization, independent target fits."""
    scaler = StandardScaler().fit(Xtr)
    A = scaler.transform(Xtr)
    B = scaler.transform(Xva)
    tm = Ttr.mean(axis=0)
    gram = A.T @ A
    eig, vec = np.linalg.eigh(gram)
    eig = np.maximum(eig, 0)
    rhs = vec.T @ (A.T @ (Ttr - tm))
    coef = np.stack([vec @ (rhs / (eig[:, None] + a)) for a in alphas], axis=0)
    return np.einsum("nd,adt->ant", B, coef, optimize=True) + tm


def ridge_fold(X, T, fold, outer):
    tr = np.flatnonzero(fold != outer)
    te = np.flatnonzero(fold == outer)
    losses = np.zeros((len(ALPHAS), T.shape[1]))
    inner_records = []
    for inner in sorted(set(fold[tr])):
        it = tr[fold[tr] != inner]
        iv = tr[fold[tr] == inner]
        assert not set(it) & set(iv) and not set(te) & (set(it) | set(iv))
        pred = ridge_path(X[it], T[it], X[iv])
        abs_sum = np.abs(pred - T[iv][None, :, :]).sum(axis=1)
        losses += abs_sum
        inner_records.append({"inner_fold": int(inner), "n_train": len(it), "n_val": len(iv),
                              "mae_eV_by_alpha_target": (abs_sum / len(iv)).tolist()})
    losses /= len(tr)
    chosen = losses.argmin(axis=0)  # first alpha in stated order wins exact ties
    full = ridge_path(X[tr], T[tr], X[te])
    pred = np.column_stack([full[chosen[j], :, j] for j in range(T.shape[1])])
    return te, pred, {"outer_fold": int(outer), "n_train": len(tr), "n_test": len(te),
                      "chosen_alpha_by_target": ALPHAS[chosen].tolist(),
                      "pooled_inner_mae_eV_by_alpha_target": losses.tolist(), "inner": inner_records}


def fit_job(k, input_name, model):
    t0 = time.time()
    stem = f"{k}__{input_name.replace('+', '_plus_')}__{model}"
    dest = OUT / f"predictions/{stem}.npz"
    if dest.exists():
        return {"job": stem, "status": "already_complete"}
    fold = np.load(OUT / "prepared/fold.npy")
    T = np.load(OUT / f"prepared/{k}_targets.npy")
    C = np.load(OUT / "prepared/composition.npy")
    X = C if input_name == "composition" else np.load(OUT / f"prepared/{k}_features.npy")
    if input_name == "features+composition":
        X = np.column_stack([X, C])
    pred = np.full_like(T, np.nan)
    records = []
    with threadpool_limits(limits=4):
        for outer in range(5):
            ft = time.time()
            if model == "ridge":
                te, p, rec = ridge_fold(X, T, fold, outer)
            else:
                tr, te = outer_train_test(SimpleNamespace(fold=fold), outer)
                p = np.empty((len(te), len(TARGETS)))
                for j in range(len(TARGETS)):
                    m = HistGradientBoostingRegressor(**HIST_GRID[0]).fit(X[tr], T[tr, j])
                    p[:, j] = m.predict(X[te])
                rec = {"outer_fold": outer, "n_train": len(tr), "n_test": len(te), "params": HIST_GRID[0]}
            pred[te] = p
            rec["seconds"] = time.time() - ft
            records.append(rec)
            print(f"{stem} fold={outer} seconds={rec['seconds']:.1f}", flush=True)
    assert np.isfinite(pred).all()
    np.savez_compressed(dest, prediction=pred, targets=np.array(TARGETS), outer_fold=fold)
    dump_json(OUT / f"predictions/{stem}_selection.json", records)
    return {"job": stem, "status": "complete", "seconds": time.time() - t0}


def fit(deadline_seconds):
    (OUT / "predictions").mkdir(exist_ok=True)
    started = time.time()
    jobs = [(k, x, m) for k in CHECKPOINTS for x in INPUTS for m in ["ridge", "hist_gbm"]]
    # Workers never exceed 3 x 4 numerical threads. Submit gradually so the deadline
    # can stop pending work without discarding completed OOF jobs.
    records = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending, cursor = {}, 0
        while cursor < len(jobs) or pending:
            while cursor < len(jobs) and len(pending) < 3 and time.time() - started < deadline_seconds:
                job = jobs[cursor]
                pending[pool.submit(fit_job, *job)] = job
                cursor += 1
            if not pending:
                break
            future = next(as_completed(pending))
            job = pending.pop(future)
            try:
                rec = future.result()
            except Exception as exc:
                rec = {"job": list(job), "status": "failed", "error": repr(exc)}
                records.append(rec)
                dump_json(OUT / "fit_status.json", records)
                raise
            records.append(rec)
            dump_json(OUT / "fit_status.json", records)
            print(json.dumps(rec), flush=True)
    print(f"fit finished: {len(records)}/{len(jobs)} jobs, {time.time()-started:.1f}s", flush=True)


def stability_row(k, x, model, score_name, stable, score, scope, fold=-1):
    n = len(stable)
    top = topk_k(0.05, n)
    return {"checkpoint": k, "input": x, "model": model, "score": score_name,
            "scope": scope, "fold": fold, "n": n, "n_stable": int(np.sum(stable)),
            "prevalence": float(np.mean(stable)), "ap": ap(stable, score), "top5_k": top,
            "precision_top5": topk_hits(stable, score, top) / top}


def regression_row(k, x, model, target, truth, pred, scope, fold=-1):
    return {"checkpoint": k, "input": x, "target": target, "model": model, "scope": scope, "fold": fold,
            "n": len(truth), "mae_meV_atom": 1000 * mean_absolute_error(truth, pred),
            "rmse_meV_atom": 1000 * np.sqrt(mean_squared_error(truth, pred)), "r2": r2_score(truth, pred),
            "explained_variance": explained_variance_score(truth, pred),
            "bias_meV_atom": 1000 * float(np.mean(pred - truth))}


def evaluate():
    fold = np.load(OUT / "prepared/fold.npy")
    ne = np.load(OUT / "prepared/num_elements.npy")
    sf = np.load(OUT / "prepared/strata.npz")
    rows, stability, strata_rows = [], [], []
    for k in CHECKPOINTS:
        T = np.load(OUT / f"prepared/{k}_targets.npy")
        hull = np.load(OUT / f"prepared/{k}_hull.npy")
        e, h, f, y, error = T.T
        stable = y <= 0
        scopes = [("pooled_oof", -1, np.ones(len(T), dtype=bool))]
        scopes += [("outer_fold", o, fold == o) for o in range(5)]
        for scope, o, mask in scopes:
            stability.append(stability_row(k, "reference", "zero_shot", "-e_hull_pred_sp", stable[mask], -hull[mask], scope, o))
        for x in INPUTS:
            for model in ["ridge", "hist_gbm"]:
                stem = f"{k}__{x.replace('+', '_plus_')}__{model}"
                path = OUT / f"predictions/{stem}.npz"
                if not path.exists():
                    continue
                with np.load(path) as z:
                    P = z["prediction"]
                    assert np.array_equal(z["outer_fold"], fold) and np.isfinite(P).all()
                scores = {"-(decoded_f-h)": -(P[:, 2] - h), "-decoded_y": -P[:, 3],
                          "-(decoded_e_k-h)": -(P[:, 0] - h)}
                # Additional composition-reference diagnostic, explicitly beyond the
                # required scores: replace the supplied h with independently decoded h.
                scores["-(decoded_e_k-decoded_h)"] = -(P[:, 0] - P[:, 1])
                for scope, o, mask in scopes:
                    for j, target in enumerate(TARGETS):
                        rows.append(regression_row(k, x, model, target, T[mask, j], P[mask, j], scope, o))
                    for score_name, score in scores.items():
                        stability.append(stability_row(k, x, model, score_name, stable[mask], score[mask], scope, o))
        flips = stable != (hull <= 0)
        false_stable = (~stable) & (hull <= 0)
        false_unstable = stable & (hull > 0)
        groups = [("all", "all", np.ones(len(T), bool))]
        for name in sf.files:
            groups += [("correction_stratum", name + "=True", sf[name]),
                       ("correction_stratum", name + "=False", ~sf[name])]
        # Joint strata are disjoint; marginal correction flags may overlap.
        flags = list(sf.files)
        for a in [False, True]:
            for b in [False, True]:
                groups.append(("correction_joint", f"{flags[0]}={a};{flags[1]}={b}", (sf[flags[0]] == a) & (sf[flags[1]] == b)))
        groups += [("num_elements", str(n), ne == n) for n in sorted(set(ne))]
        groups += [("boundary_window", f"abs_y<={m}_meV", np.abs(y) <= m / 1000) for m in [25, 50, 100]]
        for family, name, mask in groups:
            n = int(mask.sum())
            strata_rows.append({"checkpoint": k, "family": family, "stratum": name, "n": n,
                "n_stable": int(stable[mask].sum()), "mae_error_meV_atom": float(np.mean(np.abs(error[mask])) * 1000) if n else np.nan,
                "mean_error_meV_atom": float(np.mean(error[mask]) * 1000) if n else np.nan,
                "error_variance_eV2": float(np.var(error[mask])) if n else np.nan,
                "n_flips": int(flips[mask].sum()), "flip_rate_within_stratum": float(np.mean(flips[mask])) if n else np.nan,
                "share_of_all_pilot_flips": float(flips[mask].sum() / flips.sum()),
                "n_false_stable": int(false_stable[mask].sum()), "n_false_unstable": int(false_unstable[mask].sum()),
                "share_of_pilot": n / len(T), "share_of_pilot_abs_error": float(np.abs(error[mask]).sum() / np.abs(error).sum())})
    pd.DataFrame(rows).to_csv(OUT / "probe_metrics.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "probe_stability_ap.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(OUT / "error_strata.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics[(metrics.scope == "pooled_oof") & (metrics.target == "error_k")].to_csv(OUT / "error_variance_explained.csv", index=False)
    print(f"evaluated {len(rows)} metric rows; {len(stability)} stability rows; {len(strata_rows)} strata rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "fit", "evaluate"])
    parser.add_argument("--deadline-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "fit":
        fit(args.deadline_seconds)
    else:
        evaluate()


if __name__ == "__main__":
    main()
