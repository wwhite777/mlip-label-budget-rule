"""Coverage diagnostics (YAML descriptive_specs.coverage_diagnostics; descriptive). Label-free except performance_link.

Per representation (each confirmatory backbone's INITIAL features; the element-fraction composition vector), split and
outer fold f:
  standardization  mean / standard deviation of fold f's training structures (sklearn StandardScaler convention: a
                   zero-variance column keeps scale 1), float64.
  k-NN (primary)   Euclidean distance from every test structure of fold f to its k nearest training structures of fold
                   f; statistic = the mean of the k distances. Reference: a random sample of n_reference training
                   structures (default_rng(base + f).choice(len(train), size=n_reference, replace=False), training in
                   canonical order), each with its leave-self-out k-NN mean distance to the rest of training (the
                   sample point itself is excluded by index; exact duplicates elsewhere in training count as
                   neighbours). threshold = numpy.percentile(reference, percentile) (linear); a test structure is
                   OUT_OF_COVERAGE if its statistic > threshold.
                   Neighbours: batched torch on the run device (float64 expansion |q|^2 + |t|^2 - 2 q.t, top k+extra
                   candidates), candidates re-ranked by the exact float64 difference norm with ties by training index;
                   a row whose candidate margin cannot exclude an unseen tie falls back to the exact full row (CPU).
                   cpu_crosscheck() recomputes a subset by numpy brute force; neighbour sets must agree exactly
                   except at a k-th-position float64 rounding tie (counted; ).
  UMAP + KDE       umap-learn (YAML parameters; random_state in the YAML set) fitted on the standardized reference
  (secondary)      sample, applied to the test structures; scipy.stats.gaussian_kde (Scott) on the training embedding;
                   threshold = numpy.percentile(training densities, kde_percentile); LOW_DENSITY if density < threshold.
performance_link (CONFIRM primary split, n from the YAML): AP of each confirmatory backbone's best head (and, for band
gap, the best control) within covered vs OUT_OF_COVERAGE test structures of every representation (the row flags the
method's own representation), M = mean over replicates; covered - OOC difference; 90 % percentile intervals from the
frozen paired chemical-system cluster bootstrap over all systems of the part (the primary evaluation's draws).
"""
from __future__ import annotations

import json
import time

import numpy as np

from analysis import inference as I
from analysis import metrics as Mx
from analysis.config import AnalysisError


def standardizer(Xtr: np.ndarray) -> tuple:
    X = np.asarray(Xtr, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    return mu, sd


def _exact_rows(Q: np.ndarray, T: np.ndarray, k: int, self_idx=None) -> tuple:
    """numpy brute force (float64 difference norms) with ties by training index: (idx (nq, k), dist (nq, k))."""
    Q, T = np.asarray(Q, dtype=np.float64), np.asarray(T, dtype=np.float64)
    nb = np.empty((len(Q), k), dtype=np.int64)
    dd = np.empty((len(Q), k), dtype=np.float64)
    ar = np.arange(len(T))
    for i in range(len(Q)):
        d = np.sqrt(np.sum((T - Q[i]) ** 2, axis=1))
        if self_idx is not None:
            d[self_idx[i]] = np.inf
        o = np.lexsort((ar, d))[:k]
        nb[i], dd[i] = o, d[o]
    return nb, dd


def knn(Q: np.ndarray, T: np.ndarray, k: int, device: str = "cpu", self_idx=None, batch: int = 2048,
        extra: int = 16) -> dict:
    """k nearest rows of T for every row of Q (Euclidean, exact float64 distances, ties by T index).
    self_idx[i] = the row of T excluded for query i (leave-self-out) or None."""
    import torch
    Q, T = np.asarray(Q, dtype=np.float64), np.asarray(T, dtype=np.float64)
    K = min(k + extra, len(T) - (1 if self_idx is not None else 0))
    if K < k:
        raise AnalysisError(f"k-NN: {len(T)} training rows for k = {k}")
    Tt = torch.as_tensor(T, dtype=torch.float64, device=device)
    t2 = (Tt * Tt).sum(dim=1)
    nb = np.empty((len(Q), k), dtype=np.int64)
    dd = np.empty((len(Q), k), dtype=np.float64)
    n_fallback = 0
    for s in range(0, len(Q), batch):
        q = torch.as_tensor(Q[s:s + batch], dtype=torch.float64, device=device)
        d2 = (q * q).sum(dim=1)[:, None] + t2[None, :] - 2.0 * (q @ Tt.T)
        d2.clamp_(min=0.0)
        if self_idx is not None:
            si = torch.as_tensor(np.asarray(self_idx[s:s + batch], dtype=np.int64), device=device)
            d2[torch.arange(len(q), device=device), si] = float("inf")
        vals, cand = torch.topk(d2, K, dim=1, largest=False, sorted=True)
        ex = torch.sqrt(((Tt[cand] - q[:, None, :]) ** 2).sum(dim=2))
        cand_np, ex_np, vals_np = cand.cpu().numpy(), ex.cpu().numpy(), vals.cpu().numpy()
        order = np.lexsort((cand_np, ex_np), axis=-1)[:, :k]
        rows = np.arange(len(cand_np))[:, None]
        bi, bd = cand_np[rows, order], ex_np[rows, order]
        # safety: the K-th candidate's approximate squared distance must clear the k-th exact one by a margin that
        # covers the expansion's rounding; otherwise an unseen training row could tie or beat it -> exact full row
        scale = (Q[s:s + batch] ** 2).sum(axis=1) + float(t2.max().item())
        unsafe = vals_np[:, K - 1] <= bd[:, k - 1] ** 2 + 1e-9 * scale + 1e-12
        if K == len(T) - (1 if self_idx is not None else 0):
            unsafe[:] = False  # every candidate was considered
        for i in np.flatnonzero(unsafe):
            n_fallback += 1
            si = None if self_idx is None else [self_idx[s + i]]
            fi, fd = _exact_rows(Q[s + i:s + i + 1], T, k, si)
            bi[i], bd[i] = fi[0], fd[0]
        nb[s:s + batch], dd[s:s + batch] = bi, bd
    return {"idx": nb, "dist": dd, "mean": dd.mean(axis=1), "n_fallback_rows": n_fallback}


def cpu_crosscheck(Q, T, k, res: dict, rows, self_idx=None, rtol: float = 1e-12) -> dict:
    """Brute-force numpy neighbours for `rows` of Q against res. Per row: the sorted k distances must agree within
    rtol * max(1, numpy's k-th distance), and the neighbour index sets must be equal, except at a k-th-position
    ROUNDING TIE: every index in only one of the two sets lies within that tolerance of numpy's k-th distance (numpy
    distance). The torch path sums the squared differences in another order than numpy, so training rows equidistant
    from the query can differ by one ulp and resolve the training-index tie rule differently; the k-NN mean then agrees to
    rounding (the fixed coverage comparison). Any other difference raises. Before the tie
    exception ( finding 1, ): a row fails if a reported distance is non-finite, an index repeats, an index lies
    outside T, or the excluded self index is returned."""
    rows = np.asarray(rows, dtype=np.int64)
    Q = np.asarray(Q, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    si = None if self_idx is None else np.asarray(self_idx)[rows]
    nb, dd = _exact_rows(Q[rows], T, k, si)
    got, gd = res["idx"][rows], res["dist"][rows]
    n_tie, bad = 0, []
    for i in range(len(rows)):
        g = np.asarray(got[i])
        if (not np.all(np.isfinite(np.asarray(gd[i], dtype=np.float64))) or len(np.unique(g)) != len(g)
                or np.any(g < 0) or np.any(g >= len(T)) or (si is not None and np.any(g == si[i]))):
            bad.append(int(rows[i]))
            continue
        tol = rtol * max(1.0, abs(float(dd[i, k - 1])))
        if np.max(np.abs(np.sort(dd[i]) - np.sort(gd[i]))) > tol:
            bad.append(int(rows[i]))
            continue
        sym = sorted(set(nb[i].tolist()) ^ set(got[i].tolist()))
        if sym:
            dsym = np.sqrt(np.sum((T[sym] - Q[rows[i]]) ** 2, axis=1))
            if np.all(np.abs(dsym - dd[i, k - 1]) <= tol):
                n_tie += 1
            else:
                bad.append(int(rows[i]))
    if bad:
        raise AnalysisError(f"k-NN CPU cross-check: {len(bad)} of {len(rows)} checked rows differ beyond a k-th-position "
                            f"rounding tie (rows {bad[:5]})")
    same = bool(np.array_equal(np.sort(nb, axis=1), np.sort(got, axis=1)))
    same_order = bool(np.array_equal(nb, got))
    md = float(np.max(np.abs(dd - gd))) if len(rows) else 0.0
    return {"rows": int(len(rows)), "neighbour_sets_equal": same, "neighbour_order_equal": same_order,
            "n_rounding_tie_rows": int(n_tie), "max_abs_distance_diff": md}


def reference_sample(train: np.ndarray, f: int, spec: dict) -> np.ndarray:
    """Positions (into `train`, canonical order) of the reference sample of fold f."""
    n = spec["n_reference"]
    if n > len(train):
        raise AnalysisError(f"coverage reference sample {n} exceeds the {len(train)} training structures")
    return np.random.default_rng(spec["ref_rng_base"] + f).choice(len(train), size=n, replace=False)


def knn_fold(X: np.ndarray, fold: np.ndarray, f: int, spec: dict, device: str = "cpu", n_check: int = 64) -> dict:
    """k-NN coverage of fold f in one representation."""
    t0 = time.time()
    train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    mu, sd = standardizer(X[train])
    T = (np.asarray(X[train], dtype=np.float64) - mu) / sd
    Qt = (np.asarray(X[test], dtype=np.float64) - mu) / sd
    pos = reference_sample(train, f, spec)
    k = spec["k"]
    rt = knn(Qt, T, k, device)
    rr = knn(T[pos], T, k, device, self_idx=pos)
    thr = float(np.percentile(rr["mean"], spec["percentile"]))
    ooc = rt["mean"] > thr
    chk_rows = np.linspace(0, len(test) - 1, num=min(n_check, len(test))).astype(np.int64)
    chk_ref = np.linspace(0, len(pos) - 1, num=min(n_check, len(pos))).astype(np.int64)
    chk = {"test": cpu_crosscheck(Qt, T, k, rt, chk_rows), "reference": cpu_crosscheck(T[pos], T, k, rr, chk_ref,
                                                                                       self_idx=pos)}
    return {"train": train, "test": test, "ref_pos": pos, "mu": mu, "sd": sd, "test_mean": rt["mean"],
            "test_idx": train[rt["idx"]], "test_dist": rt["dist"], "ref_mean": rr["mean"], "threshold": thr,
            "ooc": ooc, "crosscheck": chk, "n_fallback_rows": rt["n_fallback_rows"] + rr["n_fallback_rows"],
            "seconds_knn": round(time.time() - t0, 2), "Z_ref": T[pos], "Z_test": Qt}


def umap_kde(Z_ref: np.ndarray, Z_test: np.ndarray, spec: dict, random_state: int) -> dict:
    t0 = time.time()
    import umap
    from scipy.stats import gaussian_kde
    u = spec["umap"]
    m = umap.UMAP(n_neighbors=u["n_neighbors"], min_dist=u["min_dist"], n_components=u["n_components"],
                  metric=u["metric"], random_state=random_state)
    E_ref = m.fit_transform(np.asarray(Z_ref, dtype=np.float32))
    E_test = m.transform(np.asarray(Z_test, dtype=np.float32))
    kde = gaussian_kde(np.asarray(E_ref, dtype=np.float64).T, bw_method="scott")
    d_ref = kde(np.asarray(E_ref, dtype=np.float64).T)
    d_test = kde(np.asarray(E_test, dtype=np.float64).T)
    thr = float(np.percentile(d_ref, spec["kde_percentile"]))
    return {"E_ref": E_ref, "E_test": E_test, "density_ref": d_ref, "density_test": d_test, "threshold": thr,
            "low": d_test < thr, "seconds": round(time.time() - t0, 2)}


def representation_rows(split: str, rep: str, folds: list, n_part: int) -> tuple:
    """Per-fold and pooled summary rows of one (split, representation); folds = list of fold result dicts."""
    rows = []
    for fr in folds:
        r = {"split": split, "representation": rep, "fold": fr["fold"], "n_train": int(len(fr["train"])),
             "n_test": int(len(fr["test"])), "n_reference": int(len(fr["ref_pos"])),
             "knn_threshold": fr["threshold"], "n_out_of_coverage": int(fr["ooc"].sum()),
             "frac_out_of_coverage": float(fr["ooc"].mean()), "test_knn_mean_median": float(np.median(fr["test_mean"])),
             "reference_knn_mean_median": float(np.median(fr["ref_mean"])), "n_fallback_rows": fr["n_fallback_rows"]}
        for rs, u in sorted(fr["umap"].items()):
            r[f"kde_threshold_rs{rs}"] = u["threshold"]
            r[f"n_low_density_rs{rs}"] = int(u["low"].sum())
            r[f"frac_low_density_rs{rs}"] = float(u["low"].mean())
        rows.append(r)
    pooled = {"split": split, "representation": rep, "fold": "pooled",
              "n_train": "", "n_test": int(sum(len(fr["test"]) for fr in folds)),
              "n_reference": int(sum(len(fr["ref_pos"]) for fr in folds)), "knn_threshold": "",
              "n_out_of_coverage": int(sum(fr["ooc"].sum() for fr in folds))}
    pooled["frac_out_of_coverage"] = pooled["n_out_of_coverage"] / pooled["n_test"]
    pooled.update(test_knn_mean_median=float(np.median(np.concatenate([fr["test_mean"] for fr in folds]))),
                  reference_knn_mean_median=float(np.median(np.concatenate([fr["ref_mean"] for fr in folds]))),
                  n_fallback_rows=int(sum(fr["n_fallback_rows"] for fr in folds)))
    for rs in sorted(folds[0]["umap"]):
        pooled[f"kde_threshold_rs{rs}"] = ""
        pooled[f"n_low_density_rs{rs}"] = int(sum(fr["umap"][rs]["low"].sum() for fr in folds))
        pooled[f"frac_low_density_rs{rs}"] = pooled[f"n_low_density_rs{rs}"] / pooled["n_test"]
    if pooled["n_test"] != n_part:
        raise AnalysisError(f"coverage {split}/{rep}: test structures {pooled['n_test']} != part {n_part}")
    rows.append(pooled)
    return rows


def per_structure_arrays(folds: list, n_part: int, k: int, random_states: list) -> dict:
    """Part-aligned arrays (every part structure is a test structure of exactly one fold)."""
    out = {"fold": np.full(n_part, -1, dtype=np.int64), "knn_mean": np.full(n_part, np.nan),
           "out_of_coverage": np.zeros(n_part, dtype=bool), "knn_neighbour_index": np.full((n_part, k), -1, np.int64),
           "knn_neighbour_dist": np.full((n_part, k), np.nan)}
    for rs in random_states:
        out[f"density_rs{rs}"] = np.full(n_part, np.nan)
        out[f"low_density_rs{rs}"] = np.zeros(n_part, dtype=bool)
        out[f"umap_rs{rs}"] = np.full((n_part, 2), np.nan)
    for fr in folds:
        t = fr["test"]
        out["fold"][t] = fr["fold"]
        out["knn_mean"][t] = fr["test_mean"]
        out["out_of_coverage"][t] = fr["ooc"]
        out["knn_neighbour_index"][t] = fr["test_idx"]
        out["knn_neighbour_dist"][t] = fr["test_dist"]
        for rs in random_states:
            out[f"density_rs{rs}"][t] = fr["umap"][rs]["density_test"]
            out[f"low_density_rs{rs}"][t] = fr["umap"][rs]["low"]
            out[f"umap_rs{rs}"][t] = fr["umap"][rs]["E_test"]
    if (out["fold"] < 0).any():
        raise AnalysisError("coverage: a part structure is in no test fold")
    return out


# ------------------------------------------------------------------------------------------------ performance link
def performance_link(cfg, part, y: np.ndarray, task: str, n: int, reps: list, scores: dict, ooc: dict,
                     own: dict, log=print) -> list:
    """scores: method -> (len(reps), N) pooled out-of-fold scores; ooc: representation -> bool (N,) OUT_OF_COVERAGE;
    own: method -> its own representation (or None). Rows per (representation, method): AP on covered and on OOC
    structures, the difference, counts, prevalence, 90 % intervals (bootstrap over all systems of the part)."""
    y = np.asarray(y)
    ones = np.ones(part.n, dtype=np.int64)
    specs, point, pops = {}, {}, {}
    for rep, flag in ooc.items():
        pops[(rep, "covered")] = np.flatnonzero(~flag)
        pops[(rep, "out_of_coverage")] = np.flatnonzero(flag)
    for m, rows in scores.items():
        if rows.shape != (len(reps), part.n) or not np.isfinite(rows).all():
            raise AnalysisError(f"performance_link {task}: scores of {m} incomplete")
        for (rep, pn), idx in pops.items():
            key = f"{m}@{rep}@{pn}"
            if len(idx) == 0 or y[idx].sum() == 0:
                point[key] = float("nan")  # AP undefined (no structure / no positive): reported, not bootstrapped
                continue
            ss, aps = [], []
            for row in rows:
                s = Mx.SortedScore(row, y, idx)
                a = Mx.ap(y[idx], row[idx])
                if s.ap(ones) != a:
                    raise AnalysisError("weighted AP != sklearn AP (performance_link)")
                ss.append(s)
                aps.append(a)
            specs[key], point[key] = ss, float(np.mean(aps))
    t0 = time.time()
    mstar = I.bootstrap(specs, part.sys_index, len(part.systems), cfg.B, cfg.boot_seed) if specs else {}
    log(f"[coverage_link] {task}: {len(specs)} specs, bootstrap {time.time() - t0:.1f}s")
    out = []
    for rep in ooc:
        for m in scores:
            r = {"analysis": "coverage_performance_link", "task": task, "n": n, "representation": rep, "method": m,
                 "own_representation": bool(own.get(m) == rep), "replicates": json.dumps(list(reps))}
            for pn in ("covered", "out_of_coverage"):
                idx = pops[(rep, pn)]
                key = f"{m}@{rep}@{pn}"
                npos = int(y[idx].sum())
                lo, hi = I.interval(mstar[key], cfg.coverage) if key in mstar else [float("nan")] * 2
                r.update({f"n_{pn}": int(len(idx)), f"n_positive_{pn}": npos,
                          f"prevalence_{pn}": (npos / len(idx)) if len(idx) else float("nan"),
                          f"AP_{pn}": point[key], f"AP_{pn}_ci_lo": lo, f"AP_{pn}_ci_hi": hi})
            kc, ko = f"{m}@{rep}@covered", f"{m}@{rep}@out_of_coverage"
            if kc in mstar and ko in mstar:
                lo, hi = I.interval(mstar[kc] - mstar[ko], cfg.coverage)
                r.update(diff_covered_minus_ooc=point[kc] - point[ko], diff_ci_lo=lo, diff_ci_hi=hi)
            else:
                r.update(diff_covered_minus_ooc=float("nan"), diff_ci_lo=float("nan"), diff_ci_hi=float("nan"))
            r["role"] = "descriptive"
            out.append(r)
    return out
