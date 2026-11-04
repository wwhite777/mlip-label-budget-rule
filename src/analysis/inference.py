"""Paired chemical-system cluster bootstrap, percentile intervals, one-sided p-values, Holm, readings, pilot branch.

Bootstrap draw b (one numpy default_rng(seed) for the whole run, draws in order b = 0..B-1): counts =
bincount(rng.integers(0, S, size=S), minlength=S) over the S systems of the part (sorted); structure weight =
count of its system. Every spec (method x population) is re-evaluated on the same weights (paired), each replicate
row separately (weighted AP = AP on the resampled data, metrics.SortedScore), and M* = mean over the spec's rows.
"""
from __future__ import annotations

import re

import numpy as np

from analysis.config import AnalysisError
from analysis.metrics import SortedScore


def cluster_weights(rng, sys_index: np.ndarray, S: int) -> np.ndarray:
    counts = np.bincount(rng.integers(0, S, size=S), minlength=S)
    return counts[sys_index]


def bootstrap(specs: dict, sys_index: np.ndarray, S: int, B: int, seed: int, with_auroc: bool = False):
    """specs: name -> list of SortedScore (one per replicate row). Returns name -> M* (B,) float64; with_auroc=True
    returns (M*, AUROC*) where AUROC* is the replicate-mean AUROC on the SAME draws (descriptive_specs.auroc_intervals)."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B, dtype=np.float64) for k in specs}
    auc = {k: np.empty(B, dtype=np.float64) for k in specs} if with_auroc else None
    for b in range(B):
        w = cluster_weights(rng, sys_index, S)
        for k, rows in specs.items():
            out[k][b] = float(np.mean([r.ap(w) for r in rows]))
            if with_auroc:
                auc[k][b] = float(np.mean([r.auroc(w) for r in rows]))
    return (out, auc) if with_auroc else out


def interval_finite(dstar, coverage: float) -> tuple:
    """Percentile interval over the finite draws (AUROC is undefined on a draw lacking a class); returns
    ([lo, hi], number of undefined draws)."""
    d = np.asarray(dstar, dtype=np.float64)
    fin = np.isfinite(d)
    if not fin.any():
        return [float("nan"), float("nan")], int(len(d))
    return interval(d[fin], coverage), int((~fin).sum())


def p_up(dstar, m: float) -> float:
    dstar = np.asarray(dstar)
    return float((1 + np.sum(dstar <= m)) / (len(dstar) + 1))


def p_down(dstar, m: float) -> float:
    dstar = np.asarray(dstar)
    return float((1 + np.sum(dstar >= m)) / (len(dstar) + 1))


def check_p_formulas(y: dict) -> None:
    pv = y["inference"]["p_values"]
    if pv["up"] != "(1 + #{d* <= m}) / (B + 1)" or pv["down"] != "(1 + #{d* >= m}) / (B + 1)":
        raise AnalysisError(f"p-value formulas in the YAML differ from the implemented ones: {pv}")


def holm(p, names=None) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1); ties ordered by name (src/gates/README_bundle.md)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    names = list(range(m)) if names is None else list(names)
    order = sorted(range(m), key=lambda i: (p[i], names[i]))
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def interval(dstar, coverage: float) -> list:
    pct = int(round(coverage * 100))          # 90 -> percentiles [5, 95] exactly (no float residue)
    a = (100 - pct) / 2
    lo, hi = np.percentile(np.asarray(dstar), [a, 100 - a])
    return [float(lo), float(hi)]


_HYP = re.compile(r"(<=|>=)\s*(-?)(m_[a-z]+)")


def family_margin(y: dict, family: str) -> tuple:
    """('<=', sign, margin name) parsed from the YAML family hypothesis text."""
    m = _HYP.search(y["families"][family]["hypotheses"])
    if not m:
        raise AnalysisError(f"cannot parse the hypothesis of {family}")
    return m.group(1), (-1.0 if m.group(2) else 1.0), m.group(3)


def bundle_id(claim: str, contrast: str) -> str:
    """'d_sp:mace_medium' -> 'E1:d_sp:mace_medium' (src/gates/README_bundle.md contrast ids)."""
    return f"{claim}:{contrast}"


def read_family(y: dict, claim: str, contrasts: list, dstars: dict, points: dict, alpha: float, margins: dict,
                coverage: float, sizes: dict) -> list:
    """contrasts: names in the claim's families. Returns rows with p, Holm-adjusted p and the reading."""
    up_f, down_f = f"{claim}_up", f"{claim}_down"
    ids = [bundle_id(claim, c) for c in contrasts]
    _, sgn, mname = family_margin(y, up_f)
    m = sgn * margins[mname]
    pu = [p_up(dstars[c], m) for c in contrasts]
    rows = []
    if up_f in sizes and sizes[up_f] != len(contrasts):
        raise AnalysisError(f"{up_f}: {len(contrasts)} hypotheses, YAML size {sizes[up_f]}")
    adj_u = holm(pu, ids)
    has_down = down_f in y["families"] and down_f in sizes
    if has_down:
        _, sgn_d, mname_d = family_margin(y, down_f)
        md = sgn_d * margins[mname_d]
        pd_ = [p_down(dstars[c], md) for c in contrasts]
        if sizes[down_f] != len(contrasts):
            raise AnalysisError(f"{down_f}: {len(contrasts)} hypotheses, YAML size {sizes[down_f]}")
        adj_d = holm(pd_, ids)
    labels = list(y["claims"][claim]["readings"].keys())
    for i, c in enumerate(contrasts):
        up_ok = adj_u[i] <= alpha
        down_ok = has_down and adj_d[i] <= alpha
        if up_ok and down_ok:
            raise AnalysisError(f"{c}: both directions read (impossible by construction)")
        if up_ok:
            reading = labels[0]
        elif down_ok:
            reading = labels[1]
        else:
            reading = y["claims"][claim]["readings"]["otherwise"]
        rows.append({"contrast": c, "bundle_id": ids[i], "claim": claim, "estimate": points[c], "margin": m,
                     "interval": interval(dstars[c], coverage), "p_up": pu[i], "p_up_holm": float(adj_u[i]),
                     "p_down": (pd_[i] if has_down else None), "p_down_holm": (float(adj_d[i]) if has_down else None),
                     "family_up": up_f, "family_down": (down_f if has_down else None), "reading": reading})
    return rows


def pilot_branch(readings: dict, backbones: list) -> dict:
    """readings: contrast name -> reading. E2 part = BENEFIT on every pilot backbone; E1 part = d_sp in
    {ADDS, DOES_NOT_EXCEED} on every pilot backbone. Extension trigger: any INCONCLUSIVE among those contrasts."""
    e2 = [readings[f"g:{b}"] for b in backbones]
    e1 = [readings[f"d_sp:{b}"] for b in backbones]
    e2_holds = all(r == "BENEFIT" for r in e2)
    e1_holds = all(r in ("ADDS", "DOES_NOT_EXCEED") for r in e1)
    if e2_holds and e1_holds:
        branch = "GO"
    elif e2_holds or e1_holds:
        branch = "KILL_WITH_SURVIVOR"
    else:
        branch = "KILL"
    inconclusive = [c for c in [f"g:{b}" for b in backbones] + [f"d_sp:{b}" for b in backbones]
                    if readings[c] == "INCONCLUSIVE"]
    return {"branch": branch, "E2_part_holds": e2_holds, "E1_part_holds": e1_holds,
            "reading_first_look": ("NEEDS_EXTENSION" if inconclusive else branch),
            "survivor": ("E2" if e2_holds and not e1_holds else "E1" if e1_holds and not e2_holds else None),
            "inconclusive_branch_contrasts": inconclusive, "extension_triggered": bool(inconclusive)}


def success_rule(readings: dict, backbones: list) -> dict:
    e2 = all(readings[f"g:{b}"] == "BENEFIT" for b in backbones)
    e1 = all(readings[f"d_sp:{b}"] in ("ADDS", "DOES_NOT_EXCEED") for b in backbones)
    return {"success": bool(e2 and e1), "E2_all_benefit": e2, "E1_all_d_sp_resolved": e1}
