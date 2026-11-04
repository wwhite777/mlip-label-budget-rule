"""Descriptive analyses that pair a descriptive-split run with the primary (chemsys seed-0) run of the same mode
(YAML descriptive_specs;  findings 1, 2, 8). No readings, no Holm; 90 % percentile intervals from the paired
chemical-system cluster bootstrap with the frozen B and seed.

random_vs_chemsys  (random-reference run) delta_split = M(best head, random split) - M(best head, chemsys seed-0
                   split) per confirmatory backbone and task at the primary n; structures paired by material_id; the
                   bootstrap resamples the chemical systems carrying both scores (sorted; when every system carries
                   both, the draws are those of the primary evaluation). AUROC difference reported the same way.
fold_sensitivity   (chemsys seed-1 / seed-2 runs) the E1/E2 point estimates and intervals of this split next to the
                   seed-0 estimates, intervals and readings copied from the seed-0 run's files; seed-0 files are only
                   read (hash-checked against the seed-0 run manifest), never written.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis import inference as I
from analysis import metrics as Mx
from analysis.config import AnalysisError, Config, sha256_file


def primary_run(cfg: Config, pair_dir, look: str) -> dict:
    """The seed-0 run manifest of `pair_dir`, checked to be the primary split of the same mode and YAML."""
    pair_dir = Path(pair_dir)
    mp = pair_dir / "run_manifest.json"
    if not mp.is_file():
        raise AnalysisError(f"--pair-with {pair_dir}: no run_manifest.json")
    man = json.loads(mp.read_text(encoding="utf-8"))
    sp = man.get("split") or {}
    if sp.get("role") != "primary" or sp.get("key") != "chemsys_primary":
        raise AnalysisError(f"--pair-with {pair_dir}: not a primary chemsys seed-0 run (split {sp})")
    if man.get("mode") != cfg.mode or man.get("part") != cfg.part:
        raise AnalysisError(f"--pair-with {pair_dir}: mode/part {man.get('mode')}/{man.get('part')} != "
                            f"{cfg.mode}/{cfg.part}")
    if man.get("prereg_sha256") != sha256_file(cfg.prereg_path):
        raise AnalysisError(f"--pair-with {pair_dir}: produced under another YAML ({man.get('prereg_sha256')})")
    if look not in (man.get("looks") or []):
        raise AnalysisError(f"--pair-with {pair_dir}: look {look!r} absent ({man.get('looks')})")
    return man


def _verified(pair_dir: Path, man: dict, rel: str) -> Path:
    """A file of the seed-0 run, checked against the sha256 its run manifest recorded (before use)."""
    for run in (man.get("runs") or {}).values():
        for f in run.get("files", []):
            if f.get("path") == rel:
                p = pair_dir / rel
                if not p.is_file() or sha256_file(p) != f["sha256"]:
                    raise AnalysisError(f"seed-0 file {p} missing or differs from its recorded sha256")
                return p
    raise AnalysisError(f"seed-0 run manifest records no file {rel}")


def random_vs_chemsys(cfg: Config, part, ev: dict, pair_dir, look: str) -> list:
    if cfg.role != "random_reference":
        raise AnalysisError("random_vs_chemsys is computed in the random-reference run only")
    pair_dir = Path(pair_dir)
    man = primary_run(cfg, pair_dir, look)
    task, n, reps, y = ev["task"], ev["n"], list(ev["replicates"]), ev["y"]
    z = np.load(_verified(pair_dir, man, f"{look}/scores_{task}_n{n}.npz"), allow_pickle=False)
    if [int(r) for r in z["replicate"]] != reps:
        raise AnalysisError(f"seed-0 replicates {z['replicate'].tolist()} != {reps}")
    mids = z["material_id"].astype(str)
    pos = {m: i for i, m in enumerate(mids)}
    if len(pos) != len(mids):
        raise AnalysisError("seed-0 scores: duplicate material_id")
    methods = [str(m) for m in z["method"]]
    j = np.array([pos.get(m, -1) for m in part.ids])
    specs, meta = {}, {}
    rows_c = {}
    scope_bb = cfg.rvc_backbones(n, task)  # descriptive_specs.random_vs_chemsys.scope
    if not scope_bb:
        raise AnalysisError(f"random_vs_chemsys scope excludes task {task} n {n}")
    for bb in scope_bb:
        m = f"best_head:{bb}"
        if m not in methods or m not in ev["rows_of"]:
            raise AnalysisError(f"{m} missing in the random or the seed-0 scores")
        sc = np.full((len(reps), part.n), np.nan)
        ok = j >= 0
        sc[:, ok] = z["score"][methods.index(m)][:, j[ok]]
        rows_c[bb] = sc
    both = np.ones(part.n, dtype=bool)
    for bb in scope_bb:
        both &= np.isfinite(rows_c[bb]).all(axis=0) & np.isfinite(ev["rows_of"][f"best_head:{bb}"]).all(axis=0)
    idx = np.flatnonzero(both)
    if len(idx) == 0:
        raise AnalysisError("no structure carries both scores")
    sys_both, sys_local = np.unique(part.sys_index[idx], return_inverse=True)
    # bootstrap over the systems carrying both scores; part-level weights are gathered through a local index
    local = np.full(part.n, -1, dtype=np.int64)
    local[idx] = sys_local
    ones = np.ones(part.n, dtype=np.int64)
    for bb in scope_bb:
        for tag, rows in (("random", ev["rows_of"][f"best_head:{bb}"]), ("chemsys0", rows_c[bb])):
            ss, aps, aus = [], [], []
            for row in rows:
                s = Mx.SortedScore(row, y, idx)
                a = Mx.ap(y[idx], row[idx])
                if s.ap(ones) != a:
                    raise AnalysisError("weighted AP != sklearn AP (random_vs_chemsys)")
                ss.append(s)
                aps.append(a)
                aus.append(Mx.auroc(y[idx], row[idx]) if y[idx].min() != y[idx].max() else float("nan"))
            specs[f"{bb}:{tag}"] = ss
            meta[f"{bb}:{tag}"] = (float(np.mean(aps)), float(np.mean(aus)))
    S = len(sys_both)
    # structures outside the paired population get local index 0; their weight is never gathered (every spec's
    # population is the paired set), so the draws resample exactly the S systems carrying both scores
    w_index = np.where(local >= 0, local, 0)
    mstar, astar = I.bootstrap(specs, w_index, S, cfg.B, cfg.boot_seed, with_auroc=True)
    n_pos = int(y[idx].sum())
    out = []
    for bb in scope_bb:
        r_, c_ = f"{bb}:random", f"{bb}:chemsys0"
        d = mstar[r_] - mstar[c_]
        lo, hi = I.interval(d, cfg.coverage)
        (alo, ahi), und = I.interval_finite(astar[r_] - astar[c_], cfg.coverage)
        out.append({"analysis": "random_vs_chemsys", "task": task, "n": n, "backbone": bb,
                    "a": f"best_head:{bb} (random split)", "b": f"best_head:{bb} (chemsys seed-0 split)",
                    "n_population": int(len(idx)), "n_positive": n_pos, "prevalence": n_pos / len(idx),
                    "n_systems_bootstrap": S, "M_random": meta[r_][0], "M_chemsys0": meta[c_][0],
                    "estimate": meta[r_][0] - meta[c_][0], "ci_lo": lo, "ci_hi": hi,
                    "auroc_estimate": meta[r_][1] - meta[c_][1], "auroc_ci_lo": alo, "auroc_ci_hi": ahi,
                    "auroc_undefined_draws": und, "replicates": json.dumps(reps), "role": "descriptive"})
    return out


def fold_sensitivity(cfg: Config, ev: dict, contrast_rows: list, pair_dir, look: str) -> list:
    """This split's E1/E2 estimates and intervals next to seed 0 (estimates, intervals, readings read from the seed-0
    run). Nothing of the seed-0 run is modified."""
    if cfg.role != "fold_sensitivity":
        raise AnalysisError("fold_sensitivity is computed in the chemsys seed-1 / seed-2 runs only")
    if ev["n"] != cfg.primary_n:
        return []
    pair_dir = Path(pair_dir)
    man = primary_run(cfg, pair_dir, look)
    tag = f"{ev['task']}_n{ev['n']}"
    with open(_verified(pair_dir, man, f"{look}/contrasts_{tag}.csv"), newline="", encoding="utf-8") as fh:
        base = {r["contrast"]: r for r in csv.DictReader(fh)}
    rd = json.loads(_verified(pair_dir, man, f"{look}/readings.json").read_text(encoding="utf-8"))
    reading0 = {r["contrast"]: r["reading"] for r in rd["readings"]}
    out = []
    for r in contrast_rows:
        name = r["contrast"]
        if "|" in name or name.split(":")[0] not in ("d_sp", "d_is", "g"):
            continue
        b = base.get(name)
        if b is None:
            raise AnalysisError(f"seed-0 run has no contrast {name}")
        out.append({"analysis": "fold_sensitivity", "task": ev["task"], "n": ev["n"], "contrast": name,
                    "claim": r["claim"], "seed0_estimate": float(b["estimate"]), "seed0_ci_lo": float(b["ci_lo"]),
                    "seed0_ci_hi": float(b["ci_hi"]), "seed0_reading": reading0.get(name, ""),
                    "split_seed": cfg.split_seed, "estimate": r["estimate"], "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"], "n_population": r["n_population"], "n_positive": r["n_positive"],
                    "prevalence": r["prevalence"],
                    "role": "descriptive; never changes the seed-0 reading"})
    return out
