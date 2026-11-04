"""Label-blind PILOT / CONFIRM partition of the P3 universe by chemical system, with fold manifests inside each part.

python3 -m splits.make_partition build --wbm WBM_SUMMARY.csv.gz --mp MP_ENERGIES.csv.gz --strata STRATA.csv.gz
        --out-dir result/splits/partition_v1 [--seed 20261002] [--pilot-frac 0.2] [--k 5] [--fold-seeds 0 1 2]
        [--random-seeds 0]
python3 -m splits.make_partition run    (same arguments)  build + G3 (partition mode) + gate pytest + run_record

Universe U, chemical system, sys_in_mp and the fold algorithms are those of splits.make_splits (imported, unchanged):
U = WBM rows with unique_prototype == True; chemsys = sorted set of [A-Z][a-z]? symbols of `formula` joined by "-".

Partition (label-blind; uses only the chemical-system keys):
  systems = sorted list of the distinct chemsys keys of U; order = [systems[i] for i in default_rng(seed).permutation(n)];
  PILOT = the first ceil(pilot_frac * n_systems) systems of `order` (the smallest prefix holding >= pilot_frac of the
  systems, system-weighted); CONFIRM = the rest. No re-draw is ever made; balance is reported, not optimised.
Inside each part (part universe = the part's structures in U order):
  chemsys family, fold-assignment seeds --fold-seeds: splits.make_splits.assign_chemsys (seeded shuffle, stable sort by
  structure count descending, greedy into the lightest fold);  CONFIRM only: random family, seeds --random-seeds,
  splits.make_splits.assign_random.
Outputs in --out-dir:
  partition.csv.gz                      material_id, chemsys, part (PILOT|CONFIRM); U order; gzip mtime=0
  partition_summary.json                rule, sizes, balance tables (arity, sys_in_mptrj, element frequency top-20),
                                        report-only flags, columns_loaded, descriptive_columns_loaded, sha256
  loaded_columns.json                   every column loaded per input file (proof that no label column was read)
  pilot/, confirm/ splits_<family>_seed<s>.csv.gz + _summary.json  (same CSV schema as result/splits:
                                        material_id, fold, chemsys, sys_in_mp)
The strata file (label-free: material_id, chemsys, sys_in_mptrj, arity) is read only to describe the parts and folds;
it never influences an assignment. Its chemsys must equal the formula-derived chemsys for every structure of U.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from splits.make_splits import (LABEL_PATTERN, assign_chemsys, assign_random, ids_hash, load_universe, sha256_file,
                                write_csv_gz, wjson)

PARTS = ("PILOT", "CONFIRM")
STRATA_COLUMNS = ("material_id", "chemsys", "sys_in_mptrj", "arity")
CODE = Path(__file__).resolve()
REPO = CODE.parents[2]
FLAG_ABS_DIFF = 0.02  # report-only threshold (coder's choice) for |part - U| in an arity share or sys_in_mptrj fraction


def load_strata(path) -> tuple[pd.DataFrame, dict]:
    cols = STRATA_COLUMNS
    assert not any(LABEL_PATTERN.search(c) for c in cols), f"label-like column in the strata allow-list {cols}"
    df = pd.read_csv(path, usecols=list(cols), keep_default_na=False, dtype=str)
    assert tuple(sorted(df.columns)) == tuple(sorted(cols)), f"{path}: loaded {list(df.columns)} != {cols}"
    assert not any(LABEL_PATTERN.search(c) for c in df.columns), f"{path}: label-like column loaded"
    bad = sorted(set(df["sys_in_mptrj"]) - {"True", "False"})
    if bad:
        raise ValueError(f"{path}: sys_in_mptrj has values outside True/False: {bad[:5]}")
    df["sys_in_mptrj"] = df["sys_in_mptrj"] == "True"
    df["arity"] = df["arity"].astype(int)
    return df, {"path": str(path), "sha256": sha256_file(path), "columns": sorted(df.columns)}


def partition_systems(systems: list, seed: int, frac: float) -> tuple[set, int]:
    systems = sorted(systems)
    order = [systems[i] for i in np.random.default_rng(seed).permutation(len(systems))]
    n_pilot = math.ceil(frac * len(systems))
    return set(order[:n_pilot]), n_pilot


def r4(x) -> float:
    return round(float(x), 4)


def arity_table(df: pd.DataFrame) -> dict:
    sys_ar = df.drop_duplicates("chemsys").set_index("chemsys")["arity"]
    out = {}
    for a in sorted(df["arity"].unique()):
        ns, nst = int((sys_ar == a).sum()), int((df["arity"] == a).sum())
        out[str(a)] = {"n_systems": ns, "n_structures": nst, "system_share": r4(ns / len(sys_ar)),
                       "structure_share": r4(nst / len(df))}
    return out


def mptrj_stats(df: pd.DataFrame) -> dict:
    sys_m = df.drop_duplicates("chemsys")["sys_in_mptrj"]
    return {"structure_weighted": r4(df["sys_in_mptrj"].mean()), "system_weighted": r4(sys_m.mean()),
            "n_structures_sys_in_mptrj": int(df["sys_in_mptrj"].sum()), "n_systems_in_mptrj": int(sys_m.sum())}


def element_freq(df: pd.DataFrame) -> pd.Series:
    """Fraction of structures whose chemical system contains each element."""
    return df["chemsys"].str.split("-").explode().value_counts() / len(df)


def balance(u: pd.DataFrame) -> dict:
    groups = {"U": u, **{p: u[u["part"] == p] for p in PARTS}}
    out = {"arity": {g: arity_table(d) for g, d in groups.items()},
           "sys_in_mptrj": {g: mptrj_stats(d) for g, d in groups.items()}}
    freq = {g: element_freq(d) for g, d in groups.items()}
    top = list(freq["U"].sort_values(ascending=False, kind="stable").index[:20])
    out["element_frequency_top20"] = [{"element": e, **{g: r4(freq[g].get(e, 0.0)) for g in groups}} for e in top]
    diffs = {p: (freq[p].reindex(freq["U"].index, fill_value=0.0) - freq["U"]).abs() for p in PARTS}
    out["element_frequency_max_abs_diff"] = {p: {"element": str(diffs[p].idxmax()), "abs_diff": r4(diffs[p].max())}
                                             for p in PARTS}
    flags = []
    for p in PARTS:
        for a, row in out["arity"]["U"].items():
            pr = out["arity"][p].get(a)
            if pr is None:
                flags.append(f"{p}: no arity-{a} system (U has {row['n_systems']})")
                continue
            for key in ("system_share", "structure_share"):
                if abs(pr[key] - row[key]) > FLAG_ABS_DIFF:
                    flags.append(f"{p}: arity {a} {key} {pr[key]} vs U {row[key]}")
        for key in ("structure_weighted", "system_weighted"):
            a_, b_ = out["sys_in_mptrj"][p][key], out["sys_in_mptrj"]["U"][key]
            if abs(a_ - b_) > FLAG_ABS_DIFF:
                flags.append(f"{p}: sys_in_mptrj {key} {a_} vs U {b_}")
        absent = sorted(set(freq["U"].index) - set(freq[p].index))
        if absent:
            flags.append(f"{p}: {len(absent)} element(s) of U absent: {absent}")
    out["flag_rule"] = (f"report only (no re-draw): an arity class or element of U absent from a part, or an absolute "
                        f"difference > {FLAG_ABS_DIFF} between a part and U in an arity share or sys_in_mptrj fraction")
    out["flags"] = flags
    return out


def fold_rows(man: pd.DataFrame, k: int) -> list:
    rows = []
    for f in range(k):
        m = man["fold"] == f
        te = man[m]
        rows.append({"fold": f, "n_test": int(m.sum()), "n_train": int((~m).sum()),
                     "n_systems_test": int(te["chemsys"].nunique()),
                     "arity_structures_test": {str(a): int(n) for a, n in sorted(te["arity"].value_counts().items())},
                     "arity_systems_test": {str(a): int(n) for a, n in
                                            sorted(te.drop_duplicates("chemsys")["arity"].value_counts().items())},
                     "sys_in_mp_frac_test": r4(te["sys_in_mp"].mean()),
                     "sys_in_mptrj_frac_test": r4(te["sys_in_mptrj"].mean()),
                     "test_ids_sha256": ids_hash(te["material_id"]),
                     "train_ids_sha256": ids_hash(man.loc[~m, "material_id"])})
    return rows


def build(wbm, mp, strata, out_dir, seed=20261002, frac=0.2, k=5, fold_seeds=(0, 1, 2), random_seeds=(0,)) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, loaded = load_universe(wbm, mp)
    st, st_rec = load_strata(strata)
    if not st["material_id"].is_unique or set(st["material_id"]) != set(u["material_id"]):
        raise ValueError("strata material_id set differs from the universe")
    st = st.set_index("material_id")
    if not (st.loc[u["material_id"], "chemsys"].to_numpy() == u["chemsys"].to_numpy()).all():
        raise ValueError("strata chemsys differs from the formula-derived chemsys")
    u["sys_in_mptrj"] = st.loc[u["material_id"], "sys_in_mptrj"].to_numpy()
    u["arity"] = st.loc[u["material_id"], "arity"].to_numpy()
    if not (u["arity"] == u["chemsys"].str.count("-") + 1).all():
        raise ValueError("strata arity differs from the number of elements in chemsys")
    systems = sorted(u["chemsys"].unique())
    pilot, n_pilot = partition_systems(systems, seed, frac)
    u["part"] = np.where(u["chemsys"].isin(pilot), "PILOT", "CONFIRM")
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    code = {"path": str(CODE), "sha256": sha256_file(CODE),
            "make_splits_sha256": sha256_file(CODE.parent / "make_splits.py")}
    part_p = out_dir / "partition.csv.gz"
    write_csv_gz(u[["material_id", "chemsys", "part"]], part_p)
    written = [part_p]
    manifests = {}
    for p in PARTS:
        up = u[u["part"] == p].reset_index(drop=True)
        sub = out_dir / p.lower()
        sub.mkdir(exist_ok=True)
        jobs = [("chemsys", s) for s in fold_seeds] + ([("random", s) for s in random_seeds] if p == "CONFIRM" else [])
        manifests[p] = []
        for fam, s in jobs:
            fold = assign_chemsys(up, k, s) if fam == "chemsys" else assign_random(up, k, s)
            man = pd.DataFrame({"material_id": up["material_id"], "fold": fold, "chemsys": up["chemsys"],
                                "sys_in_mp": up["sys_in_mp"]})
            stem = f"splits_{fam}_seed{s}"
            man_p = sub / f"{stem}.csv.gz"
            write_csv_gz(man, man_p)
            desc = man.assign(sys_in_mptrj=up["sys_in_mptrj"].to_numpy(), arity=up["arity"].to_numpy())
            folds = fold_rows(desc, k)
            summ = {"family": fam, "seed": s, "k": k, "part": p, "partition_seed": seed,
                    "definition": (f"{fam} family of splits.make_splits applied to the {p} part only"),
                    "universe": f"structures of U (WBM unique_prototype == True) whose chemical system is in {p}",
                    "n_universe": int(len(up)), "n_systems": int(up["chemsys"].nunique()),
                    "label_blind": True, "columns_loaded": loaded,
                    "descriptive_columns_loaded": {"strata": st_rec}, "prototype_disjoint": False,
                    "code": code, "created_utc": created, "folds": folds,
                    "fold_sizes": [f["n_test"] for f in folds],
                    "manifest": {"path": man_p.name, "sha256": sha256_file(man_p), "columns": list(man.columns),
                                 "n_rows": int(len(man))}}
            summ_p = sub / f"{stem}_summary.json"
            wjson(summ_p, summ)
            written += [man_p, summ_p]
            manifests[p].append({"path": f"{sub.name}/{man_p.name}", "sha256": summ["manifest"]["sha256"],
                                 "fold_sizes": summ["fold_sizes"]})
            print(f"{p} {stem}: fold sizes {summ['fold_sizes']} -> {man_p} sha256 {summ['manifest']['sha256']}")
    sizes = {}
    for p in PARTS:
        d = u[u["part"] == p]
        sizes[p] = {"n_structures": int(len(d)), "n_systems": int(d["chemsys"].nunique()),
                    "structure_share": r4(len(d) / len(u)), "system_share": r4(d["chemsys"].nunique() / len(systems))}
    summary = {
        "partition_seed": seed, "pilot_frac": frac, "k": k, "fold_seeds": list(fold_seeds),
        "random_seeds_confirm": list(random_seeds),
        "rule": ("systems = sorted distinct chemsys keys of U; order = [systems[i] for i in "
                 "numpy.random.default_rng(seed).permutation(len(systems))]; PILOT = first ceil(pilot_frac * "
                 "n_systems) systems of order; CONFIRM = the rest; label-blind; no re-draw"),
        "universe": "WBM summary rows with unique_prototype == True", "n_universe": int(len(u)),
        "n_systems": len(systems), "n_pilot_systems_rule": n_pilot, "sizes": sizes,
        "balance": balance(u), "label_blind": True, "columns_loaded": loaded,
        "descriptive_columns_loaded": {"strata": st_rec}, "code": code, "created_utc": created,
        "partition": {"path": part_p.name, "sha256": sha256_file(part_p), "columns": ["material_id", "chemsys", "part"],
                      "n_rows": int(len(u))},
        "manifests": manifests,
    }
    summ_p = out_dir / "partition_summary.json"
    wjson(summ_p, summary)
    lc_p = out_dir / "loaded_columns.json"
    wjson(lc_p, {"label_pattern": LABEL_PATTERN.pattern, "assignment_inputs": loaded,
                 "descriptive_inputs": {"strata": st_rec},
                 "any_loaded_column_matches_label_pattern": any(
                     LABEL_PATTERN.search(c) for r in [*loaded.values(), st_rec] for c in r["columns"]),
                 "note": "pandas usecols; the WBM file is read with material_id, formula, unique_prototype only"})
    written += [summ_p, lc_p]
    print(f"partition: PILOT {sizes['PILOT']} CONFIRM {sizes['CONFIRM']}")
    for fl in summary["balance"]["flags"]:
        print(f"BALANCE FLAG (report only): {fl}")
    return written


# ------------------------------------------------------------------------------------------------ run + run_record
def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def logged(cmd: list, log: Path, env=None) -> int:
    with open(log, "wb") as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def md_balance(b: dict) -> list:
    g = ["U", *PARTS]
    L = ["### Arity (systems / structures; shares)", "",
         "| arity | " + " | ".join(f"{x} systems | {x} sys share | {x} structures | {x} str share" for x in g) + " |",
         "|---|" + "---|" * (4 * len(g))]
    for a in b["arity"]["U"]:
        cells = []
        for x in g:
            r = b["arity"][x].get(a, {"n_systems": 0, "system_share": 0, "n_structures": 0, "structure_share": 0})
            cells += [str(r["n_systems"]), str(r["system_share"]), str(r["n_structures"]), str(r["structure_share"])]
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "### sys_in_mptrj (label-free MPtrj chemical-system membership)", "",
          "| group | structure-weighted | system-weighted | structures in | systems in |", "|---|---|---|---|---|"]
    for x in g:
        r = b["sys_in_mptrj"][x]
        L.append(f"| {x} | {r['structure_weighted']} | {r['system_weighted']} | {r['n_structures_sys_in_mptrj']} | "
                 f"{r['n_systems_in_mptrj']} |")
    L += ["", "### Element frequency, top 20 in U (fraction of structures containing the element)", "",
          "| element | " + " | ".join(g) + " |", "|---|" + "---|" * len(g)]
    for r in b["element_frequency_top20"]:
        L.append(f"| {r['element']} | " + " | ".join(str(r[x]) for x in g) + " |")
    L += ["", "Largest |part - U| element-frequency difference over all elements: " + "; ".join(
        f"{p} {v['element']} {v['abs_diff']}" for p, v in b["element_frequency_max_abs_diff"].items()), "",
          f"Flag rule: {b['flag_rule']}.", "",
          "Flags: " + ("none" if not b["flags"] else ""), *[f"- {f}" for f in b["flags"]]]
    return L


def run_all(a):
    """Generate the specified partition with the public partition function."""
    build(a.wbm, a.mp, a.strata, a.out_dir, a.seed, a.pilot_frac, a.k, a.fold_seeds, a.random_seeds)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="splits.make_partition", description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=("build", "run"))
    p.add_argument("--wbm", required=True)
    p.add_argument("--mp", required=True)
    p.add_argument("--strata", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--pilot-frac", type=float, default=0.2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--fold-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--random-seeds", type=int, nargs="+", default=[0])
    a = p.parse_args(argv)
    for x in (a.wbm, a.mp, a.strata):
        if not Path(x).is_file():
            print(f"missing input {x}", file=sys.stderr)
            return 2
    if a.mode == "run":
        return run_all(a)
    build(a.wbm, a.mp, a.strata, a.out_dir, a.seed, a.pilot_frac, a.k, a.fold_seeds, a.random_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
