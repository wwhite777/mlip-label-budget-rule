"""Build the label-blind P3 split manifests: 3 families x fold-assignment seeds, K folds each.

python3 -m splits.make_splits --wbm WBM_SUMMARY.csv.gz --mp MP_ENERGIES.csv.gz --out-dir result/splits
        [--seeds 0 1 2] [--k 5]

Universe: WBM rows with unique_prototype == True. Chemical system of a structure = the set of element symbols in its
`formula` (regex [A-Z][a-z]?), written sorted and joined by "-". Every text column is read with
keep_default_na=False (the formula "NaN" is sodium nitride).

Families (one manifest per family x seed):
  random   structures are shuffled with numpy default_rng(seed) and dealt round-robin into K folds (KFold-style
           shuffle: uniformly random assignment with fold sizes differing by at most 1).
  chemsys  whole chemical systems go to one fold: systems are shuffled with default_rng(seed), stably sorted by
           structure count (largest first), and each is placed into the fold with the currently smallest structure
           count (ties: lowest fold index). A test fold therefore shares no chemical system with its training folds.
  element  exploratory MatFold-like rung. The elements present in the universe are shuffled with default_rng(seed)
           and, in that seeded order, greedily placed into the group (of K) with the smallest summed per-element
           structure count (no largest-first sort: with nearly all element counts distinct it would make the groups
           identical for every seed). Test fold k = structures containing >= 1
           element of group k; training set of fold k = structures containing NO element of group k. A structure
           whose elements fall into several groups is a test item of each such fold and is excluded from the training
           set of each such fold; folds are NOT a partition of the universe.

Outputs in --out-dir (gzip with mtime=0, so the manifests are byte-reproducible):
  splits_<family>_seed<s>.csv.gz      random/chemsys: material_id, fold, chemsys, sys_in_mp
                                      element:        material_id, test_folds ("0;3"), chemsys, sys_in_mp
  splits_element_seed<s>_groups.json  element sidecar: definition, element groups, per fold sha256 of the sorted
                                      newline-joined test ids and train ids, sizes
  splits_<family>_seed<s>_summary.json fold sizes, systems per fold, elements per group, sys_in_mp fraction per fold,
                                      sha256 of the manifest (and sidecar), columns_loaded record, code sha256
sys_in_mp (label-free): True iff the structure's chemical system equals the chemical system of at least one MP entry
in the mp-energies file (only its material_id and formula columns are read).

Label blindness: only the columns in ALLOWED_COLUMNS are loaded (pandas usecols); the code asserts that no loaded
column matches LABEL_PATTERN, and records the loaded columns per input in each summary (G3 checks that record).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
ALLOWED_COLUMNS = {"wbm": ("material_id", "formula", "unique_prototype"), "mp": ("material_id", "formula")}
LABEL_PATTERN = re.compile(r"e_above_hull|e_form|energy|enthalpy|hull|bandgap|correction|stable", re.I)
FAMILIES = ("random", "chemsys", "element")
CODE = Path(__file__).resolve()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def chemsys(formula: str) -> str:
    els = sorted(set(ELEMENT_RE.findall(str(formula))))
    if not els:
        raise ValueError(f"formula {formula!r} has no element symbols")
    return "-".join(els)


def load_label_blind(path, which: str) -> tuple[pd.DataFrame, dict]:
    cols = ALLOWED_COLUMNS[which]
    assert not any(LABEL_PATTERN.search(c) for c in cols), f"label-like column in the allow-list {cols}"
    df = pd.read_csv(path, usecols=list(cols), keep_default_na=False, dtype={"material_id": str, "formula": str})
    assert tuple(sorted(df.columns)) == tuple(sorted(cols)), f"{path}: loaded {list(df.columns)} != {cols}"
    assert not any(LABEL_PATTERN.search(c) for c in df.columns), f"{path}: label-like column loaded"
    rec = {"path": str(path), "sha256": sha256_file(path), "columns": sorted(df.columns)}
    return df, rec


def load_universe(wbm_path, mp_path):
    wbm, rec_w = load_label_blind(wbm_path, "wbm")
    mp, rec_m = load_label_blind(mp_path, "mp")
    flag = wbm["unique_prototype"].astype(str).str.strip().str.lower()
    bad = sorted(set(flag) - {"true", "false"})
    if bad:
        raise ValueError(f"unique_prototype has values outside True/False: {bad[:5]}")
    u = wbm.loc[flag == "true", ["material_id", "formula"]].reset_index(drop=True)
    if not u["material_id"].is_unique:
        raise ValueError("duplicate material_id in the universe")
    u["chemsys"] = u["formula"].map(chemsys)
    mp_sys = set(mp["formula"].map(chemsys))
    u["sys_in_mp"] = u["chemsys"].isin(mp_sys)
    return u, {"wbm": rec_w, "mp": rec_m}


def assign_random(u: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(len(u))
    fold = np.empty(len(u), dtype=int)
    fold[perm] = np.arange(len(u)) % k
    return fold


def greedy_balance(keys: list, weights: dict, k: int, seed: int, largest_first: bool = True) -> dict:
    """Seeded shuffle, optionally stable sort by weight (largest first), place each key into the lightest bin."""
    order = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    if largest_first:
        order.sort(key=lambda x: -weights[x])  # stable: ties keep the seeded order
    load = [0] * k
    out = {}
    for x in order:
        b = min(range(k), key=lambda j: (load[j], j))
        out[x] = b
        load[b] += weights[x]
    return out


def assign_chemsys(u: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    counts = u["chemsys"].value_counts().to_dict()
    systems = sorted(counts)
    sys_fold = greedy_balance(systems, counts, k, seed)
    return u["chemsys"].map(sys_fold).to_numpy(dtype=int)


def assign_element(u: pd.DataFrame, k: int, seed: int):
    els_per = u["chemsys"].str.split("-")
    counts: dict = {}
    for els in els_per:
        for e in els:
            counts[e] = counts.get(e, 0) + 1
    elements = sorted(counts)
    # largest-first sorting would make the groups seed-invariant (85 elements, nearly all counts distinct, so the
    # seeded shuffle only breaks ties); the seeded order itself is therefore kept (greedy into the lightest group)
    el_group = greedy_balance(elements, counts, k, seed, largest_first=False)
    test_folds = els_per.map(lambda els: ";".join(str(g) for g in sorted({el_group[e] for e in els})))
    groups = {g: sorted(e for e in elements if el_group[e] == g) for g in range(k)}
    return test_folds, groups, counts


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def wjson(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


DEFINITIONS = {
    "random": "Structures shuffled with numpy default_rng(seed) and dealt round-robin into K folds; folds partition "
              "the universe.",
    "chemsys": "Whole chemical systems (set of elements in the formula) assigned to folds: seeded shuffle, stable sort "
               "by structure count descending, greedy into the fold with the smallest structure count; folds "
               "partition the universe and no chemical system occurs in two folds.",
    "element": "Exploratory MatFold-like rung. Elements partitioned into K seeded groups (seeded shuffle; in that order "
               "each element goes greedily into the group with the smallest summed per-element structure count). "
               "Test fold k = structures containing >= 1 element of group k; training set of fold k = structures "
               "containing no element of group k. A structure with elements in several groups is a test item of "
               "each such fold and excluded from the training set of each; folds do not partition the universe.",
}


def build(wbm_path, mp_path, out_dir, seeds, k) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u, loaded = load_universe(wbm_path, mp_path)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = []
    for fam in FAMILIES:
        for s in seeds:
            stem = f"splits_{fam}_seed{s}"
            man_p = out_dir / f"{stem}.csv.gz"
            summ = {"family": fam, "seed": s, "k": k, "definition": DEFINITIONS[fam],
                    "universe": "WBM summary rows with unique_prototype == True", "n_universe": int(len(u)),
                    "chemsys_definition": "sorted set of [A-Z][a-z]? symbols in formula, joined by '-'",
                    "sys_in_mp_definition": "chemical system equals that of >= 1 entry of the MP energies file",
                    "label_blind": True, "columns_loaded": loaded, "prototype_disjoint": False,
                    "code": {"path": str(CODE), "sha256": sha256_file(CODE)}, "created_utc": created}
            folds = []
            if fam in ("random", "chemsys"):
                fold = assign_random(u, k, s) if fam == "random" else assign_chemsys(u, k, s)
                man = pd.DataFrame({"material_id": u["material_id"], "fold": fold, "chemsys": u["chemsys"],
                                    "sys_in_mp": u["sys_in_mp"]})
                for f in range(k):
                    m = man["fold"] == f
                    folds.append({"fold": f, "n_test": int(m.sum()), "n_train": int((~m).sum()),
                                  "n_systems_test": int(man.loc[m, "chemsys"].nunique()),
                                  "sys_in_mp_frac_test": round(float(man.loc[m, "sys_in_mp"].mean()), 4),
                                  "test_ids_sha256": ids_hash(man.loc[m, "material_id"]),
                                  "train_ids_sha256": ids_hash(man.loc[~m, "material_id"])})
                write_csv_gz(man, man_p)
            else:
                test_folds, groups, counts = assign_element(u, k, s)
                man = pd.DataFrame({"material_id": u["material_id"], "test_folds": test_folds,
                                    "chemsys": u["chemsys"], "sys_in_mp": u["sys_in_mp"]})
                tf_sets = man["test_folds"].str.split(";").map(lambda x: {int(v) for v in x})
                for f in range(k):
                    m = tf_sets.map(lambda x, f=f: f in x)
                    folds.append({"fold": f, "group_elements": groups[f], "n_elements": len(groups[f]),
                                  "group_structure_count_sum": int(sum(counts[e] for e in groups[f])),
                                  "n_test": int(m.sum()), "n_train": int((~m).sum()),
                                  "n_systems_test": int(man.loc[m, "chemsys"].nunique()),
                                  "sys_in_mp_frac_test": round(float(man.loc[m, "sys_in_mp"].mean()), 4),
                                  "test_ids_sha256": ids_hash(man.loc[m, "material_id"]),
                                  "train_ids_sha256": ids_hash(man.loc[~m, "material_id"])})
                write_csv_gz(man, man_p)
                side_p = out_dir / f"{stem}_groups.json"
                side = {"family": fam, "seed": s, "k": k, "definition": DEFINITIONS[fam],
                        "groups": {str(g): groups[g] for g in range(k)},
                        "folds": [{x: f[x] for x in ("fold", "n_test", "n_train", "test_ids_sha256",
                                                     "train_ids_sha256")} for f in folds],
                        "multi_fold_structures": int((tf_sets.map(len) > 1).sum()),
                        "test_membership_total": int(tf_sets.map(len).sum())}
                wjson(side_p, side)
                summ["sidecar"] = {"path": side_p.name, "sha256": sha256_file(side_p)}
                summ["elements_per_group"] = {str(g): len(groups[g]) for g in range(k)}
                summ["multi_fold_structures"] = side["multi_fold_structures"]
                summ["test_membership_total"] = side["test_membership_total"]
            summ["folds"] = folds
            summ["fold_sizes"] = [f["n_test"] for f in folds]
            summ["manifest"] = {"path": man_p.name, "sha256": sha256_file(man_p), "columns": list(man.columns),
                                "n_rows": int(len(man))}
            summ_p = out_dir / f"{stem}_summary.json"
            wjson(summ_p, summ)
            written += [man_p, summ_p] + ([out_dir / f"{stem}_groups.json"] if fam == "element" else [])
            print(f"{stem}: fold sizes {summ['fold_sizes']} -> {man_p} sha256 {summ['manifest']['sha256']}")
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="splits.make_splits", description=__doc__.splitlines()[0])
    p.add_argument("--wbm", required=True)
    p.add_argument("--mp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args(argv)
    for x in (a.wbm, a.mp):
        if not Path(x).is_file():
            print(f"missing input {x}", file=sys.stderr)
            return 2
    build(a.wbm, a.mp, a.out_dir, a.seeds, a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
