"""Universe U, the part (PILOT / CONFIRM) with its outer folds, and the label vault.

Canonical order of a part: material_id sorted lexicographically (numpy string sort). Every array of the pipeline is
aligned to that order; "material_id order" (top-k ties, draws) means this order.

Labels
- dryrun: SYNTHETIC Bernoulli labels (config.SYNTHETIC_*), drawn for stability then band gap from one
  numpy default_rng(424242) over the part in canonical order; no label column is opened.
- pilot / confirm: the summary file is first read for material_id only; the rows of the part are selected; only then
  are the label columns parsed, for those rows only (pandas skiprows). The manifest records columns and row counts.
The LabelVault hands a replicate pipeline only labels inside its draw (else BudgetViolation) and checks at close that
the union of touched indices equals the draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.config import (SYNTHETIC_BASE_RATE, SYNTHETIC_SEED, TASKS, AnalysisError, BudgetViolation, Config,
                             PartitionError, apply_positive_rule, check_sha, resolve)


# ---------------------------------------------------------------------------------------------------------- universe
def load_universe(cfg: Config) -> pd.DataFrame:
    """U = summary rows with unique_prototype == True; columns material_id, formula (no label column is read)."""
    pop = cfg.y["population"]
    path = resolve(pop["summary_file"]["path"])
    check_sha(path, pop["summary_file"]["sha256"], "WBM summary")
    if pop["universe_filter"] != "unique_prototype == True":
        raise AnalysisError(f"unexpected universe filter {pop['universe_filter']!r}")
    df = pd.read_csv(path, usecols=["material_id", "formula", "unique_prototype"], dtype=str,
                     keep_default_na=bool(not pop["read_text_columns_with_keep_default_na_false"]))
    up = df["unique_prototype"]
    if not set(up.unique()) <= {"True", "False"}:
        raise AnalysisError(f"unique_prototype has values {sorted(set(up.unique()))[:5]}")
    u = df.loc[up == "True", ["material_id", "formula"]].reset_index(drop=True)
    if len(u) != int(pop["universe_n"]) or u["material_id"].duplicated().any():
        raise AnalysisError(f"U has {len(u)} rows (expected {pop['universe_n']}) or duplicate ids")
    return u


def load_partition(cfg: Config) -> pd.DataFrame:
    pp = cfg.y["population"]["partition"]
    path = resolve(pp["path"])
    check_sha(path, pp["sha256"], "partition")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    counts = df["part"].value_counts().to_dict()
    if counts.get("PILOT") != pp["pilot_structures"] or counts.get("CONFIRM") != pp["confirm_structures"]:
        raise AnalysisError(f"partition counts {counts} differ from the YAML")
    sysc = df.groupby("part")["chemsys"].nunique().to_dict()
    if sysc.get("PILOT") != pp["pilot_systems"] or sysc.get("CONFIRM") != pp["confirm_systems"]:
        raise AnalysisError(f"partition system counts {sysc} differ from the YAML")
    return df


def assert_in_part(ids, part: str, part_of: dict, what: str) -> None:
    """Hard error if any id belongs to the other part or is outside U."""
    ids = np.asarray(ids)
    bad = [m for m in ids if part_of.get(m) != part]
    if bad:
        other = sorted({str(part_of.get(m)) for m in bad})
        raise PartitionError(f"{what}: {len(bad)} id(s) not in {part} (found parts {other}); first {bad[:3]}")


@dataclass
class PartData:
    part: str
    ids: np.ndarray
    chemsys: np.ndarray
    fold: np.ndarray
    formula: np.ndarray
    systems: np.ndarray
    sys_index: np.ndarray
    part_of: dict = field(repr=False, default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    def index_of(self, ids) -> np.ndarray:
        pos = np.searchsorted(self.ids, ids)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        if not np.array_equal(self.ids[pos], np.asarray(ids)):
            raise PartitionError("ids not in the part")
        return pos


def build_part(part: str, ids, chemsys, fold, formula, part_of: dict, manifest=None,
               split_type: str = "chemsys") -> PartData:
    """Align to canonical order and run the partition guard (also used by the fixtures). The PILOT/CONFIRM and id guards
    apply to every split; the within-fold system-disjointness check applies to chemsys splits only (the random-reference
    split shares systems between folds by construction;  finding 2)."""
    if split_type not in ("chemsys", "random"):
        raise AnalysisError(f"unknown split type {split_type!r}")
    ids = np.asarray(ids).astype(str)
    assert_in_part(ids, part, part_of, "split manifest")
    o = np.argsort(ids, kind="mergesort")
    ids, chemsys, fold, formula = ids[o], np.asarray(chemsys)[o].astype(str), np.asarray(fold)[o].astype(np.int64), \
        np.asarray(formula)[o].astype(str)
    if len(np.unique(ids)) != len(ids):
        raise AnalysisError("duplicate material_id in the split manifest")
    systems, sys_index = np.unique(chemsys, return_inverse=True)
    fs = pd.DataFrame({"s": sys_index, "f": fold}).groupby("s")["f"].nunique()
    n_shared = int((fs > 1).sum())
    if split_type == "chemsys" and n_shared:  # chemsys-disjoint folds: every system in exactly one fold
        raise AnalysisError("a chemical system spans more than one outer fold")
    manifest = dict(manifest or {}, split_type=split_type, n_systems_spanning_folds=n_shared)
    return PartData(part, ids, chemsys, fold, formula, systems, sys_index.astype(np.int64), part_of, manifest or {})


def load_part(cfg: Config) -> PartData:
    u = load_universe(cfg)
    p = load_partition(cfg)
    if set(u["material_id"]) != set(p["material_id"]):
        raise AnalysisError("partition ids != U")
    part_of = dict(zip(p["material_id"], p["part"]))
    sys_of = dict(zip(p["material_id"], p["chemsys"]))
    check_sha(cfg.split_path, cfg.split_sha, "split manifest")
    s = pd.read_csv(cfg.split_path, dtype=str, keep_default_na=False)
    want = set(p.loc[p["part"] == cfg.part, "material_id"])
    assert_in_part(s["material_id"].to_numpy(), cfg.part, part_of, f"split {cfg.split_path.name}")
    if set(s["material_id"]) != want:
        raise AnalysisError(f"split manifest does not cover the {cfg.part} part exactly")
    if any(sys_of[m] != c for m, c in zip(s["material_id"], s["chemsys"])):
        raise AnalysisError("split chemsys disagrees with the partition chemsys")
    folds = s["fold"].astype(int)
    if sorted(folds.unique()) != list(range(cfg.n_folds)):
        raise AnalysisError(f"folds {sorted(folds.unique())} != 0..{cfg.n_folds - 1}")
    form = dict(zip(u["material_id"], u["formula"]))
    man = {"split": str(cfg.split_path.relative_to(resolve("."))), "split_sha256": cfg.split_sha,
           "partition_sha256": cfg.y["population"]["partition"]["sha256"],
           "summary_sha256": cfg.y["population"]["summary_file"]["sha256"],
           "summary_columns_read_for_universe": ["material_id", "formula", "unique_prototype"]}
    man.update(split_key=cfg.split_key, split_role=cfg.role, split_seed=cfg.split_seed)
    return build_part(cfg.part, s["material_id"].to_numpy(), s["chemsys"].to_numpy(), folds.to_numpy(),
                      np.array([form[m] for m in s["material_id"]]), part_of, man, split_type=cfg.split_type)


# ---------------------------------------------------------------------------------------------------------- labels
def synthetic_labels(part: PartData) -> tuple:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    out = {}
    for t in TASKS:  # fixed order, independent of --task
        out[t] = (rng.random(part.n) < SYNTHETIC_BASE_RATE[t]).astype(np.int8)
    man = {"source": "SYNTHETIC", "seed": SYNTHETIC_SEED, "base_rates": dict(SYNTHETIC_BASE_RATE),
           "rule": "for task in (stability, band_gap): y = default_rng(424242).random(N) < rate (one generator, "
                   "canonical material_id order); independent of every feature",
           "label_columns_read": [], "rows_read": 0}
    return out, {}, man


def load_real_labels(cfg: Config, part: PartData, tasks) -> tuple:
    """Part rows are selected from material_id alone; label columns are parsed for those rows only."""
    if cfg.mode == "dryrun":
        raise AnalysisError("real labels are never read in dryrun mode")
    pop = cfg.y["population"]
    path = resolve(pop["summary_file"]["path"])
    sha = check_sha(path, pop["summary_file"]["sha256"], "WBM summary")
    all_ids = pd.read_csv(path, usecols=["material_id"], dtype=str, keep_default_na=False)["material_id"].to_numpy()
    in_part = pd.Index(part.ids).get_indexer(all_ids) >= 0
    keep_lines = set((np.flatnonzero(in_part) + 1).tolist())  # +1: header is line 0
    cols = []
    for t in tasks:
        c = cfg.label_column(t)
        if c not in cols:
            cols.append(c)
    df = pd.read_csv(path, usecols=["material_id"] + cols, dtype=str, keep_default_na=False,
                     skiprows=lambda i: i != 0 and i not in keep_lines)
    assert_in_part(df["material_id"].to_numpy(), cfg.part, part.part_of, "label rows")
    if len(df) != part.n or set(df["material_id"]) != set(part.ids):
        raise AnalysisError("label rows do not match the part")
    df = df.set_index("material_id").loc[part.ids]
    labels, values = {}, {}
    for t in tasks:
        c = cfg.label_column(t)
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(v).all():
            raise AnalysisError(f"{c}: {int((~np.isfinite(v)).sum())} non-finite value(s) in the {cfg.part} part")
        labels[t] = apply_positive_rule(v, cfg.label_rule(t))
        values[t] = v
    man = {"source": "REAL", "file": str(path), "sha256": sha, "label_columns_read": cols,
           "rows_read": int(len(df)), "part": cfg.part,
           "filter": "material_id column read alone; part rows selected; label columns parsed for those rows only"}
    return labels, values, man


class LabelVault:
    """Labels of one task over the part; replicate pipelines see only their draw."""

    def __init__(self, y: np.ndarray, task: str):
        self._y = np.asarray(y).astype(np.int8)
        self.task = task
        self._draws: dict = {}
        self._touched: dict = {}
        self.closed: dict = {}

    def open(self, key, draw_idx) -> "LabelView":
        if key in self._draws or key in self.closed:
            raise BudgetViolation(f"replicate pipeline {key} opened twice")
        d = np.asarray(draw_idx, dtype=np.int64)
        if len(np.unique(d)) != len(d):
            raise BudgetViolation(f"{key}: draw has duplicates")
        self._draws[key] = frozenset(d.tolist())
        self._touched[key] = set()
        return LabelView(self, key)

    def _get(self, key, idx) -> np.ndarray:
        if key not in self._draws:
            raise BudgetViolation(f"{key}: no open draw")
        idx = np.asarray(idx, dtype=np.int64)
        outside = [i for i in idx.tolist() if i not in self._draws[key]]
        if outside:
            raise BudgetViolation(f"{key}: {len(outside)} label(s) requested outside the replicate's draw "
                                  f"(first index {outside[0]})")
        self._touched[key].update(idx.tolist())
        return self._y[idx].copy()

    def close(self, key) -> dict:
        d, t = self._draws.pop(key), self._touched.pop(key)
        if t != set(d):
            raise BudgetViolation(f"{key}: touched {len(t)} label(s) != draw {len(d)} "
                                  f"(missing {len(set(d) - t)}, extra {len(t - set(d))})")
        self.closed[key] = len(d)
        return {"key": list(key) if isinstance(key, tuple) else key, "n_draw": len(d), "n_touched": len(t)}

    def evaluation_labels(self) -> np.ndarray:
        if self._draws:
            raise BudgetViolation(f"evaluation labels requested while pipelines are open: {list(self._draws)}")
        return self._y.copy()


class LabelView:
    def __init__(self, vault: LabelVault, key):
        self._vault, self.key = vault, key

    def __call__(self, idx) -> np.ndarray:
        return self._vault._get(self.key, idx)
