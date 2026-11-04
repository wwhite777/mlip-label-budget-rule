"""Label draws and inner-validation splits (YAML budgets block).

draw(f, r): the outer training structures of fold f in canonical (material_id) order; n of them uniformly without
  replacement: numpy default_rng(a*f + r).choice(len(train), size=n, replace=False) with (a, 0) parsed from the YAML.
inner_split(f, r): the distinct chemical systems of the drawn labels (sorted), permuted by
  default_rng(a*f + r + c).permutation; systems are taken in that order until the cumulative number of drawn labels
  first reaches fraction * n; those labels form the inner validation set, the rest the inner training set.
"""
from __future__ import annotations

import numpy as np

from analysis.config import AnalysisError, Config
from analysis.data import PartData


def outer_train_test(part: PartData, f: int) -> tuple:
    return np.flatnonzero(part.fold != f), np.flatnonzero(part.fold == f)


def draw_labels(cfg: Config, part: PartData, f: int, r: int, n: int) -> np.ndarray:
    train, _ = outer_train_test(part, f)
    if n > len(train):
        raise AnalysisError(f"budget n={n} exceeds the {len(train)} outer training structures of fold {f}")
    a, c = cfg.draw_rng
    rng = np.random.default_rng(a * f + r + c)
    return train[rng.choice(len(train), size=n, replace=False)]


def inner_split(cfg: Config, part: PartData, draw: np.ndarray, f: int, r: int) -> tuple:
    a, c = cfg.inner_rng
    rng = np.random.default_rng(a * f + r + c)
    sys_of = part.sys_index[draw]
    systems = np.unique(sys_of)
    order = systems[rng.permutation(len(systems))]
    counts = np.bincount(sys_of, minlength=len(part.systems))[order]
    target = cfg.inner_frac * len(draw)
    k = int(np.searchsorted(np.cumsum(counts), target, side="left")) + 1
    val_sys = order[:k]
    is_val = np.isin(sys_of, val_sys)
    va, tr = draw[is_val], draw[~is_val]
    if len(va) == 0 or len(tr) == 0:
        raise AnalysisError(f"empty inner split at f={f} r={r}")
    return tr, va


def describe(draw, tr, va, part: PartData) -> dict:
    return {"n": int(len(draw)), "n_inner_train": int(len(tr)), "n_inner_val": int(len(va)),
            "inner_val_fraction": round(len(va) / len(draw), 6),
            "n_systems_inner_val": int(len(np.unique(part.sys_index[va]))),
            "systems_disjoint": bool(not set(part.sys_index[va]) & set(part.sys_index[tr]))}
