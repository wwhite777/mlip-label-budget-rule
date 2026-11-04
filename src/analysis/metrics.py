"""Metrics: sklearn average precision (primary), AUROC, prevalence, top-k precision; and the weighted AP over a fixed
sort order used by the bootstrap.

SortedScore.ap(w) reproduces sklearn.metrics.average_precision_score(y, s) evaluated on the data set in which
structure i appears w_i times (w_i a non-negative integer): the scores are sorted once (mergesort, descending, as in
sklearn's _binary_clf_curve); for a weight vector the cumulative positive / total weights are taken at the ends of
the tie groups; groups of total weight 0 are dropped (they are absent from the resampled data); precision and recall
are then formed and integrated with sklearn's own expressions. With integer weights every cumulative sum is an exact
float64 integer, so the result equals explicit resampling bit for bit (test/analysis/test_inference.py).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ap(y, s) -> float:
    return float(average_precision_score(np.asarray(y), np.asarray(s, dtype=np.float64)))


def auroc(y, s) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(s, dtype=np.float64)))


def prevalence(y) -> float:
    return float(np.mean(np.asarray(y)))


class SortedScore:
    """Fixed sort order of one score vector over one population (indices into the part)."""

    def __init__(self, score, y, pop_idx=None):
        score = np.asarray(score, dtype=np.float64)
        y = np.asarray(y)
        pop_idx = np.arange(len(score)) if pop_idx is None else np.asarray(pop_idx, dtype=np.int64)
        s = score[pop_idx]
        if not np.isfinite(s).all():
            raise ValueError("non-finite score inside the population")
        order = np.argsort(s, kind="mergesort")[::-1]
        ss = s[order]
        self.ends = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
        self.y = (y[pop_idx][order] == 1).astype(np.float64)
        self.gidx = pop_idx[order]  # weight gather index (part-level weights)

    def ap(self, w_part) -> float:
        ws = np.asarray(w_part, dtype=np.float64)[self.gidx]
        call = np.cumsum(ws)[self.ends]
        cpos = np.cumsum(ws * self.y)[self.ends]
        keep = np.diff(call, prepend=0.0) > 0
        tps = cpos[keep]
        fps = call[keep] - tps
        ps = tps + fps
        precision = np.zeros_like(tps)
        np.divide(tps, ps, out=precision, where=(ps != 0))
        if tps[-1] == 0:
            recall = np.ones_like(tps)
        else:
            recall = tps / tps[-1]
        precision = np.hstack((precision[::-1], 1))
        recall = np.hstack((recall[::-1], 0))
        return float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))

    def auroc(self, w_part) -> float:
        """AUROC on the data set in which structure i appears w_i times (Mann-Whitney form, ties counted 1/2); equals
        sklearn.metrics.roc_auc_score on the explicitly resampled data up to floating-point rounding. NaN when the
        weighted population lacks a class."""
        ws = np.asarray(w_part, dtype=np.float64)[self.gidx]
        cpos = np.cumsum(ws * self.y)[self.ends]
        cneg = np.cumsum(ws * (1.0 - self.y))[self.ends]
        P, N = cpos[-1], cneg[-1]
        if P == 0 or N == 0:
            return float("nan")
        gpos = np.diff(cpos, prepend=0.0)
        gneg = np.diff(cneg, prepend=0.0)
        return float(np.sum(gpos * (N - cneg + 0.5 * gneg)) / (P * N))


def topk_hits(y_test, s_test, k: int) -> int:
    """True positives among the top k test structures; ties broken by canonical (material_id) order."""
    order = np.lexsort((np.arange(len(s_test)), -np.asarray(s_test, dtype=np.float64)))
    return int(np.asarray(y_test)[order[:k]].sum())


def topk_k(p_hat: float, n_test: int) -> int:
    return int(np.floor(p_hat * n_test + 0.5))
