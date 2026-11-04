"""Nested selection inside one replicate's n labels (YAML selection block).

For every learner type of a menu: every grid point is fitted on the inner-training labels and scored on the
inner-validation labels (average precision); the type's configuration is the first grid point with the highest
inner-validation AP (grid order = YAML order). The menu's BEST type is the type with the highest inner-validation AP
(ties -> earlier in the YAML tie_order). Every type's chosen configuration is then refitted on all n labels and scores
the outer test fold. Labels are obtained only through the replicate's LabelView (outside-draw requests raise).
"""
from __future__ import annotations

import time

import numpy as np

from analysis import metrics, models
from analysis.config import AnalysisError


def _check_val(y_va):
    if y_va.min() == y_va.max():
        raise AnalysisError("inner validation labels are single-class; AP undefined")


def select_and_refit(types: list, tie_order: list, grids: dict, Xs: dict, view, tr, va, draw, test, y_cfg: dict,
                     device: str = "cpu", cgcnn=None) -> dict:
    """types: learner types in this menu (subset of tie_order). Xs: type -> feature matrix over the part (or None for
    cgcnn). cgcnn: optional dict(graphs=callable idx -> list, factory=callable params -> CGCNNControl).
    Returns {'best': type, 'scores': {type: test scores}, 'records': [...], 'seconds': {...}}."""
    y_tr, y_va = view(tr), view(va)
    _check_val(y_va)
    records, chosen, inner_ap, secs = [], {}, {}, {}
    for t in types:
        t0 = time.time()
        best_ap, best_cfg, best_extra = -np.inf, None, {}
        for gi, params in enumerate(grids[t]):
            if t == "cgcnn":
                ctl = cgcnn["factory"](params)
                a, ep, hist, s = ctl.fit_early_stopping(cgcnn["graphs"](tr), y_tr, cgcnn["graphs"](va), y_va,
                                                        metrics.ap)
                extra = {"best_epoch": ep, "epochs_run": len(hist), "seconds": round(s, 3)}
            else:
                X = Xs[t]
                mdl = models.make(t, params, y_cfg, device).fit(X[tr], y_tr)
                a = metrics.ap(y_va, mdl.score(X[va]))
                extra = {}
            records.append({"type": t, "grid_index": gi, "params": params, "inner_val_ap": float(a), **extra})
            if a > best_ap:
                best_ap, best_cfg, best_extra = a, params, extra
        chosen[t], inner_ap[t] = (best_cfg, best_extra), best_ap
        secs[t] = time.time() - t0
    best = None
    for t in tie_order:
        if t in types and (best is None or inner_ap[t] > inner_ap[best]):
            best = t
    y_all = view(draw)
    scores = {}
    for t in types:
        t0 = time.time()
        params, extra = chosen[t]
        if t == "cgcnn":
            ctl = cgcnn["factory"](params).refit(cgcnn["graphs"](draw), y_all, extra["best_epoch"])
            scores[t] = ctl.score(cgcnn["graphs"](test))
        else:
            X = Xs[t]
            scores[t] = models.make(t, params, y_cfg, device).fit(X[draw], y_all).score(X[test])
        secs[t] += time.time() - t0
    for rec in records:
        rec["chosen_for_type"] = bool(rec["params"] == chosen[rec["type"]][0] and
                                      rec["inner_val_ap"] == inner_ap[rec["type"]])
        rec["selected_best"] = bool(rec["type"] == best and rec["chosen_for_type"])
    return {"best": best, "scores": scores, "records": records, "seconds": secs,
            "inner_ap": {t: float(v) for t, v in inner_ap.items()},
            "chosen": {t: chosen[t][0] for t in types}}
