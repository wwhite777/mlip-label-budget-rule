"""Composition-linear energy re-referencing, using raw element fractions.

The fitted objective is sum((f - e_form_sp - b - X @ delta)**2)
+ alpha * sum(delta**2). The intercept is unpenalized. No scaling is applied.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def fit_rereference(X, f, e_form_pred_sp, alpha: float = 1.0) -> Ridge:
    """Fit one already-selected alpha on training labels only; return a Ridge model.

    X has one row per label and the frozen 89 element-fraction columns in use.
    Smaller synthetic bases are supported for unit tests. With sum(X, axis=1)=1,
    b and a common shift of delta are not separately identifiable at alpha=0;
    positive ridge selects the minimum-norm elemental coefficients.
    """
    X = np.asarray(X, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    e_form_pred_sp = np.asarray(e_form_pred_sp, dtype=np.float64)
    if X.ndim != 2 or f.shape != (len(X),) or e_form_pred_sp.shape != f.shape:
        raise ValueError("X, formation labels and predictions must have aligned rows")
    if not all(np.isfinite(a).all() for a in (X, f, e_form_pred_sp)):
        raise ValueError("non-finite training input")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    # SVD also handles the fraction/intercept degeneracy as alpha approaches 0.
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    return model.fit(X, f - e_form_pred_sp)


def score_rereference(model: Ridge, X, e_hull_pred_sp) -> np.ndarray:
    """Return -(e_hull_pred_sp + b + X @ delta); larger means more stable.

    h = e_form_pred_sp - e_hull_pred_sp is held fixed, so equivalently this
    scores -(corrected formation energy - h), without estimating another hull.
    Keep the fitted intercept when pooling different outer folds.
    """
    X = np.asarray(X, dtype=np.float64)
    e_hull_pred_sp = np.asarray(e_hull_pred_sp, dtype=np.float64)
    if X.ndim != 2 or e_hull_pred_sp.shape != (len(X),):
        raise ValueError("composition and hull predictions must have aligned rows")
    if not np.isfinite(e_hull_pred_sp).all():
        raise ValueError("non-finite hull prediction")
    return -(e_hull_pred_sp + model.predict(X))
