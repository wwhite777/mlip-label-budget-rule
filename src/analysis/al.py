"""C5 active-learning replication (YAML descriptive_specs.c5_active_learning; descriptive, never in a family or reading).

Per (task, backbone, outer fold f, seed s), on the backbone's INITIAL features:
  pool      the outer-training structures of fold f in canonical (material_id) order; test structures are never
            acquired, scored for acquisition or used (PoolOracle raises on any non-pool index).
  initial   default_rng(base + a*f + s).choice(len(pool), size=initial, replace=False) (base, a from the YAML); the
            same initial set for every strategy.
  model     heads.mlp_ensemble with its frozen hyperparameters (analysis.models.MLPEnsemble: StandardScaler fitted on
            the labelled set of the round, member m seeded torch.manual_seed(m)); score = mean member probability;
            uncertainty = standard deviation (numpy ddof 0) of the member probabilities.
  rounds    budgets initial, initial+step, ..., final; at each budget the round model scores fold f's test
            structures; between budgets `step` structures are acquired from the unlabelled pool:
              random                 the generator that drew the initial set continues: rng.choice(len(unlabelled),
                                     size=step, replace=False) over the unlabelled pool in canonical order;
              uncertainty            the `step` highest-uncertainty structures (ties: material_id order);
              uncertainty_diversity  shortlist = the mult*step highest-uncertainty structures (same order); greedy
                                     farthest-point: repeatedly the shortlisted structure with the largest minimum
                                     Euclidean distance (scipy cdist, float64) to the labelled set and the batch chosen
                                     so far, in the round model's standardized space (the labelled-set scaler); ties:
                                     material_id order.
  The budget-initial model is identical for every strategy (same set, deterministic fit) and is fitted once.
Metric: at each budget, the pooled out-of-fold AP over the part (each fold's test structures scored by that fold's
round model), per seed, then the mean over seeds. AUC_AL = trapezoid of AP over the budgets with labels in units of the
YAML unit (100). Contrasts d_unc = AP(uncertainty) - AP(random), d_div = AP(uncertainty_diversity) - AP(random) at the
final budget and for AUC_AL; 90 % percentile intervals from the frozen paired chemical-system cluster bootstrap
(inference.bootstrap, B and seed from the YAML; every M* = mean over seeds of the weighted AP on the same draw).
"""
from __future__ import annotations

import json
import time

import numpy as np

from analysis import inference as I
from analysis import metrics as Mx
from analysis.config import AnalysisError, BudgetViolation
from analysis.models import MLPEnsemble


class PoolOracle:
    """Labels of one (task, fold, seed, strategy) AL run: only pool (outer-training) indices, each at most once."""

    def __init__(self, y: np.ndarray, pool: np.ndarray, test: np.ndarray):
        self._y = np.asarray(y).astype(np.int8)
        self._pool = np.zeros(len(self._y), dtype=bool)
        self._pool[np.asarray(pool, dtype=np.int64)] = True
        self._test = np.zeros(len(self._y), dtype=bool)
        self._test[np.asarray(test, dtype=np.int64)] = True
        if (self._pool & self._test).any():
            raise AnalysisError("pool and test overlap")
        self.acquired: list = []
        self._seen = np.zeros(len(self._y), dtype=bool)

    def __call__(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if self._test[idx].any():
            raise BudgetViolation(f"C5: {int(self._test[idx].sum())} test structure(s) requested for acquisition")
        if not self._pool[idx].all():
            raise BudgetViolation("C5: an index outside the outer-training pool was requested")
        if self._seen[idx].any() or len(np.unique(idx)) != len(idx):
            raise BudgetViolation("C5: a structure was acquired twice")
        self._seen[idx] = True
        self.acquired.extend(idx.tolist())
        return self._y[idx].copy()

    @property
    def n_test_acquired(self) -> int:
        return int(self._test[np.asarray(self.acquired, dtype=np.int64)].sum()) if self.acquired else 0


def initial_draw(train: np.ndarray, f: int, seed: int, spec: dict) -> tuple:
    """(initial indices, the generator after the draw)."""
    base, a = spec["init_rng"]
    rng = np.random.default_rng(base + a * f + seed)
    if spec["initial"] > len(train):
        raise AnalysisError(f"C5 initial {spec['initial']} exceeds the {len(train)} pool structures of fold {f}")
    return train[rng.choice(len(train), size=spec["initial"], replace=False)], rng


def top_uncertain(std: np.ndarray, cand: np.ndarray, m: int) -> np.ndarray:
    """The m candidates with the highest std; ties by candidate index (= material_id order). cand ascending."""
    order = np.lexsort((cand, -np.asarray(std, dtype=np.float64)))
    return cand[order[:m]]


def farthest_point(Z_short: np.ndarray, short_idx: np.ndarray, Z_lab: np.ndarray, step: int) -> np.ndarray:
    """Greedy farthest-point batch of `step` shortlisted structures (Euclidean, float64); ties by part index."""
    from scipy.spatial.distance import cdist
    Z_short = np.asarray(Z_short, dtype=np.float64)
    dmin = cdist(Z_short, np.asarray(Z_lab, dtype=np.float64)).min(axis=1)
    avail = np.ones(len(short_idx), dtype=bool)
    chosen = []
    for _ in range(step):
        cand = np.flatnonzero(avail)
        best = dmin[cand].max()
        ties = cand[dmin[cand] == best]
        j = ties[np.argmin(short_idx[ties])]
        chosen.append(j)
        avail[j] = False
        dmin = np.minimum(dmin, cdist(Z_short, Z_short[j:j + 1])[:, 0])
    return short_idx[np.asarray(chosen, dtype=np.int64)]


def run_fold(X: np.ndarray, y: np.ndarray, fold: np.ndarray, f: int, seed: int, spec: dict, mlp_spec: dict,
             device: str = "cpu", strategies=None) -> dict:
    """One (task, backbone, fold, seed): {'scores': {strategy: (n_budgets, n_test)}, 'acquired': {strategy: idx},
    'test': idx, 'n_test_acquired': {strategy: 0}, 'seconds': s}."""
    t0 = time.time()
    strategies = list(strategies or spec["strategies"])
    train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    budgets, step, mult = spec["budgets"], spec["step"], spec["shortlist_mult"]
    out = {"scores": {}, "acquired": {}, "n_test_acquired": {}, "test": test}
    round0 = None
    for strat in strategies:
        init, rng = initial_draw(train, f, seed, spec)
        oracle = PoolOracle(y, train, test)
        lab = np.asarray(init, dtype=np.int64)
        y_lab = oracle(lab)
        S = np.empty((len(budgets), len(test)), dtype=np.float64)
        for k, b in enumerate(budgets):
            if len(lab) != b:
                raise AnalysisError(f"C5 {strat}: {len(lab)} labels at budget {b}")
            if k == 0 and round0 is not None:
                model = round0
            else:
                model = MLPEnsemble(mlp_spec, device).fit(X[lab], y_lab)
                if k == 0:
                    round0 = model
            S[k] = model.score(X[test])
            if k == len(budgets) - 1:
                break
            unl = np.setdiff1d(train, lab, assume_unique=True)  # sorted = canonical order
            if strat == "random":
                new = unl[rng.choice(len(unl), size=step, replace=False)]
            else:
                P = model.member_probs(X[unl])
                std = P.std(axis=0)
                if strat == "uncertainty":
                    new = top_uncertain(std, unl, step)
                elif strat == "uncertainty_diversity":
                    short = top_uncertain(std, unl, mult * step)
                    short = np.sort(short)
                    new = farthest_point(model.sc.transform(X[short]), short, model.sc.transform(X[lab]), step)
                else:
                    raise AnalysisError(f"unknown C5 strategy {strat}")
            y_lab = np.concatenate([y_lab, oracle(new)])
            lab = np.concatenate([lab, new])
        if oracle.n_test_acquired or np.isin(lab, test).any():
            raise BudgetViolation(f"C5 {strat}: a test structure was acquired")
        out["scores"][strat] = S
        out["acquired"][strat] = lab
        out["n_test_acquired"][strat] = oracle.n_test_acquired
    out["seconds"] = round(time.time() - t0, 2)
    return out


def trapezoid(values, budgets, unit: int):
    """Trapezoid over budgets/unit; values (..., n_budgets)."""
    x = np.asarray(budgets, dtype=np.float64) / float(unit)
    v = np.asarray(values, dtype=np.float64)
    return np.sum((x[1:] - x[:-1]) * (v[..., 1:] + v[..., :-1]) / 2.0, axis=-1)


def evaluate(cfg, part, y: np.ndarray, oof: np.ndarray, spec: dict, task: str, bb: str, log=print) -> dict:
    """oof: (n_strategies, n_seeds, n_budgets, N) pooled out-of-fold scores. Returns rows and bootstrap arrays."""
    strategies, budgets, seeds = spec["strategies"], spec["budgets"], spec["seeds"]
    if not np.isfinite(oof).all():
        raise AnalysisError(f"C5 {task} {bb}: missing out-of-fold scores")
    y = np.asarray(y)
    ones = np.ones(part.n, dtype=np.int64)
    specs, ap = {}, np.empty((len(strategies), len(seeds), len(budgets)))
    for i, s in enumerate(strategies):
        for k, b in enumerate(budgets):
            rows = []
            for j in range(len(seeds)):
                ss = Mx.SortedScore(oof[i, j, k], y)
                a = Mx.ap(y, oof[i, j, k])
                if ss.ap(ones) != a:
                    raise AnalysisError(f"C5 weighted AP != sklearn AP ({s}, {b}, seed {seeds[j]})")
                rows.append(ss)
                ap[i, j, k] = a
            specs[f"{s}@{b}"] = rows
    t0 = time.time()
    mstar = I.bootstrap(specs, part.sys_index, len(part.systems), cfg.B, cfg.boot_seed)
    t_boot = time.time() - t0
    M = ap.mean(axis=1)                                            # (strategies, budgets)
    Ms = np.stack([np.stack([mstar[f"{s}@{b}"] for b in budgets], axis=-1) for s in strategies])  # (strat, B, nb)
    auc, auc_star = trapezoid(M, budgets, spec["auc_unit"]), trapezoid(Ms, budgets, spec["auc_unit"])
    n_pos = int(y.sum())
    pop = {"n_population": int(part.n), "n_positive": n_pos, "prevalence": n_pos / part.n}
    curve, aucs, con = [], [], []
    for i, s in enumerate(strategies):
        for k, b in enumerate(budgets):
            lo, hi = I.interval(Ms[i, :, k], cfg.coverage)
            curve.append({"analysis": "C5_active_learning", "task": task, "backbone": bb, "strategy": s, "budget": b,
                          "AP": float(M[i, k]), "ci_lo": lo, "ci_hi": hi,
                          "AP_per_seed": json.dumps([round(float(v), 12) for v in ap[i, :, k]]), **pop})
        lo, hi = I.interval(auc_star[i], cfg.coverage)
        aucs.append({"analysis": "C5_active_learning", "task": task, "backbone": bb, "strategy": s,
                     "AUC_AL": float(auc[i]), "ci_lo": lo, "ci_hi": hi, "budgets": json.dumps(budgets),
                     "label_unit": spec["auc_unit"], **pop})
    ir = strategies.index("random")
    for name, other in (("d_unc", "uncertainty"), ("d_div", "uncertainty_diversity")):
        io = strategies.index(other)
        for at, est, dstar in (("final_budget", M[io, -1] - M[ir, -1], Ms[io, :, -1] - Ms[ir, :, -1]),
                               ("AUC_AL", auc[io] - auc[ir], auc_star[io] - auc_star[ir])):
            lo, hi = I.interval(dstar, cfg.coverage)
            con.append({"analysis": "C5_active_learning", "task": task, "backbone": bb, "contrast": name, "at": at,
                        "budget": budgets[-1] if at == "final_budget" else "", "a": other, "b": "random",
                        "estimate": float(est), "ci_lo": lo, "ci_hi": hi, "n_seeds": len(seeds), **pop,
                        "role": "descriptive; never in any family or reading"})
    log(f"[c5] {task} {bb}: {len(specs)} specs x {len(seeds)} seeds; bootstrap B={cfg.B} {t_boot:.1f}s")
    return {"curve": curve, "auc": aucs, "contrasts": con, "mstar": Ms, "auc_star": auc_star, "ap": ap,
            "seconds_bootstrap": round(t_boot, 1)}
