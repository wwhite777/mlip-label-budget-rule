"""Pipeline core: inputs, per-(task, n, fold, replicate) units, and the evaluation (metrics, bootstrap, readings).

Method names
  head:<h>:<bb>, best_head:<bb>, residual:<m>:<bb> (stability only), control:<c>, best_control,
  ref:r_sp:<bb>, ref:r_is2re:<bb> and their sensitivity variants ref:<ref>:<bb>:{missing_least,clipped,diag};
  C3 (CONFIRM, primary split, n = primary_n, confirmatory backbones): head_relaxed:<h>:<bb>, best_head_relaxed:<bb>.
Only the primary chemsys split (Config.role == 'primary') produces E1/E2/C4 family members and readings; the
fold-sensitivity and random-reference splits are descriptive ( findings 1, 2).
A unit = one replicate pipeline (task, n, f, r): it opens the replicate's LabelView, fits every menu with nested
selection inside the n labels, scores the outer test fold f, and closes the view (budget accounting).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from analysis import features as F
from analysis import inference as I
from analysis import metrics as Mx
from analysis.config import AnalysisError, Config, aux_entries, verified_aux
from analysis.data import LabelVault, PartData, load_part, load_real_labels, synthetic_labels
from analysis.draws import describe, draw_labels, inner_split, outer_train_test
from analysis.selection import select_and_refit


@dataclass
class Inputs:
    part: PartData
    feats: dict
    control_X: dict
    refs: dict
    labels: dict
    values: dict
    strata: dict
    diag: dict = field(default_factory=dict)
    graphs: object = None
    manifest: dict = field(default_factory=dict)
    feats_relaxed: dict = field(default_factory=dict)   # C3 oracle arm (confirmatory backbones)
    mptrj: object = None                                # C1: bool array, structure's system in full MPtrj


def load_inputs(cfg: Config, log=print) -> Inputs:
    t0 = time.time()
    part = load_part(cfg)
    log(f"[inputs] part {cfg.part}: {part.n} structures, {len(part.systems)} systems ({time.time() - t0:.1f}s)")
    # every auxiliary input is hash-checked up front (and again right before its use)
    aux = {k: {"path": e["path"], "sha256": e["sha256"], "verified": bool(verified_aux(cfg.y, k))}
           for k, e in aux_entries(cfg.y).items()}
    reg = F.load_registry(cfg)
    man = {"part": part.manifest, "features": {}, "references": {}, "registry_sha256":
           cfg.y["backbones"]["registry"]["sha256"], "auxiliary_inputs": aux}
    feats = {}
    for bb in cfg.backbones:
        feats[bb], man["features"][bb] = F.backbone_features(cfg, part, bb, reg)
    els, man["elements"] = F.element_symbols(cfg)
    if len(els) != 89:
        raise AnalysisError(f"{len(els)} elements in the MP file, expected 89 (element_fraction_89)")
    from analysis.data import load_universe
    man["elements"]["n_wbm_elements_in_U"] = F.assert_wbm_elements_included(load_universe(cfg)["formula"], els)
    man["elements"]["symbols"] = els
    comp = F.composition_features(part.formula, els)
    W = F.random_projection_matrix(cfg, comp.shape[1])
    ctl = {}
    for c in cfg.controls:
        spec = cfg.y["controls"][c]
        if c == "cgcnn":
            ctl[c] = None
        elif c == "random_projection_logistic":
            ctl[c] = comp @ W
        elif c == "randinit_encoder_logistic":
            enc = spec["features"]
            if enc != cfg.y["backbones"]["control_encoder"]:
                raise AnalysisError("control encoder mismatch")
            ra = aux_entries(cfg.y)["randinit_features"]
            if (reg[enc]["features_initial"], reg[enc]["features_initial_sha256"]) != (ra["path"], ra["sha256"]):
                raise AnalysisError("auxiliary_inputs.randinit_features differs from the registry row of "
                                    f"{enc}: {ra} vs {reg[enc]['features_initial']}")
            verified_aux(cfg.y, "randinit_features")
            ctl[c], man["features"][enc] = F.backbone_features(cfg, part, enc, reg)
        elif str(spec["features"]).startswith("element_fraction"):
            ctl[c] = comp
        else:
            raise AnalysisError(f"control {c}: unknown features {spec['features']}")
    refs, diag = {}, {}
    if cfg.references and "stability" in cfg.tasks and "stability" in cfg.y["references"]["tasks"]:
        for bb in cfg.backbones:
            r = F.references(cfg, part, bb, reg)
            refs[bb] = {k: v for k, v in r["scores"].items() if k in cfg.references}  # run's reference scope
            if "r_sp" not in refs[bb]:
                raise AnalysisError(f"{bb}: R-sp is outside the run's references {cfg.references}")
            man["references"][bb] = r["info"]
            if not np.isfinite(r["scores"]["r_sp"]["e_hull"]).all():
                raise AnalysisError(f"{bb}: R-sp hull distance missing in the part (residual heads need it)")
            diag[bb], man["references"][bb]["r_sp_diagnostic"] = F.sp_diagnostic(cfg, part, bb,
                                                                                r["scores"]["r_sp"]["e_hull"])
    strata, man["strata"] = F.correction_strata(cfg, part)
    feats_relaxed, mptrj = {}, None
    if cfg.c3:
        man["features_relaxed"] = {}
        for bb in cfg.scope_backbones(cfg.c3_scope):
            feats_relaxed[bb], man["features_relaxed"][bb] = F.relaxed_features(cfg, part, bb, reg)
    if cfg.c1:
        mptrj, man["c1_strata"] = F.mptrj_membership(cfg, part)
    if cfg.synthetic:
        labels, values, man["labels"] = synthetic_labels(part)
        labels = {t: labels[t] for t in cfg.tasks}
    else:
        labels, values, man["labels"] = load_real_labels(cfg, part, cfg.tasks)
    man["seconds_load"] = round(time.time() - t0, 1)
    log(f"[inputs] loaded in {time.time() - t0:.1f}s; labels {man['labels']['source']}")
    return Inputs(part, feats, ctl, refs, labels, values, strata, diag, None, man, feats_relaxed, mptrj)


def model_methods(cfg: Config, task: str, n: int) -> list:
    out = []
    for bb in cfg.backbones:
        out += [f"head:{h}:{bb}" for h in cfg.heads] + [f"best_head:{bb}"]
        if task == "stability":
            out += [f"residual:{m}:{bb}" for m in cfg.residual_models()]
    for bb in cfg.c3_backbones(n, task):
        out += [f"head_relaxed:{h}:{bb}" for h in cfg.heads] + [f"best_head_relaxed:{bb}"]
    ctl = cfg.controls_at(n, task)
    out += [f"control:{c}" for c in ctl] + (["best_control"] if ctl else [])
    return out


def run_id(task, n, f, r) -> str:
    return f"{task}__n{n}__f{f}__r{r}"


class State:
    def __init__(self, cfg: Config, inp: Inputs):
        self.oof, self.p_hat, self.units, self.choices = {}, {}, {}, []
        self.vaults = {t: LabelVault(inp.labels[t], t) for t in cfg.tasks}
        for t in cfg.tasks:
            for n in cfg.budgets:
                self.oof[(t, n)] = {m: np.full((cfg.replicates_max, inp.part.n), np.nan)
                                    for m in model_methods(cfg, t, n)}
        self.gpu_seconds = 0.0


def run_unit(cfg: Config, inp: Inputs, st: State, n: int, f: int, r: int, device: str, units_dir=None, log=print):
    part = inp.part
    draw = draw_labels(cfg, part, f, r, n)
    tr, va = inner_split(cfg, part, draw, f, r)
    _, test = outer_train_test(part, f)
    y = cfg.y
    for task in cfg.tasks:
        t0 = time.time()
        key = (task, n, f, r)
        view = st.vaults[task].open(key, draw)
        st.p_hat[key] = float(np.mean(view(draw)))
        oof = st.oof[(task, n)]
        recs, secs = [], {}
        for bb in cfg.backbones:
            X = inp.feats[bb]
            res = select_and_refit(cfg.heads, list(y["heads"]["tie_order"]), {h: cfg.head_grid(h) for h in cfg.heads},
                                   {h: X for h in cfg.heads}, view, tr, va, draw, test, y, device)
            for h, s in res["scores"].items():
                oof[f"head:{h}:{bb}"][r, test] = s
            oof[f"best_head:{bb}"][r, test] = res["scores"][res["best"]]
            recs += [{"menu": "heads", "backbone": bb, **x} for x in res["records"]]
            secs[f"heads:{bb}"] = res["seconds"]
            if bb in cfg.c3_backbones(n, task):  # C3: same draw, inner split, seeds and menu on DFT-relaxed features
                Xq = inp.feats_relaxed[bb]
                rq = select_and_refit(cfg.heads, list(y["heads"]["tie_order"]),
                                      {h: cfg.head_grid(h) for h in cfg.heads}, {h: Xq for h in cfg.heads}, view, tr,
                                      va, draw, test, y, device)
                for h, s in rq["scores"].items():
                    oof[f"head_relaxed:{h}:{bb}"][r, test] = s
                oof[f"best_head_relaxed:{bb}"][r, test] = rq["scores"][rq["best"]]
                recs += [{"menu": "heads_relaxed", "backbone": bb, **x} for x in rq["records"]]
            if task == "stability" and cfg.residual_models():  # residual heads need R-sp; absent in reference-free scopes
                Xr = np.hstack([X.astype(np.float64), inp.refs[bb]["r_sp"]["e_hull"][:, None]])
                for m in cfg.residual_models():
                    rr = select_and_refit([m], [m], {m: cfg.head_grid(m)}, {m: Xr}, view, tr, va, draw, test, y,
                                          device)
                    oof[f"residual:{m}:{bb}"][r, test] = rr["scores"][m]
                    recs += [{"menu": "residual", "backbone": bb, **x} for x in rr["records"]]
        ctls = cfg.controls_at(n, task)
        cg = None
        if not ctls:  # no control menu in this run's scope (random-reference run)
            pass
        elif "cgcnn" in ctls:
            from analysis.models_cgcnn import CGCNNControl
            seed = cfg.draw_rng[0] * f + r + cfg.draw_rng[1]
            cg = {"graphs": inp.graphs, "factory": lambda p, s=seed: CGCNNControl(p, y["controls"]["cgcnn"], device,
                                                                                 s)}
        if ctls:
            res = select_and_refit(ctls, list(y["controls"]["tie_order"]), {c: cfg.control_grid(c) for c in ctls},
                                   inp.control_X, view, tr, va, draw, test, y, device, cgcnn=cg)
            for c, s in res["scores"].items():
                oof[f"control:{c}"][r, test] = s
            oof["best_control"][r, test] = res["scores"][res["best"]]
            recs += [{"menu": "controls", "backbone": "", **x} for x in res["records"]]
        closed = st.vaults[task].close(key)
        rid = run_id(task, n, f, r)
        info = {"run_id": rid, "task": task, "n": n, "fold": f, "replicate": r, **describe(draw, tr, va, part),
                "p_hat": st.p_hat[key], "n_test": int(len(test)), "budget_check": closed,
                "seconds": round(time.time() - t0, 2)}
        st.units[rid] = info
        for x in recs:
            x["run_id"] = rid
        st.choices += recs
        if units_dir is not None:
            from analysis.outputs import write_unit_csv
            info["file"] = write_unit_csv(units_dir, rid, recs)
        log(f"[unit] {rid} done in {info['seconds']}s; best heads/controls recorded")


# ------------------------------------------------------------------------------------------------------ evaluation
def _pop_masks(cfg: Config, inp: Inputs, task: str) -> dict:
    N = inp.part.n
    pops = {"all": np.ones(N, bool)}
    if task == "stability":
        for bb, rr in _run_refs(cfg, inp):
            for ref, d in rr.items():
                pops[f"finite:{ref}:{bb}"] = np.isfinite(d["e_hull"])
        if "stability" in inp.values:
            rule = cfg.y["sensitivities"]["near_boundary_exclusion"]["rule"]
            m = re.search(r"\|\s*(\w+)\s*\|\s*<\s*([0-9.]+)", rule)
            if not m or m.group(1) != cfg.label_column("stability"):
                raise AnalysisError(f"cannot parse the near-boundary rule {rule!r}")
            pops["near_boundary_excluded"] = ~(np.abs(inp.values["stability"]) < float(m.group(2)))
        for k, v in inp.strata.items():
            pops[f"stratum:{k}"] = v
    if inp.mptrj is not None:
        pops["mptrj:present"] = np.asarray(inp.mptrj, dtype=bool)
        pops["mptrj:absent"] = ~np.asarray(inp.mptrj, dtype=bool)
    return pops


def _run_refs(cfg: Config, inp: Inputs) -> list:
    """References of the run's backbones and reference set only (scope of a fold-sensitivity rerun)."""
    return [(bb, {k: v for k, v in inp.refs[bb].items() if k in cfg.references})
            for bb in cfg.backbones if bb in inp.refs]


def _ref_scores(cfg: Config, inp: Inputs) -> dict:
    """Reference score vectors (score = -e_hull_pred) and their sensitivity variants."""
    out = {}
    lim = cfg.y["references"]["non_physical"]["sensitivity"]
    lo, hi = [float(v) for v in re.search(r"\[(-?[0-9.]+),\s*(-?[0-9.]+)\]", lim).groups()]
    for bb, rr in _run_refs(cfg, inp):
        for ref, d in rr.items():
            s = -d["e_hull"]
            out[f"ref:{ref}:{bb}"] = s
            fin = np.isfinite(s)
            if (~fin).any():
                least = s.copy()
                least[~fin] = np.min(s[fin]) - 1.0
                out[f"ref:{ref}:{bb}:missing_least"] = least
            out[f"ref:{ref}:{bb}:clipped"] = -F.clip_e_hull(d["e_form"], d["e_hull"], lo, hi)
        if inp.diag.get(bb) is not None:
            out[f"ref:r_sp:{bb}:diag"] = -inp.diag[bb]
    return out


def contrast_list(cfg: Config, inp: Inputs, task: str, n: int, pops: dict, ref_scores: dict) -> list:
    """(name, method_a, method_b, population, group, claim): group = the claim if the contrast enters its family,
    else a descriptive / sensitivity tag; claim fixes the margin used for the descriptive p-values."""
    C = []
    primary = n == cfg.primary_n
    fam = set(cfg.family_backbones)

    def grp(claim, bb):
        if not cfg.readings_allowed:  # fold-sensitivity / random-reference splits: never a family member
            return f"descriptive:{cfg.role}:{claim}"
        return claim if (primary and bb in fam) else f"descriptive:{claim}"

    run_refs = dict(_run_refs(cfg, inp))
    if task == "stability":
        for bb in cfg.backbones:
            if not run_refs.get(bb):
                continue
            for ref, tag in (("r_sp", "d_sp"), ("r_is2re", "d_is")):
                if ref not in run_refs[bb]:
                    continue
                pop = f"finite:{ref}:{bb}"
                pop = "all" if pops[pop].all() else pop  # no missing reference value: the intersection is the part
                C.append((f"{tag}:{bb}", f"best_head:{bb}", f"ref:{ref}:{bb}", pop, grp("E1", bb), "E1"))
                if f"ref:{ref}:{bb}:missing_least" in ref_scores:
                    C.append((f"{tag}:{bb}|missing_least", f"best_head:{bb}", f"ref:{ref}:{bb}:missing_least", "all",
                              "sensitivity:missing_least", "E1"))
                C.append((f"{tag}:{bb}|clipped", f"best_head:{bb}", f"ref:{ref}:{bb}:clipped", pop,
                          "sensitivity:non_physical_clip", "E1"))
                for p in pops:
                    if p == "near_boundary_excluded" or p.startswith("stratum:"):
                        C.append((f"{tag}:{bb}|{p}", f"best_head:{bb}", f"ref:{ref}:{bb}",
                                  p if pop == "all" else f"{pop}&{p}",
                                  f"sensitivity:{p}", "E1"))
            if f"ref:r_sp:{bb}:diag" in ref_scores:
                C.append((f"d_sp:{bb}|r_sp_diagnostic", f"best_head:{bb}", f"ref:r_sp:{bb}:diag", "all",
                          "descriptive:r_sp_diagnostic", "E1"))
            for m in cfg.residual_models():
                C.append((f"residual_{m}_vs_r_sp:{bb}", f"residual:{m}:{bb}", f"ref:r_sp:{bb}", "all",
                          "descriptive:residual", "E1"))
                C.append((f"residual_{m}_vs_best_head:{bb}", f"residual:{m}:{bb}", f"best_head:{bb}", "all",
                          "descriptive:residual", "E1"))
        for bb in cfg.backbones:
            if "logistic" in cfg.heads and "mlp_ensemble" in cfg.heads:
                if not cfg.readings_allowed:
                    g = f"descriptive:{cfg.role}:C4"
                elif "C4_up" in cfg.family_sizes:
                    g = grp("C4", bb)
                else:
                    g = "descriptive:C4"
                C.append((f"q:{bb}", f"head:logistic:{bb}", f"head:mlp_ensemble:{bb}", "all", g, "C4"))
                for p in pops:
                    if p == "near_boundary_excluded" or p.startswith("stratum:"):
                        C.append((f"q:{bb}|{p}", f"head:logistic:{bb}", f"head:mlp_ensemble:{bb}", p,
                                  f"sensitivity:{p}", "C4"))
    elif cfg.controls_at(n, task):
        for bb in cfg.backbones:
            C.append((f"g:{bb}", f"best_head:{bb}", "best_control", "all", grp("E2", bb), "E2"))
    return C


def _pop_mask(pops: dict, name: str) -> np.ndarray:
    m = np.ones_like(pops["all"])
    for p in name.split("&"):
        m = m & pops[p]
    return m


def _pop_stats(pops: dict, name: str, y: np.ndarray) -> dict:
    """Structure count, positive count and prevalence of a (possibly intersected) population ( finding 4)."""
    m = _pop_mask(pops, name)
    n_pop, n_pos = int(m.sum()), int(np.asarray(y)[m].sum())
    return {"n_population": n_pop, "n_positive": n_pos, "prevalence": (n_pos / n_pop) if n_pop else float("nan")}


def _auroc_point(y, row, idx) -> float:
    yy = y[idx]
    if yy.min() == yy.max():
        return float("nan")
    return Mx.auroc(yy, row[idx])


def descriptive_spec_keys(cfg: Config, task: str, n: int, pops: dict) -> list:
    """(method, population) specs of the C3 and C1 descriptive analyses (bootstrapped on the same draws)."""
    keys = []
    for bb in cfg.c3_backbones(n, task):
        keys += [(f"best_head_relaxed:{bb}", "all"), (f"best_head:{bb}", "all")]
    if cfg.c1_backbones(n, task) and "mptrj:present" in pops:
        for p in ("mptrj:present", "mptrj:absent"):
            keys += [(f"best_head:{bb}", p) for bb in cfg.c1_backbones(n, task)]
            keys += [(f"control:{c}", p) for c in cfg.controls_at(n, task)]
    return keys


def evaluate(cfg: Config, inp: Inputs, st: State, task: str, n: int, reps: list, log=print) -> dict:
    """Metrics, bootstrap (AP and AUROC on the same draws) and readings for one (task, n) over replicates `reps`.
    Readings only for the primary split at the primary n; descriptive analyses (C3, C1) at the primary n."""
    t0 = time.time()
    part = inp.part
    y = st.vaults[task].evaluation_labels()
    oof = st.oof[(task, n)]
    rows_of = {}
    for m, A in oof.items():
        sub = A[reps]
        if not np.isfinite(sub).all():
            raise AnalysisError(f"{task} n={n} {m}: missing out-of-fold scores for replicates {reps}")
        rows_of[m] = sub
    ref_scores = _ref_scores(cfg, inp) if task == "stability" else {}
    for m, s in ref_scores.items():
        rows_of[m] = s[None, :]
    pops = _pop_masks(cfg, inp, task)
    contrasts = contrast_list(cfg, inp, task, n, pops, ref_scores)
    if not cfg.readings_allowed and any(c[4] in ("E1", "E2", "C4") for c in contrasts):
        raise AnalysisError(f"split role {cfg.role}: a contrast entered a hypothesis family")
    # specs = (method, population) pairs used by any contrast + every method on its natural population + descriptive
    spec_keys = []
    for _, a, b, p, _, _ in contrasts:
        for k in ((a, p), (b, p)):
            if k not in spec_keys:
                spec_keys.append(k)
    for m in rows_of:
        p = natural_pop(m, rows_of[m])
        if (m, p) not in spec_keys:
            spec_keys.append((m, p))
    for k in descriptive_spec_keys(cfg, task, n, pops):
        if k not in spec_keys:
            spec_keys.append(k)
    sorted_specs, point, point_rows, point_auc, check = {}, {}, {}, {}, 0
    ones = np.ones(part.n, dtype=np.int64)
    for m, p in spec_keys:
        idx = np.flatnonzero(_pop_mask(pops, p))
        ss, aps, aus = [], [], []
        for row in rows_of[m]:
            s = Mx.SortedScore(row, y, idx)
            ss.append(s)
            a = Mx.ap(y[idx], row[idx])
            if s.ap(ones) != a:
                raise AnalysisError(f"weighted AP != sklearn AP for {m} on {p}")
            u = _auroc_point(y, row, idx)
            wu = s.auroc(ones)
            if not ((np.isnan(u) and np.isnan(wu)) or abs(u - wu) <= 1e-9):
                raise AnalysisError(f"weighted AUROC {wu} != sklearn AUROC {u} for {m} on {p}")
            check += 1
            aps.append(a)
            aus.append(u)
        sorted_specs[f"{m}@{p}"] = ss
        point[f"{m}@{p}"] = float(np.mean(aps))
        point_rows[f"{m}@{p}"] = aps
        point_auc[f"{m}@{p}"] = float(np.mean(aus))
    log(f"[eval] {task} n={n} R={len(reps)}: {len(sorted_specs)} specs, {len(contrasts)} contrasts; "
        f"weighted-AP == sklearn on {check} rows; bootstrap B={cfg.B} (AP and AUROC, same draws) ...")
    tb = time.time()
    mstar, astar = I.bootstrap(sorted_specs, part.sys_index, len(part.systems), cfg.B, cfg.boot_seed,
                               with_auroc=True)
    t_boot = time.time() - tb
    dstars, dpoint, crow = {}, {}, []
    for name, a, b, p, g, _ in contrasts:
        dstars[name] = mstar[f"{a}@{p}"] - mstar[f"{b}@{p}"]
        dpoint[name] = point[f"{a}@{p}"] - point[f"{b}@{p}"]
    # readings (primary split, primary n only)
    readings = []
    if n == cfg.primary_n and cfg.readings_allowed:
        for claim in ("E1", "E2", "C4"):
            names = [c[0] for c in contrasts if c[4] == claim]
            if names:
                readings += I.read_family(cfg.y, claim, names, dstars, dpoint, cfg.alpha, cfg.margins, cfg.coverage,
                                          cfg.family_sizes)
    margin_of = {"E1": cfg.margins["m_stab"], "E2": cfg.margins["m_gap"], "C4": -cfg.margins["m_probe"]}
    read_by = {r["contrast"]: r for r in readings}
    pop_rows = {}
    for name, a, b, p, g, claim in contrasts:
        mm = margin_of[claim]
        r = read_by.get(name, {})
        lo, hi = I.interval(dstars[name], cfg.coverage)
        (alo, ahi), a_undef = I.interval_finite(astar[f"{a}@{p}"] - astar[f"{b}@{p}"], cfg.coverage)
        ps = _pop_stats(pops, p, y)
        pop_rows.setdefault(p, {"population": p, **ps})
        crow.append({"contrast": name, "a": a, "b": b, "population": p, "group": g, "claim": claim,
                     "n_population": ps["n_population"], "n_positive": ps["n_positive"],
                     "prevalence": ps["prevalence"], "estimate": dpoint[name],
                     "ci_lo": lo, "ci_hi": hi,
                     "margin": mm, "p_up_unadjusted": I.p_up(dstars[name], mm),
                     "p_down_unadjusted": I.p_down(dstars[name], mm),
                     "p_up_holm": r.get("p_up_holm"), "p_down_holm": r.get("p_down_holm"),
                     "reading": r.get("reading", ""),
                     "auroc_estimate": point_auc[f"{a}@{p}"] - point_auc[f"{b}@{p}"], "auroc_ci_lo": alo,
                     "auroc_ci_hi": ahi, "auroc_undefined_draws": a_undef})
    # per-method table on the method's natural population
    mrows = []
    for m in rows_of:
        p = natural_pop(m, rows_of[m])
        key = f"{m}@{p}"
        idx = np.flatnonzero(_pop_mask(pops, p))
        au = [Mx.auroc(y[idx], row[idx]) for row in rows_of[m]]
        (alo, ahi), a_undef = I.interval_finite(astar[key], cfg.coverage)
        mrows.append({"method": m, "population": p, "n_structures": int(len(idx)),
                      "n_positive": int(y[idx].sum()),
                      "prevalence": Mx.prevalence(y[idx]), "M": point[key],
                      "M_r": json.dumps([round(v, 12) for v in point_rows[key]]), "auroc": float(np.mean(au)),
                      "auroc_ci_lo": alo, "auroc_ci_hi": ahi, "auroc_undefined_draws": a_undef,
                      "topk_precision": _topk(cfg, part, st, task, n, reps, rows_of[m], y),
                      "deterministic_reference": bool(m.startswith("ref:"))})
    c3 = _c3_rows(cfg, task, n, pops, y, point, point_auc, mstar, astar)
    c1 = _c1_rows(cfg, task, n, pops, y, point_auc, astar)
    out = {"task": task, "n": n, "replicates": list(reps), "metrics": mrows, "contrasts": crow, "readings": readings,
           "mstar": mstar, "astar": astar, "dstar": dstars, "seconds_bootstrap": round(t_boot, 1),
           "seconds_eval": round(time.time() - t0, 1), "weighted_ap_checks": check,
           "prevalence_part": Mx.prevalence(y), "population_sizes": {k: int(v.sum()) for k, v in pops.items()},
           "populations": [{"task": task, "n": n, **v} for v in pop_rows.values()], "c3": c3, "c1": c1,
           "rows_of": rows_of, "y": y}
    log(f"[eval] {task} n={n} done: bootstrap {t_boot:.1f}s, total {time.time() - t0:.1f}s")
    return out


def _c3_rows(cfg, task, n, pops, y, point, point_auc, mstar, astar) -> list:
    """C3 input-structure contrast: M(best head on DFT-relaxed features) - M(best head on initial features), paired on
    the whole part, same draws; AP and AUROC with 90 % intervals. Oracle arm, descriptive."""
    rows = []
    ps = _pop_stats(pops, "all", y)
    for bb in cfg.c3_backbones(n, task):
        a, b = f"best_head_relaxed:{bb}@all", f"best_head:{bb}@all"
        lo, hi = I.interval(mstar[a] - mstar[b], cfg.coverage)
        (alo, ahi), und = I.interval_finite(astar[a] - astar[b], cfg.coverage)
        rows.append({"analysis": "C3_input_structure", "task": task, "n": n, "backbone": bb,
                     "a": a.split("@")[0], "b": b.split("@")[0], "population": "all", **ps,
                     "M_relaxed": point[a], "M_initial": point[b], "estimate": point[a] - point[b], "ci_lo": lo,
                     "ci_hi": hi, "auroc_relaxed": point_auc[a], "auroc_initial": point_auc[b],
                     "auroc_estimate": point_auc[a] - point_auc[b], "auroc_ci_lo": alo, "auroc_ci_hi": ahi,
                     "auroc_undefined_draws": und, "role": "descriptive oracle arm; excluded from deployable claims"})
    return rows


def _c1_rows(cfg, task, n, pops, y, point_auc, astar) -> list:
    """C1 membership association: per backbone, AUROC gap (MPtrj-present minus absent) of the best head and of each
    control k, interaction I_k = head gap - control-k gap; 90 % intervals on the same draws. Exploratory."""
    if not (cfg.c1_backbones(n, task) and "mptrj:present" in pops):
        return []
    rows = []
    sp, sa = _pop_stats(pops, "mptrj:present", y), _pop_stats(pops, "mptrj:absent", y)

    def gap(m):
        pt = point_auc[f"{m}@mptrj:present"] - point_auc[f"{m}@mptrj:absent"]
        return pt, astar[f"{m}@mptrj:present"] - astar[f"{m}@mptrj:absent"]

    for bb in cfg.c1_backbones(n, task):
        hp, hd = gap(f"best_head:{bb}")
        (hlo, hhi), hu = I.interval_finite(hd, cfg.coverage)
        for c in cfg.controls_at(n, task):
            cp, cd = gap(f"control:{c}")
            (clo, chi), cu = I.interval_finite(cd, cfg.coverage)
            (ilo, ihi), iu = I.interval_finite(hd - cd, cfg.coverage)
            rows.append({"analysis": "C1_membership_association", "task": task, "n": n, "backbone": bb,
                         "control": c, "head_gap": hp, "head_gap_ci_lo": hlo, "head_gap_ci_hi": hhi,
                         "control_gap": cp, "control_gap_ci_lo": clo, "control_gap_ci_hi": chi,
                         "interaction": hp - cp, "interaction_ci_lo": ilo, "interaction_ci_hi": ihi,
                         "undefined_draws": max(hu, cu, iu),
                         "n_present": sp["n_population"], "n_positive_present": sp["n_positive"],
                         "prevalence_present": sp["prevalence"], "n_absent": sa["n_population"],
                         "n_positive_absent": sa["n_positive"], "prevalence_absent": sa["prevalence"],
                         "role": "exploratory association only"})
    return rows


def _topk(cfg, part, st, task, n, reps, rows, y) -> float:
    if cfg.y["metrics"]["top_k"]["k"] != "floor(p_hat_{f,r} * N_f + 0.5)":
        raise AnalysisError("top-k rule changed")
    vals = []
    for i, r in enumerate(reps):
        row = rows[0] if rows.shape[0] == 1 else rows[i]
        row = np.where(np.isfinite(row), row, np.nanmin(row) - 1.0)
        tp = kk = 0
        for f in range(cfg.n_folds):
            test = np.flatnonzero(part.fold == f)
            k = Mx.topk_k(st.p_hat[(task, n, f, r)], len(test))
            tp += Mx.topk_hits(y[test], row[test], k)
            kk += k
        vals.append(tp / kk if kk else float("nan"))
    return float(np.mean(vals))


def natural_pop(m: str, rows: np.ndarray) -> str:
    """'all' if the method scores every structure; a reference with missing values -> its finite set."""
    if np.isfinite(rows).all():
        return "all"
    parts = m.split(":")
    if parts[0] != "ref":
        raise AnalysisError(f"{m}: missing scores")
    return f"finite:{parts[1]}:{parts[2]}"
