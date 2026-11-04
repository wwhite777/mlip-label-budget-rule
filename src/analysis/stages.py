"""Stages of `analysis.run --stage ...` (S5): the CONFIRM work split into independent processes that can be spread over
GPUs 1-3, each with its own run manifest (G5 format), and an assembler that builds the CONFIRM bundle (G4 / G6) and one
combined run manifest (G5) from the stage directories.

Stage            split                      reads                                   writes (in its --out directory)
primary          chemsys_primary            inputs, labels                          units, metrics, contrasts, readings
                                                                                    (per task; C3 / C1 excluded: own
                                                                                    stages); no bundle
fold_sensitivity chemsys_sensitivity_<i>    inputs, labels, primary dir (--pair-with) as the legacy fold-sensitivity run
random_reference random_reference           inputs, labels, primary dir (--pair-with) as the legacy random-reference run
c3               chemsys_primary            relaxed features, labels, primary best-head scores
c1               chemsys_primary            labels, MPtrj strata, primary best-head / control scores
c5               chemsys_primary            initial features, labels (pool only during acquisition)
coverage         --split (YAML scope)       features + composition only (label-free; no label column is read)
coverage_link    chemsys_primary            coverage dirs (--coverage-dir), primary scores, labels
assemble         -                          every stage directory under --out (hash-checked)
Every stage of a confirm run calls check_deferred() and the local specification hash check (analysis.config.verify_freeze) before
anything is loaded (analysis.run.main). The legacy single-process run (no --stage) is unchanged.
Equivalence: per-task primary stages + assemble give the bundle of the legacy both-task run byte for byte; the c3 / c1
stages give the legacy C3 / C1 rows (same draws: the bootstrap depends only on the seed and the part's systems)
(test/analysis/test_s5_runner.py).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis import al as AL
from analysis import coverage as COV
from analysis import descriptive as D
from analysis import features as F
from analysis import inference as I
from analysis import metrics as Mx
from analysis import outputs as O
from analysis.config import AnalysisError, Config, REPO, c5_spec, coverage_spec, sha256_file

STAGES = ("primary", "fold_sensitivity", "random_reference", "c3", "c1", "c5", "coverage", "coverage_link",
          "assemble")
COMPOSITION = "composition"


def look_of(cfg: Config) -> str:
    return "final" if cfg.mode == "confirm" else "first"


def split_for_stage(stage: str, split_arg) -> str:
    if stage not in STAGES:
        raise AnalysisError(f"unknown stage {stage!r}; stages {STAGES}")
    if stage == "fold_sensitivity":
        if not split_arg or not str(split_arg).startswith("chemsys_sensitivity_"):
            raise AnalysisError("--stage fold_sensitivity needs --split chemsys_sensitivity_<i>")
        return split_arg
    if stage == "random_reference":
        if split_arg not in (None, "random_reference"):
            raise AnalysisError("--stage random_reference uses --split random_reference")
        return "random_reference"
    if stage == "coverage":
        return split_arg or "chemsys_primary"
    if split_arg not in (None, "chemsys_primary"):
        raise AnalysisError(f"--stage {stage} runs on the primary chemsys split only (got --split {split_arg})")
    return "chemsys_primary"


# ------------------------------------------------------------------------------------------------ manifests
def new_manifest(cfg: Config, stage: str, argv=None, freeze=None, device=None, threads=None) -> dict:
    return {"stage": stage, "mode": cfg.mode, "part": cfg.part, "tasks": list(cfg.tasks), "argv": argv,
            "freeze": freeze, "device": device, "threads": threads, "prereg_sha256": sha256_file(cfg.prereg_path),
            "prereg_path": str(cfg.prereg_path),
            "split": {"key": cfg.split_key, "role": cfg.role, "type": cfg.split_type, "seed": cfg.split_seed,
                      "path": str(cfg.split_path), "sha256": cfg.split_sha, "readings_allowed": cfg.readings_allowed},
            "code_sha256": {str(p.relative_to(REPO)): sha256_file(p)
                            for p in sorted((REPO / "src" / "analysis").glob("*.py"))},
            "timings": {}, "runs": {}, "covers": [], "_t_start": time.time()}


def finish_manifest(out: Path, man: dict) -> dict:
    t0 = man.pop("_t_start", None)
    if t0 is not None:
        man["timings"]["wall_s"] = round(time.time() - t0, 1)
    try:
        import torch
        if str((man.get("device") or {}).get("device", "")).startswith("cuda"):
            man["device"]["max_memory_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001 - informational only
        pass
    O.write_json(Path(out) / "run_manifest.json", man)
    return man


def _file(out: Path, rel: str, sha: str, rows=None, metrics=None) -> dict:
    d = {"path": rel, "sha256": sha}
    if rows is not None:
        d["rows"] = int(rows)
    if metrics:
        d["metric_columns"] = list(metrics)
    return d


# ------------------------------------------------------------------------------------------------ primary pairing
def primary_covers(man: dict) -> list:
    """Keys 'primary:<task>:n<n>' a primary run directory covers (stage manifests record them; a legacy primary run
    covers every task and budget it evaluated)."""
    if man.get("covers"):
        return [c for c in man["covers"] if c.startswith("primary:")]
    budgets = sorted({int(k.split("__")[2][1:]) for k in man.get("runs", {}) if k.startswith("eval__")})
    return [f"primary:{t}:n{n}" for t in man.get("tasks") or [] for n in budgets]


def primary_dirs(cfg: Config, pair_dirs, tasks, look: str, n: int | None = None) -> dict:
    """task -> (dir, manifest) of the primary chemsys seed-0 run(s) among --pair-with (same mode / part / YAML) that
    evaluated budget n (default: the primary n)."""
    n = cfg.primary_n if n is None else n
    out = {}
    for p in pair_dirs or []:
        p = Path(p)
        man = D.primary_run(cfg, p, look)
        cov = primary_covers(man)
        for t in tasks:
            if f"primary:{t}:n{n}" in cov:
                if t in out:
                    raise AnalysisError(f"two primary directories cover task {t} at n = {n}: {out[t][0]} and {p}")
                out[t] = (p, man)
    missing = [t for t in tasks if t not in out]
    if missing:
        raise AnalysisError(f"no primary run directory (--pair-with) covers task(s) {missing} at n = {n}")
    return out


def primary_scores(cfg: Config, part, pdir: Path, man: dict, look: str, task: str, n: int, methods: list,
                   reps: list) -> dict:
    """method -> (len(reps), N) out-of-fold scores read from the primary run's verified scores file."""
    z = np.load(D._verified(pdir, man, f"{look}/scores_{task}_n{n}.npz"), allow_pickle=False)
    if [int(r) for r in z["replicate"]] != list(reps):
        raise AnalysisError(f"primary replicates {z['replicate'].tolist()} != {list(reps)}")
    if not np.array_equal(z["material_id"].astype(str), part.ids):
        raise AnalysisError("primary scores are not aligned to this part (material_id order)")
    have = [str(m) for m in z["method"]]
    out = {}
    for m in methods:
        if m not in have:
            raise AnalysisError(f"primary scores {task} n={n}: method {m} absent")
        out[m] = np.asarray(z["score"][have.index(m)], dtype=np.float64)
    return out


# ------------------------------------------------------------------------------------------------ inputs
@dataclass
class StageInputs:
    part: object
    labels: dict = field(default_factory=dict)        # task -> int8 (N,)
    feats: dict = field(default_factory=dict)         # representation -> (N, d)
    feats_relaxed: dict = field(default_factory=dict)
    mptrj: object = None
    manifest: dict = field(default_factory=dict)


def load_stage_inputs(cfg: Config, stage: str, backbones: list, log=print) -> StageInputs:
    """Only what the stage needs. coverage never opens a label column."""
    from analysis.data import load_part, load_real_labels, synthetic_labels
    t0 = time.time()
    part = load_part(cfg)
    man = {"part": part.manifest}
    si = StageInputs(part=part, manifest=man)
    if stage in ("c3", "c5", "coverage"):
        reg = F.load_registry(cfg)
        man["features"] = {}
        for bb in backbones:
            si.feats[bb], man["features"][bb] = F.backbone_features(cfg, part, bb, reg)
        if stage == "coverage":
            els, man["elements"] = F.element_symbols(cfg)
            si.feats[COMPOSITION] = F.composition_features(part.formula, els)
        if stage == "c3":
            man["features_relaxed"] = {}
            for bb in backbones:
                si.feats_relaxed[bb], man["features_relaxed"][bb] = F.relaxed_features(cfg, part, bb, reg)
    if stage == "c1":
        si.mptrj, man["c1_strata"] = F.mptrj_membership(cfg, part)
    if stage == "coverage":
        man["labels"] = {"source": "NONE", "label_columns_read": [], "rows_read": 0,
                         "note": "coverage is label-free (descriptive_specs.coverage_diagnostics.labels)"}
    elif cfg.synthetic:
        lab, _, man["labels"] = synthetic_labels(part)
        si.labels = {t: lab[t] for t in cfg.tasks}
    else:
        si.labels, _, man["labels"] = load_real_labels(cfg, part, cfg.tasks)
    man["seconds_load"] = round(time.time() - t0, 1)
    log(f"[inputs:{stage}] {part.n} structures; labels {man['labels']['source']} ({man['seconds_load']}s)")
    return si


# ------------------------------------------------------------------------------------------------ C3
def run_c3(cfg: Config, si: StageInputs, out, device: str, man: dict, pair_dirs, log=print) -> dict:
    """C3 oracle arm in its own process: relaxed-feature heads with the primary run's draws, inner splits, seeds and
    menu; paired with the primary run's best-head (initial) scores; rows as pipeline._c3_rows."""
    from analysis.data import LabelVault
    from analysis.draws import draw_labels, inner_split, outer_train_test
    from analysis.pipeline import _c3_rows, run_id
    from analysis.selection import select_and_refit
    if not cfg.c3 or not cfg.c3_scope:
        raise AnalysisError("C3 runs in a CONFIRM primary-split configuration only (descriptive_specs.c3_input_structure)")
    out = Path(out)
    look, n = look_of(cfg), cfg.c3_scope["n"]
    tasks = [t for t in cfg.tasks if cfg.c3_backbones(n, t)]
    pdirs = primary_dirs(cfg, pair_dirs, tasks, look, n)
    part, reps, y_cfg = si.part, list(cfg.replicates_first), cfg.y
    heads, tie = list(cfg.heads), list(y_cfg["heads"]["tie_order"])
    vaults = {t: LabelVault(si.labels[t], t) for t in tasks}
    oof = {(t, bb): {h: np.full((len(reps), part.n), np.nan) for h in heads + ["best"]}
           for t in tasks for bb in cfg.c3_backbones(n, t)}
    recs, units = [], {}
    t0 = time.time()
    for ri, r in enumerate(reps):
        for f in range(cfg.n_folds):
            draw = draw_labels(cfg, part, f, r, n)
            tr, va = inner_split(cfg, part, draw, f, r)
            _, test = outer_train_test(part, f)
            for t in tasks:
                tu = time.time()
                key = (t, n, f, r)
                view = vaults[t].open(key, draw)
                for bb in cfg.c3_backbones(n, t):
                    Xq = si.feats_relaxed[bb]
                    rq = select_and_refit(heads, tie, {h: cfg.head_grid(h) for h in heads}, {h: Xq for h in heads},
                                          view, tr, va, draw, test, y_cfg, device)
                    for h, s in rq["scores"].items():
                        oof[(t, bb)][h][ri, test] = s
                    oof[(t, bb)]["best"][ri, test] = rq["scores"][rq["best"]]
                    recs += [{"run_id": run_id(t, n, f, r), "menu": "heads_relaxed", "backbone": bb, **x}
                             for x in rq["records"]]
                units[run_id(t, n, f, r)] = {"budget_check": vaults[t].close(key), "seconds": round(time.time() - tu, 2)}
                log(f"[c3] {run_id(t, n, f, r)} done in {units[run_id(t, n, f, r)]['seconds']}s")
    man["timings"]["units_s"] = round(time.time() - t0, 1)
    for t in tasks:
        y = vaults[t].evaluation_labels()
        bbs = cfg.c3_backbones(n, t)
        init = primary_scores(cfg, part, *pdirs[t], look, t, n, [f"best_head:{bb}" for bb in bbs], reps)
        rows_of = {}
        for bb in bbs:
            rows_of[f"best_head_relaxed:{bb}@all"] = oof[(t, bb)]["best"]
            rows_of[f"best_head:{bb}@all"] = init[f"best_head:{bb}"]
        point, point_auc, specs = _points(rows_of, y, {"all": np.arange(part.n)})
        tb = time.time()
        mstar, astar = I.bootstrap(specs, part.sys_index, len(part.systems), cfg.B, cfg.boot_seed, with_auroc=True)
        man["timings"][f"bootstrap_{t}_s"] = round(time.time() - tb, 1)
        rows = _c3_rows(cfg, t, n, {"all": np.ones(part.n, bool)}, y, point, point_auc, mstar, astar)
        tag = f"{t}_n{n}"
        files = [_file(out, f"{look}/c3_input_structure_{tag}.csv",
                       O.write_csv(out / look / f"c3_input_structure_{tag}.csv", rows), len(rows),
                       ["estimate", "ci_lo", "ci_hi"])]
        methods = [f"head_relaxed:{h}:{bb}" for bb in bbs for h in heads] + [f"best_head_relaxed:{bb}" for bb in bbs]
        S = np.stack([oof[(t, m.split(":")[-1])][m.split(":")[1] if m.startswith("head_relaxed") else "best"]
                      for m in methods])
        files.append(_file(out, f"{look}/scores_relaxed_{tag}.npz",
                           O.save_npz(out / look / f"scores_relaxed_{tag}.npz", compress=True, material_id=part.ids,
                                      fold=part.fold, replicate=np.asarray(reps), method=np.asarray(methods),
                                      score=S)))
        man["runs"][f"c3__{tag}"] = {"files": files}
        man["covers"].append(f"c3:{t}")
    ch = [{k: (json.dumps(v, sort_keys=True) if k == "params" else v) for k, v in x.items()} for x in recs]
    man["runs"]["c3__choices"] = {"files": [_file(out, f"{look}/selection_choices_relaxed.csv",
                                                  O.write_csv(out / look / "selection_choices_relaxed.csv", ch),
                                                  len(ch), ["inner_val_ap"])]}
    man["units"] = units
    man["pair_with"] = {t: str(pdirs[t][0]) for t in tasks}
    return man


def _points(rows_of: dict, y: np.ndarray, idx_of: dict) -> tuple:
    """spec 'method@pop' -> replicate-mean AP / AUROC on the population, SortedScore rows (weighted AP == sklearn)."""
    from analysis.pipeline import _auroc_point
    ones = np.ones(len(y), dtype=np.int64)
    point, point_auc, specs = {}, {}, {}
    for key, rows in rows_of.items():
        idx = idx_of[key.split("@", 1)[1]]
        ss, aps, aus = [], [], []
        for row in rows:
            s = Mx.SortedScore(row, y, idx)
            a = Mx.ap(y[idx], row[idx])
            if s.ap(ones) != a:
                raise AnalysisError(f"weighted AP != sklearn AP for {key}")
            ss.append(s)
            aps.append(a)
            aus.append(_auroc_point(y, row, idx))
        specs[key], point[key], point_auc[key] = ss, float(np.mean(aps)), float(np.mean(aus))
    return point, point_auc, specs


# ------------------------------------------------------------------------------------------------ C1
def run_c1(cfg: Config, si: StageInputs, out, man: dict, pair_dirs, log=print) -> dict:
    """C1 membership association from the primary run's scores (rows as pipeline._c1_rows)."""
    from analysis.pipeline import _c1_rows
    if not cfg.c1 or not cfg.c1_scope:
        raise AnalysisError("C1 runs in a CONFIRM primary-split configuration only")
    out = Path(out)
    look, n = look_of(cfg), cfg.c1_scope["n"]
    tasks = [t for t in cfg.tasks if cfg.c1_backbones(n, t)]
    pdirs = primary_dirs(cfg, pair_dirs, tasks, look, n)
    part, reps = si.part, list(cfg.replicates_first)
    present = np.asarray(si.mptrj, dtype=bool)
    pops = {"all": np.ones(part.n, bool), "mptrj:present": present, "mptrj:absent": ~present}
    idx_of = {p: np.flatnonzero(v) for p, v in pops.items()}
    for t in tasks:
        y = np.asarray(si.labels[t])
        methods = [f"best_head:{bb}" for bb in cfg.c1_backbones(n, t)] + [f"control:{c}" for c in cfg.controls_at(n, t)]
        sc = primary_scores(cfg, part, *pdirs[t], look, t, n, methods, reps)
        rows_of = {f"{m}@{p}": sc[m] for p in ("mptrj:present", "mptrj:absent") for m in methods}
        _, point_auc, specs = _points(rows_of, y, idx_of)
        tb = time.time()
        _, astar = I.bootstrap(specs, part.sys_index, len(part.systems), cfg.B, cfg.boot_seed, with_auroc=True)
        man["timings"][f"bootstrap_{t}_s"] = round(time.time() - tb, 1)
        rows = _c1_rows(cfg, t, n, pops, y, point_auc, astar)
        tag = f"{t}_n{n}"
        man["runs"][f"c1__{tag}"] = {"files": [_file(out, f"{look}/c1_membership_{tag}.csv",
                                                     O.write_csv(out / look / f"c1_membership_{tag}.csv", rows),
                                                     len(rows), ["interaction", "interaction_ci_lo",
                                                                 "interaction_ci_hi"])]}
        man["covers"].append(f"c1:{t}")
        log(f"[c1] {t}: {len(rows)} rows")
    man["pair_with"] = {t: str(pdirs[t][0]) for t in tasks}
    return man


# ------------------------------------------------------------------------------------------------ C5
def c5_backbones(cfg: Config, only=None) -> list:
    bbs = list(cfg.family_backbones)  # c5 scope 'confirmatory' (the pilot backbones in the dryrun configuration)
    if only:
        bad = [b for b in only if b not in bbs]
        if bad:
            raise AnalysisError(f"--backbone {bad} outside the C5 scope {bbs}")
        bbs = [b for b in bbs if b in only]
    return bbs


def run_c5(cfg: Config, si: StageInputs, out, device: str, man: dict, backbones: list, log=print) -> dict:
    spec = c5_spec(cfg.y)
    out = Path(out)
    look = look_of(cfg)
    part = si.part
    mlp = dict(cfg.y["heads"]["mlp_ensemble"])
    tasks = [t for t in cfg.tasks if t in spec["tasks"]]
    S_, R_, K_ = len(spec["strategies"]), len(spec["seeds"]), len(spec["budgets"])
    man["c5_spec"] = {k: v for k, v in spec.items() if k != "scope"}
    for t in tasks:
        y = np.asarray(si.labels[t])
        for bb in backbones:
            t0 = time.time()
            X = si.feats[bb]
            oof = np.full((S_, R_, K_, part.n), np.nan)
            acq_keys, acq = [], []
            for f in range(cfg.n_folds):
                for j, seed in enumerate(spec["seeds"]):
                    res = AL.run_fold(X, y, part.fold, f, seed, spec, mlp, device)
                    for i, s in enumerate(spec["strategies"]):
                        for k in range(K_):
                            oof[i, j, k, res["test"]] = res["scores"][s][k]
                        acq_keys.append(f"{f}:{seed}:{s}")
                        acq.append(part.ids[res["acquired"][s]])
                        if res["n_test_acquired"][s] != 0:
                            raise AnalysisError("C5: a test structure was acquired")
                    log(f"[c5] {t} {bb} fold {f} seed {seed}: {res['seconds']}s")
            t_al = time.time() - t0
            ev = AL.evaluate(cfg, part, y, oof, spec, t, bb, log)
            tag = f"{t}__{bb}"
            d = out / look
            files = [
                _file(out, f"{look}/c5_curve_{tag}.csv", O.write_csv(d / f"c5_curve_{tag}.csv", ev["curve"]),
                      len(ev["curve"]), ["AP", "ci_lo", "ci_hi", "prevalence"]),
                _file(out, f"{look}/c5_auc_{tag}.csv", O.write_csv(d / f"c5_auc_{tag}.csv", ev["auc"]),
                      len(ev["auc"]), ["AUC_AL", "ci_lo", "ci_hi"]),
                _file(out, f"{look}/c5_contrasts_{tag}.csv", O.write_csv(d / f"c5_contrasts_{tag}.csv",
                                                                         ev["contrasts"]),
                      len(ev["contrasts"]), ["estimate", "ci_lo", "ci_hi"]),
                _file(out, f"{look}/c5_scores_{tag}.npz",
                      O.save_npz(d / f"c5_scores_{tag}.npz", compress=True, material_id=part.ids, fold=part.fold,
                                 strategy=np.asarray(spec["strategies"]), seed=np.asarray(spec["seeds"]),
                                 budget=np.asarray(spec["budgets"]), score=oof)),
                _file(out, f"{look}/c5_acquired_{tag}.npz",
                      O.save_npz(d / f"c5_acquired_{tag}.npz", compress=True, key=np.asarray(acq_keys),
                                 material_id=np.stack(acq))),
                _file(out, f"{look}/c5_bootstrap_{tag}.npz",
                      O.save_npz(d / f"c5_bootstrap_{tag}.npz", compress=True, strategy=np.asarray(spec["strategies"]),
                                 budget=np.asarray(spec["budgets"]), mstar=ev["mstar"], auc_star=ev["auc_star"]))]
            man["runs"][f"c5__{tag}"] = {"files": files}
            man["timings"][f"c5__{tag}"] = {"al_s": round(t_al, 1), "bootstrap_s": ev["seconds_bootstrap"]}
            man["covers"].append(f"c5:{t}:{bb}")
            man.setdefault("c5_checks", {})[tag] = {"n_test_acquired": 0, "n_runs": len(acq_keys),
                                                    "acquired_per_run": int(spec["final"])}
    return man


# ------------------------------------------------------------------------------------------------ coverage
def coverage_reprs(cfg: Config, only=None) -> list:
    reps = list(cfg.family_backbones) + [COMPOSITION]
    if only:
        bad = [r for r in only if r not in reps]
        if bad:
            raise AnalysisError(f"--repr {bad} outside the coverage representations {reps}")
        reps = [r for r in reps if r in only]
    return reps


def run_coverage(cfg: Config, si: StageInputs, out, device: str, man: dict, reprs: list, log=print) -> dict:
    spec = coverage_spec(cfg.y)
    if cfg.split_key not in spec["splits"]:
        raise AnalysisError(f"coverage split {cfg.split_key} outside the YAML scope {spec['splits']}")
    out = Path(out)
    look = look_of(cfg)
    part = si.part
    man["coverage_spec"] = spec
    man["crosscheck"] = {}
    for rep in reprs:
        X = si.feats[rep]
        folds = []
        t0 = time.time()
        for f in range(cfg.n_folds):
            fr = COV.knn_fold(X, part.fold, f, spec, device)
            fr["fold"] = f
            fr["umap"] = {}
            for rs in spec["random_states"]:
                fr["umap"][rs] = COV.umap_kde(fr["Z_ref"], fr["Z_test"], spec, rs)
            del fr["Z_ref"], fr["Z_test"]
            man["crosscheck"][f"{cfg.split_key}:{rep}:{f}"] = fr["crosscheck"]
            man["timings"][f"coverage__{cfg.split_key}__{rep}__f{f}"] = {
                "knn_s": fr["seconds_knn"], **{f"umap_kde_rs{rs}_s": u["seconds"] for rs, u in fr["umap"].items()}}
            log(f"[coverage] {cfg.split_key} {rep} fold {f}: OOC {int(fr['ooc'].sum())}/{len(fr['test'])}, knn "
                f"{fr['seconds_knn']}s, umap+kde " + ", ".join(f"rs{rs} {u['seconds']}s" for rs, u in fr["umap"].items()))
            folds.append(fr)
        rows = COV.representation_rows(cfg.split_key, rep, folds, part.n)
        arr = COV.per_structure_arrays(folds, part.n, spec["k"], spec["random_states"])
        tag = f"{cfg.split_key}__{rep}"
        d = out / look
        man["runs"][f"coverage__{tag}"] = {"files": [
            _file(out, f"{look}/coverage_{tag}.csv", O.write_csv(d / f"coverage_{tag}.csv", rows), len(rows),
                  ["n_test", "n_out_of_coverage", "frac_out_of_coverage"] +
                  [f"frac_low_density_rs{rs}" for rs in spec["random_states"]]),
            _file(out, f"{look}/coverage_{tag}.npz",
                  O.save_npz(d / f"coverage_{tag}.npz", compress=True, material_id=part.ids, **arr))]}
        man["timings"][f"coverage__{tag}_s"] = round(time.time() - t0, 1)
        man["covers"].append(f"coverage:{cfg.split_key}:{rep}")
    return man


def run_coverage_link(cfg: Config, si: StageInputs, out, man: dict, pair_dirs, coverage_dirs, log=print) -> dict:
    spec = coverage_spec(cfg.y)
    out = Path(out)
    look, n = look_of(cfg), spec["link_n"]
    part, reps = si.part, list(cfg.replicates_first)
    if n not in cfg.budgets:
        raise AnalysisError(f"performance_link n = {n} is not a budget of this configuration {cfg.budgets}")
    pdirs = primary_dirs(cfg, pair_dirs, cfg.tasks, look, n)
    want = coverage_reprs(cfg)
    ooc, src = {}, {}
    for cd in coverage_dirs or []:
        cd = Path(cd)
        cm = json.loads((cd / "run_manifest.json").read_text(encoding="utf-8"))
        if cm.get("stage") != "coverage" or cm.get("mode") != cfg.mode or cm.get("part") != cfg.part \
                or cm.get("prereg_sha256") != sha256_file(cfg.prereg_path):
            raise AnalysisError(f"--coverage-dir {cd}: not a coverage stage of this mode / part / YAML")
        if (cm.get("split") or {}).get("key") != "chemsys_primary":
            continue
        for rep in want:
            rel = f"{look}/coverage_chemsys_primary__{rep}.npz"
            if any(f["path"] == rel for r in cm["runs"].values() for f in r["files"]):
                if rep in ooc:
                    raise AnalysisError(f"representation {rep} in two coverage directories")
                z = np.load(D._verified(cd, cm, rel), allow_pickle=False)
                if not np.array_equal(z["material_id"].astype(str), part.ids):
                    raise AnalysisError(f"coverage {rep}: material_id order differs from the part")
                ooc[rep], src[rep] = np.asarray(z["out_of_coverage"], dtype=bool), str(cd)
    miss = [r for r in want if r not in ooc]
    if miss:
        raise AnalysisError(f"performance_link: chemsys_primary coverage missing for {miss}")
    ooc = {r: ooc[r] for r in want}
    for t in cfg.tasks:
        methods = [f"best_head:{bb}" for bb in cfg.family_backbones] + (["best_control"] if t == "band_gap" else [])
        own = {f"best_head:{bb}": bb for bb in cfg.family_backbones}
        own["best_control"] = COMPOSITION
        sc = primary_scores(cfg, part, *pdirs[t], look, t, n, methods, reps)
        tb = time.time()
        rows = COV.performance_link(cfg, part, si.labels[t], t, n, reps, sc, ooc, own, log)
        man["timings"][f"coverage_link_{t}_s"] = round(time.time() - tb, 1)
        tag = f"{t}_n{n}"
        man["runs"][f"coverage_link__{tag}"] = {"files": [
            _file(out, f"{look}/coverage_link_{tag}.csv", O.write_csv(out / look / f"coverage_link_{tag}.csv", rows),
                  len(rows), ["n_covered", "n_out_of_coverage"])]}
        man["covers"].append(f"coverage_link:{t}")
    man["pair_with"] = {t: str(pdirs[t][0]) for t in cfg.tasks}
    man["coverage_dirs"] = src
    return man


# ------------------------------------------------------------------------------------------------ assemble
def required_covers(cfg: Config) -> list:
    """Stage coverage keys a complete run of this mode needs (CONFIRM: every stage; dryrun / pilot configuration:
    primary, c5, coverage on chemsys_primary, coverage_link - the other analyses exist for CONFIRM only)."""
    tasks = list(cfg.tasks)
    req = [f"primary:{t}:n{n}" for t in tasks for n in cfg.budgets]
    c5 = c5_spec(cfg.y)
    req += [f"c5:{t}:{bb}" for t in tasks if t in c5["tasks"] for bb in c5_backbones(cfg)]
    cs = coverage_spec(cfg.y)
    splits = cs["splits"] if cfg.mode == "confirm" else ["chemsys_primary"]
    req += [f"coverage:{s}:{r}" for s in splits for r in coverage_reprs(cfg)]
    req += [f"coverage_link:{t}" for t in tasks]
    if cfg.mode == "confirm":
        fs = (cfg.y["descriptive_specs"]["fold_sensitivity"]["scope"]["tasks"])
        nsens = len(cfg.y["splits"]["confirm"]["chemsys_sensitivity"])
        req += [f"fold_sensitivity:chemsys_sensitivity_{i}:{t}" for i in range(nsens) for t in tasks if t in fs]
        req += [f"random_reference:{t}" for t in tasks if t in (cfg.rvc_scope or {}).get("tasks", [])]
        if cfg.c3_scope:
            req += [f"c3:{t}" for t in tasks if t in cfg.c3_scope["tasks"]]
        if cfg.c1_scope:
            req += [f"c1:{t}" for t in tasks if t in cfg.c1_scope["tasks"]]
    return req


def assemble(cfg: Config, out, man: dict, log=print, need=None) -> dict:
    """Builds <out>/<look>/{bundle.json, draws.npz, readings.json} from the per-task primary stages and the combined
    G5 run manifest <out>/run_manifest.json over every stage directory (immediate subdirectories of <out>).
    need: the stage keys that must be present (default required_covers(cfg); fixtures pass a subset)."""
    out = Path(out)
    look = look_of(cfg)
    ysha = sha256_file(cfg.prereg_path)
    stages, covered = {}, {}
    for sub in sorted(p for p in out.iterdir() if p.is_dir() and (p / "run_manifest.json").is_file()):
        sm = json.loads((sub / "run_manifest.json").read_text(encoding="utf-8"))
        if "stage" not in sm or sm["stage"] == "assemble":
            raise AnalysisError(f"{sub}: not a stage directory (no stage / an assembly)")
        probs = []
        if sm.get("mode") != cfg.mode or sm.get("part") != cfg.part:
            probs.append(f"mode/part {sm.get('mode')}/{sm.get('part')}")
        if sm.get("prereg_sha256") != ysha:
            probs.append(f"YAML sha256 {str(sm.get('prereg_sha256'))[:12]} != {ysha[:12]}")
        if cfg.mode == "confirm" and (sm.get("freeze") or {}).get("status") != "PASS":
            probs.append("freeze verify not PASS")
        for rid, r in (sm.get("runs") or {}).items():
            for f in r.get("files", []):
                p = sub / f["path"]
                if not p.is_file() or sha256_file(p) != f["sha256"]:
                    probs.append(f"{f['path']} missing or sha256 differs")
        if sm.get("missing_run_ids"):
            probs.append(f"missing run ids {sm['missing_run_ids'][:3]}")
        if probs:
            raise AnalysisError(f"stage directory {sub.name}: " + "; ".join(probs))
        stages[sub.name] = sm
        for c in sm.get("covers") or []:
            if c in covered:
                raise AnalysisError(f"{c} covered twice ({covered[c]}, {sub.name})")
            covered[c] = sub.name
    need = required_covers(cfg) if need is None else list(need)
    missing = [c for c in need if c not in covered]
    if missing:
        raise AnalysisError(f"incomplete run: {len(missing)} stage key(s) missing, e.g. {missing[:6]}")
    # ---- bundle from the per-task primary stages (legacy order: tasks in cfg.tasks order)
    evs, readings, all_rows, reps0 = [], {}, [], None
    for t in cfg.tasks:
        sub = out / covered[f"primary:{t}:n{cfg.primary_n}"]
        sm = stages[sub.name]
        rd = json.loads(D._verified(sub, sm, f"{look}/readings.json").read_text(encoding="utf-8"))
        rows = [r for r in rd["readings"] if r["claim"] in _claims_of_task(t)]
        z = np.load(D._verified(sub, sm, f"{look}/bootstrap_{t}_n{cfg.primary_n}.npz"), allow_pickle=False)
        names = [str(c) for c in z["contrast"]]
        dstar = {c: np.asarray(z["dstar"][names.index(c)], dtype=np.float64) for c in names}
        if reps0 is not None and rd["replicates"] != reps0:
            raise AnalysisError("primary stages used different replicates")
        reps0 = rd["replicates"]
        evs.append({"readings": rows, "dstar": dstar, "replicates": rd["replicates"], "n": cfg.primary_n})
        all_rows += rows
        for r in rows:
            readings[r["contrast"]] = r["reading"]
    fam = cfg.family_backbones
    d = out / look
    if cfg.mode in ("dryrun", "pilot"):
        br = I.pilot_branch(readings, fam)
        decision = {"rule": "pilot_branch", "reading": br["reading_first_look"], "parts": {
            "E1": br["E1_part_holds"], "E2": br["E2_part_holds"]},
            "survivor": None if br["extension_triggered"] else br["survivor"]}
        detail = dict(br, note="staged runs do not run the pilot extension; NEEDS_EXTENSION -> legacy runner")
        stage_name = "pilot"
    else:
        sr = I.success_rule(readings, fam)
        decision = {"rule": "success_rule", "reading": "MET" if sr["success"] else "NOT_MET",
                    "parts": {"E1": sr["E1_all_d_sp_resolved"], "E2": sr["E2_all_benefit"]}, "survivor": None}
        detail = sr
        stage_name = "confirm"
    rsha = O.write_json(d / "readings.json", {"look": look, "replicates": reps0, "readings": all_rows,
                                               "decision": decision, "decision_detail": detail})
    b = O.write_bundle(d, cfg, stage_name, look, evs, decision)
    runs = {f"bundle__{look}": {"files": [{"path": f"{look}/bundle.json", "sha256": b["bundle_sha256"]},
                                           {"path": f"{look}/draws.npz", "sha256": b["draws_sha256"]}]},
            f"readings__{look}": {"files": [{"path": f"{look}/readings.json", "sha256": rsha}]}}
    exp, act, miss, unexp = [], [], [], []
    for name, sm in stages.items():
        for rid, r in (sm.get("runs") or {}).items():
            runs[f"{name}/{rid}"] = {"files": [{**f, "path": f"{name}/{f['path']}"} for f in r["files"]]}
        exp += [f"{name}/{x}" for x in sm.get("expected_run_ids") or []]
        act += [f"{name}/{x}" for x in sm.get("actual_run_ids") or []]
        miss += [f"{name}/{x}" for x in sm.get("missing_run_ids") or []]
        unexp += [f"{name}/{x}" for x in sm.get("unexpected_run_ids") or []]
    man.update(runs=runs, expected_run_ids=exp, actual_run_ids=act, missing_run_ids=miss, unexpected_run_ids=unexp,
               looks=[look], decisions={look: decision}, covers_required=need,
               stages={name: {"stage": sm["stage"], "tasks": sm.get("tasks"), "split": (sm.get("split") or {}).get(
                   "key"), "covers": sm.get("covers"), "wall_s": (sm.get("timings") or {}).get("wall_s"),
                   "freeze": (sm.get("freeze") or {}).get("status"),
                   "labels": ((sm.get("inputs") or {}).get("labels") or {}).get("source"),
                   "run_manifest_sha256": sha256_file(out / name / "run_manifest.json")}
                   for name, sm in stages.items()})
    log(f"[assemble] {len(stages)} stage directories; decision {decision['reading']}; {len(runs)} G5 runs")
    return man


def _claims_of_task(task: str) -> tuple:
    return ("E1", "C4") if task == "stability" else ("E2",)


def stage_config(cfg: Config, stage: str, budgets=None) -> Config:
    """The configuration a stage runs with (a copy; the YAML views are shared). budgets (primary stage only): restrict
    the run to these budgets of the configuration (the evaluation at each n is independent)."""
    c = copy.copy(cfg)
    if stage == "primary":
        c.c3 = c.c1 = False  # C3 and C1 are their own stages
    if budgets:
        if stage != "primary":
            raise AnalysisError("--budget applies to --stage primary only")
        bad = [n for n in budgets if n not in cfg.budgets]
        if bad or len(set(budgets)) != len(budgets):
            raise AnalysisError(f"--budget {budgets}: not distinct budgets of the configuration {cfg.budgets}")
        c.budgets = [n for n in cfg.budgets if n in budgets]
    return c
