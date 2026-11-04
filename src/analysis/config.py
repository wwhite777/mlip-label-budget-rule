"""Configuration: every analysis constant comes from research/protocol_legacy.yaml (amends v1.1, which amends v1).

Modes
  dryrun   the PILOT configuration on PILOT structures with SYNTHETIC labels (no real label column is opened);
           the freeze check is skipped.
  pilot    the YAML `pilot` block on PILOT structures with real labels; refuses to run unless
           the published specification hashes match.
  confirm  the confirmatory protocol on CONFIRM structures with real labels; same freeze requirement.
The only constants NOT read from the YAML are the dry-run synthetic-label constants (task card S3), which are used in
dryrun mode only.

Splits (resolve_split): chemsys_primary (seed 0, the only split that produces E1/E2/C4 readings), chemsys_sensitivity_<i>
(splits.confirm.chemsys_sensitivity[i], fold-assignment seed sensitivities.fold_assignment_seeds[i]; descriptive) and
random_reference (random folds, no system-disjointness check; descriptive). Only the primary split enters families.
Auxiliary inputs (YAML block auxiliary_inputs) are hash-checked by verified_aux() before any use.
CONFIRM runs are refused by check_deferred() while descriptive_specs.deferred_exploratory lists analyses (C5, coverage)
that have no frozen operational specification and implementation (v1.1). From v1.2 the list is empty and
descriptive_specs.c5_active_learning / coverage_diagnostics are parsed fail-closed by c5_spec() / coverage_spec()
(implemented in analysis.al / analysis.coverage).
"""
from __future__ import annotations

import hashlib
import itertools
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "research" / "protocol_legacy.yaml"
MODES = ("dryrun", "pilot", "confirm")
TASKS = ("stability", "band_gap")

# Task card S3 (dryrun only; not preregistered constants): synthetic Bernoulli labels, independent of the features.
SYNTHETIC_SEED = 424242
SYNTHETIC_BASE_RATE = {"stability": 0.17, "band_gap": 0.25}


class AnalysisError(Exception):
    """Base class: the pipeline refuses to continue."""


class PartitionError(AnalysisError):
    """A structure of the other part (or outside U) reached features, labels or splits."""


class BudgetViolation(AnalysisError):
    """A label outside the replicate's draw was requested, or the touched set differs from the draw."""


class FreezeMismatch(AnalysisError):
    """The preregistration freeze cannot be verified."""


class HashMismatch(AnalysisError):
    """An input file does not match its recorded sha256."""


class DeferredAnalysis(AnalysisError):
    """A CONFIRM run was requested while preregistered analyses are still deferred (no frozen specification)."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha(path, expected: str, what: str = "") -> str:
    got = sha256_file(path)
    if got != expected:
        raise HashMismatch(f"{what or path}: sha256 {got} != recorded {expected}")
    return got


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def load_prereg(path=PREREG) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise AnalysisError(f"{path}: not a mapping")
    return doc


_RNG_RE = re.compile(r"default_rng\((\d+)\*fold \+ replicate(?: \+ (\d+))?\)")


def parse_rng_formula(text: str) -> tuple:
    """'numpy default_rng(1000*fold + replicate + 7)' -> (1000, 7)."""
    m = _RNG_RE.search(str(text))
    if not m:
        raise AnalysisError(f"cannot parse the rng formula {text!r}")
    return int(m.group(1)), int(m.group(2) or 0)


_POS_RE = re.compile(r"^\s*(<=|>=|<|>)\s*([-+0-9.eE]+)\s*$")


def parse_positive_rule(text: str):
    """'<= 0.0' -> (op, threshold)."""
    m = _POS_RE.match(str(text))
    if not m:
        raise AnalysisError(f"cannot parse positive_if {text!r}")
    return m.group(1), float(m.group(2))


def apply_positive_rule(values, rule):
    import numpy as np
    op, thr = rule
    v = np.asarray(values, dtype=np.float64)
    return {"<=": v <= thr, ">=": v >= thr, "<": v < thr, ">": v > thr}[op].astype(np.int8)


SPLIT_KEYS = {"chemsys_primary": ("chemsys", "primary"), "random_reference": ("random", "random_reference")}
_SENS_RE = re.compile(r"chemsys_sensitivity_(0|[1-9][0-9]*)")


def resolve_split(y: dict, section: str, split_key: str) -> dict:
    """Resolve a split selector BEFORE any direct dictionary lookup ( finding 1).
    Returns {key, path, sha256, split_type ('chemsys'|'random'), role ('primary'|'fold_sensitivity'|'random_reference'),
    seed}. chemsys_sensitivity_<i> selects splits.confirm.chemsys_sensitivity[i] (index validated) and its seed
    sensitivities.fold_assignment_seeds[i]; the file name must carry that seed."""
    splits = y["splits"]
    if section not in ("pilot", "confirm") or not isinstance(splits.get(section), dict):
        raise AnalysisError(f"splits.{section} absent")
    sec = splits[section]
    m = _SENS_RE.fullmatch(str(split_key))
    if m:
        if section != "confirm":
            raise AnalysisError(f"{split_key}: fold-sensitivity splits exist for CONFIRM only")
        lst = sec.get("chemsys_sensitivity")
        if not isinstance(lst, list) or not lst:
            raise AnalysisError("splits.confirm.chemsys_sensitivity is not a non-empty list")
        i = int(m.group(1))
        if not 0 <= i < len(lst):
            raise AnalysisError(f"{split_key}: index {i} outside splits.confirm.chemsys_sensitivity[0..{len(lst) - 1}]")
        seeds = list(y["sensitivities"]["fold_assignment_seeds"])
        if len(seeds) != len(lst):
            raise AnalysisError(f"{len(lst)} sensitivity split files but fold_assignment_seeds {seeds}")
        sp, seed = lst[i], int(seeds[i])
        if not isinstance(sp, dict) or not sp.get("path") or not sp.get("sha256"):
            raise AnalysisError(f"{split_key}: entry is not a {{path, sha256}} mapping")
        if f"seed{seed}." not in Path(sp["path"]).name:
            raise AnalysisError(f"{split_key}: file {sp['path']} does not carry fold-assignment seed {seed}")
        return {"key": split_key, "path": sp["path"], "sha256": sp["sha256"], "split_type": "chemsys",
                "role": "fold_sensitivity", "seed": seed, "index": i}
    if split_key == "chemsys_sensitivity":
        raise AnalysisError("use chemsys_sensitivity_<i> (i = 0, 1) for the sensitivity split seeds")
    if split_key not in SPLIT_KEYS or split_key not in sec:
        allowed = [k for k in SPLIT_KEYS if k in sec] + (
            [f"chemsys_sensitivity_{i}" for i in range(len(sec.get("chemsys_sensitivity") or []))])
        raise AnalysisError(f"split {split_key!r} not available for {section}; allowed {allowed}")
    sp = sec[split_key]
    if not isinstance(sp, dict) or not sp.get("path") or not sp.get("sha256"):
        raise AnalysisError(f"{split_key}: entry is not a {{path, sha256}} mapping")
    typ, role = SPLIT_KEYS[split_key]
    ms = re.search(r"seed(\d+)\.", Path(sp["path"]).name)
    return {"key": split_key, "path": sp["path"], "sha256": sp["sha256"], "split_type": typ, "role": role,
            "seed": int(ms.group(1)) if ms else None, "index": None}


_SCOPE_BB = {"confirmatory backbones": "confirmatory", "every fitted backbone": "fitted"}
RVC_PHRASES = ("best head", "both tasks")  # random_vs_chemsys.scope must state both (fail-closed)


def parse_scope_text(text, what: str, all_n, primary_only: bool = True, individual_controls: bool = False,
                     need=()) -> dict:
    """A descriptive_specs scope sentence -> {backbones: confirmatory|fitted, n, tasks, tasks_stated}. Fail-closed:
    every element the code needs must be stated (CONFIRM; for C3/C1 'primary chemsys split (seed 0) only'; exactly one
    backbone rule; exactly one n in budgets.all_n; for C1 'each individual control')."""
    t = " ".join(str(text).split())
    probs = []
    if not re.search(r"\bCONFIRM\b", t):
        probs.append("CONFIRM not stated")
    if primary_only and "primary chemsys split (seed 0) only" not in t:
        probs.append("'primary chemsys split (seed 0) only' not stated")
    bb = [v for k, v in _SCOPE_BB.items() if k in t]
    if len(bb) != 1:
        probs.append(f"backbone rule not exactly one of {list(_SCOPE_BB)}")
    ns = [int(x.replace(",", "")) for x in re.findall(r"\bn = ([\d,]+)", t)]
    if len(ns) != 1 or ns[0] not in [int(v) for v in all_n]:
        probs.append(f"n not stated once within budgets.all_n (found {ns})")
    if individual_controls and "each individual control" not in t:
        probs.append("'each individual control' not stated")
    probs += [f"{w!r} not stated" for w in need if w not in t]
    if probs:
        raise AnalysisError(f"descriptive_specs.{what}.scope {t!r}: " + "; ".join(probs))
    stated = "both tasks" in t
    return {"backbones": bb[0], "n": ns[0], "tasks": list(TASKS), "tasks_stated": stated, "text": t}


FOLD_SCOPE_KEYS = ("budgets", "backbones", "heads", "controls", "references", "tasks")


def fold_scope(y: dict, tasks: list) -> dict:
    """descriptive_specs.fold_sensitivity.scope -> the restricted run configuration of a fold-sensitivity rerun."""
    fs = (y.get("descriptive_specs") or {}).get("fold_sensitivity")
    if not isinstance(fs, dict) or not isinstance(fs.get("scope"), dict):
        raise AnalysisError("descriptive_specs.fold_sensitivity.scope (mapping) is required for a fold-sensitivity run")
    sc = fs["scope"]
    if set(sc) != set(FOLD_SCOPE_KEYS):
        raise AnalysisError(f"fold_sensitivity.scope keys {sorted(sc)} != {sorted(FOLD_SCOPE_KEYS)}")
    all_n = [int(n) for n in y["budgets"]["all_n"]]
    budgets = sc["budgets"]
    if not (isinstance(budgets, list) and budgets and all(isinstance(n, int) and n in all_n for n in budgets)
            and int(y["budgets"]["primary_n"]) in budgets):
        raise AnalysisError(f"fold_sensitivity.scope.budgets {budgets!r} must be a subset of {all_n} with primary n")
    bbs = y["backbones"]
    full = list(bbs["confirmatory"]) + list(bbs["exploratory"]) + list(bbs["secondary"])
    sel = {"confirmatory": list(bbs["confirmatory"]), "all": full}
    backbones = sel.get(sc["backbones"]) if isinstance(sc["backbones"], str) else list(sc["backbones"])
    if not backbones or not set(backbones) <= set(full) or not set(bbs["confirmatory"]) <= set(backbones):
        raise AnalysisError(f"fold_sensitivity.scope.backbones {sc['backbones']!r} (confirmatory | all | list with "
                            f"every confirmatory backbone)")
    tie_h = list(y["heads"]["tie_order"])
    heads = tie_h if sc["heads"] == "all" else list(sc["heads"]) if isinstance(sc["heads"], list) else None
    if not heads or not set(heads) <= set(tie_h):
        raise AnalysisError(f"fold_sensitivity.scope.heads {sc['heads']!r} (all | list of heads.tie_order)")
    tie_c = list(y["controls"]["tie_order"])
    ctl = {"all": tie_c, "all_except_cgcnn": [c for c in tie_c if c != "cgcnn"]}
    controls = ctl.get(sc["controls"]) if isinstance(sc["controls"], str) else list(sc["controls"])
    if not controls or not set(controls) <= set(tie_c):
        raise AnalysisError(f"fold_sensitivity.scope.controls {sc['controls']!r} (all | all_except_cgcnn | list)")
    refs = sc["references"]
    if not (isinstance(refs, list) and set(refs) <= {"r_sp", "r_is2re"} and "r_sp" in refs):
        raise AnalysisError(f"fold_sensitivity.scope.references {refs!r} (subset of r_sp, r_is2re with r_sp)")
    st = sc["tasks"]
    if not (isinstance(st, list) and set(st) <= set(TASKS)) or not set(tasks) <= set(st):
        raise AnalysisError(f"requested tasks {tasks} not inside fold_sensitivity.scope.tasks {st!r}")
    return {"budgets": [int(n) for n in budgets], "backbones": backbones, "family_backbones": list(bbs["confirmatory"]),
            "heads": heads, "controls": controls, "references": list(refs), "tasks": list(tasks), "raw": dict(sc)}


def aux_entries(y: dict) -> dict:
    """Flattened auxiliary_inputs: 'key' or 'key.sub' -> {path, sha256}."""
    a = y.get("auxiliary_inputs")
    if not isinstance(a, dict):
        raise AnalysisError("the YAML has no auxiliary_inputs block (required from v1.1,  finding 6)")
    out = {}
    for k, v in a.items():
        if k == "note":
            continue
        if isinstance(v, dict) and "path" in v:
            out[k] = v
        elif isinstance(v, dict):
            for s, e in v.items():
                out[f"{k}.{s}"] = e
        else:
            raise AnalysisError(f"auxiliary_inputs.{k} is not a mapping")
    for k, e in out.items():
        if not isinstance(e, dict) or not isinstance(e.get("path"), str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                                          str(e.get("sha256"))):
            raise AnalysisError(f"auxiliary_inputs.{k} is not a {{path, sha256}} mapping")
    return out


def verified_aux(y: dict, key: str) -> Path:
    """Path of auxiliary input `key` ('key' or 'key.sub') after its sha256 matched the YAML (hard error otherwise)."""
    ents = aux_entries(y)
    if key not in ents:
        raise AnalysisError(f"auxiliary_inputs has no entry {key!r}")
    p = resolve(ents[key]["path"])
    if not p.is_file():
        raise HashMismatch(f"auxiliary input {key}: {p} is missing")
    check_sha(p, ents[key]["sha256"], f"auxiliary input {key}")
    return p


# ---------------------------------------------------------------------------------------------- v1.2 descriptive specs
C5_STRATEGIES = ("random", "uncertainty", "uncertainty_diversity")
_C5_RNG = re.compile(r"default_rng\((\d+) \+ (\d+)\*fold \+ seed\)")
_COV_RNG = re.compile(r"default_rng\((\d+) \+ fold\)")


def _need(text: str, phrases, what: str) -> list:
    t = " ".join(str(text).split())
    return [f"{what}: {p!r} not stated" for p in phrases if p not in t]


def _one_int(pattern: str, text: str, what: str, probs: list):
    t = " ".join(str(text).split())
    ms = re.findall(pattern, t)
    if len(ms) != 1:
        probs.append(f"{what}: /{pattern}/ found {len(ms)} time(s)")
        return None
    return int(str(ms[0]).replace(",", ""))


def c5_spec(y: dict) -> dict:
    """descriptive_specs.c5_active_learning -> the constants of analysis.al (fail-closed: every element the code needs
    must be stated; the same vocabulary is checked by the legacy configuration validator)."""
    ds = y.get("descriptive_specs") or {}
    c = ds.get("c5_active_learning")
    if not isinstance(c, dict):
        raise AnalysisError("descriptive_specs.c5_active_learning (mapping) is required")
    probs = []
    sc = c.get("scope")
    if not isinstance(sc, dict):
        probs.append("scope is not a mapping")
        sc = {}
    want = {"part": "CONFIRM", "split": "chemsys_primary", "backbones": "confirmatory", "features": "initial"}
    probs += [f"scope.{k} {sc.get(k)!r} != {v!r}" for k, v in want.items() if sc.get(k) != v]
    tasks = sc.get("tasks")
    if not (isinstance(tasks, list) and tasks and set(tasks) <= set(TASKS)):
        probs.append(f"scope.tasks {tasks!r} not a non-empty subset of {TASKS}")
    sch = c.get("schedule") if isinstance(c.get("schedule"), dict) else {}
    vals = {k: sch.get(k) for k in ("initial", "step", "rounds", "final")}
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in vals.values()):
        probs.append(f"schedule {sch!r} needs positive integers initial, step, rounds, final")
    elif vals["initial"] + vals["rounds"] * vals["step"] != vals["final"]:
        probs.append(f"schedule: initial + rounds*step != final ({vals})")
    seeds = c.get("seeds")
    if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) and not isinstance(s, bool) and s >= 0
                                                      for s in seeds) and len(set(seeds)) == len(seeds)):
        probs.append(f"seeds {seeds!r} not a list of distinct non-negative integers")
    st = c.get("strategies")
    if not isinstance(st, dict) or tuple(st) != C5_STRATEGIES:
        probs.append(f"strategies {list(st) if isinstance(st, dict) else st!r} != {list(C5_STRATEGIES)}")
        st = {}
    mult = _one_int(r"the (\d+)\*step highest-uncertainty", st.get("uncertainty_diversity", ""),
                    "strategies.uncertainty_diversity", probs)
    probs += _need(st.get("uncertainty_diversity", ""), ("greedy farthest-point", "Euclidean distance",
                                                        "standardized feature space", "to the labelled set",
                                                        "the batch chosen so far", "ties: material_id order"),
                   "strategies.uncertainty_diversity")
    probs += _need(st.get("uncertainty", ""), ("the step highest-uncertainty", "unlabelled pool",
                                              "ties: material_id order"), "strategies.uncertainty")
    probs += _need(st.get("random", ""), ("uniform without replacement", "unlabelled outer-training pool"),
                   "strategies.random")
    probs += _need(c.get("model", ""), ("heads.mlp_ensemble", "frozen hyperparameters", "no grid, no selection",
                                        "standardized on the labelled set of each round", "mean member probability",
                                        "standard deviation of the member probabilities"), "model")
    m = _C5_RNG.findall(" ".join(str(c.get("initial_draw", "")).split()))
    if len(m) != 1:
        probs.append(f"initial_draw: 'default_rng(<base> + <a>*fold + seed)' found {len(m)} time(s)")
    probs += _need(c.get("initial_draw", ""), ("uniform without replacement", "outer-training pool",
                                               "the same initial set for every strategy"), "initial_draw")
    probs += _need(c.get("pool", ""), ("outer-training structures of the fold only",
                                       "test structures are never acquired"), "pool")
    unit = _one_int(r"labels in units of ([\d,]+)", c.get("metric", ""), "metric", probs)
    probs += _need(c.get("metric", ""), ("pooled out-of-fold AP", "mean over seeds", "trapezoid"), "metric")
    probs += _need(c.get("contrasts", ""), ("d_unc = AP(uncertainty) - AP(random)",
                                            "d_div = AP(uncertainty_diversity) - AP(random)", "at the final budget",
                                            "AUC_AL", "paired chemical-system cluster bootstrap"), "contrasts")
    if probs:
        raise AnalysisError("descriptive_specs.c5_active_learning: " + "; ".join(probs))
    budgets = [vals["initial"] + i * vals["step"] for i in range(vals["rounds"] + 1)]
    return {"tasks": list(tasks), "budgets": budgets, "initial": vals["initial"], "step": vals["step"],
            "rounds": vals["rounds"], "final": vals["final"], "seeds": [int(s) for s in seeds],
            "strategies": list(C5_STRATEGIES), "shortlist_mult": int(mult),
            "init_rng": (int(m[0][0]), int(m[0][1])), "auc_unit": int(unit), "scope": dict(sc)}


def coverage_spec(y: dict) -> dict:
    """descriptive_specs.coverage_diagnostics -> the constants of analysis.coverage (fail-closed)."""
    ds = y.get("descriptive_specs") or {}
    c = ds.get("coverage_diagnostics")
    if not isinstance(c, dict):
        raise AnalysisError("descriptive_specs.coverage_diagnostics (mapping) is required")
    probs = []
    sc = c.get("scope") if isinstance(c.get("scope"), dict) else {}
    if sc.get("part") != "CONFIRM":
        probs.append(f"scope.part {sc.get('part')!r} != 'CONFIRM'")
    splits = sc.get("splits")
    if not (isinstance(splits, list) and splits and set(splits) <= {"chemsys_primary", "random_reference"}
            and "chemsys_primary" in splits):
        probs.append(f"scope.splits {splits!r} (subset of chemsys_primary, random_reference with chemsys_primary)")
    reps = sc.get("representations")
    want_r = ["confirmatory backbones (initial features)", "composition element fractions"]
    if reps != want_r:
        probs.append(f"scope.representations {reps!r} != {want_r}")
    kt = c.get("knn_primary", "")
    k = _one_int(r"\bk = (\d+) nearest", kt, "knn_primary", probs)
    n_ref = _one_int(r"\b([\d,]+)-structure random sample", kt, "knn_primary", probs)
    pct = _one_int(r"exceeds the (\d+)th percentile of the reference", kt, "knn_primary", probs)
    mr = _COV_RNG.findall(" ".join(str(kt).split()))
    if len(mr) != 1:
        probs.append(f"knn_primary: 'default_rng(<base> + fold)' found {len(mr)} time(s)")
    probs += _need(kt, ("Euclidean distance", "training structures of the same fold",
                        "standardization fitted on that fold's training structures", "leave-self-out",
                        "to the rest of training", "mean 5-NN distance" if k == 5 else f"mean {k}-NN distance",
                        "OUT_OF_COVERAGE"), "knn_primary")
    ut = c.get("umap_kde_secondary", "")
    nn = _one_int(r"n_neighbors (\d+)\b", ut, "umap_kde_secondary", probs)
    nc = _one_int(r"n_components (\d+)\b", ut, "umap_kde_secondary", probs)
    md = re.findall(r"min_dist ([0-9]*\.?[0-9]+)\b", " ".join(str(ut).split()))
    me = re.findall(r"metric (\w+)", " ".join(str(ut).split()))
    rs = re.findall(r"random_state in \{([\d, ]+)\}", " ".join(str(ut).split()))
    if len(md) != 1 or len(me) != 1 or len(rs) != 1:
        probs.append(f"umap_kde_secondary: min_dist {md}, metric {me}, random_state set {rs} (need one each)")
    kde_pct = _one_int(r"below the (\d+)th percentile of the training densities", ut, "umap_kde_secondary", probs)
    probs += _need(ut, ("umap-learn", "fitted on the same", "applied to the test structures",
                        "scipy.stats.gaussian_kde, Scott bandwidth", "on the training embedding", "LOW_DENSITY",
                        "reported per random_state"), "umap_kde_secondary")
    pl = c.get("performance_link", "")
    n_link = _one_int(r"\bn = ([\d,]+)", pl, "performance_link", probs)
    probs += _need(pl, ("CONFIRM primary split", "AP of the best head", "for band gap, of the best control",
                        "covered versus OUT_OF_COVERAGE", "k-NN primary", "counts, prevalence and 90 % intervals",
                        "frozen bootstrap"), "performance_link")
    if n_link is not None and n_link not in [int(v) for v in y["budgets"]["all_n"]]:
        probs.append(f"performance_link n = {n_link} not in budgets.all_n")
    probs += _need(c.get("labels", ""), ("coverage is label-free", "only performance_link reads labels"), "labels")
    probs += _need(c.get("outputs", ""), ("out-of-coverage and low-density fractions", "pooled fractions with counts"),
                   "outputs")
    if probs:
        raise AnalysisError("descriptive_specs.coverage_diagnostics: " + "; ".join(probs))
    return {"splits": list(splits), "k": int(k), "n_reference": int(n_ref), "ref_rng_base": int(mr[0]),
            "percentile": float(pct), "umap": {"n_neighbors": int(nn), "min_dist": float(md[0]),
                                               "n_components": int(nc), "metric": me[0]},
            "random_states": [int(v) for v in rs[0].replace(" ", "").split(",")], "kde_percentile": float(kde_pct),
            "link_n": int(n_link), "representations": list(reps)}


SPEC_PARSERS = {"c5_active_learning": c5_spec, "coverage_diagnostics": coverage_spec}


# Deferred exploratory analyses: name prefix in descriptive_specs.deferred_exploratory.items -> the YAML spec key a
# later amendment must add. Both are implemented from v1.2 (analysis.al, analysis.coverage); their YAML specifications
# are parsed fail-closed (c5_spec / coverage_spec) before a CONFIRM run is allowed.
DEFERRED_SPEC_KEYS = {"C5 active-learning replication": "c5_active_learning",
                      "coverage diagnostics": "coverage_diagnostics"}
IMPLEMENTED_DEFERRED: set = {"c5_active_learning", "coverage_diagnostics"}


def check_deferred(cfg) -> None:
    """CONFIRM refuses to run while C5 / coverage are deferred or unimplemented (pilot and dryrun are allowed)."""
    if cfg.mode != "confirm":
        return
    ds = cfg.y.get("descriptive_specs")
    if not isinstance(ds, dict):
        raise DeferredAnalysis("confirm refused: the YAML has no descriptive_specs block")
    items = list(((ds.get("deferred_exploratory") or {}).get("items")) or [])
    problems = []
    if items:
        problems.append(f"deferred without a frozen operational specification: {items}")
    for name, key in DEFERRED_SPEC_KEYS.items():
        if any(str(i).startswith(name) for i in items):
            continue
        if key not in ds:
            problems.append(f"{name}: no descriptive_specs.{key}")
        elif key not in IMPLEMENTED_DEFERRED:
            problems.append(f"{name}: descriptive_specs.{key} is not implemented in src/analysis")
        else:
            try:
                SPEC_PARSERS[key](cfg.y)
            except AnalysisError as e:
                problems.append(f"{name}: {e}")
    if problems:
        raise DeferredAnalysis("confirm refused (C5 active-learning replication and coverage diagnostics must be "
                               "specified in a later amendment, frozen and implemented before any CONFIRM label is "
                               "read): " + "; ".join(problems))


def grid(spec: dict, keys: list) -> list:
    """Cartesian product of the YAML list-valued keys, in YAML key order then value order."""
    vals = [spec[k] if isinstance(spec[k], list) else [spec[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


@dataclass
class Config:
    mode: str
    tasks: list
    prereg: dict
    prereg_path: Path
    part: str
    backbones: list            # backbones whose heads are fitted
    family_backbones: list     # backbones entering the hypothesis families
    heads: list
    controls: list
    references: list
    budgets: list
    primary_n: int
    replicates_first: list
    replicates_max: int
    extension_allowed: bool
    extension_replicates: list
    n_folds: int
    split_key: str
    split_path: Path
    split_sha: str
    draw_rng: tuple
    inner_rng: tuple
    inner_frac: float
    B: int
    boot_seed: int
    alpha: float
    coverage: float
    margins: dict
    family_sizes: dict
    cgcnn_budgets: list
    cgcnn_tasks: list
    synthetic: bool
    split_type: str = "chemsys"        # chemsys | random (random: no within-fold system-disjointness check)
    role: str = "primary"              # primary | fold_sensitivity | random_reference (only primary is read)
    split_seed: int | None = 0
    freeze_dir: Path | None = None     # implementation.freeze_dir
    c3: bool = False                   # descriptive_specs.c3_input_structure (CONFIRM, primary split)
    c1: bool = False                   # descriptive_specs.c1_membership_association (CONFIRM, primary split)
    c3_scope: dict | None = None       # parsed descriptive_specs.c3_input_structure.scope
    c1_scope: dict | None = None       # parsed descriptive_specs.c1_membership_association.scope
    rvc_scope: dict | None = None      # parsed descriptive_specs.random_vs_chemsys.scope
    split_scope: dict | None = None    # descriptive_specs.fold_sensitivity.scope (fold-sensitivity runs)
    residual_heads: bool = True        # False for the random-reference run (no references in its scope)
    overrides: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ generators
    @classmethod
    def build(cls, mode: str, tasks=None, prereg_path=PREREG, split_key: str | None = None, **overrides):
        if mode not in MODES:
            raise AnalysisError(f"mode {mode!r} not in {MODES}")
        y = load_prereg(prereg_path)
        tasks = list(tasks or TASKS)
        for t in tasks:
            if t not in TASKS:
                raise AnalysisError(f"task {t!r} not in {TASKS}")
        bud, rep, inf = y["budgets"], y["replicates"], y["inference"]
        pil = y["pilot"]
        draw_rng = parse_rng_formula(bud["draw"])
        inner_rng = parse_rng_formula(bud["inner_validation"]["rng"])
        if bud["inner_validation"]["unit"] != "chemical_system":
            raise AnalysisError("inner validation unit is not chemical_system")
        if bud.get("labels_outside_draw_allowed") is not False:
            raise AnalysisError("labels_outside_draw_allowed must be false")
        cov_txt = str(inf["interval"]["coverage"])
        m = re.search(r"(\d+)\s*percent", cov_txt)
        if not m or "two-sided" not in cov_txt:
            raise AnalysisError(f"cannot parse interval coverage {cov_txt!r}")
        coverage = int(m.group(1)) / 100.0
        pm = inf["interval"].get("percentile_method", "numpy.percentile default (linear)")
        if pm != "numpy.percentile default (linear)":
            raise AnalysisError(f"interval percentile_method {pm!r} is not the implemented numpy default (linear)")
        margins = {k: float(v) for k, v in y["margins"].items() if k.startswith("m_")}
        if mode in ("dryrun", "pilot"):
            if pil["part"] != "PILOT":
                raise AnalysisError("pilot.part is not PILOT")
            part = "PILOT"
            backbones = list(pil["backbones"])
            family_backbones = list(pil["backbones"])
            heads = list(pil["heads"])
            controls = list(pil["controls"])
            references = list(pil["references"])
            budgets = [int(pil["n"])]
            primary_n = int(pil["n"])
            replicates_first = list(range(int(pil["replicates_first_look"])))
            replicates_max = int(rep["pilot_max"])
            ext = pil["extension"]
            extension_allowed = bool(ext["allowed_once"])
            extension_replicates = [int(r) for r in ext["add_replicates"]]
            family_sizes = {k: int(v) for k, v in pil["families"].items()}
            split_key = split_key or "chemsys_primary"
            sp = resolve_split(y, "pilot", split_key)
        else:
            part = "CONFIRM"
            bb = y["backbones"]
            backbones = list(bb["confirmatory"]) + list(bb["exploratory"]) + list(bb["secondary"])
            family_backbones = list(bb["confirmatory"])
            heads = list(y["heads"]["tie_order"])
            controls = list(y["controls"]["tie_order"])
            references = ["r_sp", "r_is2re"]
            budgets = [int(n) for n in bud["all_n"]]
            primary_n = int(bud["primary_n"])
            replicates_first = list(range(int(rep["confirm"])))
            replicates_max = int(rep["confirm"])
            extension_allowed = False
            extension_replicates = []
            family_sizes = {k: int(v["size"]) for k, v in y["families"].items()}
            split_key = split_key or "chemsys_primary"
            sp = resolve_split(y, "confirm", split_key)
        if sp["role"] != "primary":
            family_sizes = {}  # descriptive splits never enter E1/E2/C4 families or readings ( finding 2)
        ds = y.get("descriptive_specs") or {}
        primary_confirm = mode == "confirm" and sp["role"] == "primary"
        split_scope = None
        if sp["role"] == "fold_sensitivity":  # restricted rerun read from the YAML (not the full confirm config)
            split_scope = fold_scope(y, tasks)
            budgets, backbones = split_scope["budgets"], split_scope["backbones"]
            heads, controls, references = split_scope["heads"], split_scope["controls"], split_scope["references"]
        all_n = bud["all_n"]
        c3_scope = parse_scope_text(ds["c3_input_structure"]["scope"], "c3_input_structure", all_n) \
            if "c3_input_structure" in ds else None
        c1_scope = parse_scope_text(ds["c1_membership_association"]["scope"], "c1_membership_association", all_n,
                                    individual_controls=True) if "c1_membership_association" in ds else None
        rvc_scope = parse_scope_text(ds["random_vs_chemsys"]["scope"], "random_vs_chemsys", all_n,
                                     primary_only=False, need=RVC_PHRASES) if "random_vs_chemsys" in ds else None
        residual_heads = True
        if sp["role"] == "random_reference":  # the random run only feeds delta_split: run scope from the YAML sentence
            if rvc_scope is None:
                raise AnalysisError("descriptive_specs.random_vs_chemsys.scope is required for the random-reference run")
            if not set(tasks) <= set(rvc_scope["tasks"]):
                raise AnalysisError(f"requested tasks {tasks} outside random_vs_chemsys.scope")
            budgets = [rvc_scope["n"]]
            backbones = list(bb["confirmatory"]) if rvc_scope["backbones"] == "confirmatory" else backbones
            heads = list(y["heads"]["tie_order"])  # 'best head' = heads menu with nested selection
            controls, references, residual_heads = [], [], False
        fdir = (y.get("implementation") or {}).get("freeze_dir")
        if primary_n not in [int(n) for n in bud["all_n"]]:
            raise AnalysisError("primary n not in budgets.all_n")
        if rep.get("average_scores_across_replicates") is not False:
            raise AnalysisError("average_scores_across_replicates must be false")
        cfg = cls(
            mode=mode, tasks=tasks, prereg=y, prereg_path=Path(prereg_path), part=part, backbones=backbones,
            family_backbones=family_backbones, heads=heads, controls=controls, references=references,
            budgets=budgets, primary_n=primary_n, replicates_first=replicates_first, replicates_max=replicates_max,
            extension_allowed=extension_allowed, extension_replicates=extension_replicates,
            n_folds=int(y["splits"]["n_folds"]), split_key=split_key, split_path=resolve(sp["path"]),
            split_sha=sp["sha256"], draw_rng=draw_rng, inner_rng=inner_rng,
            inner_frac=float(bud["inner_validation"]["fraction"]), B=int(inf["bootstrap"]["B"]),
            boot_seed=int(inf["bootstrap"]["seed"]), alpha=float(inf["alpha"]), coverage=coverage,
            margins=margins, family_sizes=family_sizes,
            cgcnn_budgets=[int(n) for n in y["controls"]["cgcnn"]["budgets"]],
            cgcnn_tasks=[str(t) for t in y["controls"]["cgcnn"]["tasks"]],
            synthetic=(mode == "dryrun"), split_type=sp["split_type"], role=sp["role"], split_seed=sp["seed"],
            freeze_dir=resolve(fdir) if fdir else None, c3=primary_confirm and c3_scope is not None,
            c1=primary_confirm and c1_scope is not None, c3_scope=c3_scope, c1_scope=c1_scope, rvc_scope=rvc_scope,
            split_scope=split_scope, residual_heads=residual_heads,
        )
        cfg._check_family_sizes()
        for k, v in overrides.items():  # test fixtures only (tiny synthetic problems)
            if not hasattr(cfg, k):
                raise AnalysisError(f"unknown override {k}")
            setattr(cfg, k, v)
        cfg.overrides = dict(overrides)
        return cfg

    @property
    def readings_allowed(self) -> bool:
        return self.role == "primary"

    def scope_backbones(self, scope: dict) -> list:
        return list(self.family_backbones) if scope["backbones"] == "confirmatory" else list(self.backbones)

    def c3_backbones(self, n: int, task: str) -> list:
        sc = self.c3_scope
        return self.scope_backbones(sc) if self.c3 and sc and n == sc["n"] and task in sc["tasks"] else []

    def c1_backbones(self, n: int, task: str) -> list:
        sc = self.c1_scope
        return self.scope_backbones(sc) if self.c1 and sc and n == sc["n"] and task in sc["tasks"] else []

    def rvc_backbones(self, n: int, task: str) -> list:
        sc = self.rvc_scope
        ok = self.role == "random_reference" and sc and n == sc["n"] and task in sc["tasks"]
        return self.scope_backbones(sc) if ok else []

    def _check_family_sizes(self):
        if not self.readings_allowed:
            if self.family_sizes:
                raise AnalysisError(f"split role {self.role}: no hypothesis family allowed")
            return
        k = len(self.family_backbones)
        want = {"E1_up": 2 * k, "E1_down": 2 * k, "E2_up": k, "E2_down": k, "C4_up": k}
        for fam, size in self.family_sizes.items():
            if want.get(fam) != size:
                raise AnalysisError(f"family {fam}: YAML size {size} != computed {want.get(fam)}")

    # ------------------------------------------------------------------ YAML views
    @property
    def y(self) -> dict:
        return self.prereg

    def head_grid(self, head: str) -> list:
        h = self.y["heads"]
        if head == "logistic":
            return grid(h["logistic"], ["C"])
        if head == "hist_gbm":
            return grid(h["hist_gbm"], ["learning_rate", "max_leaf_nodes"])
        if head == "mlp_ensemble":
            return [{}]  # fixed configuration (no grid in the YAML)
        raise AnalysisError(f"unknown head {head}")

    def control_grid(self, control: str) -> list:
        c = self.y["controls"][control]
        if control in ("composition_logistic", "random_projection_logistic", "randinit_encoder_logistic"):
            return grid(c, ["C"])
        if control == "composition_hist_gbm":
            return grid(c, ["learning_rate", "max_leaf_nodes"])
        if control == "cgcnn":
            return grid(c["grid"], list(c["grid"].keys()))
        raise AnalysisError(f"unknown control {control}")

    def controls_at(self, n: int, task: str) -> list:
        return [c for c in self.controls if c != "cgcnn" or (n in self.cgcnn_budgets and task in self.cgcnn_tasks)]

    def residual_models(self) -> list:
        return list(self.y["heads"]["residual_heads"]["models"]) if self.residual_heads else []

    def label_rule(self, task: str):
        return parse_positive_rule(self.y["labels"][task]["positive_if"])

    def label_column(self, task: str) -> str:
        return self.y["labels"][task]["column"]


def verify_freeze(cfg, freeze_dir):
    from integrity import verify_protocol
    return verify_protocol()
