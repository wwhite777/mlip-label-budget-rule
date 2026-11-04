"""Runner: python -m analysis.run --mode dryrun|pilot|confirm --task stability|band_gap|both --out DIR
           [--device cpu|cuda] [--freeze-dir D] [--split chemsys_primary|chemsys_sensitivity_0|chemsys_sensitivity_1|
           random_reference] [--pair-with SEED0_RUN_DIR ...] [--threads 8] [--graph-workers 8]
           [--stage primary|fold_sensitivity|random_reference|c3|c1|c5|coverage|coverage_link|assemble]
           [--budget N ...] (primary) [--backbone B ...] (c5) [--repr R ...] (coverage)
           [--coverage-dir DIR ...] (coverage_link)
(PYTHONPATH=src, from the project root, with the project venv.)
--freeze-dir defaults to the YAML's implementation.freeze_dir. Descriptive splits (chemsys_sensitivity_<i>,
random_reference; CONFIRM only) need --pair-with = the output directory of the primary chemsys seed-0 run; they never
write readings, decisions or bundles. A confirm run is refused while descriptive_specs.deferred_exploratory lists
analyses without a frozen specification and implementation (v1.1: C5 active learning, coverage diagnostics; v1.2
specifies both and empties the list).
Without --stage the legacy single-process run executes (unchanged). With --stage the work is split into processes
(analysis.stages): each writes its own run_manifest.json into --out; `--stage assemble --out TOP` (TOP = the parent of
the stage directories) builds TOP/<look>/bundle.json (G4/G6) and TOP/run_manifest.json (G5). Every stage of a confirm
run passes check_deferred() and the local specification hash check before any input is loaded.

dryrun   PILOT configuration, synthetic labels, freeze check skipped.
pilot    YAML pilot block: pilot backbones, pilot heads/controls (no CGCNN), replicates 0..first_look-1; if the branch
         rule finds an INCONCLUSIVE branch contrast, ONE extension adds the YAML extension replicates and re-judges.
confirm  CONFIRM part, all budgets (readings at primary n only), confirmatory + exploratory + secondary backbones
         (families over the confirmatory set), all controls (CGCNN at its YAML budgets).
pilot / confirm refuse to run unless the local specification hash check (with --freeze-dir) exits 0.
GPU: device cuda requires CUDA_VISIBLE_DEVICES to be set and to exclude physical GPU 0.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _device_guard(device: str) -> dict:
    import torch
    torch.use_deterministic_algorithms(True)
    info = {"device": device, "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__}
    if device.startswith("cuda"):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not vis or "0" in [v.strip() for v in vis.split(",")]:
            raise SystemExit("refusing: set CUDA_VISIBLE_DEVICES to GPUs among 1-3 (GPU 0 belongs to another user)")
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


class GraphStore:
    """Lazily built CGCNN graphs (INITIAL structures) for part indices; cached."""

    def __init__(self, cfg, part, workers: int, log=print):
        from analysis.config import check_sha, resolve
        ini = cfg.y["population"]["initial_structures"]
        self.path = resolve(ini["path"])
        check_sha(self.path, ini["sha256"], "initial structures")
        self.cfg, self.part, self.workers, self.log = cfg, part, workers, log
        self.cache, self.structs, self.seconds_build = {}, None, 0.0

    def __call__(self, idx):
        from analysis.models_cgcnn import build_graphs, load_initial_structures
        need = [int(i) for i in idx if int(i) not in self.cache]
        if need:
            t0 = time.time()
            if self.structs is None:
                self.structs = dict(zip(range(self.part.n), load_initial_structures(self.path, self.part.ids)))
                self.log(f"[graphs] loaded {self.part.n} initial structures in {time.time() - t0:.0f}s")
            g = self.cfg.y["controls"]["cgcnn"]["graph"]
            if g["element_embedding"] != "one_hot":
                raise SystemExit("cgcnn element embedding changed")
            built = build_graphs([self.structs[i] for i in need], g["cutoff_angstrom"], g["max_neighbors"],
                                 self.workers)
            self.cache.update(zip(need, built))
            self.seconds_build += time.time() - t0
        return [self.cache[int(i)] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analysis.run", description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["dryrun", "pilot", "confirm"])
    ap.add_argument("--task", required=True, choices=["stability", "band_gap", "both"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--freeze-dir")
    ap.add_argument("--split", default=None)
    ap.add_argument("--pair-with", action="append", default=None,
                    help="primary chemsys seed-0 run directory (descriptive splits; stages: one per task allowed)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--graph-workers", type=int, default=8)
    ap.add_argument("--stage", default=None, help="run one stage (analysis.stages.STAGES); default: legacy run")
    ap.add_argument("--budget", action="append", type=int, default=None,
                    help="primary stage: restrict to these budgets (one process per budget spreads the evaluation)")
    ap.add_argument("--backbone", action="append", default=None, help="c5: restrict to these confirmatory backbones")
    ap.add_argument("--repr", action="append", default=None, help="coverage: restrict to these representations")
    ap.add_argument("--coverage-dir", action="append", default=None, help="coverage_link: coverage stage directories")
    a = ap.parse_args(argv)
    if a.stage is not None:
        return main_stage(a, argv)
    if a.pair_with and len(a.pair_with) > 1:
        raise SystemExit("the legacy run takes one --pair-with directory")
    a.pair_with = a.pair_with[0] if a.pair_with else None

    import torch
    from threadpoolctl import threadpool_limits

    from analysis import inference as I
    from analysis.config import REPO, Config, check_deferred, sha256_file, verify_freeze
    from analysis.pipeline import load_inputs

    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.time() - t_start:8.1f}s] {msg}", flush=True)

    tasks = ["stability", "band_gap"] if a.task == "both" else [a.task]
    cfg = Config.build(a.mode, tasks, split_key=a.split)
    check_deferred(cfg)  # confirm: refused while C5 / coverage are deferred ( finding 8)
    freeze = verify_freeze(cfg, a.freeze_dir or cfg.freeze_dir)
    log(f"mode={cfg.mode} part={cfg.part} tasks={tasks} freeze={freeze['status']}")
    dev = _device_guard(a.device)
    torch.set_num_threads(a.threads)
    I.check_p_formulas(cfg.y)
    manifest = {"mode": cfg.mode, "part": cfg.part, "tasks": tasks, "argv": sys.argv, "freeze": freeze,
                "device": dev, "threads": a.threads, "prereg_sha256": sha256_file(cfg.prereg_path),
                "code_sha256": {str(p.relative_to(REPO)): sha256_file(p)
                                for p in sorted((REPO / "src" / "analysis").glob("*.py"))},
                "timings": {}, "runs": {}, "_t_start": t_start}
    with threadpool_limits(a.threads):
        inp = load_inputs(cfg, log)
        if any("cgcnn" in cfg.controls_at(n, t) for n in cfg.budgets for t in cfg.tasks):
            inp.graphs = GraphStore(cfg, inp.part, a.graph_workers, log)
        manifest = execute(cfg, inp, out, a.device, manifest, log, pair_with=a.pair_with)
    log(f"done: looks={manifest['looks']} decisions={manifest['decisions']} wall={manifest['timings']['wall_s']}s; "
        f"missing runs={len(manifest['missing_run_ids'])}")
    return 0 if not manifest["missing_run_ids"] else 1


def main_stage(a, argv=None) -> int:
    """One stage (analysis.stages). Confirm: check_deferred + freeze verify first, then the stage's inputs only."""
    import torch
    from threadpoolctl import threadpool_limits

    from analysis import inference as I
    from analysis import stages as S
    from analysis.config import Config, check_deferred, verify_freeze

    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.time() - t_start:8.1f}s] {msg}", flush=True)

    tasks = ["stability", "band_gap"] if a.task == "both" else [a.task]
    split = S.split_for_stage(a.stage, a.split)
    cfg = S.stage_config(Config.build(a.mode, tasks, split_key=split), a.stage, a.budget)
    check_deferred(cfg)
    freeze = verify_freeze(cfg, a.freeze_dir or cfg.freeze_dir)
    log(f"stage={a.stage} mode={cfg.mode} part={cfg.part} split={cfg.split_key} tasks={tasks} "
        f"freeze={freeze['status']}")
    dev = _device_guard(a.device)
    torch.set_num_threads(a.threads)
    I.check_p_formulas(cfg.y)
    argv_rec = sys.argv if argv is None else ["analysis.run", *argv]
    if a.stage in ("primary", "fold_sensitivity", "random_reference"):
        from analysis.pipeline import load_inputs
        pair = None
        if a.stage != "primary":
            if not a.pair_with:
                raise SystemExit(f"--stage {a.stage} needs --pair-with <primary stage directory>")
            look = S.look_of(cfg)
            pd = S.primary_dirs(cfg, a.pair_with, tasks, look)
            dirs = {str(v[0]) for v in pd.values()}
            if len(dirs) != 1:
                raise SystemExit(f"--stage {a.stage}: tasks {tasks} are in different primary directories {sorted(dirs)};"
                                 f" run one task per invocation")
            pair = dirs.pop()
        manifest = {"mode": cfg.mode, "part": cfg.part, "tasks": tasks, "argv": argv_rec, "freeze": freeze,
                    "device": dev, "threads": a.threads, "_t_start": t_start,
                    "code_sha256": S.new_manifest(cfg, a.stage)["code_sha256"]}
        with threadpool_limits(a.threads):
            inp = load_inputs(cfg, log)
            if any("cgcnn" in cfg.controls_at(n, t) for n in cfg.budgets for t in cfg.tasks):
                inp.graphs = GraphStore(cfg, inp.part, a.graph_workers, log)
            manifest = execute(cfg, inp, out, a.device, manifest, log, pair_with=pair, stage=a.stage)
        log(f"done: stage {a.stage} wall={manifest['timings']['wall_s']}s; missing runs="
            f"{len(manifest['missing_run_ids'])}")
        return 0 if not manifest["missing_run_ids"] else 1
    man = S.new_manifest(cfg, a.stage, argv_rec, freeze, dev, a.threads)
    man["_t_start"] = t_start
    with threadpool_limits(a.threads):
        if a.stage == "assemble":
            man = S.assemble(cfg, out, man, log)
        else:
            if a.stage == "c3":
                bbs = list(cfg.family_backbones)
            elif a.stage == "c5":
                bbs = S.c5_backbones(cfg, a.backbone)
            elif a.stage == "coverage":
                reps = S.coverage_reprs(cfg, a.repr)
                bbs = [r for r in reps if r != S.COMPOSITION]
            else:
                bbs = []
            si = S.load_stage_inputs(cfg, a.stage, bbs, log)
            man["inputs"] = si.manifest
            if a.stage == "c3":
                man = S.run_c3(cfg, si, out, a.device, man, a.pair_with, log)
            elif a.stage == "c1":
                man = S.run_c1(cfg, si, out, man, a.pair_with, log)
            elif a.stage == "c5":
                man = S.run_c5(cfg, si, out, a.device, man, bbs, log)
            elif a.stage == "coverage":
                man = S.run_coverage(cfg, si, out, a.device, man, reps, log)
            elif a.stage == "coverage_link":
                man = S.run_coverage_link(cfg, si, out, man, a.pair_with, a.coverage_dir, log)
    man = S.finish_manifest(out, man)
    log(f"done: stage {a.stage} wall={man['timings']['wall_s']}s; covers {man.get('covers')}")
    return 0 if not man.get("missing_run_ids") else 1


CONTRAST_COLUMNS = ["contrast", "a", "b", "population", "group", "claim", "n_population", "n_positive", "prevalence",
                    "estimate", "ci_lo", "ci_hi", "margin", "p_up_unadjusted", "p_down_unadjusted", "p_up_holm",
                    "p_down_holm", "reading", "auroc_estimate", "auroc_ci_lo", "auroc_ci_hi", "auroc_undefined_draws"]


def execute(cfg, inp, out, device: str, manifest: dict, log=print, pair_with=None, stage=None) -> dict:
    """Units, looks (with the one-time pilot extension), outputs, run manifest. Used by main and by the fixtures.
    pair_with: the primary chemsys seed-0 run directory (required for the descriptive splits).
    stage: None (legacy run) or primary | fold_sensitivity | random_reference (recorded with the keys it covers)."""
    import numpy as np
    import torch

    from analysis import descriptive as D
    from analysis import inference as I
    from analysis import outputs as O
    from analysis.config import AnalysisError, check_deferred, sha256_file
    from analysis.pipeline import State, evaluate, run_id, run_unit
    t_start = time.time()
    out = Path(out)
    tasks = cfg.tasks
    if (inp.manifest.get("labels") or {}).get("source") == "REAL":
        check_deferred(cfg)
    if not cfg.readings_allowed and pair_with is None:
        raise AnalysisError(f"split {cfg.split_key} ({cfg.role}) is descriptive: --pair-with <seed-0 run dir> "
                            f"is required")
    manifest.setdefault("timings", {})
    manifest.setdefault("runs", {})
    manifest.update(mode=cfg.mode, part=cfg.part, prereg_sha256=sha256_file(cfg.prereg_path),
                    split={"key": cfg.split_key, "role": cfg.role, "type": cfg.split_type, "seed": cfg.split_seed,
                           "path": str(cfg.split_path), "sha256": cfg.split_sha,
                           "readings_allowed": cfg.readings_allowed},
                    pair_with=(str(pair_with) if pair_with else None), tasks=list(tasks))
    if stage is not None:
        want_role = {"primary": "primary", "fold_sensitivity": "fold_sensitivity",
                     "random_reference": "random_reference"}[stage]
        if cfg.role != want_role:
            raise AnalysisError(f"stage {stage} with split role {cfg.role}")
        if stage == "primary" and (cfg.c3 or cfg.c1):
            raise AnalysisError("stage primary excludes C3 / C1 (their own stages)")
        manifest["stage"] = stage
        manifest["covers"] = ([f"primary:{t}:n{n}" for t in tasks for n in cfg.budgets] if stage == "primary" else
                              [f"{stage}:{cfg.split_key}:{t}" if stage == "fold_sensitivity" else f"{stage}:{t}"
                               for t in tasks])
    if True:
        manifest["inputs"] = inp.manifest
        O.write_json(out / "labels_manifest.json", inp.manifest.get("labels", {}))
        st = State(cfg, inp)
        units_dir = out / "units"
        looks, evals_by_look = [], {}

        def run_replicates(reps):
            for r in reps:
                for f in range(cfg.n_folds):
                    for n in cfg.budgets:
                        run_unit(cfg, inp, st, n, f, r, device, units_dir, log)

        def evaluate_look(look, reps):
            t0 = time.time()
            d = out / look
            evs = {}
            for t in tasks:
                for n in cfg.budgets:
                    ev = evaluate(cfg, inp, st, t, n, reps, log)
                    evs[(t, n)] = ev
                    tag = f"{t}_n{n}"
                    ref_rows = {}
                    if t == "stability":
                        from analysis.pipeline import _ref_scores
                        ref_rows = _ref_scores(cfg, inp)
                    files = [{"path": f"{look}/scores_{tag}.npz",
                              "sha256": O.write_scores(d / f"scores_{tag}.npz", inp.part, reps, st.oof[(t, n)],
                                                       ref_rows)},
                             {"path": f"{look}/metrics_{tag}.csv", "sha256": O.write_csv(d / f"metrics_{tag}.csv",
                                                                                          ev["metrics"]),
                              "rows": len(ev["metrics"]),
                              "metric_columns": ["M", "auroc", "auroc_ci_lo", "auroc_ci_hi", "prevalence"]},
                             {"path": f"{look}/contrasts_{tag}.csv",
                              "sha256": O.write_csv(d / f"contrasts_{tag}.csv", ev["contrasts"],
                                                    columns=None if ev["contrasts"] else CONTRAST_COLUMNS),
                              "rows": len(ev["contrasts"]),
                              "metric_columns": ["estimate", "ci_lo", "ci_hi", "p_up_unadjusted", "p_down_unadjusted",
                                                 "n_positive", "prevalence", "auroc_estimate", "auroc_ci_lo",
                                                 "auroc_ci_hi"]},
                             {"path": f"{look}/bootstrap_{tag}.npz",
                              "sha256": O.save_npz(d / f"bootstrap_{tag}.npz", compress=True,
                                                   contrast=np.asarray(list(ev["dstar"]), dtype=str),
                                                   dstar=(np.stack(list(ev["dstar"].values())) if ev["dstar"]
                                                          else np.zeros((0, cfg.B))),
                                                   spec=np.asarray(list(ev["mstar"])),
                                                   mstar=np.stack(list(ev["mstar"].values())),
                                                   astar=np.stack([ev["astar"][k] for k in ev["mstar"]]))}]
                    files += [{"path": f"{look}/populations_{tag}.csv",
                               "sha256": O.write_csv(d / f"populations_{tag}.csv", ev["populations"],
                                                     columns=None if ev["populations"] else
                                                     ["task", "n", "population", "n_population", "n_positive",
                                                      "prevalence"]),
                               "rows": len(ev["populations"]),
                               "metric_columns": ["n_population", "n_positive", "prevalence"]}]
                    desc = {"c3_input_structure": ev["c3"], "c1_membership": ev["c1"]}
                    if cfg.rvc_backbones(n, t):  # descriptive_specs.random_vs_chemsys.scope
                        desc["random_vs_chemsys"] = D.random_vs_chemsys(cfg, inp.part, ev, pair_with, look)
                    if cfg.role == "fold_sensitivity":
                        desc["fold_sensitivity"] = D.fold_sensitivity(cfg, ev, ev["contrasts"], pair_with, look)
                    for kind, rows in desc.items():
                        if rows:
                            files.append({"path": f"{look}/{kind}_{tag}.csv",
                                          "sha256": O.write_csv(d / f"{kind}_{tag}.csv", rows), "rows": len(rows),
                                          "metric_columns": ["interaction", "interaction_ci_lo", "interaction_ci_hi"]
                                          if kind == "c1_membership" else ["estimate", "ci_lo", "ci_hi"]})
                    manifest["runs"][f"eval__{look}__{tag}"] = {"files": files}
                    manifest["timings"][f"eval__{look}__{tag}"] = {"bootstrap_s": ev["seconds_bootstrap"],
                                                                   "eval_s": ev["seconds_eval"]}
            ch = O.write_csv(d / "selection_choices.csv",
                             [{k: (v if k != "params" else str(sorted(v.items()))) for k, v in x.items()}
                              for x in st.choices])
            manifest["runs"][f"choices__{look}"] = {"files": [{"path": f"{look}/selection_choices.csv", "sha256": ch,
                                                               "rows": len(st.choices),
                                                               "metric_columns": ["inner_val_ap"]}]}
            readings = {}
            for ev in evs.values():
                for r in ev["readings"]:
                    readings[r["contrast"]] = r["reading"]
            decision = None
            fam = cfg.family_backbones
            have = cfg.readings_allowed and all(f"g:{b}" in readings and f"d_sp:{b}" in readings for b in fam)
            if have and cfg.mode in ("dryrun", "pilot"):
                br = I.pilot_branch(readings, fam)
                first = look == "first"
                decision = {"rule": "pilot_branch",
                            "reading": br["reading_first_look"] if first else br["branch"],
                            "parts": {"E1": br["E1_part_holds"], "E2": br["E2_part_holds"]},
                            "survivor": (None if first and br["extension_triggered"] else br["survivor"])}
                decision_detail = br
            elif have:
                sr = I.success_rule(readings, fam)
                decision = {"rule": "success_rule", "reading": "MET" if sr["success"] else "NOT_MET",
                            "parts": {"E1": sr["E1_all_d_sp_resolved"], "E2": sr["E2_all_benefit"]}, "survivor": None}
                decision_detail = sr
            elif not cfg.readings_allowed:
                decision_detail = {"note": f"split {cfg.split_key} ({cfg.role}) is descriptive: no readings, "
                                           f"no decision"}
            else:
                decision_detail = {"note": "decision needs both tasks"}
            if cfg.readings_allowed:
                rsha = O.write_json(d / "readings.json", {"look": look, "replicates": reps, "readings": [
                    r for ev in evs.values() for r in ev["readings"]], "decision": decision,
                    "decision_detail": decision_detail})
                manifest["runs"][f"readings__{look}"] = {"files": [{"path": f"{look}/readings.json",
                                                                    "sha256": rsha}]}
            elif any(ev["readings"] for ev in evs.values()):
                raise AnalysisError(f"split role {cfg.role} produced readings")
            if decision is not None:
                prim = [evs[(t, cfg.primary_n)] for t in tasks]
                stage = "confirm" if cfg.mode == "confirm" else "pilot"
                b = O.write_bundle(d, cfg, stage, look, prim, decision)
                manifest["runs"][f"bundle__{look}"] = {"files": [
                    {"path": f"{look}/bundle.json", "sha256": b["bundle_sha256"]},
                    {"path": f"{look}/draws.npz", "sha256": b["draws_sha256"]}]}
            manifest["timings"][f"look_{look}_s"] = round(time.time() - t0, 1)
            evals_by_look[look] = {"decision": decision, "detail": decision_detail, "readings": readings}
            log(f"[look {look}] decision {decision and decision['reading']}")
            return decision, decision_detail

        t0 = time.time()
        run_replicates(cfg.replicates_first)
        manifest["timings"]["units_first_s"] = round(time.time() - t0, 1)
        look = "final" if cfg.mode == "confirm" else "first"
        decision, detail = evaluate_look(look, cfg.replicates_first)
        looks.append(look)
        if cfg.mode != "confirm" and decision is not None and detail.get("extension_triggered") \
                and cfg.extension_allowed:
            log(f"[extension] INCONCLUSIVE branch contrasts {detail['inconclusive_branch_contrasts']}: adding "
                f"replicates {cfg.extension_replicates} (once)")
            t0 = time.time()
            run_replicates(cfg.extension_replicates)
            manifest["timings"]["units_extension_s"] = round(time.time() - t0, 1)
            evaluate_look("extension", cfg.replicates_first + cfg.extension_replicates)
            looks.append("extension")
        all_reps = cfg.replicates_first + (cfg.extension_replicates if "extension" in looks else [])
        expected = [run_id(t, n, f, r) for t in tasks for n in cfg.budgets for f in range(cfg.n_folds)
                    for r in all_reps]
        for rid, u in st.units.items():
            fi = u.get("file")
            manifest["runs"][rid] = {"files": [{**fi, "path": str(Path(fi["path"]).relative_to(out))}]}
        manifest["expected_run_ids"] = expected
        manifest["actual_run_ids"] = sorted(st.units)
        manifest["missing_run_ids"] = sorted(set(expected) - set(st.units))
        manifest["unexpected_run_ids"] = sorted(set(st.units) - set(expected))
        manifest["units"] = st.units
        manifest["looks"] = looks
        manifest["decisions"] = {k: v["decision"] for k, v in evals_by_look.items()}
        if inp.graphs is not None:
            manifest["timings"]["graph_build_s"] = round(inp.graphs.seconds_build, 1)
        if device.startswith("cuda"):
            manifest.setdefault("device", {})["max_memory_allocated_MiB"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        manifest["timings"]["execute_wall_s"] = round(time.time() - t_start, 1)
        manifest["timings"]["wall_s"] = round(time.time() - manifest.get("_t_start", t_start), 1)
        manifest.pop("_t_start", None)
        O.write_json(out / "run_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
