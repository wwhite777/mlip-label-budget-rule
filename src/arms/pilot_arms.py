"""arms-a3: PILOT-only correction inputs; accepted numerical input arrays.

Gate helpers remain for the inherited synthetic tests, but are never run here.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
pilot = ROOT / "src/pilot"
sys.path[:0] = [str(ROOT / "src"), str(pilot)]

from pilot_data import (FORMATION, HULL, ConfirmLabelError, assert_pilot_ids, composition_basis,
                        load_metadata, load_pilot_labels, load_pilot_part)
from analysis import metrics, models
from analysis.draws import draw_labels, inner_split, outer_train_test
from analysis.features import _align, composition_features, load_registry, references
from posthoc.s9_additions import h_mp

CHECKPOINTS = ("mace_medium", "chgnet", "sevenn", "mattersim_5m", "orb")
G = None  # read-only arrays inherited by fork workers; each worker uses one compute thread


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def load_inputs():
    cfg, universe, partition = load_metadata()
    part = load_pilot_part(cfg, universe, partition)
    assert_pilot_ids(part.ids, part.part_of)
    elements, basis = composition_basis(cfg)
    comp = composition_features(part.formula, elements)
    labels, access = load_pilot_labels(ROOT / cfg.y["population"]["summary_file"]["path"], part.ids, part.part_of)
    registry = load_registry(cfg)
    refs, ref_info = {}, {}
    for bb in CHECKPOINTS:
        # Registry names the correct CHGNet _initial_nomp2020 file.
        row = {k: registry[bb][k] for k in ("ref_sp", "ref_sp_sha256")}
        if bb == "chgnet":
            assert Path(row["ref_sp"]).name == "zero_label_sp_chgnet_initial_nomp2020.csv.gz"
        rr = references(cfg, part, bb, {bb: row})
        refs[bb], ref_info[bb] = rr["scores"], rr["info"]
    hulls = np.column_stack([refs[b]["r_sp"]["e_hull"] for b in CHECKPOINTS])
    if not np.isfinite(hulls).all():
        raise ValueError("missing reference score")
    inp = SimpleNamespace(cfg=cfg, part=part, labels=labels, comp=comp, refs=refs, hulls=hulls, feats={})
    inp.hmp = h_mp(inp)  # reuse the frozen cross-check of the common fixed MP hull
    for bb in CHECKPOINTS:
        path = ROOT / f"cache/features/wbm_full_{bb}_initial.npz"
        with np.load(path, allow_pickle=False) as z:
            if str(z["structures_source"]) != "initial":
                raise ValueError("non-initial features")
            failed = set(z["failed_ids"].astype(str)) & set(part.ids)
            if failed:
                raise ValueError(f"failed PILOT features: {bb} {len(failed)}")
            inp.feats[bb] = np.asarray(z["feats"], dtype=np.float32)[_align(z["mids"], part, bb)]
        if not np.isfinite(inp.feats[bb]).all():
            raise ValueError(f"non-finite PILOT features: {bb}")
    inp.manifest = {"label_access": access, "composition_basis": dict(basis, elements=elements),
                    "references": ref_info, "part_n": part.n, "systems_n": len(part.systems),
                    "draw_rng": cfg.draw_rng, "inner_rng": cfg.inner_rng, "inner_fraction": cfg.inner_frac,
                    "hist_grid": cfg.head_grid("hist_gbm"), "device": "CPU", "threads_per_worker": 1}
    return inp


def split(n, f, r):
    inp = G
    draw = draw_labels(inp.cfg, inp.part, f, r, n)
    tr, va = inner_split(inp.cfg, inp.part, draw, f, r)
    _, test = outer_train_test(inp.part, f)
    return draw, tr, va, test


def label_frame(idx, draw=None):
    """All label lookups go through the tested PILOT assertion and draw guard."""
    return G.labels.lookup(G.part.ids[idx], allowed_ids=None if draw is None else G.part.ids[draw])


def perturb_features(X, seed):
    """Relative Gaussian noise; preserve the frozen extractor's float32 dtype."""
    return (X * (1 + 1e-7 * np.random.default_rng(seed).standard_normal(X.shape))).astype(X.dtype)


def gate_point_pass(params, difference, noise_floor):
    tolerance = max(1e-4, 2 * noise_floor)
    return abs(difference) <= tolerance and (params.get("C") != 0.01 or abs(difference) <= 1e-4)


def gate_unit(path):
    t0 = time.monotonic()
    records = pd.read_csv(path).to_dict("records")
    match = re.fullmatch(r"stability__n(\d+)__f(\d+)__r(\d+)", Path(path).stem)
    n, f, r = map(int, match.groups())
    draw, tr, va, _ = split(n, f, r)
    ytr, yva = label_frame(tr, draw).stable.to_numpy(), label_frame(va, draw).stable.to_numpy()
    rows = []
    for rec in records:
        if rec["menu"] not in ("heads", "residual") or rec["type"] not in ("logistic", "hist_gbm"):
            continue
        bb, kind = rec["backbone"], rec["type"]
        params = json.loads(rec["params"])
        for variant in (["regenerated", "planted_mattersim_for_mace"] if bb == "mace_medium" else ["regenerated"]):
            base = G.feats["mattersim_5m" if variant != "regenerated" else bb]
            assert base.shape == G.feats[bb].shape
            def evaluate(features):
                X = features
                if rec["menu"] == "residual":
                    X = np.column_stack([X.astype(np.float64), G.refs[bb]["r_sp"]["e_hull"]])
                with threadpool_limits(1):
                    model = models.make(kind, params, G.cfg.y, "cpu").fit(X[tr], ytr)
                    return metrics.ap(yva, model.score(X[va]))
            current = evaluate(base)
            noisy = [evaluate(perturb_features(base, seed)) for seed in range(5)]
            floor = max(abs(a-current) for a in noisy)
            old = float(rec["inner_val_ap"])
            rows.append({"variant": variant, "unit": Path(path).name, "n": n, "fold": f, "draw": r,
                     "menu": rec["menu"], "checkpoint": bb, "model": kind,
                     "grid_index": rec["grid_index"], "params": rec["params"],
                     "frozen_inner_val_ap": old, "regenerated_inner_val_ap": current,
                     "difference": current-old, "abs_difference": abs(current-old),
                     "within_1e-4": abs(current-old) <= 1e-4, "noise_floor": floor,
                     "noise_tolerance": max(1e-4, 2*floor),
                     "point_pass": gate_point_pass(params, current-old, floor),
                     **{f"perturbed_ap_seed{s}": a for s, a in enumerate(noisy)}})
    return rows, time.monotonic()-t0


def run_gate(workers):
    paths = sorted((ROOT / "result/s4_pilot/pilot/units").glob("stability__n3000__*.csv"))
    rows = []
    with futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        jobs = {pool.submit(gate_unit, p): p for p in paths}
        for job in futures.as_completed(jobs):
            new, elapsed = job.result()
            rows.extend(new)
            print(f"gate {jobs[job].name}: {len(new)} grid points, {elapsed:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values(["unit", "menu", "checkpoint", "model", "grid_index"])
    frame.to_csv(OUT / "equivalence_gate.csv", index=False)
    original = frame.loc[frame.variant == "regenerated"]
    log = original.loc[original.model == "logistic"]
    plant = frame.loc[(frame.variant != "regenerated") & (frame.model == "logistic")]
    assert len(log) == 240 and (original.model == "hist_gbm").sum() == 240
    assert len(plant) == 120
    ok = bool(log.point_pass.all() and not plant.point_pass.all())
    summary = {
        "status": "PASS" if ok else "FAIL",
        "units": len(paths), "logistic_count": len(log), "logistic_max_abs_difference": log.abs_difference.max(),
        "hist_gbm_count": int((original.model == "hist_gbm").sum()),
        "logistic_count_above_1e-4": int((~log["within_1e-4"]).sum()),
        "logistic_failed_points": int((~log.point_pass).sum()),
        "c001_count": int(log.params.map(lambda s: json.loads(s)["C"] == 0.01).sum()),
        "c001_failed_points": int((~log.loc[log.params.map(lambda s: json.loads(s)["C"] == 0.01), "point_pass"]).sum()),
        "planted_logistic_count": len(plant), "planted_failed_points": int((~plant.point_pass).sum()),
        "perturbation": "float32 X*(1+1e-7*N(0,1)), fresh default_rng(seed), seeds 0..4; hull columns unchanged",
        "rule": "C=0.01 gap <=1e-4 AND all logistic gaps <=max(1e-4,2*point noise floor); planted swap fails",
        "by_menu_checkpoint_model": []}
    for key, group in frame.groupby(["variant", "menu", "checkpoint", "model"]):
        summary["by_menu_checkpoint_model"].append(dict(zip(("variant", "menu", "checkpoint", "model"), key),
            count=len(group), max_abs_difference=group.abs_difference.max(),
            mean_abs_difference=group.abs_difference.mean(), count_nonzero=int((group.abs_difference > 0).sum()),
            count_above_tolerance=int((group.abs_difference > 1e-4).sum()), max_noise_floor=group.noise_floor.max()))
    dump(OUT / "equivalence_gate_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary["status"] == "PASS"


def main():
    global G
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16; one compute thread per worker")
    t0 = time.monotonic()
    G = load_inputs()
    dump(OUT / "run_manifest.json", G.manifest)
    if args.command == "gate":
        ok = run_gate(args.workers)
        dump(OUT / "gate_run.json", {"seconds": time.monotonic()-t0, "exit_code": 0 if ok else 2})
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
