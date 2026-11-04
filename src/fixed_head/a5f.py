"""fixed_head / fixed-modal A5, PILOT only; reuse arms and stability unchanged."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT.parent / "arms"
stability = OUT.parent / "stability"
sys.path.insert(0, str(stability))
import stability_check as stability

p, arms = stability.p, stability.arms
dump, sha = stability.dump, stability.sha
perturb_inputs, perturbation_trace = stability.perturb_inputs, stability.perturbation_trace
WORKERS = 8
TASK_START = datetime(2026, 10, 22, 6, 3, 15, tzinfo=timezone.utc).timestamp()
DEADLINE = TASK_START + 75 * 60
FIT_DEADLINE = DEADLINE - 120
THRESHOLD = 0.005
R0STAR = 0.775379  # task-specified comparator; do not replace by the unrounded value
FIXED = None


def choose_fixed(paths, grid):
    """Count every eligible unit, breaking a modal tie by frozen grid order."""
    counts = [0] * len(grid)
    keys = set()
    for path in sorted(paths):
        info = json.loads(Path(path).read_text())
        if info["arm"] != "A5" or info["n"] != 10000:
            raise ValueError(f"ineligible modal-selection unit: {path}")
        key = (info["checkpoint"], info["fold"], info["draw"])
        if key in keys:
            raise ValueError(f"duplicate modal-selection unit: {key}")
        keys.add(key)
        counts[grid.index(info["chosen"]["hist_gbm"]) ] += 1
    if not keys:
        raise ValueError("no modal-selection units")
    selected = max(range(len(grid)), key=counts.__getitem__)
    return counts, selected, keys


def select_fixed(write=True):
    manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    grid = manifest["hist_grid"]
    paths = sorted((SOURCE / "units").glob("A5__*__n10000__*.json"))
    counts, selected, keys = choose_fixed(paths, grid)
    expected = {(bb, f, r) for bb in p.CHECKPOINTS for f in range(5) for r in range(3)}
    if keys != expected:
        raise AssertionError("modal units do not cover all five checkpoints/folds/three draws")
    rows = [dict(grid_index=i, **params, count=counts[i], fraction=counts[i] / len(paths),
                 selected=i == selected) for i, params in enumerate(grid)]
    if write:
        pd.DataFrame(rows).to_csv(OUT / "grid_counts.csv", index=False)
        dump(OUT / "fixed_setting.json", dict(params=grid[selected], grid_index=selected,
             counts=counts, total_units=len(paths), grid=grid,
             rule="all arms-a3 A5 n10000 units; modal count; ties by frozen grid order"))
    return grid[selected], rows, paths


def verify_inputs():
    prior = json.loads((SOURCE / "inputs_sha256.json").read_text())
    names = [name for name in prior if name.startswith((
        "cache/", "result/reference/", "result/splits/", "research/", "src/"))]
    names += ["src/pilot/pilot_data.py",
              "src/pilot/rereference.py"]
    hashes = {}
    for name in sorted(set(names)):
        value = sha(ROOT / name)
        if value != prior[name]:
            raise ValueError(f"arms input hash mismatch: {name}")
        hashes[name] = value
    extra = [SOURCE / "inputs_sha256.json",
             SOURCE / "pilot_arms.py", SOURCE / "run_arms.py", SOURCE / "pilot_arms_ap.csv",
             SOURCE / "run_manifest.json", stability / "stability_check.py",
             stability / "test_stability_check.py"]
    for bb in p.CHECKPOINTS:
        for n in (3000, 10000):
            for r in range(3):
                for f in range(5):
                    prefix = SOURCE / "units" / f"A5__{bb}__n{n}__f{f}__r{r}"
                    extra.extend((prefix.with_suffix(".json"), prefix.with_suffix(".npz")))
    for path in extra:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    dump(OUT / "inputs_sha256.json", hashes)
    return hashes


def unit(bb, n, f, r):
    """A5's identical float64 inputs and all-draw refit, without inner selection."""
    started = time.monotonic()
    draw, tr, va, test = p.split(n, f, r)
    if len(draw) != n or not len(tr) or not len(va):
        raise AssertionError("invalid draw or empty inner split")
    y = p.label_frame(draw, draw).stable.to_numpy()
    X = np.column_stack([p.G.feats[bb].astype(np.float64), p.G.hulls])
    flags = []
    with threadpool_limits(1):
        if np.unique(y).size == 1:
            score = np.full(len(test), float(y[0]))
            flags.append("draw_single_class_constant_score")
        else:
            score = p.models.make("hist_gbm", FIXED, p.G.cfg.y, "cpu").fit(
                X[draw], y).score(X[test])
    if not np.isfinite(score).all():
        raise ValueError("non-finite A5f score")
    prefix = p.OUT / "units" / f"A5f__{bb}__n{n}__f{f}__r{r}"
    np.savez_compressed(str(prefix) + ".npz", test_idx=test, material_id=p.G.part.ids[test],
                        model=np.array(["hist_gbm"]), score=score[None, :], draw_idx=draw,
                        inner_train_idx=tr, inner_val_idx=va)
    info = dict(arm="A5f", checkpoint=bb, n=n, fold=f, draw=r, best="hist_gbm",
                chosen={"hist_gbm": FIXED}, inner_selection_used=False,
                fit_label_count=len(y), n_inner_train=len(tr), n_inner_val=len(va),
                flags=flags, seconds=time.monotonic() - started)
    dump(prefix.with_suffix(".json"), info)
    return info


def unit_args(args):
    return unit(*args)


def run_variant(name, inp, jobs):
    p.G, p.OUT = inp, OUT / "runs" / name
    (p.OUT / "units").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pool = mp.get_context("fork").Pool(WORKERS)
    completed = []
    try:
        pending = pool.imap_unordered(unit_args, jobs, chunksize=1)
        while len(completed) < len(jobs):
            if time.time() >= FIT_DEADLINE:
                raise TimeoutError("fixed_head fitting deadline reached")
            try:
                info = pending.next(timeout=1)
            except mp.TimeoutError:
                continue
            completed.append(info)
            if len(completed) % 10 == 0 or len(completed) == len(jobs):
                print(f"{name}: {len(completed)}/{len(jobs)} units, "
                      f"{time.monotonic()-started:.1f}s", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        dump(p.OUT / "timing.json", dict(seconds=time.monotonic() - started,
             completed_units=len(completed), planned_units=len(jobs)))
    keys = sorted({(bb, n, r) for bb, n, f, r in jobs})
    rows = [row for bb, n, r in keys for row in arms.assemble("A5f", bb, n, r)]
    if len(rows) != len(keys):
        raise AssertionError("incomplete pooled AP coverage")
    pd.DataFrame(rows).to_csv(p.OUT / "pooled_ap.csv", index=False)
    return rows


def write_pilot(rows):
    old = pd.read_csv(SOURCE / "pilot_arms_ap.csv", float_precision="round_trip")
    frame = pd.DataFrame(rows).rename(columns={"AP": "M"})
    ref = old.loc[(old.arm == "A5") & (old.model == "hist_gbm"),
                  ["checkpoint", "n", "draw", "AP"]].rename(columns={"AP": "A5_inner_selected_M"})
    frame = frame.merge(ref, on=["checkpoint", "n", "draw"], validate="one_to_one", how="left")
    if len(frame) != 30 or frame.A5_inner_selected_M.isna().any():
        raise AssertionError("incomplete performance or paired arms reference")
    frame["draw"] = frame.draw.astype(str)
    means = frame.groupby(["checkpoint", "n"], as_index=False)[["M", "A5_inner_selected_M"]].mean()
    means["arm"], means["model"], means["draw"] = "A5f", "hist_gbm", "mean"
    frame = pd.concat([frame, means], ignore_index=True)
    frame["design"], frame["part"] = "P", "PILOT"
    frame["R0_star_fixed"] = R0STAR
    frame["M_minus_R0star"] = frame.M - R0STAR
    frame["M_minus_A5_inner_selected"] = frame.M - frame.A5_inner_selected_M
    cols = ["arm", "model", "design", "part", "checkpoint", "n", "draw", "M",
            "R0_star_fixed", "M_minus_R0star", "A5_inner_selected_M", "M_minus_A5_inner_selected"]
    frame = frame.sort_values(["n", "checkpoint", "draw"])[cols]
    frame.to_csv(OUT / "a5f_pilot.csv", index=False)
    return frame


def write_stability(results):
    rows = []
    for bb in ("mace_medium", "orb"):
        base = next(row for row in results["unperturbed"] if
                    (row["checkpoint"], row["n"], row["draw"]) == (bb, 10000, 0))
        row = dict(arm="A5f", model="hist_gbm", design="P", part="PILOT", checkpoint=bb,
                   n=10000, draw=0, M_0=base["AP"], threshold=THRESHOLD)
        shifts = []
        for seed in range(3):
            key = f"seed{seed}"
            if key in results:
                current = next(item for item in results[key] if item["checkpoint"] == bb)
                shift = current["AP"] - base["AP"]
                row.update({f"M_seed{seed}": current["AP"], f"shift_seed{seed}": shift})
                shifts.append(abs(shift))
        row.update(max_abs_shift=max(shifts) if shifts else None, completed_seeds=len(shifts),
                   verdict=("PASS" if max(shifts) <= THRESHOLD else "FAIL")
                   if len(shifts) == 3 else "INCOMPLETE")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "stability_check.csv", index=False)
    return rows


def write_summary(grid, stability, pilot):
    lines = ["| Grid index | learning_rate | max_leaf_nodes | Count | Fixed |",
             "|---:|---:|---:|---:|:---:|"]
    for row in grid:
        lines.append(f"| {row['grid_index']} | {row['learning_rate']} | {row['max_leaf_nodes']} | "
                     f"{row['count']} | {'yes' if row['selected'] else ''} |")
    lines += ["", "| Source | M₀ | M seed 0 | M seed 1 | M seed 2 | Max abs shift | Verdict |",
              "|---|---:|---:|---:|---:|---:|:---:|"]
    for row in stability:
        values = " | ".join(f"{row[f'M_seed{s}']:.9f}" for s in range(3))
        lines.append(f"| {row['checkpoint']} | {row['M_0']:.9f} | {values} | "
                     f"{row['max_abs_shift']:.9f} | {row['verdict']} |")
    lines += ["", "| n | Source | M draw 0 | M draw 1 | M draw 2 | Mean M | Mean − R0* | A5 mean | Mean − A5 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in (3000, 10000):
        for bb in p.CHECKPOINTS:
            block = pilot.loc[(pilot.n == n) & (pilot.checkpoint == bb)].set_index("draw")
            mean = block.loc["mean"]
            values = " | ".join(f"{block.loc[str(r), 'M']:.9f}" for r in range(3))
            lines.append(f"| {n} | {bb} | {values} | {mean.M:.9f} | {mean.M_minus_R0star:+.9f} | "
                         f"{mean.A5_inner_selected_M:.9f} | {mean.M_minus_A5_inner_selected:+.9f} |")
    passed = all(row["verdict"] == "PASS" for row in stability)
    verdict = "PASS" if passed else "FAIL"
    consequence = ("A5f meets the specified numerical-stability condition for F1/F2."
                   if passed else "A5f fails the numerical-stability condition for F1/F2; all affected readings "
                   "must remain descriptive.")
    lines += ["", f"A5f fixes the modal point across all 75 arms-a3 A5 n = 10,000 PILOT units "
              f"(all five checkpoints, five folds and draws 0–2), retaining the frozen HistGBM defaults "
              f"and fitting once on all n draw labels without inner selection. Its unchanged C05 "
              f"criterion is max abs(M_s − M₀) ≤ 0.005 for every tested source: overall {verdict}. "
              f"{consequence} Design P AP is pooled over all five held-out folds (42,799 PILOT rows); "
              f"performance means average the three paired draws, with R0* fixed at the requested 0.775379. "
              f"The CSV includes per-draw and mean differences to both comparators. stability's perturbation function "
              f"is imported unchanged: seeds 0–2, one default_rng per seed, relative 1e-7 Gaussian noise on all "
              f"encoders in checkpoint order then hulls, encoders recast to float32 and hulls float64; fractions "
              f"remain bit-identical. All fitting and evaluation uses the tested PILOT-only reader and lookup guard; "
              f"zero CONFIRM label rows were read. These development results do not by themselves freeze an arm."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    global FIXED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("count", "run"))
    args = parser.parse_args()
    FIXED, grid_rows, _ = select_fixed()
    if args.command == "count":
        print((OUT / "grid_counts.csv").read_text(), flush=True)
        return 0
    started = time.monotonic()
    verify_inputs()
    base = p.load_inputs()
    p.assert_pilot_ids(base.part.ids, base.part.part_of)
    if base.cfg.head_grid("hist_gbm") != [dict(learning_rate=row["learning_rate"],
            max_leaf_nodes=row["max_leaf_nodes"]) for row in grid_rows]:
        raise AssertionError("source grid differs from frozen config")
    for array in [base.comp, base.hulls, *base.feats.values()]:
        array.setflags(write=False)
    dump(OUT / "run_manifest.json", dict(base.manifest, workers=WORKERS, device="CPU",
         compute_threads_per_worker=1, threadpools=threadpool_info(), fixed_params=FIXED,
         hist_max_iter=base.cfg.y["heads"]["hist_gbm"]["max_iter"], inner_selection_used=False,
         task_start_epoch=TASK_START, deadline_epoch=DEADLINE, threshold=THRESHOLD,
         perturbation_code="unchanged stability/stability_check.py:perturb_inputs",
         comparator_R0star=R0STAR, budgets=[3000, 10000], draws=[0, 1, 2]))
    jobs = [(bb, n, f, r) for n in (3000, 10000) for bb in p.CHECKPOINTS
            for r in range(3) for f in range(5)]
    results, traces = {}, []
    try:
        results["unperturbed"] = run_variant("unperturbed", base, jobs)
        pilot = write_pilot(results["unperturbed"])
        write_stability(results)
        for seed in range(3):
            noisy = perturb_inputs(base, seed)
            traces.append(perturbation_trace(base, noisy, seed))
            dump(OUT / "perturbation_trace.json", traces)
            stability_jobs = [(bb, 10000, f, 0) for bb in ("mace_medium", "orb") for f in range(5)]
            results[f"seed{seed}"] = run_variant(f"seed{seed}", noisy, stability_jobs)
            stability = write_stability(results)
            del noisy
        write_summary(grid_rows, stability, pilot)
        print(json.dumps(stability, indent=2), flush=True)
    finally:
        dump(OUT / "run_status.json", dict(seconds=time.monotonic() - started,
             complete_variants=list(results), complete=len(results) == 4,
             confirm_label_rows_read=base.manifest["label_access"]["confirm_label_rows_read"]))
    return 0  # a numerical FAIL is a completed scientific check


if __name__ == "__main__":
    with threadpool_limits(1):
        raise SystemExit(main())
