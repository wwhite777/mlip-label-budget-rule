"""stability / C05: full pooled-OoF numerical check, PILOT only.

Import the accepted arms-a3 code unchanged; redirect its output directory only.
One RNG per perturbation seed consumes encoder arrays in CHECKPOINTS order,
then the row-major five-column hull matrix. Composition is never perturbed.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT.parent / "arms"
sys.path.insert(0, str(SOURCE))
import pilot_arms as p
import run_arms as arms

ARMS = (("A4", "ensemble"), ("A5", "mace_medium"), ("A5", "orb"))
WORKERS = 8  # one compute thread each; reserve room for parent/pool management
TASK_START = datetime(2026, 10, 22, 5, 51, 51, tzinfo=timezone.utc).timestamp()
DEADLINE = TASK_START + 75 * 60
FIT_DEADLINE = DEADLINE - 120
THRESHOLD = 0.005


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x: x.item()) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_inputs():
    """Pin used inputs against arms-a3; hash new evidence without changing it."""
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
             SOURCE / "pilot_arms.py", SOURCE / "run_arms.py", SOURCE / "noise_floor.json",
             SOURCE / "pilot_arms_ap.csv", SOURCE / "run_manifest.json"]
    for arm, bb in ARMS:
        for fold in range(5):
            prefix = SOURCE / "units" / f"{arm}__{bb}__n10000__f{fold}__r0"
            extra.extend((prefix.with_suffix(".json"), prefix.with_suffix(".npz")))
    for path in extra:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    dump(OUT / "inputs_sha256.json", hashes)
    return hashes


def perturb_inputs(base, seed):
    rng = np.random.default_rng(seed)
    feats = {}
    for bb in p.CHECKPOINTS:
        X = base.feats[bb]
        if X.dtype != np.float32:
            raise TypeError(f"unexpected encoder dtype: {bb} {X.dtype}")
        feats[bb] = (X * (1 + 1e-7 * rng.standard_normal(X.shape))).astype(np.float32)
    if base.hulls.dtype != np.float64:
        raise TypeError(f"unexpected hull dtype: {base.hulls.dtype}")
    hulls = base.hulls * (1 + 1e-7 * rng.standard_normal(base.hulls.shape))
    refs = {bb: {**base.refs[bb], "r_sp": {**base.refs[bb]["r_sp"], "e_hull": hulls[:, i]}}
            for i, bb in enumerate(p.CHECKPOINTS)}
    return SimpleNamespace(**{**vars(base), "feats": feats, "hulls": hulls, "refs": refs})


def perturbation_trace(base, noisy, seed):
    if noisy.comp.tobytes() != base.comp.tobytes():
        raise AssertionError("composition fractions changed")
    changed = {bb: int(np.count_nonzero(base.feats[bb] != noisy.feats[bb])) for bb in p.CHECKPOINTS}
    if not all(changed.values()) or np.array_equal(base.hulls, noisy.hulls):
        raise AssertionError("perturbation did not change every input block")
    return dict(seed=seed, composition_bit_identical=True, composition_sha256=array_sha(base.comp),
                features_changed_values=changed, hulls_changed_values=int(np.count_nonzero(base.hulls != noisy.hulls)),
                feature_sha256={bb: array_sha(noisy.feats[bb]) for bb in p.CHECKPOINTS},
                hull_sha256=array_sha(noisy.hulls), feature_dtype="float32", hull_dtype=str(noisy.hulls.dtype))


def run_variant(name, inp):
    p.G, p.OUT = inp, OUT / "runs" / name
    (p.OUT / "units").mkdir(parents=True, exist_ok=True)
    jobs = [(arm, bb, 10000, fold, 0) for arm, bb in ARMS for fold in range(5)]
    started = time.monotonic()
    pool = mp.get_context("fork").Pool(WORKERS)
    completed = []
    try:
        pending = pool.imap_unordered(arms.unit_args, jobs, chunksize=1)
        while len(completed) < len(jobs):
            if time.time() >= FIT_DEADLINE:
                raise TimeoutError("stability fitting deadline reached; preserve partial units")
            try:
                info = pending.next(timeout=1)
            except mp.TimeoutError:
                continue
            completed.append(info)
            print(f"{name}: {info['arm']}/{info['checkpoint']} fold {info['fold']} "
                  f"{len(completed)}/{len(jobs)} ({info['seconds']:.1f}s)", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        dump(p.OUT / "timing.json", dict(seconds=time.monotonic() - started,
             completed_units=len(completed), planned_units=len(jobs)))
    rows = [row for arm, bb in ARMS for row in arms.assemble(arm, bb, 10000, 0)
            if row["model"] == "hist_gbm"]
    if len(rows) != 3:
        raise AssertionError("incomplete pooled arm coverage")
    pd.DataFrame(rows).to_csv(p.OUT / "pooled_ap.csv", index=False)
    return {(row["arm"], row["checkpoint"]): row for row in rows}


def stored_pooled(arm, bb, inp):
    """Recompute the stored draw-0 AP from all five accepted score files."""
    score = np.full(inp.part.n, np.nan)
    y = inp.labels.lookup(inp.part.ids).stable.to_numpy()
    for fold in range(5):
        prefix = SOURCE / "units" / f"{arm}__{bb}__n10000__f{fold}__r0"
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as z:
            idx = z["test_idx"]
            np.testing.assert_array_equal(z["material_id"], inp.part.ids[idx])
            if not np.isnan(score[idx]).all():
                raise AssertionError("overlapping stored test folds")
            score[idx] = z["score"][list(z["model"]).index("hist_gbm")]
    if not np.isfinite(score).all():
        raise AssertionError("incomplete stored test coverage")
    return p.metrics.ap(y, score)


def write_report(results, stored, r0star):
    rows = []
    for arm, bb in ARMS:
        key = (arm, bb)
        if "unperturbed" not in results:
            continue
        base = results["unperturbed"][key]
        m0 = base["AP"]
        row = dict(arm=arm, model="hist_gbm", checkpoint=bb, design="P", part="PILOT", n=10000, draw=0,
                   M_0=m0, arms_draw0_AP=stored[key], unperturbed_abs_difference=abs(m0 - stored[key]),
                   unperturbed_within_1e12=abs(m0 - stored[key]) <= 1e-12, R0_star_fixed=r0star)
        shifts, changes = [], []
        for seed in range(3):
            name = f"seed{seed}"
            if name not in results:
                continue
            current = results[name][key]
            shift = current["AP"] - m0
            old_params, new_params = json.loads(base["selected_params"]), json.loads(current["selected_params"])
            folds = [fold for fold in map(str, range(5)) if new_params[fold] != old_params[fold]]
            row.update({f"M_seed{seed}": current["AP"], f"shift_seed{seed}": shift,
                        f"grid_changed_folds_seed{seed}": json.dumps(folds),
                        f"A5_minus_R0star_shift_seed{seed}": shift if arm == "A5" else None})
            shifts.append(abs(shift))
            changes.append(bool(folds))
        row["max_abs_shift"] = max(shifts) if shifts else None
        row["grid_changed_any_fold"] = any(changes)
        row["completed_perturbation_seeds"] = len(shifts)
        row["verdict"] = ("PASS" if max(shifts) <= THRESHOLD else "FAIL") if len(shifts) == 3 else "INCOMPLETE"
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "stability_check.csv", index=False)
    lines = ["| Arm / checkpoint | M₀ | M seed 0 | M seed 1 | M seed 2 | max |ΔAP| | grid changed | |M₀−arms| | Verdict |",
             "|---|---:|---:|---:|---:|---:|:---:|---:|:---:|"]
    # Escape the absolute-value bars in the table headings.
    lines[0] = lines[0].replace("max |ΔAP|", "max abs ΔAP").replace("|M₀−arms|", "abs(M₀−arms)")
    for row in rows:
        vals = [f"{row.get(f'M_seed{s}', float('nan')):.12f}" for s in range(3)]
        maximum = row["max_abs_shift"]
        lines.append(f"| {row['arm']}/HistGBM {row['checkpoint']} | {row['M_0']:.12f} | " + " | ".join(vals) +
                     f" | {maximum if maximum is not None else float('nan'):.12f} | "
                     f"{'yes' if row['grid_changed_any_fold'] else 'no'} | {row['unperturbed_abs_difference']:.3g} | {row['verdict']} |")
    complete = len(rows) == 3 and all(row["completed_perturbation_seeds"] == 3 for row in rows)
    verdicts = "; ".join(f"{r['arm']}/{r['checkpoint']}: {r['verdict']}" for r in rows) or "no complete baseline"
    paragraph = (f"PILOT only, design P, n = 10,000 labels per outer learner, draw r = 0; AP is pooled over all five "
                 f"held-out folds (42,799 rows), with complete inner-grid selection and refit rerun for each input variant. "
                 f"C05 is applied unchanged: PASS iff max abs ΔAP ≤ 0.005; {verdicts}. "
                 f"All computations {'completed' if complete else 'have not completed'}. "
                 f"Baseline reproduction tolerance is 1e-12. Perturbations use one numpy default_rng(seed), seeds 0–2, "
                 f"consuming the five full encoder matrices in frozen checkpoint order "
                 f"(mace_medium, chgnet, sevenn, mattersim_5m, orb), then the row-major hull matrix, "
                 f"multiplying each value by 1 + 1e-7·N(0,1); encoders are recast to float32, hulls remain float64. "
                 f"Composition fractions are bit-identical. The fixed strongest measured zero-shot comparator "
                 f"R0* = {r0star:.12f} is evaluated once on original hulls; each A5 − R0* shift equals the signed "
                 f"M shift reported in the CSV. No CONFIRM label rows were read. A FAIL requires descriptive status "
                 f"before freeze under C05; PASS supports freezing with respect to this numerical criterion only.")
    (OUT / "stability_check_summary.md").write_text("\n".join(lines) + "\n\n" + paragraph + "\n")
    return rows


def main():
    started = time.monotonic()
    verify_inputs()
    base = p.load_inputs()
    p.G = base
    p.assert_pilot_ids(base.part.ids, base.part.part_of)
    for array in [base.comp, base.hulls, *base.feats.values()]:
        array.setflags(write=False)
    dump(OUT / "run_manifest.json", dict(base.manifest, workers=WORKERS, device="CPU",
         compute_threads_per_worker=1, threadpools=threadpool_info(), task_start_epoch=TASK_START,
         deadline_epoch=DEADLINE, criterion="max_s abs(M_s-M_0) <= 0.005; seeds 0,1,2", threshold=THRESHOLD,
         source_code="unchanged imports of arms pilot_arms.py and run_arms.py"))
    stored_frame = pd.read_csv(SOURCE / "pilot_arms_ap.csv", float_precision="round_trip")
    stored = {}
    for arm, bb in ARMS:
        match = stored_frame.loc[(stored_frame.arm == arm) & (stored_frame.model == "hist_gbm") &
            (stored_frame.checkpoint == bb) & (stored_frame.n == 10000) & (stored_frame.draw == 0)]
        if len(match) != 1:
            raise AssertionError("stored AP row is not unique")
        stored[(arm, bb)] = float(match.AP.iloc[0])
        if abs(stored_pooled(arm, bb, base) - stored[(arm, bb)]) > 1e-12:
            raise AssertionError("stored AP and stored fold scores differ")
    y = base.labels.lookup(base.part.ids).stable.to_numpy()
    r0aps = {bb: p.metrics.ap(y, -base.hulls[:, i]) for i, bb in enumerate(p.CHECKPOINTS)}
    r0star = max(r0aps.values())
    dump(OUT / "fixed_baselines.json", dict(R0_AP=r0aps, R0_star=r0star,
         strongest_checkpoint=max(r0aps, key=r0aps.get), fixed_original_inputs=True))
    results, traces = {}, []
    try:
        results["unperturbed"] = run_variant("unperturbed", base)
        rows = write_report(results, stored, r0star)
        if not all(row["unperturbed_within_1e12"] for row in rows):
            raise AssertionError("unperturbed reproduction tolerance unmet; do not proceed")
        for seed in range(3):
            noisy = perturb_inputs(base, seed)
            traces.append(perturbation_trace(base, noisy, seed))
            dump(OUT / "perturbation_trace.json", traces)
            results[f"seed{seed}"] = run_variant(f"seed{seed}", noisy)
            rows = write_report(results, stored, r0star)
            del noisy
        print(json.dumps(rows, indent=2), flush=True)
    finally:
        dump(OUT / "run_status.json", dict(seconds=time.monotonic() - started,
             complete_variants=list(results), complete=len(results) == 4,
             confirm_label_rows_read=base.manifest["label_access"]["confirm_label_rows_read"]))
    # A numerical FAIL is a completed check, not a software failure; retain its verdict.
    return 0


if __name__ == "__main__":
    with threadpool_limits(1):
        raise SystemExit(main())
