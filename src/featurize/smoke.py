"""S0 SMOKE test for one backbone (NOT evidence). Run in the backbone's env.

python smoke.py --backbone mace_small --ids result/s0_env/smoke_ids_200.txt --batch 32 --device cuda \
    --out-json result/s0_env/smoke_mace_small.json --feat-npz cache/features/smoke_mace_small.npz \
    [--cache-npz cache/wbm/wbm_mace_feats_n18000_s0.npz]   # mace_small only: compare with old features

Protocol: load structures and model (not timed); warm-up = featurize the first 10 structures;
pass 1 and pass 2 = featurize all ids (timed with cuda synchronize); determinism = max|pass1 - pass2|;
batch=1 vs batched on the first 50; rigid rotation (seed 0) of the first 20 vs pass 1.
For MACE backbones the per-structure path (calc.get_descriptors, the original method) is also timed
and compared with the batched path.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as E  # noqa: E402


def timed(fz, structures, batch):
    import torch
    torch.cuda.synchronize()
    t = time.time()
    f = E.featurize_all(fz, structures, batch)
    torch.cuda.synchronize()
    return f, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(E.BACKBONE_INFO))
    ap.add_argument("--ids", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--feat-npz", required=True)
    ap.add_argument("--cache-npz", default=None)
    ap.add_argument("--structures", default="relaxed", choices=sorted(E.STRUCT_FILES))
    ap.add_argument("--compare-npz", default=None, help="other features for the same ids (e.g. relaxed smoke)")
    a = ap.parse_args()
    import torch
    mids = [ln.strip() for ln in open(a.ids) if ln.strip()]
    t0 = time.time()
    structures = E.load_structures(mids, a.structures)
    rec = {"label": "SMOKE - not evidence", "backbone": a.backbone, "n": len(mids), "batch": a.batch,
           "structures_source": a.structures, "structures_file": E.STRUCT_FILES[a.structures][0],
           "structures_sha256": E.sha256_file(E.STRUCT_FILES[a.structures][0]),
           "seconds_load_structures": time.time() - t0, "pooled_tensor": E.BACKBONE_INFO[a.backbone],
           "python": platform.python_version(), "executable": sys.executable, "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "n_atoms_mean": float(np.mean([len(s) for s in structures])),
           "n_atoms_max": int(max(len(s) for s in structures))}
    t0 = time.time()
    fz = E.build_featurizer(a.backbone, a.device, "batched")
    rec["seconds_model_load"] = time.time() - t0
    rec["weights_path"] = os.path.realpath(str(fz.weights))
    rec["weights_size_bytes"] = os.path.getsize(rec["weights_path"])
    rec["weights_sha256"] = E.sha256_file(rec["weights_path"])
    E.featurize_all(fz, structures[:10], a.batch)  # warm-up
    f1, s1 = timed(fz, structures, a.batch)
    f2, s2 = timed(fz, structures, a.batch)
    rec.update(dim=int(f1.shape[1]), seconds_pass1=s1, seconds_pass2=s2,
               structures_per_s=len(mids) / s1, structures_per_s_pass2=len(mids) / s2,
               determinism_max_abs_diff=float(np.max(np.abs(f1 - f2))),
               finite=bool(np.isfinite(f1).all()),
               gpu_hours_full_wbm_256963=256963 / (len(mids) / s1) / 3600.0,
               peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    # batching check: batch=1 vs batched on the first 50 structures (no cross-talk between graphs)
    rec["batch1_vs_batched_max_abs_diff_first50"] = float(np.max(np.abs(
        E.featurize_all(fz, structures[:50], 1) - f1[:50])))
    # invariance check: rigid rotation (fixed seed) of the first 20 crystals, fractional coords kept
    from pymatgen.core import Lattice, Structure
    from scipy.spatial.transform import Rotation
    rot = Rotation.random(random_state=0).as_matrix()
    rotated = [Structure(Lattice(s.lattice.matrix @ rot.T), s.species, s.frac_coords,
                         site_properties=s.site_properties) for s in structures[:20]]
    fr = E.featurize_all(fz, rotated, a.batch)
    rec["rotation_max_abs_diff_first20"] = float(np.max(np.abs(fr - f1[:20])))
    if a.backbone.startswith("mace"):
        fz.mode, fz.needs_grad = "single", True
        E.featurize_all(fz, structures[:10], 1)  # warm-up
        fs, ss = timed(fz, structures, 1)
        rec.update(single_structures_per_s=len(mids) / ss, single_seconds=ss,
                   batched_vs_single_max_abs_diff=float(np.max(np.abs(f1 - fs))))
    if a.cache_npz:
        z = np.load(a.cache_npz, allow_pickle=True)
        pos = {m: i for i, m in enumerate(z["mids"])}
        ref = z["feats"][[pos[m] for m in mids]].astype(np.float64)
        for tag, f in [("batched", f1), ("single", fs if a.backbone.startswith("mace") else None)]:
            if f is None:
                continue
            d = np.abs(f.astype(np.float64) - ref)
            rec[f"cache_{tag}_max_abs_diff"] = float(d.max())
            rec[f"cache_{tag}_max_rel_diff"] = float(np.max(d / np.maximum(np.abs(ref), 1e-8)))
            rec[f"cache_{tag}_max_rel_diff_ref_gt_1e-3"] = float(np.max(np.where(np.abs(ref) > 1e-3,
                                                                       d / np.maximum(np.abs(ref), 1e-8), 0)))
            rec[f"cache_{tag}_max_abs_ref"] = float(np.abs(ref).max())
        rec["cache_npz"] = os.path.abspath(a.cache_npz)
        rec["cache_feats_dtype"] = str(z["feats"].dtype)
    if a.compare_npz:
        z = np.load(a.compare_npz, allow_pickle=True)
        assert list(z["mids"]) == mids, "compare-npz ids differ"
        d = np.abs(f1.astype(np.float64) - z["feats"].astype(np.float64))
        rec.update(compare_npz=os.path.abspath(a.compare_npz), compare_mean_abs_diff=float(d.mean()),
                   compare_max_abs_diff=float(d.max()),
                   compare_mean_rel_l2=float(np.mean(np.linalg.norm(d, axis=1) /
                                                     np.maximum(np.linalg.norm(z["feats"], axis=1), 1e-12))))
    os.makedirs(os.path.dirname(os.path.abspath(a.feat_npz)), exist_ok=True)
    np.savez_compressed(a.feat_npz, feats=f1, mids=np.array(mids), backbone=a.backbone, label="SMOKE")
    rec["feat_npz"] = os.path.abspath(a.feat_npz)
    rec["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(a.out_json, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
