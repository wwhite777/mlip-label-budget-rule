"""Full-WBM proof that --energy leaves the feature path unchanged, plus validation of the new feature files.

python compare_energy_pass_features.py --energy-feats-dir <dir with <b>_initial.npz from the --energy pass>
                                       --out result/reference/checks/energy_pass_feature_identity.json
python compare_energy_pass_features.py --energy-feats-dir <dir> --control mace_small,sevenn
                                       --out result/reference/checks/control_plain_vs_frozen.json   (GPU floor)
For the six frozen backbones: features written by the --energy pass vs the frozen cache/features/wbm_full_<b>_initial
.npz (same ids, same code path; GPU runs are not bit-reproducible, so max |diff| is compared with the earlier GPU
floor). For mattersim_5m (relaxed + initial): shape, id order vs cache/features/wbm_all_ids.txt, non-finite values,
NaN rows, failed ids, weights path / sha256.
"""
import argparse
import json
import os

import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FEAT = os.path.join(PROJECT, "cache", "features")
FROZEN = ["mace_small", "mace_medium", "chgnet", "mattersim", "orb", "sevenn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy-feats-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default=None, help="comma list: compare <dir>/control_plain_<b>_initial.npz "
                                                    "(rerun WITHOUT --energy) with the frozen file instead")
    a = ap.parse_args()
    if a.control:
        res = {}
        for b in a.control.split(","):
            fn = np.load(os.path.join(a.energy_feats_dir, f"control_plain_{b}_initial.npz"))
            fo = np.load(os.path.join(FEAT, f"wbm_full_{b}_initial.npz"))
            d = np.abs(fn["feats"].astype(np.float64) - fo["feats"])
            res[b] = {"ids_equal": bool((fn["mids"] == fo["mids"]).all()), "max_abs_diff": float(d.max()),
                      "mean_abs_diff": float(d.mean()), "bit_identical": fn["feats"].tobytes() == fo["feats"].tobytes()}
            pe = os.path.join(a.energy_feats_dir, f"{b}_initial.npz")  # --energy pass, same code, same session
            if os.path.exists(pe):
                fe = np.load(pe)["feats"]
                res[b]["control_vs_energy_pass_max_abs_diff"] = float(np.abs(fn["feats"].astype(np.float64) - fe).max())
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    ids = [ln.strip() for ln in open(os.path.join(FEAT, "wbm_all_ids.txt")) if ln.strip()]
    res = {"identity_vs_frozen": {}, "new_feature_files": {}}
    for b in FROZEN:
        p_new = os.path.join(a.energy_feats_dir, f"{b}_initial.npz")
        p_old = os.path.join(FEAT, f"wbm_full_{b}_initial.npz")
        zn, zo = np.load(p_new), np.load(p_old)
        fn, fo = zn["feats"], zo["feats"]
        d = np.abs(fn.astype(np.float64) - fo)
        res["identity_vs_frozen"][b] = {
            "frozen": p_old, "energy_pass": p_new, "shape_equal": fn.shape == fo.shape,
            "ids_equal": bool((zn["mids"] == zo["mids"]).all()), "bit_identical": fn.tobytes() == fo.tobytes(),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "frac_elements_differ": float((fn != fo).mean()),
            "weights_sha256_equal": str(zn["weights_sha256"]) == str(zo["weights_sha256"]),
            "seconds_featurize_energy_pass": float(zn["seconds_featurize"])}
    for tag in ["wbm_full_mattersim_5m", "wbm_full_mattersim_5m_initial"]:
        p = os.path.join(FEAT, f"{tag}.npz")
        z = np.load(p)
        f = z["feats"]
        res["new_feature_files"][tag] = {
            "path": p, "bytes": os.path.getsize(p), "shape": list(f.shape), "dtype": str(f.dtype),
            "ids_equal_wbm_all_ids_in_order": [str(m) for m in z["mids"]] == ids,
            "nan_rows": int(np.isnan(f).any(1).sum()), "nonfinite_values": int((~np.isfinite(f)).sum()),
            "failed_ids": z["failed_ids"].tolist(), "isolated_atom_ids": z["isolated_atom_ids"].tolist(),
            "weights_path": str(z["weights_path"]), "weights_sha256": str(z["weights_sha256"]),
            "structures_source": str(z["structures_source"]), "structures_sha256": str(z["structures_sha256"]),
            "seconds_featurize": float(z["seconds_featurize"]), "pooled_tensor": str(z["pooled_tensor"])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
