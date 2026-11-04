"""Prediction-vs-prediction sanity check (no labels): our single-point e_form vs the published IS2RE e_form of the SAME
checkpoint. Catches unit / per-atom / correction-convention slips; it is not an accuracy number.

python sanity_sp_vs_is2re.py --out result/reference/checks/sp_vs_is2re_predictions.json
For CHGNet, also split by whether the MP2020 term (pymatgen 2023.5.10, initial structure) is non-zero, for both the
MP2020-applied and the no-MP2020 SP files: the published CHGNet file applied no MP2020 step, so on corrected rows
the convention that matches it should sit closer.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(PROJECT, "result", "reference")
PAIRS = [("mace_medium", "mace-mp-0"), ("chgnet", "chgnet-0.3.0"), ("sevenn", "sevennet-0"),
         ("mattersim_5m", "mattersim-v1-5m"), ("orb", "orb-v3")]


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    return {"n": int(m.sum()), "median_abs_diff": float(np.median(np.abs(d))), "mean_diff": float(d.mean()),
            "pearson_r": float(np.corrcoef(a[m], b[m])[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for fs, key in PAIRS:
        sp = pd.read_csv(os.path.join(OUT, f"zero_label_sp_{fs}_initial.csv.gz"))
        ir = pd.read_csv(os.path.join(OUT, f"zero_label_is2re_{key}.csv.gz"))
        assert (sp["material_id"] == ir["material_id"]).all()
        res[f"{fs} vs {key}"] = stats(sp["e_form_pred_sp"].to_numpy(), ir["e_form_pred_is2re"].to_numpy())
    z = np.load(os.path.join(PROJECT, "cache", "reference", "mp2020_corr_pmg2023.5.10_initial.npz"))
    ids = pd.read_csv(os.path.join(OUT, "zero_label_sp_chgnet_initial.csv.gz"))["material_id"]
    corr = pd.Series(z["correction"], index=[str(m) for m in z["material_id"]]).reindex(ids).to_numpy()
    ir = pd.read_csv(os.path.join(OUT, "zero_label_is2re_chgnet-0.3.0.csv.gz"))["e_form_pred_is2re"].to_numpy()
    for tag, f in [("mp2020", "zero_label_sp_chgnet_initial.csv.gz"),
                   ("nomp2020", "zero_label_sp_chgnet_initial_nomp2020.csv.gz")]:
        sp = pd.read_csv(os.path.join(OUT, f))["e_form_pred_sp"].to_numpy()
        for sel, m in [("rows with MP2020 term != 0", corr != 0), ("rows with MP2020 term == 0", corr == 0)]:
            res[f"chgnet SP {tag} vs published chgnet-0.3.0 | {sel}"] = stats(sp[m], ir[m])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
