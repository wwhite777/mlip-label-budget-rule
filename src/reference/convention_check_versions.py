"""Cross-version part of the CONVENTION CHECK: which rows differ and why.

python convention_check_versions.py --old cache/reference/mp2020_corr_pmg2023.5.10_relaxed.npz
                                    --new cache/reference/mp2020_corr_pmg2025.10.7_relaxed.npz --out-dir result/reference
Per-row MP2020 corrections (mp2020_corrections_only.py, DFT-relaxed WBM entries, pymatgen default strict_anions) under
two pymatgen versions, turned into corrected formation energies with the matbench-discovery 1.0.0 elemental refs and
compared with the summary label e_form_per_atom_mp2020_corrected. Label columns are read ONLY to validate the energy
convention (DFT energy in, DFT label out); no model is involved.
Outputs convention_check_versions.json and convention_check_version_rows.csv (every row > 1 meV/atom under either
version, with both versions' adjustment lists and a reason category).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbd_energy as M  # noqa: E402

TOL = 1e-3


def load(path, index):
    z = np.load(path)
    df = pd.DataFrame({"correction": z["correction"], "adjustments": z["adjustments"], "error": z["error"]},
                      index=[str(m) for m in z["material_id"]]).reindex(index)
    return df, str(z["pymatgen_version"])


def anion_set(adj):
    return sorted(set(re.findall(r"MP2020 anion correction \(([^)]+)\)", adj or "")))


def u_set(adj):
    return sorted(set(re.findall(r"MP2020 GGA/GGA\+U mixing correction \(([^)]+)\)", adj or "")))


def reason(row):
    ao, an = anion_set(row["adj_old"]), anion_set(row["adj_new"])
    uo, un = u_set(row["adj_old"]), u_set(row["adj_new"])
    parts = []
    if ao != an:
        parts.append(f"anion corrections old={'+'.join(ao) or 'none'} new={'+'.join(an) or 'none'}")
    elif row["adj_old"] != row["adj_new"]:
        parts.append("same anion species, different anion amounts/values")
    if uo != un:
        parts.append(f"U corrections old={'+'.join(uo) or 'none'} new={'+'.join(un) or 'none'}")
    return "; ".join(parts) or "adjustment lists identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    summ = M.load_summary()
    cse = M.load_cse()
    mids = list(summ.index)
    e_unc = np.array([cse[m]["energy"] for m in mids], dtype=np.float64)
    refs = M.elemental_refs("mbd1.0.0")
    formulas = summ["formula"].tolist()
    lab = summ["e_form_per_atom_mp2020_corrected"].to_numpy()
    res = {"versions_main_env": M.versions(), "tolerance_eV_per_atom": TOL, "elemental_refs": "mbd1.0.0"}
    tab = pd.DataFrame({"formula": formulas, "label_e_form_mp2020_corrected": lab,
                        "label_correction_per_atom": summ["e_correction_per_atom_mp2020"].to_numpy(),
                        "n_atoms": summ["n_sites"].to_numpy()}, index=pd.Index(mids, name="material_id"))
    flag = np.zeros(len(mids), bool)
    for tag, path in [("old", a.old), ("new", a.new)]:
        df, ver = load(path, mids)
        ef = M.e_form_per_atom(formulas, e_unc + df["correction"].to_numpy(), refs)
        d = ef - lab
        big = np.abs(d) > TOL
        flag |= big
        tab[f"pymatgen_{tag}"] = ver
        tab[f"correction_per_atom_{tag}"] = df["correction"].to_numpy() / tab["n_atoms"].to_numpy()
        tab[f"adj_{tag}"] = df["adjustments"].to_numpy()
        tab[f"e_form_{tag}"] = ef
        tab[f"diff_{tag}"] = d
        res[f"pymatgen {ver} ({tag})"] = {
            "file": path, "max_abs_diff": float(np.nanmax(np.abs(d))), "n_gt_1meV": int(big.sum()),
            "n_gt_0.1meV": int((np.abs(d) > 1e-4).sum()), "n_nan": int(np.isnan(d).sum()),
            "n_error_or_rejected": int((df["error"] != "").sum()),
            "n_gt_1meV_unique_prototype": int((big & summ["unique_prototype"].to_numpy()).sum()),
            "examples_gt_1meV": tab.index[big][:10].tolist()}
    rows = tab[flag].copy()
    rows["reason"] = rows.apply(reason, axis=1)
    rows["unique_prototype"] = summ.loc[rows.index, "unique_prototype"].to_numpy()
    rows.to_csv(os.path.join(a.out_dir, "convention_check_version_rows.csv"))
    nb = rows[np.abs(rows["diff_new"]) > TOL]
    res["rows_gt_1meV_new_only"] = int(((np.abs(rows["diff_new"]) > TOL) & ~(np.abs(rows["diff_old"]) > TOL)).sum())
    res["rows_gt_1meV_old"] = int((np.abs(rows["diff_old"]) > TOL).sum())
    res["reason_counts_new_vs_old"] = nb["reason"].value_counts().head(30).to_dict()
    res["diff_new_eV_per_atom_quantiles"] = {q: float(np.quantile(nb["diff_new"], q)) for q in
                                             [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]} if len(nb) else {}
    res["rows_csv"] = os.path.join(a.out_dir, "convention_check_version_rows.csv")
    json.dump(res, open(os.path.join(a.out_dir, "convention_check_versions.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
