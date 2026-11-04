"""CONVENTION CHECK: DFT uncorrected energies through the exact prediction pipeline (mbd_energy.py) must reproduce
the WBM summary column e_form_per_atom_mp2020_corrected.

python convention_check.py --out-dir result/reference
Reads label columns ONLY to validate the energy convention (DFT energy in, DFT label out); no model is involved.
Variants (to attribute differences, not to pick a favourable one):
  strict_anions: pymatgen default (= the pinned prediction pipeline write_discovery_artifacts) and "no_check"
                 (= the pinned label script data/wbm/compile_wbm_test_set.py);
  elemental refs: matbench_discovery 1.0.0 (rounded 4 dp) and unrounded (pinned commit energy.py).
Outputs convention_check.json (+ per-row csv of every row differing by > 1 meV/atom in any variant).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbd_energy as M  # noqa: E402

TOL = 1e-3  # eV/atom


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-workers", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    res = {"versions": M.versions(), "tolerance_eV_per_atom": TOL}
    res["elemental_refs_file"] = M.ELEM_REFS
    res["elemental_refs_md5"] = md5(M.ELEM_REFS)
    res["elemental_refs_md5_expected"] = M.ELEM_REFS_MD5
    assert res["elemental_refs_md5"] == M.ELEM_REFS_MD5, "elemental reference file md5 mismatch"
    summ = M.load_summary()
    cse = M.load_cse()
    mids = list(summ.index)
    assert set(mids) == set(cse) and len(mids) == 256963
    e_unc = np.array([cse[m]["energy"] for m in mids], dtype=np.float64)
    assert all(cse[m]["correction"] == 0 for m in mids), "WBM CSEs carry corrections"
    formulas = summ["formula"].tolist()
    from pymatgen.core import Composition
    n_at = np.array([Composition(f).num_atoms for f in formulas])
    res["formula_natoms_equals_n_sites"] = bool(np.all(n_at == summ["n_sites"].to_numpy()))
    lab_unc = summ["e_form_per_atom_uncorrected"].to_numpy()
    lab_cor = summ["e_form_per_atom_mp2020_corrected"].to_numpy()
    lab_corr_pa = summ["e_correction_per_atom_mp2020"].to_numpy()
    res["label_nan"] = {"e_form_per_atom_uncorrected": int(np.isnan(lab_unc).sum()),
                        "e_form_per_atom_mp2020_corrected": int(np.isnan(lab_cor).sum())}
    refs = {k: M.elemental_refs(k) for k in ["mbd1.0.0", "unrounded"]}
    per_row = pd.DataFrame({"formula": formulas, "n_atoms": n_at, "label_e_form_uncorrected": lab_unc,
                            "label_correction_per_atom": lab_corr_pa, "label_e_form_mp2020_corrected": lab_cor},
                           index=pd.Index(mids, name="material_id"))
    flag = np.zeros(len(mids), bool)
    res["variants"] = {}
    for rk, rv in refs.items():
        ef = M.e_form_per_atom(formulas, e_unc, rv)
        d = ef - lab_unc
        res["variants"][f"uncorrected|refs={rk}"] = {
            "max_abs_diff": float(np.nanmax(np.abs(d))), "n_gt_1meV": int((np.abs(d) > TOL).sum()),
            "n_nan": int(np.isnan(d).sum())}
        per_row[f"e_form_uncorrected|{rk}"] = ef
        flag |= np.abs(d) > TOL
    for sa in [None, "no_check"]:
        sa_name = "default" if sa is None else sa
        t1 = time.time()
        df = M.mp2020_corrected(mids, e_unc, cse, structs=None, strict_anions=sa, n_workers=a.n_workers)
        corr_pa = df["correction"].to_numpy() / n_at
        per_row[f"correction_per_atom|{sa_name}"] = corr_pa
        per_row[f"adjustments|{sa_name}"] = df["adjustments"].to_numpy()
        n_rej = int((df["error"] != "").sum())
        dcorr = corr_pa - lab_corr_pa
        for rk, rv in refs.items():
            ef = M.e_form_per_atom(formulas, df["corrected_energy"].to_numpy(), rv)
            d = ef - lab_cor
            big = np.abs(d) > TOL
            flag |= big
            per_row[f"e_form_mp2020|{sa_name}|{rk}"] = ef
            ex = per_row.index[big][:10].tolist()
            res["variants"][f"mp2020 strict_anions={sa_name}|refs={rk}"] = {
                "max_abs_diff": float(np.nanmax(np.abs(d))), "n_gt_1meV": int(big.sum()),
                "n_gt_0.1meV": int((np.abs(d) > 1e-4).sum()), "n_nan": int(np.isnan(d).sum()),
                "n_rejected_or_error": n_rej, "examples_gt_1meV": ex,
                "n_rows_correction_differs_gt_1meV": int((np.abs(dcorr) > TOL).sum()),
                "seconds_mp2020": time.time() - t1}
        res["variants"][f"mp2020 strict_anions={sa_name}|refs=*"] = {
            "errors": df.loc[df["error"] != "", "error"].value_counts().head(10).to_dict()}
    rows = per_row[flag]
    os.makedirs(a.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(a.out_dir, "convention_check_diff_rows.csv"))
    res["n_rows_flagged_any_variant"] = int(flag.sum())
    res["diff_rows_csv"] = os.path.join(a.out_dir, "convention_check_diff_rows.csv")
    res["seconds_total"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out_dir, "convention_check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
