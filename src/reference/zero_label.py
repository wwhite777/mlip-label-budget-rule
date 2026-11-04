"""Zero-label stability predictions (no WBM label is used to fit anything).

python zero_label.py sp --backbone <b> [--refs mbd1.0.0|unrounded] [--no-mp2020]
python zero_label.py is2re --model <published model key>

sp:    cache/reference/sp_energy_<b>_initial.npz (one forward pass on the INITIAL structure, no relaxation)
       -> + MP2020 correction of the WBM entry with the initial structure (Matbench Discovery steps, see
       mbd_energy.py) -> formation energy (matbench-discovery 1.0.0 get_e_form_per_atom, rounded MP elemental refs)
       -> result/reference/zero_label_sp_<b>_initial.csv.gz
       The correction is taken from --corrections (default: pymatgen 2023.5.10, the version whose MP2020 behaviour
       reproduces the WBM label column exactly in the convention check; pymatgen 2025.10.7 of the main env does
       not, see convention_check_versions.json). MP2020 adjustments do not depend on the energy; this is checked
       per backbone by re-running the pinned pipeline call live (main env, entry._energy = E_sp) and comparing
       with the precomputed same-version corrections (--corrections-main).
       --no-mp2020: formation energy from the raw single-point energy (CHGNet's own convention: its targets
       already include MP2020 corrections; the legacy Matbench Discovery CHGNet join script applied no MP2020)
       -> zero_label_sp_<b>_initial_nomp2020.csv.gz
is2re: published Matbench Discovery IS2RE prediction file (Figshare article 28187990, e_form_per_atom = their
       MP2020-corrected formation energy after relaxation) -> result/reference/zero_label_is2re_<model>.csv.gz

Hull distance on the FIXED MP hull (Matbench Discovery convention):
    e_hull_pred = e_hull_DFT + (e_form_pred - e_form_DFT)
with e_hull_DFT = e_above_hull_mp2020_corrected_ppd_mp and e_form_DFT = e_form_per_atom_mp2020_corrected (pinned WBM
summary, Figshare file 64706751). Identity: e_hull_DFT = e_form_DFT - h_MP(c), where h_MP(c) is the formation energy
of the MP convex hull at the candidate's composition c, so e_hull_pred = e_form_pred - h_MP(c): the candidate's own
DFT energy cancels; only the MP hull (built from MP DFT energies, not the WBM label) enters.
Rows: every one of the 256,963 summary ids in summary order; missing / rejected predictions stay NaN (never dropped).
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

PROJECT = M.PROJECT
OUT = os.path.join(PROJECT, "result", "reference")
PRED_FILES = {  # published IS2RE predictions (Figshare article 28187990); md5 from models/<family>/<model>.yml @ e3d7966
    "mace-mp-0": ("cache/mbd/preds/mace-mp-0_discovery.csv.gz", "5e231a2b3a029714b984890a6fbe5002"),
    "chgnet-0.3.0": ("cache/mbd/preds/chgnet-0.3.0_discovery.csv.gz", "fd7cd3781a24be465aaeadf97663ce58"),
    "sevennet-0": ("cache/mbd/preds/sevennet-0_discovery.csv.gz", "76a001eb17dbcfe952c2df74f0b18e8c"),
    "mattersim-v1-5m": ("cache/mbd/preds/mattersim-v1-5m_discovery.csv.gz", "3779100dfaa5b2f733464006287ed9e0"),
    "orb-v3": ("cache/mbd/preds/orb-v3_discovery.csv.gz", "87b2195d204240eea7e1801afd0ea71a"),
    "mace-mpa-0": ("cache/mbd/preds/mace-mpa-0_discovery.csv.gz", "69c5b43ed1bc972ba623627438e7653b"),
}
E_FORM_DFT, E_HULL_DFT = "e_form_per_atom_mp2020_corrected", "e_above_hull_mp2020_corrected_ppd_mp"


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hull_shift(summ, e_form_pred):
    """e_hull_pred = e_hull_DFT + (e_form_pred - e_form_DFT) = e_form_pred - h_MP(c)."""
    return summ[E_HULL_DFT].to_numpy() + (e_form_pred - summ[E_FORM_DFT].to_numpy())


def write(df, path, meta):
    df.to_csv(path, index=False, float_format="%.6f")
    meta.update(output=path, output_sha256=file_hash(path, "sha256"), written_utc=time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(path.replace(".csv.gz", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def coverage(summ, e_form_pred):
    uniq = summ["unique_prototype"].to_numpy()
    nan = ~np.isfinite(e_form_pred)
    return {"n_rows": int(len(e_form_pred)), "n_nan_all": int(nan.sum()),
            "n_unique_prototype": int(uniq.sum()), "n_nan_unique_prototype": int((nan & uniq).sum())}


def load_corr(path, index):
    z = np.load(path)
    assert str(z["structures"]) == "initial", path
    c = pd.Series(z["correction"], index=[str(m) for m in z["material_id"]]).reindex(index).to_numpy()
    return c, str(z["pymatgen_version"]), file_hash(path, "sha256")


def run_sp(a, summ):
    for f in ["convention_check.json", "convention_check_versions.json"]:  # gate: the convention check must exist
        assert os.path.exists(os.path.join(OUT, f)), f"run the convention check first ({f})"
    cc = json.load(open(os.path.join(OUT, "convention_check.json")))
    if cc["variants"][f"uncorrected|refs={a.refs}"]["n_gt_1meV"] != 0:
        sys.exit(f"convention check: energy->e_form path fails for refs={a.refs}; refusing to run")
    cv = json.load(open(os.path.join(OUT, "convention_check_versions.json")))
    zc = np.load(a.corrections)
    ver = str(zc["pymatgen_version"])
    passed = [k for k, v in cv.items() if k.startswith(f"pymatgen {ver} ") and v["n_gt_1meV"] == 0]
    if not a.no_mp2020 and not passed:
        sys.exit(f"MP2020 corrections from pymatgen {ver} did not pass the convention check; refusing to run")
    path = os.path.join(PROJECT, "cache", "reference", f"sp_energy_{a.backbone}_initial.npz")
    z = np.load(path)
    mids = [str(m) for m in z["material_id"]]
    assert len(mids) == len(set(mids)) == len(summ) and set(mids) == set(summ.index)
    e = pd.Series(z["energy_total"], index=mids).reindex(summ.index).to_numpy()
    formulas = summ["formula"].tolist()
    n_at = summ["n_sites"].to_numpy()
    assert np.array_equal(pd.Series(z["n_atoms"], index=mids).reindex(summ.index).to_numpy(), n_at)
    refs = M.elemental_refs(a.refs)
    meta = {"kind": "single-point (initial structure, one forward pass, no relaxation)", "backbone": a.backbone,
            "energy_npz": path, "energy_npz_sha256": file_hash(path, "sha256"),
            "checkpoint_sha256": str(z["weights_sha256"]), "checkpoint_path": str(z["weights_path"]),
            "energy_source": str(z["energy_source"]), "elemental_refs": a.refs, "versions_main_env": M.versions(),
            "n_energy_nonfinite": int((~np.isfinite(e)).sum()), "energy_failed_ids": z["failed_ids"].tolist()}
    if a.no_mp2020:
        corr = np.zeros(len(e))
        meta["mp2020"] = "NOT applied (raw single-point energy)"
        out = os.path.join(OUT, f"zero_label_sp_{a.backbone}_initial_nomp2020.csv.gz")
    else:
        corr, ver, sha = load_corr(a.corrections, summ.index)
        meta.update(mp2020=f"MaterialsProject2020Compatibility() (pymatgen default strict_anions), process_entry("
                           f"clean=True) on the WBM entry with the INITIAL structure, pymatgen {ver}; this version "
                           f"reproduces the label column in the convention check",
                    mp2020_corrections_npz=a.corrections, mp2020_corrections_sha256=sha,
                    mp2020_pymatgen=ver, mp2020_n_nan=int(np.isnan(corr).sum()))
        # energy independence of the correction, shown on this backbone's own energies: the live pinned-pipeline
        # call (main env, entry._energy = E_sp) must give the same correction as the precomputed same-version file
        cse = M.load_cse()
        structs = M.load_initial_structures()
        t0 = time.time()
        live = M.mp2020_corrected(list(summ.index), e, cse, structs=structs, strict_anions=None,
                                  n_workers=a.n_workers)
        c_live = live["corrected_energy"].to_numpy() - e
        c_new, ver_new, _ = load_corr(a.corrections_main, summ.index)
        d_live = np.abs(c_live - c_new)
        d_ver = np.abs(corr - c_new) / n_at
        meta.update(energy_independence_check={
            "live_pymatgen": M.versions()["pymatgen"], "precomputed_same_version": ver_new,
            "max_abs_diff_eV": float(np.nanmax(d_live)), "n_nan_live": int(np.isnan(c_live).sum()),
            "n_live_rejected_or_error": int((live["error"] != "").sum()), "seconds": time.time() - t0},
            n_rows_correction_differs_vs_main_env_gt_1meV_per_atom=int((d_ver > 1e-3).sum()))
        assert np.nanmax(d_live) < 1e-6, "MP2020 correction depends on the energy?"
        out = os.path.join(OUT, f"zero_label_sp_{a.backbone}_initial.csv.gz")
    e_form = M.e_form_per_atom(formulas, e + corr, refs)
    meta["coverage"] = coverage(summ, e_form)
    write(pd.DataFrame({"material_id": summ.index, "e_form_pred_sp": e_form,
                        "e_hull_pred_sp": hull_shift(summ, e_form)}), out, meta)


def run_is2re(a, summ):
    rel, md5_exp = PRED_FILES[a.model]
    path = os.path.join(PROJECT, rel)
    md5 = file_hash(path, "md5")
    assert md5 == md5_exp, f"{path}: md5 {md5} != {md5_exp}"
    p = pd.read_csv(path)
    ids = p["material_id"].astype(str)
    n_dup = int(ids.duplicated().sum())
    assert n_dup == 0, f"{n_dup} duplicate ids in {path}"
    s = pd.Series(p["e_form_per_atom"].to_numpy(dtype=np.float64), index=ids)
    extra = sorted(set(s.index) - set(summ.index))
    e_form = s.reindex(summ.index).to_numpy()
    meta = {"kind": "published IS2RE (Matbench Discovery relaxation)", "model": a.model, "pred_file": path,
            "pred_file_md5": md5, "pred_file_sha256": file_hash(path, "sha256"), "n_rows_in_file": int(len(p)),
            "n_nan_in_file": int(s.isna().sum()), "ids_in_file_not_in_summary": extra[:50],
            "n_ids_in_file_not_in_summary": len(extra), "ids_missing_from_file": int((~summ.index.isin(s.index)).sum())}
    meta["coverage"] = coverage(summ, e_form)
    write(pd.DataFrame({"material_id": summ.index, "e_form_pred_is2re": e_form,
                        "e_hull_pred_is2re": hull_shift(summ, e_form)}),
          os.path.join(OUT, f"zero_label_is2re_{a.model}.csv.gz"), meta)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sp")
    s1.add_argument("--backbone", required=True)
    s1.add_argument("--refs", default="mbd1.0.0", choices=["mbd1.0.0", "unrounded"])
    s1.add_argument("--no-mp2020", action="store_true")
    s1.add_argument("--corrections", default=os.path.join(PROJECT, "cache", "reference",
                                                          "mp2020_corr_pmg2023.5.10_initial.npz"))
    s1.add_argument("--corrections-main", default=os.path.join(PROJECT, "cache", "reference",
                                                               "mp2020_corr_pmg2025.10.7_initial.npz"))
    s1.add_argument("--n-workers", type=int, default=16)
    s2 = sub.add_parser("is2re")
    s2.add_argument("--model", required=True, choices=sorted(PRED_FILES))
    a = ap.parse_args()
    summ = M.load_summary()
    (run_sp if a.cmd == "sp" else run_is2re)(a, summ)


if __name__ == "__main__":
    main()
