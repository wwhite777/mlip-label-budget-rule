"""Label-free pretraining-overlap strata v2: chemical system present in MPtrj (the backbones' training set) vs present
in the 2025-02-01 MP energies list (the v1 proxy).

python3 src/membership/strata_v2.py

Universe and chemical systems: src/splits/make_splits.load_universe (WBM summary v64706751, unique_prototype == True,
only material_id / formula / unique_prototype loaded; MP energies: only material_id / formula). sys_in_mp2025 is
make_splits' sys_in_mp. sys_in_mptrj: chemical system equals the chemical_system of >= 1 MPtrj material in
cache/membership/mptrj_frames_by_material.csv.gz (built by mptrj_membership.py from the species columns).
No label column is read anywhere.

Outputs: result/membership/wbm_strata_v2.csv.gz (material_id, chemsys, sys_in_mptrj, sys_in_mp2025, arity; gzip
mtime=0) and result/membership/wbm_strata_v2_summary.json (fractions, cross-tabs, per split family/seed/fold).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from splits.make_splits import LABEL_PATTERN, load_universe, sha256_file  # noqa: E402

WBM = ROOT / "cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz"
MP = ROOT / "cache/mbd/2025-02-01-mp-energies.csv.gz"
MPTRJ = ROOT / "cache/membership/mptrj_frames_by_material.csv.gz"
SPLITS = ROOT / "result/splits"
OUT_CSV = ROOT / "result/membership/wbm_strata_v2.csv.gz"
OUT_JSON = ROOT / "result/membership/wbm_strata_v2_summary.json"


def frac(x) -> float:
    return round(float(pd.Series(x).mean()), 4)


def main() -> int:
    u, rec = load_universe(WBM, MP)
    for r in rec.values():
        assert not any(LABEL_PATTERN.search(c) for c in r["columns"]), r
    assert len(u) == 215488, len(u)
    mt = pd.read_csv(MPTRJ, keep_default_na=False, dtype=str, usecols=["material_id", "chemical_system"])
    mptrj_sys = set(mt["chemical_system"])
    mp_df = pd.read_csv(MP, keep_default_na=False, dtype=str, usecols=["material_id", "formula"])
    from splits.make_splits import chemsys
    mp_sys = set(mp_df["formula"].map(chemsys))

    u = u.rename(columns={"sys_in_mp": "sys_in_mp2025"})
    u["sys_in_mptrj"] = u["chemsys"].isin(mptrj_sys)
    u["arity"] = u["chemsys"].str.count("-") + 1
    out = u[["material_id", "chemsys", "sys_in_mptrj", "sys_in_mp2025", "arity"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(out.to_csv(index=False).encode("utf-8"))

    systems = u.drop_duplicates("chemsys")[["chemsys", "sys_in_mptrj", "sys_in_mp2025", "arity"]]
    xt_s = pd.crosstab(u["sys_in_mptrj"], u["sys_in_mp2025"])
    xt_y = pd.crosstab(systems["sys_in_mptrj"], systems["sys_in_mp2025"])

    def tab(xt):
        return {f"mptrj={a}": {f"mp2025={b}": int(xt.loc[a, b]) if (a in xt.index and b in xt.columns) else 0
                               for b in (True, False)} for a in (True, False)}

    by_arity = {}
    for a, g in u.groupby("arity"):
        gs = systems[systems["arity"] == a]
        by_arity[str(a)] = {"n_structures": int(len(g)), "n_systems": int(len(gs)),
                            "struct_frac_sys_in_mptrj": frac(g["sys_in_mptrj"]),
                            "struct_frac_sys_in_mp2025": frac(g["sys_in_mp2025"]),
                            "system_frac_in_mptrj": frac(gs["sys_in_mptrj"]),
                            "system_frac_in_mp2025": frac(gs["sys_in_mp2025"])}

    lookup = u.set_index("material_id")
    per_split = []
    split_files = sorted(SPLITS.glob("splits_*_seed*.csv.gz"))
    for f in split_files:
        fam, seed = f.name[len("splits_"):-len(".csv.gz")].split("_seed")
        s = pd.read_csv(f, keep_default_na=False, dtype=str)
        assert not any(LABEL_PATTERN.search(c) for c in s.columns), s.columns
        assert set(s["material_id"]) <= set(lookup.index), f
        assert (s["chemsys"].to_numpy() == lookup.loc[s["material_id"], "chemsys"].to_numpy()).all(), f
        inm = lookup.loc[s["material_id"], "sys_in_mptrj"].to_numpy()
        inp = lookup.loc[s["material_id"], "sys_in_mp2025"].to_numpy()
        if "fold" in s.columns:
            folds = sorted(s["fold"].astype(int).unique())
            masks = {k: (s["fold"].astype(int) == k).to_numpy() for k in folds}
        else:
            tf = s["test_folds"].str.split(";")
            folds = sorted({int(x) for v in tf for x in v if x != ""})
            masks = {k: tf.map(lambda v, k=k: str(k) in v).to_numpy() for k in folds}
        for k in folds:
            m = masks[k]
            per_split.append({"family": fam, "seed": int(seed), "fold": int(k), "n_test": int(m.sum()),
                              "n_systems_test": int(s.loc[m, "chemsys"].nunique()),
                              "test_frac_sys_in_mptrj": frac(inm[m]),
                              "test_frac_sys_in_mp2025": frac(inp[m]),
                              "manifest": f.name, "manifest_sha256": sha256_file(f)})

    summary = {
        "n_structures": int(len(u)), "n_systems": int(len(systems)),
        "n_mptrj_systems": len(mptrj_sys), "n_mp2025_systems": len(mp_sys),
        "mptrj_systems_not_in_mp2025": len(mptrj_sys - mp_sys), "mp2025_systems_not_in_mptrj": len(mp_sys - mptrj_sys),
        "structure_weighted": {"frac_sys_in_mptrj": frac(u["sys_in_mptrj"]),
                               "frac_sys_in_mp2025": frac(u["sys_in_mp2025"]),
                               "n_sys_in_mptrj": int(u["sys_in_mptrj"].sum()),
                               "n_sys_in_mp2025": int(u["sys_in_mp2025"].sum())},
        "system_weighted": {"frac_in_mptrj": frac(systems["sys_in_mptrj"]),
                            "frac_in_mp2025": frac(systems["sys_in_mp2025"]),
                            "n_in_mptrj": int(systems["sys_in_mptrj"].sum()),
                            "n_in_mp2025": int(systems["sys_in_mp2025"].sum())},
        "crosstab_structures": tab(xt_s), "crosstab_systems": tab(xt_y),
        "by_arity": by_arity,
        "per_split_fold": per_split,
        "definitions": {"sys_in_mptrj": "WBM chemical system equals the species-set chemical system of >= 1 MPtrj "
                                        "material (2024-09-03-mp-trj.extxyz.zip, all frames)",
                        "sys_in_mp2025": "WBM chemical system equals that of >= 1 entry of 2025-02-01-mp-energies "
                                         "(= sys_in_mp of the v1 split manifests)",
                        "test set (element family)": "structures whose test_folds contain the fold"},
        "inputs": {**rec, "mptrj": {"path": str(MPTRJ.relative_to(ROOT)), "sha256": sha256_file(MPTRJ),
                                     "columns": ["chemical_system", "material_id"]}},
        "labels_read": False,
        "output": {"csv": str(OUT_CSV.relative_to(ROOT)), "sha256": sha256_file(OUT_CSV),
                   "bytes": OUT_CSV.stat().st_size},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__))},
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_split_fold", "inputs")}, indent=1))
    print("family seed fold n_test frac_mptrj frac_mp2025")
    for r in per_split:
        print(r["family"], r["seed"], r["fold"], r["n_test"], r["test_frac_sys_in_mptrj"], r["test_frac_sys_in_mp2025"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
