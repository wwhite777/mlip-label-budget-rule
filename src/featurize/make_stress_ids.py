"""Deterministic geometry stress sets from cache/features/wbm_struct_scan.csv.gz (SMOKE robustness only).

python make_stress_ids.py --scan cache/features/wbm_struct_scan.csv.gz --outdir result/s0_env

Per source (initial, relaxed): union of the 10 largest and 10 smallest volume/atom, the 10 smallest minimum
interatomic distances, the 10 largest n_sites, and every structure with an atom isolated at 5 A or 6 A.
Ties broken by material_id; output order = sorted material_id.
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.scan, index_col="material_id").sort_index()
    for src in ["initial", "relaxed"]:
        v, d, n = df[f"vol_per_atom_{src}"], df[f"min_dist_{src}"], df[f"n_sites_{src}"]
        pick = {}
        pick["vol_max"] = v.sort_values(ascending=False, kind="stable").index[:10]
        pick["vol_min"] = v.sort_values(kind="stable").index[:10]
        pick["min_dist_min"] = d.sort_values(kind="stable").index[:10]
        pick["n_sites_max"] = n.sort_values(ascending=False, kind="stable").index[:10]
        pick["isolated_5A_or_6A"] = df.index[(df[f"n_isolated_5A_{src}"] > 0) | (df[f"n_isolated_6A_{src}"] > 0)]
        ids = sorted(set().union(*[set(x) for x in pick.values()]))
        with open(os.path.join(a.outdir, f"stress_ids_{src}.txt"), "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"[stress] {src}: {len(ids)} ids; " + "; ".join(f"{k}={len(x)}" for k, x in pick.items()) +
              f"; vol/atom range in set {v.loc[ids].min():.2f}..{v.loc[ids].max():.2f}; "
              f"min_dist range {d.loc[ids].min():.3f}..{d.loc[ids].max():.3f}", flush=True)


if __name__ == "__main__":
    main()
