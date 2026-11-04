"""CPU-only geometry scan of all WBM relaxed and initial structures (no model, no GPU).

python scan_structures.py --out cache/features/wbm_struct_scan.csv.gz [--workers 12]

Per material_id and source: n_sites, volume per atom, minimum interatomic distance (within 6 A, periodic),
number of atoms with no neighbour within 5 A and within 6 A ("isolated"; CHGNet's atom-graph cutoff is 6 A
and its converter raises on isolated atoms by default).
"""
import argparse
import bz2
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as E  # noqa: E402


def _geom(d):
    from pymatgen.core import Structure
    s = Structure.from_dict(d)
    c, _, _, dist = s.get_neighbor_list(r=6.0)
    n = len(s)
    has5 = np.zeros(n, bool)
    has5[c[dist <= 5.0]] = True
    has6 = np.zeros(n, bool)
    has6[c] = True
    return n, s.volume / n, float(dist.min()) if len(dist) else float("nan"), int((~has5).sum()), int((~has6).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import pandas as pd
    cols = {}
    for src in ["relaxed", "initial"]:
        path, key, sub = E.STRUCT_FILES[src]
        with bz2.open(path, "rt") as fh:
            raw = json.load(fh)
        rows = sorted(raw["material_id"].keys(), key=int)
        mids = [raw["material_id"][r] for r in rows]
        dicts = [raw[key][r][sub] if sub else raw[key][r] for r in rows]
        del raw
        with Pool(a.workers) as p:
            res = p.map(_geom, dicts, chunksize=500)
        df = pd.DataFrame(res, columns=["n_sites", "vol_per_atom", "min_dist", "n_isolated_5A", "n_isolated_6A"],
                          index=pd.Index(mids, name="material_id")).add_suffix(f"_{src}")
        cols[src] = df
        print(f"[scan] {src}: {len(df)} structures from {path}", flush=True)
    out = cols["relaxed"].join(cols["initial"], how="outer")
    out.to_csv(a.out)
    for src in ["relaxed", "initial"]:
        v = out[f"vol_per_atom_{src}"]
        print(f"[scan] {src}: vol/atom min {v.min():.3f} p0.1 {v.quantile(.001):.3f} median {v.median():.3f} "
              f"p99.9 {v.quantile(.999):.3f} max {v.max():.3f}; min_dist min {out[f'min_dist_{src}'].min():.3f}; "
              f"structures with isolated atoms: 5A {(out[f'n_isolated_5A_{src}'] > 0).sum()}, "
              f"6A {(out[f'n_isolated_6A_{src}'] > 0).sum()}", flush=True)


if __name__ == "__main__":
    main()
