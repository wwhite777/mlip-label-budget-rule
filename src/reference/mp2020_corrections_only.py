"""MP2020 correction (eV, total) per WBM entry under whatever pymatgen is installed - for the cross-version diagnosis.

python mp2020_corrections_only.py --structures relaxed|initial --out <npz> [--strict-anions X] [--n-workers 16]
Same steps as mbd_energy._work (entry = WBM CSE; entry._structure = the chosen structure; MaterialsProject2020
Compatibility(**kw).process_entry(entry, clean=True)). The correction does not depend on the entry energy (MP2020
adjustments are per-atom constants chosen from composition, structure and run parameters), so the DFT uncorrected
energy is kept. Depends only on numpy + pymatgen (runs in an old-pymatgen scratch venv); no labels are read.
"""
import argparse
import bz2
import json
import os
import warnings
from multiprocessing import get_context

import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WBM_CSE = os.path.join(PROJECT, "cache", "wbm", "wbm_cse.json.bz2")
WBM_INIT = os.path.join(PROJECT, "cache", "mbd", "2022-10-19-wbm-init-structs.json.bz2")
_G = {}


def _work(bounds):
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    warnings.filterwarnings("ignore")
    kw = {} if _G["sa"] is None else {"strict_anions": _G["sa"]}
    compat = MaterialsProject2020Compatibility(**kw)
    out = []
    for i in range(*bounds):
        mid = _G["mids"][i]
        try:
            entry = ComputedStructureEntry.from_dict(_G["cse"][mid])
            if _G["structs"] is not None:
                entry._structure = Structure.from_dict(_G["structs"][mid])  # noqa: SLF001
            p = compat.process_entry(entry, clean=True)
            if p is None:
                out.append((np.nan, "", "MP2020 compatibility rejected entry"))
                continue
            out.append((float(p.correction), ";".join(f"{a.name}={a.value:.6f}" for a in p.energy_adjustments), ""))
        except Exception as exc:  # noqa: BLE001
            out.append((np.nan, "", f"{type(exc).__name__}: {exc}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", required=True, choices=["relaxed", "initial"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-anions", default=None)
    ap.add_argument("--n-workers", type=int, default=16)
    a = ap.parse_args()
    import pymatgen.core
    try:
        import importlib.metadata as md
        pmg = md.version("pymatgen")
    except Exception:  # noqa: BLE001
        pmg = getattr(pymatgen.core, "__version__", "unknown")
    with bz2.open(WBM_CSE, "rt") as fh:
        raw = json.load(fh)
    cse = {raw["material_id"][k]: raw["computed_structure_entry"][k] for k in raw["material_id"]}
    mids = sorted(cse)
    structs = None
    if a.structures == "initial":
        with bz2.open(WBM_INIT, "rt") as fh:
            r2 = json.load(fh)
        structs = {r2["material_id"][k]: r2["initial_structure"][k] for k in r2["material_id"]}
    _G.update(mids=mids, cse=cse, structs=structs, sa=a.strict_anions)
    bounds = [(i, min(i + 2000, len(mids))) for i in range(0, len(mids), 2000)]
    with get_context("fork").Pool(a.n_workers) as pool:
        rows = [r for part in pool.map(_work, bounds) for r in part]
    np.savez_compressed(a.out, material_id=np.array(mids), correction=np.array([r[0] for r in rows]),
                        adjustments=np.array([r[1] for r in rows]), error=np.array([r[2] for r in rows]),
                        pymatgen_version=pmg, strict_anions=str(a.strict_anions), structures=a.structures)
    n_err = sum(1 for r in rows if r[2])
    print(f"[mp2020_corrections_only] pymatgen={pmg} structures={a.structures} strict_anions={a.strict_anions} "
          f"n={len(rows)} errors/rejected={n_err} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
