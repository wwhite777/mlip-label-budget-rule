"""Total energy -> MP2020-corrected formation energy per atom, Matbench Discovery convention (P3, study).

Not a re-implementation: the two steps are the ones Matbench Discovery applies to predicted energies.
  1. MP2020 correction. Pinned repo commit e3d7966 (cache/reference/mbd_src/matbench_discovery_discovery.py,
     write_discovery_artifacts): entry = copy of the WBM ComputedStructureEntry; entry._energy = predicted energy;
     entry._structure = the structure the model evaluated; processed = MaterialsProject2020Compatibility()
     .process_entry(entry, clean=True); processed is None -> no prediction ("MP2020 compatibility rejected entry").
     strict_anions is the pymatgen default unless a caller passes another value (the convention check does, to
     attribute differences; the pinned label script data/wbm/compile_wbm_test_set.py uses strict_anions="no_check").
  2. Formation energy: matbench_discovery 1.0.0 (pip) energy.get_e_form_per_atom({"composition": formula,
     "energy": processed.energy}, refs), with formula = WBM summary `formula` (as write_discovery_artifacts) and
     refs = matbench_discovery.energy.mp_elemental_ref_energies (1.0.0: MP elemental reference entries file
     2023-02-07-mp-elemental-reference-entries.json.gz, Figshare file 40387775, energy_per_atom ROUNDED to 4 dp) or
     the unrounded energy_per_atom of the same entries (pinned commit energy.py: mp_elemental_ref_energies, no
     rounding). Which one reproduces the WBM label column is decided by convention_check.py.
Energies only; this module never reads a label column.
"""
import bz2
import json
import os
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
MBD_CACHE = os.path.join(PROJECT, "cache", "reference", "mbd_cache")
os.environ.setdefault("MATBENCH_DISCOVERY_CACHE_DIR", MBD_CACHE)  # must precede any matbench_discovery import
SUMMARY = os.path.join(PROJECT, "cache", "mbd", "2023-12-13-wbm-summary_v64706751.csv.gz")
WBM_CSE = os.path.join(PROJECT, "cache", "wbm", "wbm_cse.json.bz2")
WBM_INIT = os.path.join(PROJECT, "cache", "mbd", "2022-10-19-wbm-init-structs.json.bz2")
ELEM_REFS = os.path.join(MBD_CACHE, "mp", "2023-02-07-mp-elemental-reference-entries.json.gz")
ELEM_REFS_MD5 = "6e93b6f38d6e27d6c811d3cafb23a070"  # data-files.yml @ e3d7966, mp_elemental_ref_entries


def versions():
    import importlib.metadata as m
    return {k: m.version(k) for k in ["matbench-discovery", "pymatgen", "numpy", "pandas"]}


def load_summary():
    """WBM summary, formulas read with keep_default_na=False (so no formula string becomes NaN)."""
    df = pd.read_csv(SUMMARY, keep_default_na=False)
    for c in df.columns:
        if c not in ("material_id", "formula", "protostructure_spglib_initial_structure", "protostructure_spglib",
                     "unique_prototype"):
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
    if df["unique_prototype"].dtype != bool:
        df["unique_prototype"] = df["unique_prototype"].map({"True": True, "False": False})
    assert df["unique_prototype"].dtype == bool and int(df["unique_prototype"].sum()) == 215488
    return df.set_index("material_id")


def load_cse():
    with bz2.open(WBM_CSE, "rt") as fh:
        raw = json.load(fh)
    return {raw["material_id"][k]: raw["computed_structure_entry"][k] for k in raw["material_id"]}


def load_initial_structures():
    with bz2.open(WBM_INIT, "rt") as fh:
        raw = json.load(fh)
    return {raw["material_id"][k]: raw["initial_structure"][k] for k in raw["material_id"]}


def elemental_refs(kind):
    """'mbd1.0.0' = matbench_discovery 1.0.0 mp_elemental_ref_energies (rounded 4 dp); 'unrounded' = same entries."""
    from matbench_discovery import energy as mbd_energy
    if kind == "mbd1.0.0":
        return dict(mbd_energy.mp_elemental_ref_energies)
    if kind == "unrounded":
        return {el: e.energy_per_atom for el, e in mbd_energy.mp_elem_reference_entries.items()}
    raise ValueError(kind)


# ---------------------------------------------------------------- MP2020 (worker globals set before fork)
_G = {}


def _work(bounds):
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    warnings.filterwarnings("ignore")
    kw = {} if _G["strict_anions"] is None else {"strict_anions": _G["strict_anions"]}
    compat = MaterialsProject2020Compatibility(**kw)
    out = []
    for i in range(*bounds):
        mid, e = _G["mids"][i], _G["energies"][i]
        if not np.isfinite(e):
            out.append((np.nan, np.nan, "", "no energy"))
            continue
        try:
            entry = ComputedStructureEntry.from_dict(_G["cse"][mid])
            entry._energy = float(e)  # noqa: SLF001 - as write_discovery_artifacts
            if _G["structs"] is not None:
                entry._structure = Structure.from_dict(_G["structs"][mid])  # noqa: SLF001
            p = compat.process_entry(entry, clean=True)
            if p is None:
                out.append((np.nan, np.nan, "", "MP2020 compatibility rejected entry"))
                continue
            adj = ";".join(f"{a.name}={a.value:.6f}" for a in p.energy_adjustments)
            out.append((float(p.energy), float(p.correction), adj, ""))
        except Exception as exc:  # noqa: BLE001 - recorded per row, never silent
            out.append((np.nan, np.nan, "", f"{type(exc).__name__}: {exc}"))
    return out


def mp2020_corrected(mids, energies, cse, structs=None, strict_anions=None, n_workers=24, chunk=2000):
    """Corrected total energy per id. structs=None keeps each CSE's own (DFT-relaxed) structure; otherwise
    mid -> structure dict of the structure the model evaluated. Returns DataFrame (index material_id):
    corrected_energy, correction, adjustments, error."""
    _G.update(mids=list(mids), energies=np.asarray(energies, dtype=np.float64), cse=cse, structs=structs,
              strict_anions=strict_anions)
    bounds = [(i, min(i + chunk, len(mids))) for i in range(0, len(mids), chunk)]
    with get_context("fork").Pool(n_workers) as pool:
        rows = [r for part in pool.map(_work, bounds) for r in part]
    return pd.DataFrame(rows, index=pd.Index(list(mids), name="material_id"),
                        columns=["corrected_energy", "correction", "adjustments", "error"])


def e_form_per_atom(formulas, total_energies, refs):
    """matbench_discovery 1.0.0 get_e_form_per_atom per row; NaN energy -> NaN."""
    from matbench_discovery.energy import get_e_form_per_atom
    return np.array([get_e_form_per_atom({"composition": f, "energy": float(e)}, refs) if np.isfinite(e) else np.nan
                     for f, e in zip(formulas, total_energies)])
