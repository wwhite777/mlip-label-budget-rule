"""Label-free MP2020 correction-regime descriptors per unique-prototype WBM structure.

python3 src/membership/correction_flags.py

Inputs (label-blind; only material_id, formula, unique_prototype of the WBM summary and material_id, formula of the MP
energies file are loaded, through src/splits/make_splits.load_universe, the same universe as the split manifests):
  cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz, cache/mbd/2025-02-01-mp-energies.csv.gz
No energy, correction (e_correction_per_atom_mp2020), hull or band-gap column is read.

Rule source (pymatgen v2023.5.10, the behaviour Matbench Discovery states reproduces the released WBM values):
  pymatgen/entries/MP2020Compatibility.yaml  CompositionCorrections: oxide/peroxide/superoxide/ozonide (O), S, F, Cl,
      Br, I, N, Se, Si, Sb, Te, H; GGAUMixingCorrections keyed by O and F for V, Cr, Mn, Fe, Co, Ni, W, Mo
  pymatgen/entries/compatibility.py  MaterialsProject2020Compatibility.get_adjustments: unary compositions get no
      correction; O -> oxide-type correction whenever present; S -> correction if sulfide_type is sulfide/polysulfide
      (structure-dependent); Br I Se Si Sb Te H N F Cl -> correction if the guessed oxidation state is < 0, else only
      if the element is the most electronegative one (Pauling X, pymatgen periodic_table.json); U mixing correction
      for the metals keyed under the most electronegative element (O or F).
Local copies of the three source files: cache/membership/pymatgen_v2023.5.10_*.

Composition-level flags (do not reproduce the corrections; they mark where the scheme can act):
  arity                       number of distinct elements
  mp2020_anion_els            MP2020 anion-correction elements present (O S F Cl Br I N H Se Si Sb Te), ";"-joined
  contains_mp2020_anion_el    >= 1 of them present (pure containment, as on the task card)
  mp2020_anion_correctable    contains_mp2020_anion_el and arity >= 2 (pymatgen skips unary compositions)
  most_electroneg_el          element with the largest pymatgen X (empty if any element lacks X)
  x_undefined                 an element has no X in pymatgen (He, Ne, Ar, ...): pymatgen's sort order is undefined
  oxi_dependent_anion_els     Br I Se Si Sb Te H N F Cl present but NOT the most electronegative element: their
                              correction depends on oxi_state_guesses (the version-sensitive path, MBD issue #358)
  anion_assignment_oxi_dependent  arity >= 2 and oxi_dependent_anion_els non-empty
  hubbard_u_tm_els            V Cr Mn Fe Co Ni W Mo present, ";"-joined
  hubbard_u_tm_with_O_or_F    >= 1 U metal and (O or F) present (task-card definition)
  hubbard_u_pymatgen_rule     >= 1 U metal and most_electroneg_el in {O, F} (compatibility.py rule)
Outputs: result/membership/correction_regime_flags.csv.gz (gzip mtime=0), result/membership/correction_regime_counts.json
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from splits.make_splits import ELEMENT_RE, LABEL_PATTERN, load_universe, sha256_file  # noqa: E402

WBM = ROOT / "cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz"
MP = ROOT / "cache/mbd/2025-02-01-mp-energies.csv.gz"
PT = ROOT / "cache/membership/pymatgen_v2023.5.10_periodic_table.json"
SRC_FILES = ["pymatgen_v2023.5.10_MP2020Compatibility.yaml", "pymatgen_v2023.5.10_compatibility.py",
             "pymatgen_v2023.5.10_periodic_table.json"]
SRC_URL = "https://raw.githubusercontent.com/materialsproject/pymatgen/v2023.5.10/pymatgen/"
OUT_CSV = ROOT / "result/membership/correction_regime_flags.csv.gz"
OUT_JSON = ROOT / "result/membership/correction_regime_counts.json"

ANION_ELS = ("O", "S", "F", "Cl", "Br", "I", "N", "H", "Se", "Si", "Sb", "Te")
OXI_DEPENDENT = ("Br", "I", "Se", "Si", "Sb", "Te", "H", "N", "F", "Cl")
U_METALS = ("V", "Cr", "Mn", "Fe", "Co", "Ni", "W", "Mo")


def main() -> int:
    u, rec = load_universe(WBM, MP)
    for r in rec.values():
        assert not any(LABEL_PATTERN.search(c) for c in r["columns"]), r
    ptab = json.loads(PT.read_text())
    xval = {el: d.get("X") for el, d in ptab.items()}
    rows = []
    for mid, formula, csys in zip(u["material_id"], u["formula"], u["chemsys"]):
        els = sorted(set(ELEMENT_RE.findall(formula)))
        assert "-".join(els) == csys
        arity = len(els)
        anions = [e for e in ANION_ELS if e in els]
        x_undef = any(not isinstance(xval.get(e), (int, float)) for e in els)
        if x_undef:
            men = ""
        else:
            xmax = max(xval[e] for e in els)
            tops = [e for e in els if xval[e] == xmax]
            men = tops[0] if len(tops) == 1 else ";".join(tops)
        oxi_dep = [e for e in OXI_DEPENDENT if e in els and e != men]
        umet = [e for e in U_METALS if e in els]
        has_of = ("O" in els) or ("F" in els)
        rows.append({
            "material_id": mid, "chemsys": csys, "arity": arity,
            "mp2020_anion_els": ";".join(anions),
            "contains_mp2020_anion_el": bool(anions),
            "mp2020_anion_correctable": bool(anions) and arity >= 2,
            "most_electroneg_el": men, "x_undefined": x_undef,
            "oxi_dependent_anion_els": ";".join(oxi_dep),
            "anion_assignment_oxi_dependent": arity >= 2 and bool(oxi_dep),
            "hubbard_u_tm_els": ";".join(umet),
            "hubbard_u_tm_with_O_or_F": bool(umet) and has_of,
            "hubbard_u_pymatgen_rule": bool(umet) and men in ("O", "F"),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))

    n = len(df)
    bool_cols = ["contains_mp2020_anion_el", "mp2020_anion_correctable", "x_undefined",
                 "anion_assignment_oxi_dependent", "hubbard_u_tm_with_O_or_F", "hubbard_u_pymatgen_rule"]
    per_el = {e: int(df["mp2020_anion_els"].str.split(";").map(lambda s, e=e: e in s).sum()) for e in ANION_ELS}
    per_oxi = {e: int(df["oxi_dependent_anion_els"].str.split(";").map(lambda s, e=e: e in s).sum())
               for e in OXI_DEPENDENT}
    per_u = {e: int(df["hubbard_u_tm_els"].str.split(";").map(lambda s, e=e: e in s).sum()) for e in U_METALS}
    disagree = df[df["hubbard_u_tm_with_O_or_F"] != df["hubbard_u_pymatgen_rule"]]
    summary = {
        "n_structures": n,
        "counts": {c: int(df[c].sum()) for c in bool_cols},
        "fractions": {c: round(float(df[c].mean()), 4) for c in bool_cols},
        "arity_counts": {str(k): int(v) for k, v in df["arity"].value_counts().sort_index().items()},
        "structures_containing_anion_el": per_el,
        "structures_with_oxi_dependent_anion_el": per_oxi,
        "structures_containing_u_metal": per_u,
        "most_electroneg_el_top10": {k: int(v) for k, v in df["most_electroneg_el"].value_counts().head(10).items()},
        "hubbard_u_rule_disagreements": {"n": int(len(disagree)),
                                         "examples": disagree[["material_id", "chemsys", "most_electroneg_el"]]
                                         .head(10).values.tolist()},
        "inputs": rec,
        "rule_sources": {f: {"url": SRC_URL + {"MP2020Compatibility.yaml": "entries/MP2020Compatibility.yaml",
                                                "compatibility.py": "entries/compatibility.py",
                                                "periodic_table.json": "core/periodic_table.json"}[f.split("_", 2)[2]],
                             "sha256": sha256_file(ROOT / "cache/membership" / f)} for f in SRC_FILES},
        "labels_read": False,
        "output": {"csv": str(OUT_CSV.relative_to(ROOT)), "sha256": sha256_file(OUT_CSV),
                   "bytes": OUT_CSV.stat().st_size},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__))},
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_structures", "counts", "fractions", "arity_counts",
                                               "hubbard_u_rule_disagreements")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
