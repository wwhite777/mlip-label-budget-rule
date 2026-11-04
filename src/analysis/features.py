"""Inputs of every method: backbone features (INITIAL structures, registry + sha256), element-fraction composition
vectors, the fixed Gaussian random projection, the random-init encoder features, and the zero-label references."""
from __future__ import annotations

import csv
import json
import re

import numpy as np
import pandas as pd

from analysis.config import AnalysisError, Config, aux_entries, check_sha, resolve, verified_aux
from analysis.data import PartData, PartitionError

# Auxiliary inputs (element basis source, its partition column record, MP2020 correction arrays, single-point
# energies, random-init features) come from the YAML block auxiliary_inputs; verified_aux() checks each expected
# sha256 BEFORE the file is opened ( finding 6).
_TOKEN = re.compile(r"^([A-Z][a-z]?)([0-9]*\.?[0-9]*)$")


def load_registry(cfg: Config) -> dict:
    reg = cfg.y["backbones"]["registry"]
    path = resolve(reg["path"])
    check_sha(path, reg["sha256"], "reference registry")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["backbone"]: r for r in rows}


def _align(mids: np.ndarray, part: PartData, what: str) -> np.ndarray:
    """Row index into `mids` for every part id (canonical order); every part id must be present."""
    idx = pd.Index(np.asarray(mids).astype(str))
    if idx.duplicated().any():
        raise AnalysisError(f"{what}: duplicate material_id")
    pos = idx.get_indexer(part.ids)
    if (pos < 0).any():
        raise AnalysisError(f"{what}: {int((pos < 0).sum())} part structure(s) missing")
    return pos


def backbone_features(cfg: Config, part: PartData, backbone: str, registry: dict) -> tuple:
    if cfg.y["backbones"]["feature_input"] != "initial" or cfg.y["inputs"]["primary"] != "initial":
        raise AnalysisError("feature input is not 'initial'")
    row = registry[backbone]
    path = resolve(row["features_initial"])
    sha = check_sha(path, row["features_initial_sha256"], f"features {backbone}")
    z = np.load(path, allow_pickle=False)
    mids = z["mids"].astype(str)
    failed = set(z["failed_ids"].astype(str).tolist()) & set(part.ids.tolist())
    if failed:
        raise AnalysisError(f"{backbone}: {len(failed)} part structure(s) failed featurisation")
    if str(z["structures_source"]) != "initial":
        raise AnalysisError(f"{backbone}: features are from {z['structures_source']} structures")
    X = np.asarray(z["feats"], dtype=np.float32)[_align(mids, part, backbone)]
    if not np.isfinite(X).all():
        raise AnalysisError(f"{backbone}: non-finite features")
    return X, {"path": row["features_initial"], "sha256": sha, "dim": int(X.shape[1])}


def element_symbols(cfg: Config) -> tuple:
    """The elements present in the MP energies file formulas (auxiliary_inputs.element_basis_source), sorted by atomic
    number. The partition column record must name the same file and hash (consistency of the frozen input graph)."""
    from pymatgen.core.periodic_table import Element
    ents = aux_entries(cfg.y)
    src = ents["element_basis_source"]
    if src["path"] not in str((cfg.y.get("implementation") or {}).get("element_basis", "")):
        raise AnalysisError(f"implementation.element_basis does not name {src['path']}")
    rec_path = verified_aux(cfg.y, "partition_loaded_columns")
    rec = json.loads(rec_path.read_text())["assignment_inputs"]["mp"]
    if rec.get("path") != src["path"] or rec.get("sha256") != src["sha256"]:
        raise AnalysisError(f"loaded_columns.json records {rec.get('path')} {rec.get('sha256')} but the YAML "
                            f"element_basis_source is {src['path']} {src['sha256']}")
    path = verified_aux(cfg.y, "element_basis_source")
    f = pd.read_csv(path, usecols=["formula"], dtype=str, keep_default_na=False)["formula"]
    els = sorted(set(re.findall(r"[A-Z][a-z]?", " ".join(f.tolist()))), key=lambda e: Element(e).Z)
    return els, {"path": src["path"], "sha256": src["sha256"], "n_elements": len(els), "columns_read": ["formula"],
                 "verified_before_use": True}


def parse_formula(text: str) -> dict:
    out = {}
    for tok in str(text).split():
        m = _TOKEN.match(tok)
        if not m:
            raise AnalysisError(f"cannot parse formula token {tok!r} in {text!r}")
        out[m.group(1)] = out.get(m.group(1), 0.0) + (float(m.group(2)) if m.group(2) else 1.0)
    if not out:
        raise AnalysisError(f"empty formula {text!r}")
    return out


def composition_features(formulas, elements: list) -> np.ndarray:
    col = {e: i for i, e in enumerate(elements)}
    X = np.zeros((len(formulas), len(elements)), dtype=np.float64)
    for i, f in enumerate(formulas):
        d = parse_formula(f)
        tot = sum(d.values())
        for e, c in d.items():
            if e not in col:
                raise AnalysisError(f"element {e} of {f!r} not in the element list")
            X[i, col[e]] = c / tot
    return X


def assert_wbm_elements_included(universe_formulas, elements: list) -> int:
    present = set()
    for f in universe_formulas:
        present.update(parse_formula(f).keys())
    missing = present - set(elements)
    if missing:
        raise AnalysisError(f"WBM elements not in the composition basis: {sorted(missing)}")
    return len(present)


def random_projection_matrix(cfg: Config, d_in: int) -> np.ndarray:
    rp = cfg.y["controls"]["random_projection_logistic"]
    if rp["distribution"] != "gaussian" or not str(rp["input"]).startswith("element_fraction"):
        raise AnalysisError("random projection spec changed")
    return np.random.default_rng(int(rp["seed"])).standard_normal((d_in, int(rp["dim"])))


# ------------------------------------------------------------------------------------------------------ references
def references(cfg: Config, part: PartData, backbone: str, registry: dict) -> dict:
    """R-sp and R-is2re hull-distance predictions over the part (NaN = missing); score = -e_hull_pred."""
    row = registry[backbone]
    out, info = {}, {}
    for ref, col, scol in (("r_sp", "ref_sp", "ref_sp_sha256"), ("r_is2re", "ref_is2re_as_published",
                                                                   "ref_is2re_sha256")):
        if not row.get(col):
            continue
        path = resolve(row[col])
        sha = check_sha(path, row[scol], f"{ref} {backbone}")
        suf = "sp" if ref == "r_sp" else "is2re"
        df = pd.read_csv(path, usecols=["material_id", f"e_form_pred_{suf}", f"e_hull_pred_{suf}"],
                         dtype={"material_id": str}, keep_default_na=True)
        if df["material_id"].duplicated().any():
            raise AnalysisError(f"{ref} {backbone}: duplicate material_id")
        sub = df.set_index("material_id").reindex(part.ids)
        e_form = sub[f"e_form_pred_{suf}"].to_numpy(dtype=np.float64)
        e_hull = sub[f"e_hull_pred_{suf}"].to_numpy(dtype=np.float64)
        out[ref] = {"e_form": e_form, "e_hull": e_hull}
        lo, hi = -10.0, 10.0
        lim = cfg.y["references"]["non_physical"]["sensitivity"]
        m = re.search(r"\[(-?[0-9.]+),\s*(-?[0-9.]+)\]", lim)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
        fin = np.isfinite(e_form)
        info[ref] = {"path": row[col], "sha256": sha, "n_missing": int((~np.isfinite(e_hull)).sum()),
                     "n_non_physical": int(((e_form < lo) | (e_form > hi))[fin].sum()), "clip": [lo, hi]}
    return {"scores": out, "info": info}


def clip_e_hull(e_form, e_hull, lo, hi):
    """Clip e_form_pred to [lo, hi]; e_hull_pred = e_form_pred - h_MP(c) shifts by the same amount."""
    return e_hull + (np.clip(e_form, lo, hi) - e_form)


def sp_diagnostic(cfg: Config, part: PartData, backbone: str, e_hull_sp: np.ndarray) -> tuple:
    """R-sp with RELAXED-structure MP2020 corrections for the structures of U whose correction differs by > 1 meV/atom.
    Not applicable for CHGNet (no MP2020 step)."""
    reg_conv = load_registry(cfg)[backbone]["convention"]
    if "NO second MP2020" in reg_conv:
        return None, {"applicable": False, "reason": reg_conv}
    keys = ("mp2020_corrections_initial", "mp2020_corrections_relaxed", f"sp_energy.{backbone}")
    p_ci, p_cr, p_en = [verified_aux(cfg.y, k) for k in keys]  # hash-checked BEFORE np.load
    ents = aux_entries(cfg.y)
    ci, cr = np.load(p_ci, allow_pickle=False), np.load(p_cr, allow_pickle=False)
    en = np.load(p_en, allow_pickle=False)
    ids = ci["material_id"].astype(str)
    ir = pd.Index(cr["material_id"].astype(str)).get_indexer(ids)
    ie = pd.Index(en["material_id"].astype(str)).get_indexer(ids)
    if (ir < 0).any() or (ie < 0).any():
        raise AnalysisError("diagnostic inputs do not cover the same material_ids")
    d = (cr["correction"][ir] - ci["correction"]) / en["n_atoms"][ie]
    in_u = pd.Index(list(part.part_of.keys())).get_indexer(ids) >= 0
    flag = np.abs(d) > 1e-3
    n_u = int((flag & in_u).sum())
    idx = _align(ids, part, "diagnostic")
    shift = np.where(flag[idx], d[idx], 0.0)
    info = {"applicable": True, "n_flagged_in_U": n_u, "n_flagged_in_part": int(flag[idx].sum()),
            "expected_in_U": 472, "count_matches": n_u == 472,
            "inputs": {ents[k]["path"]: ents[k]["sha256"] for k in keys}, "inputs_verified_before_use": True}
    return e_hull_sp + shift, info


def correction_strata(cfg: Config, part: PartData) -> tuple:
    s = cfg.y["sensitivities"]["correction_regime_strata"]
    path = resolve(s["file"])
    sha = check_sha(path, s["sha256"], "correction strata")
    colmap = {"oxidation_state_dependent_anion": "anion_assignment_oxi_dependent",
              "hubbard_u_metal_with_O_or_F": "hubbard_u_tm_with_O_or_F"}
    df = pd.read_csv(path, usecols=["material_id"] + [colmap[k] for k in s["strata"]], dtype=str,
                     keep_default_na=False)
    if set(df["material_id"]) - set(part.part_of):
        raise PartitionError("strata ids outside U")
    df = df.set_index("material_id").reindex(part.ids)
    out = {k: (df[colmap[k]] == "True").to_numpy() for k in s["strata"]}
    return out, {"path": s["file"], "sha256": sha, "column_map": colmap,
                 "counts_in_part": {k: int(v.sum()) for k, v in out.items()}}


def relaxed_features(cfg: Config, part: PartData, backbone: str, registry: dict) -> tuple:
    """DFT-relaxed-structure features (registry features_relaxed + sha256) for the C3 oracle arm (descriptive only)."""
    if cfg.y["inputs"].get("oracle_arm_only") != "relaxed":
        raise AnalysisError("inputs.oracle_arm_only is not 'relaxed'")
    row = registry[backbone]
    if not row.get("features_relaxed") or not row.get("features_relaxed_sha256"):
        raise AnalysisError(f"{backbone}: no features_relaxed in the registry")
    path = resolve(row["features_relaxed"])
    sha = check_sha(path, row["features_relaxed_sha256"], f"relaxed features {backbone}")
    z = np.load(path, allow_pickle=False)
    if "structures_source" in z.files and str(z["structures_source"]) == "initial":
        raise AnalysisError(f"{backbone}: features_relaxed holds initial-structure features")
    mids = z["mids"].astype(str)
    if "failed_ids" in z.files:
        failed = set(z["failed_ids"].astype(str).tolist()) & set(part.ids.tolist())
        if failed:
            raise AnalysisError(f"{backbone}: {len(failed)} part structure(s) failed relaxed featurisation")
    X = np.asarray(z["feats"], dtype=np.float32)[_align(mids, part, f"{backbone} relaxed")]
    if not np.isfinite(X).all():
        raise AnalysisError(f"{backbone}: non-finite relaxed features")
    return X, {"path": row["features_relaxed"], "sha256": sha, "dim": int(X.shape[1])}


_TEXT_REF = re.compile(r"([\w./-]+\.[\w.]+)(?:,\s*|\s+\()sha256\s+([0-9a-f]{64})")


def mptrj_membership(cfg: Config, part: PartData) -> tuple:
    """C1 strata: sys_in_mptrj per part structure from the file named (with its sha256) in
    descriptive_specs.c1_membership_association.strata; hash-checked before reading."""
    txt = str(cfg.y["descriptive_specs"]["c1_membership_association"]["strata"])
    m = _TEXT_REF.search(txt)
    col = re.match(r"\s*(\w+)\s+from\s", txt)
    if not m or not col:
        raise AnalysisError(f"cannot parse the C1 strata reference {txt!r}")
    path = resolve(m.group(1))
    sha = check_sha(path, m.group(2), "C1 membership strata")
    df = pd.read_csv(path, usecols=["material_id", col.group(1)], dtype=str, keep_default_na=False)
    if set(df["material_id"]) - set(part.part_of):
        raise PartitionError("C1 strata ids outside U")
    sub = df.set_index("material_id").reindex(part.ids)[col.group(1)]
    if sub.isna().any() or not set(sub.unique()) <= {"True", "False"}:
        raise AnalysisError(f"C1 strata column {col.group(1)} missing or not boolean for part structures")
    present = (sub == "True").to_numpy()
    return present, {"path": m.group(1), "sha256": sha, "column": col.group(1), "n_present": int(present.sum()),
                     "n_absent": int((~present).sum())}
