"""Only explicitly allowed non-label columns are parsed in this module."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .common import ROOT, ProtocolError, checked, resolve, load_yaml
from analysis.data import build_part
from analysis.features import composition_features

SUMMARY_COLUMNS = frozenset({'material_id', 'formula', 'unique_prototype', 'n_sites'})
TABLE_COLUMNS = {
    'summary': SUMMARY_COLUMNS,
    'partition': frozenset({'material_id', 'part', 'chemsys'}),
    'folds': frozenset({'material_id', 'chemsys', 'fold'}),
    'campaign': frozenset({'material_id', 'part', 'variant', 'chemsys', 'round', 'assignment'}),
    'element_basis': frozenset({'formula'}),
    'reference': frozenset({'material_id', 'e_form_pred_sp', 'e_hull_pred_sp'}),
    'strata': frozenset({'material_id', 'anion_assignment_oxi_dependent', 'hubbard_u_tm_with_O_or_F'}),
}
TRACE = []
PERIODIC = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
            "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


def metadata_csv(path, columns, *, table='summary'):
    # Summary access is always restricted, even if a caller supplies another schema.
    is_summary = 'wbm-summary' in Path(path).name or Path(path).resolve() == resolve(load_yaml()['population']['summary']['path']).resolve()
    allowed = SUMMARY_COLUMNS if is_summary else TABLE_COLUMNS.get(table, frozenset())
    if not columns or not set(columns) <= allowed:
        raise ProtocolError("label column or unapproved metadata column attempted outside guarded reader")
    TRACE.append(dict(path=str(path), columns=list(columns)))
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)


def align(source_ids, target_ids):
    source_ids = pd.Index(np.asarray(source_ids).astype(str))
    if source_ids.has_duplicates:
        raise ProtocolError("duplicate source ids")
    idx = source_ids.get_indexer(target_ids)
    if (idx < 0).any():
        raise ProtocolError("incomplete input coverage")
    return idx


def load_part(doc, part_name="CONFIRM", root=ROOT):
    pop = doc["population"]
    p = metadata_csv(checked(pop["partition"], root), ["material_id", "part", "chemsys"], table="partition")
    if p.material_id.duplicated().any():
        raise ProtocolError("duplicate partition ids")
    for name in ("CONFIRM", "PILOT"):
        pp = p.loc[p.part == name]
        if len(pp) != pop[name.lower() + "_n"] or pp.chemsys.nunique() != pop[name.lower() + "_systems"]:
            raise ProtocolError("partition count drift")
    if set(p.loc[p.part == "PILOT", "chemsys"]) & set(p.loc[p.part == "CONFIRM", "chemsys"]):
        raise ProtocolError("PILOT/CONFIRM systems overlap")
    u = metadata_csv(checked(pop["summary"], root), ["material_id", "formula", "unique_prototype"])
    u = u.loc[u.unique_prototype == "True"]
    if len(u) != pop["universe_n"] or u.material_id.duplicated().any() or set(u.material_id) != set(p.material_id):
        raise ProtocolError("unique-prototype universe mismatch")
    spec = doc["designs"]["P"]["folds"] if part_name == "CONFIRM" else doc["validation_inputs"]["pilot_folds"]
    folds = metadata_csv(checked(spec, root), ["material_id", "chemsys", "fold"], table="folds")
    pp = p.loc[p.part == part_name].set_index("material_id")
    if folds.material_id.duplicated().any() or set(folds.material_id) != set(pp.index):
        raise ProtocolError("fold coverage mismatch")
    if not np.array_equal(pp.loc[folds.material_id, "chemsys"], folds.chemsys):
        raise ProtocolError("fold systems mismatch")
    if sorted(folds.fold.astype(int).unique()) != list(range(doc["designs"]["P"]["n_folds"])):
        raise ProtocolError("fold count mismatch")
    part_of = dict(zip(p.material_id, p.part))
    form = u.set_index("material_id").loc[folds.material_id, "formula"].to_numpy()
    return build_part(part_name, folds.material_id, folds.chemsys, folds.fold.astype(int), form, part_of)


def campaign_assignments(doc, part, root=ROOT):
    frames, masks = {}, {}
    for variant in ("S-a", "S-b"):
        spec = doc["designs"]["S"][variant]
        f = metadata_csv(checked(spec["assignment"], root), ["material_id", "part", "variant", "chemsys", "round", "assignment"], table="campaign")
        idx = align(f.material_id, part.ids)
        if len(f) != part.n:
            raise ProtocolError("campaign population mismatch")
        f = f.iloc[idx].reset_index(drop=True)
        if set(f.part) != {part.part} or set(f.variant) != {variant} or not np.array_equal(f.chemsys, part.chemsys):
            raise ProtocolError("campaign identity mismatch")
        rd = np.array([int(m.split("-")[1]) for m in part.ids])
        train = f.assignment.to_numpy() == "train"
        test = f.assignment.to_numpy() == "test"
        shared = np.isin(part.chemsys, part.chemsys[train])
        wanted = ~train if variant == "S-a" else ~train & ~shared
        if not np.array_equal(f["round"].astype(int), rd) or not set(rd) <= {1, 2, 3, 4, 5}:
            raise ProtocolError("campaign round mismatch")
        expected_assignment = np.where(train, "train", np.where(wanted, "test", "excluded"))
        # pilot uses an explicit excluded reason; only its train/test masks govern.
        if not np.array_equal(train, np.isin(rd, doc["designs"]["S"]["train_rounds"])) or not np.array_equal(test, wanted):
            raise ProtocolError("campaign split mismatch")
        if train.sum() != doc["designs"]["S"]["train_n"] or len(np.unique(part.chemsys[train])) != doc["designs"]["S"]["train_systems"]:
            raise ProtocolError("campaign training counts mismatch")
        if test.sum() != spec["test_n"] or len(np.unique(part.chemsys[test])) != spec["test_systems"]:
            raise ProtocolError("campaign test counts mismatch")
        if len(set(part.chemsys[train]) & set(part.chemsys[test])) != spec["shared_train_test_systems"]:
            raise ProtocolError("campaign shared systems mismatch")
        masks[variant] = np.flatnonzero(test)
        frames[variant] = f
    train = np.flatnonzero(frames["S-a"].assignment.to_numpy() == "train")
    if not np.array_equal(train, np.flatnonzero(frames["S-b"].assignment.to_numpy() == "train")):
        raise ProtocolError("campaign train pools differ")
    if not set(masks["S-b"]) <= set(masks["S-a"]):
        raise ProtocolError("S-b is not an S-a subset")
    return train, masks, frames


def load_inputs(doc, part_name="CONFIRM", root=ROOT):
    part = load_part(doc, part_name, root)
    formulas = metadata_csv(checked(doc["inputs"]["element_basis"], root), ["formula"], table="element_basis").formula
    present = set(re.findall(r"[A-Z][a-z]?", " ".join(formulas)))
    elements = [e for e in PERIODIC if e in present]
    if len(elements) != doc["inputs"]["composition_dim"] or set(elements) != present:
        raise ProtocolError("element basis mismatch")
    comp = composition_features(part.formula, elements)
    if not np.isfinite(comp).all() or not np.allclose(comp.sum(axis=1), 1):
        raise ProtocolError("invalid composition fractions")
    refs, feats, report = {}, {}, {}
    for bb in doc["backbones"]["order"]:
        refpath = checked(doc["inputs"]["reference_inputs"][bb], root)
        if bb == "chgnet" and not refpath.name.endswith("_initial_nomp2020.csv.gz"):
            raise ProtocolError("CHGNet double-correction path")
        ref = metadata_csv(refpath, ["material_id", "e_form_pred_sp", "e_hull_pred_sp"], table="reference")
        pos = align(ref.material_id, part.ids)
        e, h = ref.iloc[pos][["e_form_pred_sp", "e_hull_pred_sp"]].astype(float).to_numpy().T
        if not np.isfinite(e).all() or not np.isfinite(h).all():
            raise ProtocolError(f"nonfinite references: {bb}")
        refs[bb] = dict(r_sp=dict(e_form=e, e_hull=h))
        path = checked(doc["inputs"]["feature_inputs"][bb], root)
        with np.load(path, allow_pickle=False) as z:
            if str(z["structures_source"]) != "initial" or str(z["weights_sha256"]) != doc["inputs"]["checkpoint_inputs"][bb]["sha256"]:
                raise ProtocolError(f"feature provenance mismatch: {bb}")
            if set(z["failed_ids"].astype(str)) & set(part.ids):
                raise ProtocolError(f"failed features: {bb}")
            pos = align(z["mids"], part.ids)
            feats[bb] = np.asarray(z["feats"], dtype=np.float32)[pos]
            if not np.isfinite(feats[bb]).all():
                raise ProtocolError(f"nonfinite features: {bb}")
            report[bb] = dict(rows=part.n, feature_shape=list(feats[bb].shape),
                              feature_nonfinite=0, reference_nonfinite=0, missing_ids=0,
                              source_feature_rows=len(z["mids"]), source_reference_rows=len(ref),
                              weights_sha256=str(z["weights_sha256"]), pooled_tensor=str(z["pooled_tensor"]))
        print(f"covered {part_name} {bb}: {feats[bb].shape}", flush=True)
    hulls = np.column_stack([refs[b]["r_sp"]["e_hull"] for b in doc["backbones"]["order"]])
    hmp = refs["mace_medium"]["r_sp"]["e_form"] - hulls[:, 0]
    deviations = {b: float(np.max(np.abs(refs[b]["r_sp"]["e_form"] - refs[b]["r_sp"]["e_hull"] - hmp))) for b in refs}
    if max(deviations.values()) > doc["probe"]["common_h_tolerance_eV_atom"]:
        raise ProtocolError("fixed MP hull disagreement")
    return SimpleNamespace(part=part, comp=comp, refs=refs, hulls=hulls, feats=feats, hmp=hmp,
        coverage=dict(part=part_name, structures=part.n, systems=len(part.systems),
                      fold_sizes=np.bincount(part.fold).tolist(), checkpoints=report,
                      common_h_max_deviations=deviations, labels_opened=False, csv_column_trace=list(TRACE)))
