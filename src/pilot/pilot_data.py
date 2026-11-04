"""Read-only access to frozen inputs; the only label reader is PILOT guarded."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from analysis.config import Config, BudgetViolation, check_sha, resolve, verified_aux
from analysis.data import build_part, load_partition, load_universe
from analysis.features import element_symbols

FORMATION = "e_form_per_atom_mp2020_corrected"
HULL = "e_above_hull_mp2020_corrected_ppd_mp"


class ConfirmLabelError(AssertionError):
    """A non-PILOT material_id reached a label read or lookup."""


def assert_pilot_ids(ids, part_of: dict) -> None:
    """An explicit assertion remains active even under python -O."""
    bad = [str(mid) for mid in ids if part_of.get(str(mid)) != "PILOT"]
    if bad:
        raise ConfirmLabelError(f"PILOT-only label assertion: non-PILOT id(s) {bad[:3]}")


def load_metadata():
    """Verify frozen protocol bytes and load only label-free universe/partition data.

    This exploratory task checks the frozen contract/YAML and each input it uses;
    it does not invoke the full v1.2 gate over unrelated model/structure archives.
    """
    cfg = Config.build("pilot", tasks=["stability"])
    freeze = cfg.freeze_dir / "PREREGISTERED_p3_study_v1.2.sha256"
    for line in freeze.read_text().splitlines():
        sha, path = line.split(maxsplit=1)
        check_sha(resolve(path), sha, "frozen protocol")
    u, p = load_universe(cfg), load_partition(cfg)
    if set(u.material_id) != set(p.material_id) or p.material_id.duplicated().any():
        raise ValueError("partition must cover unique-prototype universe exactly once")
    return cfg, u, p


def load_pilot_part(cfg, universe, partition):
    check_sha(cfg.split_path, cfg.split_sha, "PILOT folds")
    s = pd.read_csv(cfg.split_path, dtype=str, keep_default_na=False)
    want = partition.loc[partition.part == "PILOT"].set_index("material_id")
    if set(s.material_id) != set(want.index):
        raise ValueError("PILOT folds do not cover PILOT exactly")
    if not np.array_equal(want.loc[s.material_id, "chemsys"].to_numpy(), s.chemsys.to_numpy()):
        raise ValueError("fold and partition chemical systems differ")
    if sorted(s.fold.astype(int).unique()) != list(range(cfg.n_folds)):
        raise ValueError("unexpected outer folds")
    part_of = dict(zip(partition.material_id, partition.part))
    formula = universe.set_index("material_id").loc[s.material_id, "formula"].to_numpy()
    return build_part("PILOT", s.material_id, s.chemsys, s.fold.astype(int), formula, part_of)


def composition_basis(cfg):
    """Use the frozen helper, or exactly reproduce its atomic-number sort without pymatgen."""
    try:
        elements, info = element_symbols(cfg)
        info["implementation"] = "analysis.features.element_symbols"
    except ModuleNotFoundError as error:
        if error.name != "pymatgen":
            raise
        source = cfg.y["auxiliary_inputs"]["element_basis_source"]
        if source["path"] not in cfg.y["implementation"]["element_basis"]:
            raise ValueError("element basis source differs from implementation specification")
        record = json.loads(verified_aux(cfg.y, "partition_loaded_columns").read_text())
        mp = record["assignment_inputs"]["mp"]
        if (mp["path"], mp["sha256"]) != (source["path"], source["sha256"]):
            raise ValueError("partition and frozen element basis sources disagree")
        formulas = pd.read_csv(verified_aux(cfg.y, "element_basis_source"), usecols=["formula"],
                               dtype=str, keep_default_na=False).formula
        periodic = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
                    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
                    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
                    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
        present = set(re.findall(r"[A-Z][a-z]?", " ".join(formulas)))
        elements = sorted(present, key=periodic.index)
        info = dict(source, n_elements=len(elements), columns_read=["formula"], verified_before_use=True,
                    implementation="equivalent atomic-number sort; pymatgen unavailable")
    if len(elements) != 89:
        raise ValueError(f"expected frozen 89-element basis, got {len(elements)}")
    return elements, info


class PilotLabels:
    """PILOT-only label storage; assert partition membership at every lookup.

    Fitting/tuning callers pass allowed_ids=their draw, enforcing the n-label cap.
    Evaluation callers omit it only when computing held-out metrics or prevalence.
    """
    def __init__(self, frame, part_of):
        assert_pilot_ids(frame.index, part_of)
        self._frame, self._part_of = frame, dict(part_of)

    def lookup(self, ids, *, allowed_ids=None):
        ids = np.asarray(ids).astype(str)
        assert_pilot_ids(ids, self._part_of)
        if allowed_ids is not None:
            assert_pilot_ids(allowed_ids, self._part_of)
            if not set(ids) <= set(allowed_ids):
                raise BudgetViolation("label lookup outside the replicate's n-label draw")
        return self._frame.loc[ids].copy()


def load_pilot_labels(path, ids, part_of):
    """Select PILOT physical rows from ids alone BEFORE parsing any label column.

    As in analysis.data.load_real_labels, skiprows removes all other data rows
    before pandas parses the selected label columns. No band-gap column is read.
    """
    ids = np.asarray(ids).astype(str)
    assert_pilot_ids(ids, part_of)  # must precede even opening the path
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate requested label ids")
    all_ids = pd.read_csv(path, usecols=["material_id"], dtype=str,
                          keep_default_na=False).material_id.to_numpy()
    selected = pd.Index(ids).get_indexer(all_ids) >= 0
    selected_ids = all_ids[selected]
    assert_pilot_ids(selected_ids, part_of)  # before label-column read
    if len(selected_ids) != len(ids) or set(selected_ids) != set(ids):
        raise ValueError("summary does not cover requested PILOT ids exactly once")
    lines = set((np.flatnonzero(selected) + 1).tolist())
    frame = pd.read_csv(path, usecols=["material_id", FORMATION, HULL], dtype=str,
                        keep_default_na=False, skiprows=lambda i: i != 0 and i not in lines)
    assert_pilot_ids(frame.material_id, part_of)
    frame = frame.set_index("material_id").loc[ids]
    if not np.array_equal(frame.index.to_numpy(), ids):
        raise ValueError("label row alignment failed")
    for col in (FORMATION, HULL):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col]).all():
            raise ValueError(f"non-finite PILOT labels: {col}")
    frame["stable"] = (frame[HULL] <= 0).astype(np.int8)
    manifest = {"source": str(path), "part": "PILOT", "rows_read": len(frame),
                "label_columns_read": [FORMATION, HULL], "confirm_label_rows_read": 0,
                "filter": "ids-only pass; assert PILOT ids; skip non-PILOT rows before parsing labels",
                "assertion": "assert_pilot_ids before reads and every PilotLabels.lookup"}
    return PilotLabels(frame, part_of), manifest
