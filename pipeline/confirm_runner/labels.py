"""The only real-label reader; synthetic mode returns before opening source data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import ProtocolError, ROOT, YAML, resolve, checked

FORMATION = "e_form_per_atom_mp2020_corrected"
HULL = "e_above_hull_mp2020_corrected_ppd_mp"


class LabelAccessError(AssertionError):
    pass


class Labels:
    def __init__(self, frame, part, trace):
        if frame.index.has_duplicates or not np.array_equal(frame.index.to_numpy(), part.ids):
            raise LabelAccessError("label alignment failed")
        if not np.isfinite(frame[[FORMATION, HULL]].to_numpy()).all():
            raise LabelAccessError("nonfinite labels at run start: STOP, no fallback/intersection")
        self.frame, self.part, self.trace = frame, part, trace
        self.active_draw = None
        self.accesses = 0

    def lookup(self, ids, *, allowed_ids=None):
        ids = np.asarray(ids).astype(str)
        self.part.index_of(ids)
        if any(self.part.part_of.get(m) != self.part.part for m in ids):
            raise LabelAccessError("label partition mismatch")
        if self.active_draw is not None:
            if allowed_ids is None or not set(allowed_ids) <= self.active_draw:
                raise LabelAccessError("unguarded evaluation or altered draw inside fitting")
        if allowed_ids is not None and not set(ids) <= set(allowed_ids):
            raise LabelAccessError("label lookup outside n-label draw")
        self.accesses += 1
        return self.frame.loc[ids].copy()


def read_labels(doc, part, *, synthetic=False, seed=20261022, yaml_path=YAML, root=ROOT):
    if synthetic:
        # Independent random labels and numeric targets; never inspect a data input.
        rng = np.random.default_rng(seed)
        y = (rng.random(part.n) < 0.15).astype(np.int8)
        magnitude = rng.uniform(0.01, 0.5, part.n)
        hull = np.where(y == 1, -magnitude, magnitude)
        formation = rng.normal(-1, 0.5, part.n)
        frame = pd.DataFrame({FORMATION: formation, HULL: hull, "stable": y}, index=part.ids)
        return Labels(frame, part, dict(source="SYNTHETIC", seed=seed, prevalence_parameter=0.15,
            independence="Bernoulli and continuous targets depend only on seed and canonical row count, independent of every input",
            label_columns_opened=[], real_label_rows_parsed=0))
    # Shared guard precedes source I/O, including the IDs pass.
    require_protocol(yaml_path=yaml_path,root=root)
    if part.part != 'CONFIRM':
        raise LabelAccessError('non-CONFIRM population cannot authorize real labels')
    path = checked(doc['population']['summary'], root)
    all_ids = pd.read_csv(path, usecols=['material_id'], dtype=str, keep_default_na=False).material_id.to_numpy()
    selected = pd.Index(part.ids).get_indexer(all_ids) >= 0
    if len(all_ids[selected]) != part.n or set(all_ids[selected]) != set(part.ids):
        raise LabelAccessError("summary label IDs do not cover population exactly")
    lines = set((np.flatnonzero(selected) + 1).tolist())
    # Labels of excluded rows are skipped before parsing; no gap column is opened.
    frame = pd.read_csv(path, usecols=['material_id', FORMATION, HULL], dtype=str,
                        keep_default_na=False, skiprows=lambda i: i != 0 and i not in lines)
    if frame.material_id.duplicated().any():
        raise LabelAccessError("duplicate label IDs")
    frame = frame.set_index('material_id').loc[part.ids]
    for column in (FORMATION, HULL):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    frame['stable'] = (frame[HULL] <= doc['labels']['stability']['threshold_eV_atom']).astype(np.int8)
    return Labels(frame, part, dict(source='REAL_POST_FREEZE', label_columns_opened=[FORMATION, HULL],
                                    real_label_rows_parsed=part.n, protocol_sha256=__import__('hashlib').sha256(Path(yaml_path).read_bytes()).hexdigest()))


def require_protocol(*, yaml_path=YAML, root=ROOT):
    """Verify the public specification before real label or training-energy access."""
    from integrity import verify_protocol
    try:
        return verify_protocol(yaml_path=yaml_path, root=root)
    except Exception as exc:
        raise LabelAccessError(f'published specification verification failed: {exc}') from exc
