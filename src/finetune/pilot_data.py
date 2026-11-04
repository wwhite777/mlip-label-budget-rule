"""PILOT-only, row-filtered I/O shared by training, identity checks and metrics."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'src'))
from analysis.config import Config
from analysis.data import assert_in_part, load_part

SUMMARY = PROJECT/'cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz'
PARTITION = PROJECT/'result/splits/partition_v1/partition.csv.gz'
CORRECTIONS = PROJECT/'cache/reference/mp2020_corr_pmg2023.5.10_initial.npz'
REFS = PROJECT/'cache/reference/mbd_cache/mp/2023-02-07-mp-elemental-reference-entries.json.gz'


def assert_pilot(ids):
    p = pd.read_csv(PARTITION, usecols=['material_id', 'part'], keep_default_na=False)
    assert_in_part(ids, 'PILOT', dict(zip(p.material_id, p.part)), 'source PILOT guard')


def selected_rows(path, ids, columns):
    """Guard first, read ids alone, then parse requested columns on selected rows only."""
    ids = list(ids)
    assert_pilot(ids)
    all_ids = pd.read_csv(path, usecols=['material_id'], dtype=str,
                          keep_default_na=False).material_id
    wanted = set(ids)
    lines = set((np.flatnonzero(all_ids.isin(wanted)) + 1).tolist())
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(['material_id'] + list(columns))),
                        keep_default_na=False, skiprows=lambda i: i != 0 and i not in lines)
    if frame.material_id.duplicated().any() or set(frame.material_id) != wanted:
        raise ValueError(f'Missing or duplicate selected rows: {path}')
    return frame.set_index('material_id').loc[ids]


def pilot_part():
    # Config dryrun is used only to load label-free metadata, never synthetic labels.
    cfg = Config.build('dryrun', tasks=['stability'])
    return cfg, load_part(cfg)


def checkpoint(backbone):
    with np.load(PROJECT/f'cache/reference/sp_energy_{backbone}_initial.npz') as z:
        return Path(str(z['weights_path'])), str(z['weights_sha256'])


def score_totals(ids, raw, counts, summary=SUMMARY, corrections=CORRECTIONS, refs_path=REFS,
                 *, correction_multiplier=1):
    """The zero_label.run_sp convention, with its actual fixed-hull function."""
    from finetune_common import elemental_refs
    from pymatgen.core import Composition
    from reference.zero_label import hull_shift, E_FORM_DFT, E_HULL_DFT
    s = selected_rows(summary, ids, ['formula', 'n_sites', E_FORM_DFT, E_HULL_DFT])
    counts = np.asarray(counts, dtype=np.int64)
    if not np.array_equal(counts, s.n_sites.to_numpy(dtype=np.int64)):
        raise ValueError('Summary/structure atom count mismatch')
    with np.load(corrections) as z:
        if str(z['structures']) != 'initial' or str(z['pymatgen_version']) != '2023.5.10':
            raise ValueError('Expected pinned pymatgen 2023.5.10 INITIAL corrections')
        corr = pd.Series(z['correction'], index=z['material_id'].astype(str)).loc[ids].to_numpy()
    refs = elemental_refs(refs_path)
    raw = np.asarray(raw, dtype=np.float64)
    # Same total-energy subtraction/division as get_e_form_per_atom.
    eform = np.array([(e + correction_multiplier*c - sum(
        refs[str(el)]*amount for el, amount in Composition(formula).items()))/n
        for formula, e, c, n in zip(s.formula, raw, corr, counts)])
    hull = hull_shift(s, eform)
    out = pd.DataFrame({'material_id': ids, 'n_atoms': counts, 'energy_total_eV': raw,
                        'mp2020_correction_eV': corr, 'e_form_pred_sp': eform,
                        'e_hull_pred_sp': hull})
    if not np.isfinite(out.drop(columns='material_id').to_numpy()).all():
        raise RuntimeError('Nonfinite test prediction')
    return out
