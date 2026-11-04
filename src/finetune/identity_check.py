#!/usr/bin/env python3
"""Zero optimizer steps, original checkpoint E0, 50 fixed PILOT INITIAL ids."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from finetune_common import load_selected, read_ids, sha256, write_json, seed_all, initialize_e0
from finetune_mace import MaceBackend
from finetune_orb import OrbBackend
from pilot_data import PROJECT, checkpoint, selected_rows, score_totals, assert_pilot

TOLERANCE = 2e-6


def assert_identity(actual, stored):
    delta = np.abs(np.asarray(actual) - np.asarray(stored))
    assert np.isfinite(delta).all() and np.max(delta) <= TOLERANCE, (
        f'zero-step identity failed: max error={np.max(delta):.12g} eV/atom > {TOLERANCE}')
    return float(np.max(delta))


def check(backbone, ids):
    assert_pilot(ids)
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError('Identity check requires 50 unique PILOT ids')
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    initial = PROJECT/'cache/mbd/2022-10-19-wbm-init-structs.json.bz2'
    structures = load_selected(initial, 'initial_structure', ids)
    atoms = {m: AseAtomsAdaptor.get_atoms(Structure.from_dict(structures[m])) for m in ids}
    path, expected_sha = checkpoint(backbone)
    assert sha256(path) == expected_sha, 'Checkpoint differs from zero-shot extraction'
    seed_all(0)
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() > 8:
        torch.set_num_interop_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    backend = {'mace_medium': MaceBackend, 'orb': OrbBackend}[backbone](
        SimpleNamespace(checkpoint=path, device='cpu'), atoms)
    initialize_e0(backend, 'foundation', [], {}, atoms, 8)
    backend.configure_trainable('full', 'foundation')
    assert not backend.e0_parameter().requires_grad
    backend.model.eval()
    original = backend.e0_parameter().detach().clone()
    raw = []
    with torch.no_grad():
        for offset in range(0, len(ids), 8):
            raw.extend(backend.energy(ids[offset:offset+8], training=False).double().cpu().tolist())
    assert torch.equal(original, backend.e0_parameter()), 'Identity check changed E0'
    counts = [len(atoms[m]) for m in ids]
    out = score_totals(ids, raw, counts)
    reference = PROJECT/f'result/reference/zero_label_sp_{backbone}_initial.csv.gz'
    stored = selected_rows(reference, ids, ['e_hull_pred_sp']).e_hull_pred_sp.to_numpy()
    error = assert_identity(out.e_hull_pred_sp, stored)
    double = score_totals(ids, raw, counts, correction_multiplier=2)
    try:
        assert_identity(double.e_hull_pred_sp, stored)
    except AssertionError as exc:
        planted = str(exc)
    else:
        raise AssertionError('Planted double MP2020 correction was not detected')
    out['stored_e_hull_pred_sp'] = stored
    return out, {'backbone': backbone, 'n_ids': len(ids), 'device': 'cpu', 'dtype': 'float32',
        'e0_mode': 'foundation', 'optimizer_steps': 0, 'original_e0_unchanged': True, 'max_abs_error_eV_per_atom': error,
        'tolerance_eV_per_atom': TOLERANCE, 'double_correction_detected': True,
        'planted_failure': planted, 'checkpoint': str(path), 'checkpoint_sha256': expected_sha,
        'reference_sha256': sha256(reference), 'initial_sha256': sha256(initial)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=['mace_medium', 'orb'], required=True)
    p.add_argument('--ids', type=Path, default=Path(__file__).parent/'identity_ids.txt')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    out, result = check(a.backbone, read_ids(a.ids))
    out.to_csv(a.output/f'{a.backbone}.csv', index=False)
    write_json(a.output/f'{a.backbone}.json', result)
    print(result)
