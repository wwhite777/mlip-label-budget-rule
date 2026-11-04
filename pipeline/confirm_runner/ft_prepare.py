"""finetune prepare adapted only for guarded I/O and the declared S-a overlap.

The training loop, optimizer, epoch selection, fresh refit and backends are imported
unchanged. Source correspondence is recorded in README and the change list.
"""
import math
import os
import time
import numpy as np
import torch

def prepare(a, common, access):
    from .common import sha256
    seed_all, read_ids, chemsys = common.seed_all, common.read_ids, common.chemsys
    selected_rows = access.selected_rows
    load_selected = lambda path,column,ids: access.load_selected(common,path,column,ids)
    PARTITION = common.PARTITION
    # Check every offline prerequisite before constructing either library's model.
    files = {k: getattr(a, k) for k in ['checkpoint', 'train_ids', 'valid_ids', 'test_ids',
                                      'cse', 'summary', 'initial', 'corrections', 'elemental_refs']}
    for p in files.values():
        if not p.is_file():
            raise FileNotFoundError(f'Local input missing; no download attempted: {p}')
    if not 0 <= a.epochs <= 100 or a.patience < 1 or a.batch_size < 1:
        raise ValueError('Epochs must be in [0, 100]; patience and batch size must be positive')
    if not 1 <= a.threads <= 8:
        raise ValueError('Threads must be in [1, 8]')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if a.dry_run and a.device != 'cpu':
        raise ValueError('Dry runs are CPU only')
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', str(a.output/'mplconfig'))
    torch.set_num_threads(a.threads)
    if torch.get_num_interop_threads() > 8:
        torch.set_num_interop_threads(a.threads)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    seed_all(a.seed)
    draw, valid, test = map(read_ids, [a.train_ids, a.valid_ids, a.test_ids])
    access.assert_ids(draw + test)
    if dict(train=draw, fit=[m for m in draw if m not in set(valid)], valid=valid, test=test) != access.ids:
        raise ValueError("FT driver IDs differ from authorized manifest")
    if not set(valid) < set(draw) or set(draw) & set(test):
        raise ValueError('Validation must be a proper subset of draw; test must be disjoint')
    if len(valid) < math.ceil(0.2 * len(draw)):
        raise ValueError('Validation must contain at least 20% of draw labels')
    fit = [m for m in draw if m not in set(valid)]
    if a.dry_run and len(set(draw + test)) != 8:
        raise ValueError('Dry run requires exactly 8 unique structures across draw and test')
    # Construct fit targets only from campaign rows; test targets never enter fitting.
    summ = selected_rows(a.summary, draw + test, ['formula', 'n_sites'])
    sy = summ['formula'].map(chemsys)
    if set(sy.loc[fit]) & set(sy.loc[valid]):
        raise ValueError('Inner fit and validation chemical systems overlap')
    if access.unit["design"] == "P" and set(sy.loc[draw]) & set(sy.loc[test]):
        raise ValueError('Outer draw and test chemical systems overlap')
    raw_summary = selected_rows(a.summary, draw, ['uncorrected_energy_from_cse'])
    t0 = time.perf_counter()
    entries = load_selected(a.cse, 'computed_structure_entry', draw)
    target = {m: float(entries[m]['energy']) for m in draw}
    if not np.isfinite(list(target.values())).all():
        raise ValueError('Nonfinite campaign energy')
    init_ids = draw + test if a.train_geometry == 'initial' else test
    init = load_selected(a.initial, 'initial_structure', init_ids)
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = {m: AseAtomsAdaptor.get_atoms(Structure.from_dict(entries[m]['structure'])) for m in draw}
    if a.train_geometry == 'initial':
        for m in draw:
            initial_atom = AseAtomsAdaptor.get_atoms(Structure.from_dict(init[m]))
            assert len(initial_atom) == len(atoms[m]), f'Initial/CSE atom count mismatch: {m}'
            atoms[m] = initial_atom
    atoms.update({m: AseAtomsAdaptor.get_atoms(Structure.from_dict(init[m])) for m in test})
    for m in draw:
        if not np.isclose(entries[m]['energy'], float(raw_summary.loc[m, 'uncorrected_energy_from_cse']), atol=5.1e-7, rtol=0):
            raise ValueError(f'CSE and pinned summary raw energy disagree: {m}')
    for m in draw + test:
        if len(atoms[m]) != int(float(summ.loc[m, 'n_sites'])):
            raise ValueError(f'Summary/structure atom count mismatch: {m}')
    inputs = {k: {'path': str(p.resolve()), 'sha256': sha256(p)} for k, p in files.items()}
    inputs['partition'] = {'path': str(PARTITION), 'sha256': sha256(PARTITION)}
    return draw, fit, valid, test, target, atoms, summ, {
        'inputs': inputs, 'load_seconds': time.perf_counter()-t0,
        'ids': {'draw': draw, 'fit': fit, 'validation': valid, 'test': test},
        'n_atoms': {m: len(atoms[m]) for m in atoms},
        'target': 'uncorrected DFT total energy (eV), relaxed WBM CSE energy; ',
        'target_values_eV': target,
        'target_max_difference_vs_summary_raw_eV': max(
            abs(target[m]-float(raw_summary.loc[m,'uncorrected_energy_from_cse'])) for m in draw),
        'train_geometry': a.train_geometry,
        'test_geometry': 'INITIAL; no relaxation',
    }

