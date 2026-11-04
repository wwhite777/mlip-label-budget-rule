"""Offline campaign I/O and shared energy-only selection/refit loop."""
import argparse
import bz2
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from pilot_data import assert_pilot, selected_rows, score_totals, PARTITION

PROJECT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def elemental_refs(path):
    import gzip
    from pymatgen.entries.computed_entries import ComputedEntry
    with gzip.open(path, 'rt') as f:
        entries = json.load(f)
    return {el: round(ComputedEntry.from_dict(entry).energy_per_atom, 4) for el,entry in entries.items()}


def column_items(path, wanted_column, wanted_ids):
    """Stream the local pandas column-oriented JSON without a multi-GB object."""
    decoder = json.JSONDecoder()
    with bz2.open(path, 'rt') as f:
        buf, pos, eof = '', 0, False

        def fill():
            nonlocal buf, pos, eof
            chunk = f.read(1 << 20)
            buf = buf[pos:] + chunk
            pos = 0
            eof = not chunk

        def peek():
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    return buf[pos]
                fill()
                if eof:
                    raise ValueError('Unexpected end of JSON')

        def token(char):
            nonlocal pos
            if peek() != char:
                raise ValueError(f'Expected {char!r}')
            pos += 1

        def value():
            nonlocal pos
            peek()
            while True:
                try:
                    obj, pos = decoder.raw_decode(buf, pos)
                    return obj
                except json.JSONDecodeError:
                    before = len(buf) - pos
                    fill()
                    if len(buf) == before:
                        raise ValueError('Truncated JSON value')

        def skip_value():
            # Skip unselected JSON values lexically: never deserialize another
            # row's CSE (which contains its DFT label).
            nonlocal pos
            first = peek()
            string = re.compile(r'"(?:[^"\\]|\\.)*"')
            if first == '"':
                while True:
                    quoted = string.match(buf, pos)
                    if quoted is not None:
                        pos = quoted.end()
                        return
                    fill()
                    if eof:
                        raise ValueError('Truncated skipped string')
            if first not in '{[':
                while True:
                    match = re.search(r'[,}\]]', buf[pos:])
                    if match is not None:
                        pos += match.start()
                        return
                    pos = len(buf)
                    fill()
                    if eof:
                        raise ValueError('Truncated skipped primitive')
            depth = 0
            special = re.compile(r'[{}\[\]"]')
            while True:
                match = special.search(buf, pos)
                if match is None:
                    pos = len(buf)
                    fill()
                    if eof:
                        raise ValueError('Truncated skipped JSON')
                    continue
                pos = match.start()
                char = buf[pos]
                if char == '"':
                    quoted = string.match(buf, pos)
                    if quoted is None:
                        fill()
                        if eof:
                            raise ValueError('Truncated skipped string')
                        continue
                    pos = quoted.end()
                else:
                    pos += 1
                    depth += 1 if char in '{[' else -1
                    if depth == 0:
                        return

        mapping = {}
        token('{')
        while peek() != '}':
            col = value()
            token(':')
            token('{')
            while peek() != '}':
                key = value()
                token(':')
                if col == 'material_id':
                    obj = value()
                    mapping[key] = obj
                elif col == wanted_column and mapping[key] in wanted_ids:
                    obj = value()
                    yield col, key, obj, mapping[key]
                else:
                    skip_value()
                if peek() == ',':
                    token(',')
                else:
                    break
            token('}')
            if peek() == ',':
                token(',')
            else:
                break
        token('}')


def load_selected(path, column, ids):
    assert_pilot(ids)
    wanted, out = set(ids), {}
    for col, key, obj, mid in column_items(path, column, wanted):
        out[mid] = obj
        if len(out) == len(wanted):
            break
    if set(out) != wanted:
        raise ValueError(f'Missing ids in {path}: {sorted(wanted - set(out))[:10]}')
    return out


def read_ids(path):
    p = Path(path)
    ids = json.loads(p.read_text()) if p.suffix == '.json' else p.read_text().split()
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError('ID file must be a JSON string array or whitespace-separated ids')
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate ids')
    return ids


def chemsys(formula):
    return '-'.join(sorted(set(re.findall('[A-Z][a-z]?', formula))))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--checkpoint', required=True, type=Path, help='Existing local checkpoint; URLs forbidden')
    p.add_argument('--train-ids', required=True, type=Path, help='Full n-label draw, INCLUDING validation ids')
    p.add_argument('--valid-ids', required=True, type=Path, help='v1.2 validation subset of the full draw')
    p.add_argument('--test-ids', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='New run directory')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--variant', choices=['full', 'head'], required=True)
    p.add_argument('--e0', choices=['residual', 'foundation'], default='residual',
                   help='Residual least-squares shift (default), or frozen original checkpoint references')
    p.add_argument('--train-geometry', choices=['relaxed', 'initial'], default='relaxed',
                   help='Labelled fit/validation/refit geometry; targets remain relaxed raw CSE totals')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--dry-run', action='store_true', help='8 unique structures; at most 1 step per selection/refit phase')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cse', type=Path, default=PROJECT/'cache/wbm/wbm_cse.json.bz2')
    p.add_argument('--summary', type=Path, default=PROJECT/'cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz')
    p.add_argument('--initial', type=Path, default=PROJECT/'cache/mbd/2022-10-19-wbm-init-structs.json.bz2')
    p.add_argument('--corrections', type=Path, default=PROJECT/'cache/reference/mp2020_corr_pmg2023.5.10_initial.npz')
    p.add_argument('--elemental-refs', type=Path, default=PROJECT/'cache/reference/mbd_cache/mp/2023-02-07-mp-elemental-reference-entries.json.gz')
    return p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare(a):
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
    assert_pilot(draw + valid + test)
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
    if set(sy.loc[draw]) & set(sy.loc[test]):
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


def residual_e0(ids, target, foundation, atoms, original):
    """minimum-norm residual shift, added to the original checkpoint E0."""
    from ase.data import chemical_symbols
    zs = sorted({int(z) for m in ids for z in atoms[m].numbers})
    missing = set(zs) - set(original)
    if missing:
        raise ValueError(f'Elements unsupported by checkpoint: {missing}')
    A = np.array([[np.count_nonzero(atoms[m].numbers == z) for z in zs] for m in ids], dtype=np.float64)
    y = np.array([target[m] - foundation[m] for m in ids], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite foundation residual')
    coef, _, rank, singular = np.linalg.lstsq(A, y, rcond=None)
    values = dict(original)
    values.update({z: original[z] + float(d) for z, d in zip(zs, coef)})
    counts = np.array([len(atoms[m]) for m in ids], dtype=np.float64)
    return values, {'ids': ids, 'observed_atomic_numbers': zs, 'rank': int(rank),
                    'n_columns': len(zs), 'singular_values': singular.tolist(),
                    'rank_deficient': bool(rank < len(zs)),
                    'residual_rmse_before_total_eV': float(np.sqrt(np.mean(y**2))),
                    'residual_rmse_after_total_eV': float(np.sqrt(np.mean((y - A @ coef)**2))),
                    'residual_rmse_before_eV_per_atom': float(np.sqrt(np.mean((y/counts)**2))),
                    'residual_rmse_after_eV_per_atom': float(np.sqrt(np.mean(((y - A @ coef)/counts)**2))),
                    'delta_eV': {str(z): float(d) for z, d in zip(zs, coef)},
                    'delta_by_element_eV': {chemical_symbols[z]: float(d) for z, d in zip(zs, coef)},
                    'original_values_eV': {str(z): float(v) for z, v in original.items()},
                    'values_eV': {str(z): float(v) for z, v in values.items()},
                    'foundation_total_eV': {m: float(foundation[m]) for m in ids},
                    'unseen_elements': 'checkpoint reference retained; no campaign estimate',
                    'solver': 'numpy.linalg.lstsq(A_counts, DFT_raw_total - foundation_original_E0_total, rcond=None); minimum norm; E0 = original + delta'}


def predict_totals(backend, ids, batch_size):
    backend.model.eval()
    raw = []
    with torch.no_grad():
        for offset in range(0, len(ids), batch_size):
            raw.extend(backend.energy(ids[offset:offset+batch_size], training=False).detach().cpu().double().tolist())
    if len(raw) != len(ids) or not np.isfinite(raw).all():
        raise RuntimeError('Missing or nonfinite model total energy')
    return dict(zip(ids, raw))


def initialize_e0(backend, mode, ids, target, atoms, batch_size):
    original = backend.original_e0()
    if mode == 'foundation':
        # Do not round-trip or overwrite the checkpoint references.
        return {'mode': mode, 'ids': [], 'trainable': False,
                'original_values_eV': {str(z): float(v) for z, v in original.items()},
                'values_eV': {str(z): float(v) for z, v in original.items()},
                'solver': None, 'reference': 'original checkpoint; no shift or fit ()'}
    foundation = predict_totals(backend, ids, batch_size)
    values, result = residual_e0(ids, target, foundation, atoms, original)
    backend.set_e0(values)
    return {**result, 'mode': mode, 'trainable': True}


def per_atom_mse(prediction, ids, target, atoms):
    return float(np.mean([((prediction[m] - target[m])/len(atoms[m]))**2 for m in ids]))


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def run(a, backend_class):
    started = time.perf_counter()
    draw, fit, valid, test, target, atoms, summ, run_record = prepare(a)
    a.lr = a.lr if a.lr is not None else (1e-4 if a.variant == 'full' else 1e-3)
    if a.lr <= 0:
        raise ValueError('Learning rate must be positive')
    run_record.update(model=backend_class.name, variant=a.variant, e0_mode=a.e0, dry_run=a.dry_run, seed=a.seed,
                   device=a.device, torch_threads=a.threads, args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                   recipe={'optimizer': 'AdamW', 'lr': a.lr, 'weight_decay': 0.0, 'betas': [0.9,0.999],
                           'eps': 1e-8, 'batch_size': a.batch_size, 'max_epochs': a.epochs,
                           'patience': a.patience, 'early_stop': 'validation per-atom MSE; strict improvement; epoch 0 selectable',
                           'loss': 'mean(((predicted_total - target_total)/n_atoms)**2)',
                           'weights': {'energy': 1.0, 'forces': 0.0, 'stress': 0.0},
                           'scheduler': None, 'precision': 'float32-highest', 'replay': False,
                           'protocol': ('residual E0 on inner fit only; selection includes epoch 0; fresh all-label residual E0 and refit for selected epoch count'
                                        if a.e0 == 'residual' else
                                        'frozen foundation E0 in selection and fresh refit; epoch 0 is zero-shot and selectable; refit for selected epoch count')})
    run_record['versions'] = {k: importlib.metadata.version(k) for k in ['torch','mace-torch','orb-models','numpy','pandas','ase','pymatgen']}
    if a.device == 'cuda':
        run_record['gpu'] = {'name': torch.cuda.get_device_name(),
                          'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'count': 1, 'timing': 'synchronized wall time on one allocated GPU'}
    history, phases = [], []
    selected_epoch = 0
    for phase, train_ids in [('selection', fit), ('refit', draw)]:
        seed_all(a.seed)
        t0 = time.perf_counter()
        backend = backend_class(a, atoms)
        e0_run_record = initialize_e0(backend, a.e0, train_ids, target, atoms, a.batch_size)
        backend.configure_trainable(a.variant, a.e0)
        model = backend.model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        phase_run_record = {'phase': phase, 'e0': e0_run_record, 'setup_seconds': time.perf_counter()-t0,
                         'parameters_total': sum(p.numel() for p in model.parameters()),
                         'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                         'trainable_names': [n for n,p in model.named_parameters() if p.requires_grad],
                         'steps': [], 'epochs': [],
                         'setup_peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if a.device == 'cuda' else 0,
                         'setup_peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if a.device == 'cuda' else 0}
        rng = np.random.default_rng(a.seed)
        epoch0_begin = time.perf_counter()
        epoch0_pred = predict_totals(backend, list(dict.fromkeys(train_ids + valid)), a.batch_size)
        epoch0 = {'epoch': 0, 'fit_mse': per_atom_mse(epoch0_pred, train_ids, target, atoms),
                  'validation_mse': per_atom_mse(epoch0_pred, valid, target, atoms)}
        if phase == 'selection':
            history.append(dict(epoch0))
        else:
            # Fresh refit initialization, before any gradient step. Test labels
            # are used only by score_totals after training, never for selection.
            test0_begin = time.perf_counter()
            epoch0_test = predict_totals(backend, test, a.batch_size)
            synchronize(a.device)
            run_record['epoch0_test_seconds'] = time.perf_counter()-test0_begin
        synchronize(a.device)
        epoch0_seconds = time.perf_counter()-epoch0_begin
        phase_run_record['epochs'].append({**epoch0, 'wall_seconds': epoch0_seconds,
            'gpu_seconds': epoch0_seconds if a.device == 'cuda' else 0.0,
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if a.device == 'cuda' else 0,
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if a.device == 'cuda' else 0})
        print(json.dumps({'phase': phase, **epoch0}), flush=True)
        best, stale = epoch0['validation_mse'], 0
        max_epochs = a.epochs if phase == 'selection' else selected_epoch
        if a.dry_run:
            max_epochs = min(1, max_epochs)
        for epoch in range(1, max_epochs + 1):
            synchronize(a.device)
            if a.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            epoch_begin = time.perf_counter()
            model.train()
            order = [train_ids[i] for i in rng.permutation(len(train_ids))]
            epoch_loss = 0.0
            for offset in range(0, len(order), a.batch_size):
                mids = order[offset:offset+a.batch_size]
                optimizer.zero_grad(set_to_none=True)
                before = {n: p.detach().clone() for n,p in model.named_parameters()} if a.dry_run else {}
                synchronize(a.device)
                begin = time.perf_counter()
                pred = backend.energy(mids, training=True)
                synchronize(a.device)
                forward_end = time.perf_counter()
                gold = torch.tensor([target[m] for m in mids], dtype=pred.dtype, device=pred.device)
                counts = torch.tensor([len(atoms[m]) for m in mids], dtype=pred.dtype, device=pred.device)
                loss = (((pred - gold) / counts)**2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite energy loss')
                loss.backward()
                e0_grad = backend.e0_parameter().grad
                if a.e0 == 'residual':
                    if e0_grad is None or not torch.isfinite(e0_grad).all():
                        raise RuntimeError('E0 gradient absent/nonfinite')
                elif backend.e0_parameter().requires_grad or e0_grad is not None:
                    raise RuntimeError('Foundation E0 must be frozen with no gradient')
                e0_before = backend.e0_parameter().detach().clone()
                optimizer.step()
                e0_unchanged = torch.equal(backend.e0_parameter().detach(), e0_before)
                if a.e0 == 'foundation' and not e0_unchanged:
                    raise RuntimeError('Foundation E0 changed during optimizer step')
                synchronize(a.device)
                end = time.perf_counter()
                changed = [n for n,p in model.named_parameters() if a.dry_run and not torch.equal(p.detach(), before[n])]
                if a.dry_run:
                    frozen_changed = [n for n,p in model.named_parameters() if n in changed and not p.requires_grad]
                    if frozen_changed:
                        raise RuntimeError(f'Frozen parameters changed: {frozen_changed}')
                    if not any(n.startswith(('readouts.', 'heads.energy.mlp.')) for n in changed):
                        raise RuntimeError('No energy readout parameter updated')
                epoch_loss += loss.item()*len(mids)
                phase_run_record['steps'].append({'epoch': epoch, 'ids': mids, 'n_structures': len(mids),
                    'forward_seconds': forward_end-begin, 'backward_optimizer_seconds': end-forward_end,
                    'total_seconds': end-begin, 'loss_eV2_per_atom2': loss.item(),
                    'e0_gradient_norm': float(e0_grad.norm()) if e0_grad is not None else None,
                    'e0_bit_identical': e0_unchanged,
                    'e0_changed_max_eV': float((backend.e0_parameter().detach()-e0_before).abs().max()),
                    'changed_parameter_names': changed if a.dry_run else None,
                    'frozen_parameter_changes': 0 if a.dry_run else None})
                if a.dry_run:
                    break
            if phase == 'selection':
                model.eval()
                tval = time.perf_counter()
                errors = []
                with torch.no_grad():
                    for offset in range(0, len(valid), a.batch_size):
                        mids = valid[offset:offset+a.batch_size]
                        preds = backend.energy(mids, training=False).detach().cpu().numpy()
                        errors.extend([(float(p)-target[m])/len(atoms[m]) for p,m in zip(preds,mids)])
                synchronize(a.device)
                val_mse = float(np.mean(np.square(errors)))
                if not math.isfinite(val_mse):
                    raise RuntimeError('Nonfinite validation loss')
                history.append({'epoch': epoch, 'fit_mse': epoch_loss/len(train_ids), 'validation_mse': val_mse,
                                'validation_seconds': time.perf_counter()-tval})
                if val_mse < best:
                    best, stale, selected_epoch = val_mse, 0, epoch
                else:
                    stale += 1
            synchronize(a.device)
            epoch_seconds = time.perf_counter()-epoch_begin
            epoch_record = {'epoch': epoch, 'wall_seconds': epoch_seconds,
                            'gpu_seconds': epoch_seconds if a.device == 'cuda' else 0.0,
                            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if a.device == 'cuda' else 0,
                            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if a.device == 'cuda' else 0}
            phase_run_record['epochs'].append(epoch_record)
            print(json.dumps({'phase': phase, **epoch_record}), flush=True)
            if phase == 'selection' and stale >= a.patience:
                break
        phases.append(phase_run_record)
        if phase == 'selection':
            # Release the selection graph/parameters before loading the fresh refit.
            del optimizer, backend, model
            if phase_run_record['steps']:
                del pred, loss, gold, counts, e0_grad, e0_before
    model.eval()
    begin = time.perf_counter()
    final_test = predict_totals(backend, test, a.batch_size)
    synchronize(a.device)
    run_record['test_seconds'] = time.perf_counter()-begin
    for filename, prediction in [('test_energies.csv', final_test), ('epoch0_test_energies.csv', epoch0_test)]:
        out = score_totals(test, [prediction[m] for m in test], [len(atoms[m]) for m in test],
                           a.summary, a.corrections, a.elemental_refs)
        out.to_csv(a.output/filename, index=False)
    run_record.update(selected_epoch=selected_epoch, history=history, phases=phases,
                   optimizer_steps=sum(len(p['steps']) for p in phases), elapsed_seconds=time.perf_counter()-started,
                   status='cpu_smoke_complete' if a.dry_run else 'complete',
                   scoring_convention='raw INITIAL model energy + one pinned MP2020 correction; rounded elemental refs; zero_label.hull_shift ()',
                   output_sha256={name: sha256(a.output/name) for name in ['test_energies.csv', 'epoch0_test_energies.csv']})
    if not a.dry_run:
        torch.save(model.state_dict(), a.output/'model_state.pt')
        run_record['output_sha256']['model_state.pt'] = sha256(a.output/'model_state.pt')
    synchronize(a.device)
    run_record['elapsed_seconds'] = time.perf_counter()-started
    run_record['gpu_seconds'] = run_record['elapsed_seconds'] if a.device == 'cuda' else 0.0
    run_record['gpu_hours'] = run_record['gpu_seconds']/3600
    run_record['gpu_seconds_definition'] = 'one GPU times complete driver wall seconds, including CPU I/O and setup; CPU dry run = 0'
    run_record['epoch_gpu_seconds'] = sum(e['gpu_seconds'] for p in phases for e in p['epochs'])
    run_record['peak_memory_allocated_bytes'] = max([e['peak_memory_allocated_bytes'] for p in phases for e in p['epochs']] +
        [p['setup_peak_memory_allocated_bytes'] for p in phases] +
        [torch.cuda.max_memory_allocated() if a.device == 'cuda' else 0])
    run_record['peak_memory_reserved_bytes'] = max([e['peak_memory_reserved_bytes'] for p in phases for e in p['epochs']] +
        [p['setup_peak_memory_reserved_bytes'] for p in phases] +
        [torch.cuda.max_memory_reserved() if a.device == 'cuda' else 0])
    write_json(a.output/'run_record.json', run_record)
    print(json.dumps({'status': run_record['status'], 'output': str(a.output), 'steps': run_record['optimizer_steps'],
                      'selected_epoch': selected_epoch, 'elapsed_seconds': run_record['elapsed_seconds']}))
