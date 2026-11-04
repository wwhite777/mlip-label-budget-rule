"""probe P-only full-training mechanism adapter; every result is descriptive.

The original ridge selection and fitting functions are imported unchanged.
The sole reporting override is C14 ceil top-five-percent rounding.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import ROOT, ProtocolError, checked, dump
from .inputs import align, metadata_csv
from .labels import FORMATION, HULL

PROBE = None


def import_probe():
    global PROBE
    if PROBE is None:
        spec = importlib.util.spec_from_file_location('frozen_mechanism_source', ROOT / 'src/probe/probe.py')
        PROBE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PROBE)
    return PROBE


def fit_one(job):
    return PROBE.fit_job(*job)


def execute(doc, inp, out, workers=3):
    """Use all outer-training rows, never an n-label learner or campaign fit."""
    p = import_probe()
    p.OUT = Path(out)
    p.OUT.mkdir(parents=True, exist_ok=True)
    (p.OUT / 'prepared').mkdir(exist_ok=True)
    (p.OUT / 'predictions').mkdir(exist_ok=True)
    p.topk_k = lambda fraction, n: int(np.ceil(fraction * n))
    if inp.part.part != 'CONFIRM':
        raise ProtocolError('mechanism adapter requires CONFIRM P context')
    labels = inp.labels.lookup(inp.part.ids)
    f, y = labels[FORMATION].to_numpy(), labels[HULL].to_numpy()
    for bb in doc['backbones']['order']:
        ref = inp.refs[bb]['r_sp']
        targets = np.column_stack([ref['e_form'], ref['e_form'] - ref['e_hull'], f, y, f-ref['e_form']])
        np.save(p.OUT / f'prepared/{bb}_features.npy', inp.feats[bb].astype(np.float64))
        np.save(p.OUT / f'prepared/{bb}_targets.npy', targets)
        np.save(p.OUT / f'prepared/{bb}_hull.npy', ref['e_hull'])
    for name, arr in [('composition', inp.comp), ('ids', inp.part.ids), ('fold', inp.part.fold),
                      ('chemsys', inp.part.chemsys), ('num_elements', np.count_nonzero(inp.comp, axis=1))]:
        np.save(p.OUT / f'prepared/{name}.npy', arr)
    flags = doc['probe']['correction_strata']
    colmap = {'oxidation_state_dependent_anion': 'anion_assignment_oxi_dependent',
              'hubbard_u_metal_with_O_or_F': 'hubbard_u_tm_with_O_or_F'}
    frame = metadata_csv(checked(dict(path=flags['file'], sha256=flags['sha256'])),
                         ['material_id'] + [colmap[k] for k in flags['strata']], table='strata')
    frame = frame.iloc[align(frame.material_id, inp.part.ids)]
    strata = {}
    for name in flags['strata']:
        if not set(frame[colmap[name]]) <= {'True', 'False'}:
            raise ProtocolError('nonboolean mechanism stratum')
        strata[name] = frame[colmap[name]].to_numpy() == 'True'
    np.savez(p.OUT / 'prepared/strata.npz', **strata)
    jobs = [(bb, x, model) for bb in doc['backbones']['order'] for x in p.INPUTS for model in ('ridge', 'hist_gbm')]
    started = time.monotonic()
    with mp.get_context('fork').Pool(min(workers, 3)) as pool:
        records = list(pool.imap_unordered(fit_one, jobs))
    p.evaluate()
    path = p.OUT / 'error_strata.csv'
    table = pd.read_csv(path)
    table.rename(columns={c: c.replace('pilot', 'confirm') for c in table.columns}, inplace=True)
    table.to_csv(path, index=False)
    dump(p.OUT / 'run.json', dict(status='PASS', design='P', scope='all outer-training rows', descriptive_only=True,
        top5_rounding='ceil', workers=min(workers, 3), threads_per_worker=4, records=records,
        synthetic_only=inp.labels.trace['source'] == 'SYNTHETIC', seconds=time.monotonic()-started))
