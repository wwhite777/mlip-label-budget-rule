"""Recompute descriptive mechanism tables from archived predictions and targets."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'src'), str(ROOT / 'tools')]
from confirm_runner.mechanism import import_probe
from check_stages import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    verify(a.data_root / 'data/stages', a.data_root / 'data/extension/CPU-R')
    if a.out_root.exists() and any(a.out_root.iterdir()):
        raise ValueError('use a fresh output directory')
    source = a.data_root / 'data/stages/mechanism'
    for folder, pattern in (('prepared', '*'), ('predictions', '*.npz')):
        dest = a.out_root / folder
        dest.mkdir(parents=True, exist_ok=True)
        for f in (source / folder).glob(pattern):
            if f.name.endswith(('_features.npy', 'composition.npy', 'ids.npy', 'chemsys.npy')):
                continue
            shutil.copyfile(f, dest / f.name)
    probe = import_probe()
    probe.OUT = a.out_root
    probe.topk_k = lambda fraction, n: int(np.ceil(fraction * n))
    with threadpool_limits(1):
        probe.evaluate()
    path = a.out_root / 'error_strata.csv'
    table = pd.read_csv(path)
    table.rename(columns={c: c.replace('pilot', 'confirm') for c in table.columns}, inplace=True)
    table.to_csv(path, index=False)
    names = ('probe_metrics.csv', 'probe_stability_ap.csv', 'error_strata.csv', 'error_variance_explained.csv')
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(source / name), pd.read_csv(a.out_root / name), check_exact=True)
    print(json.dumps(dict(pass_all=True, tables=len(names), fitting_performed=False, publisher_label_files_opened=0)))


if __name__ == '__main__':
    main()
