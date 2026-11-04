"""Recompute every practical reading and both judges from archived inference inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'tools')]
from confirm_runner.common import load_yaml, dump
from confirm_runner import statistics, relaxed_statistics
from check_stages import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    verify(a.data_root / 'data/stages', a.data_root / 'data/extension/CPU-R')
    total = 0
    for stage in ('CPU-P', 'CPU-S', 'FT-P', 'FT-S', 'CPU-R'):
        extension = stage == 'CPU-R'
        folder = a.data_root / ('data/extension/CPU-R' if extension else 'data/stages/' + stage)
        doc = load_yaml(ROOT / ('pipeline/protocol_v2.4.yaml' if extension else 'pipeline/protocol.yaml'))
        module = relaxed_statistics if extension else statistics
        actual = []
        for record in sorted(folder.rglob('inference_*.json')):
            info = json.loads(record.read_text())
            design = info['design']
            expected = info['readings']
            budgets = doc['amendment_A2']['budgets'] if extension else sorted({r['n'] for r in expected if r['family'] != 'F4'})
            with np.load(record.with_name('bootstrap_' + design + '.npz'), allow_pickle=False) as arrays:
                stars = {k: arrays[k] for k in arrays.files}
            kwargs = {} if extension else dict(allow_incomplete_f3=stage == 'CPU-P')
            rows = module.read_families(doc, info['points'], stars, budgets, design,
                                       synthetic=False, M_r=info['M_r'], **kwargs)
            assert rows == expected, stage + ': readings differ from archived inference'
            actual.extend(rows)
        expected = json.loads((folder / 'readings.json').read_text())
        key = lambda r: (r['design'], r['family'], r['contrast'])
        assert sorted(actual, key=key) == sorted(expected, key=key), stage
        dump(a.out_root / stage / 'readings.json', actual)
        if stage in ('CPU-P', 'CPU-R'):
            decision = module.judge(doc, actual)
            assert decision == json.loads((folder / 'judge.json').read_text()), stage + ': judge changed'
            dump(a.out_root / stage / 'judge.json', decision)
        total += len(actual)
    print(json.dumps(dict(pass_all=True, readings=total, stages=5, judges=2, label_files_opened=0)))


if __name__ == '__main__':
    main()
