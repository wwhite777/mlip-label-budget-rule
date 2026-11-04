"""Verify every restored scientific output and its complete local predecessor chain."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from confirm_runner.common import YAML, ProtocolError, sha256
from confirm_runner.stages import ORDER, require_previous


def verify(root, extension=None):
    root = Path(root).resolve()
    require_previous(root, 'CPU-R')
    count = sum(len(json.loads((root / name / 'stage_manifest.json').read_text())['outputs']) for name in ORDER[:-1])
    if extension is not None:
        extension = Path(extension).resolve()
        manifest = json.loads((extension / 'stage_manifest.json').read_text())
        if (manifest['stage'] != 'CPU-R' or manifest['status'] != 'COMPLETE' or manifest['synthetic_only']
                or manifest['yaml_sha256'] != sha256(ROOT / 'pipeline/protocol_v2.4.yaml')
                or manifest['previous_manifests'] != {name: sha256(root / name / 'stage_manifest.json') for name in ORDER[:-1]}):
            raise ProtocolError('extension specification or predecessor changed')
        for rel, digest in manifest['outputs'].items():
            path = (extension.parent / rel).resolve()
            if not path.is_relative_to(extension) or sha256(path) != digest:
                raise ProtocolError('extension output changed')
        count += len(manifest['outputs'])
    return dict(pass_all=True, stages=7 if extension is not None else 6, output_bindings=count)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-root', type=Path, default=ROOT / 'data/stages')
    p.add_argument('--extension-root', type=Path, default=ROOT / 'data/extension/CPU-R')
    a = p.parse_args()
    print(json.dumps(verify(a.stage_root, a.extension_root)))
