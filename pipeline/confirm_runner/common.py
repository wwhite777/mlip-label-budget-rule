from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
YAML = HERE / "protocol.yaml"
CONTRACT = HERE / "CONTRACT.md"
sys.path.insert(0, str(ROOT / "src"))


class ProtocolError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    out = {}
    for key, val in node.value:
        key = loader.construct_object(key, deep=deep)
        if key in out:
            raise ProtocolError(f"duplicate YAML key: {key}")
        out[key] = loader.construct_object(val, deep=deep)
    return out


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_yaml(path=YAML):
    doc = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    pins = json.loads((HERE / 'implementation_pins.json').read_text())
    if doc['implementation']['imports'] != pins['imports']:
        raise ProtocolError('implementation imports differ from published hashes')
    return doc


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path, root=ROOT):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def checked(spec, root=ROOT):
    path = resolve(spec["path"], root)
    if sha256(path) != spec["sha256"]:
        raise ProtocolError(f"input hash mismatch: {path}")
    return path


def dump(path, value):
    import numpy as np
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")
