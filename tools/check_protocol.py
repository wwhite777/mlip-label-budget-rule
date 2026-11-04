"""Check the original scientific inventory and the published extension settings."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from confirm_runner.common import load_yaml


def flat(value, prefix=''):
    if isinstance(value, dict):
        return {a: b for k, v in value.items() for a, b in flat(v, prefix + '.' + str(k) if prefix else str(k)).items()}
    if isinstance(value, list):
        return {a: b for i, v in enumerate(value) for a, b in flat(v, prefix + '.' + str(i)).items()}
    return {prefix: value}


def main():
    base = load_yaml()
    values = flat(base)
    inventory = json.loads((ROOT / 'SCIENTIFIC_FIELDS.json').read_text())
    for row in inventory['scientific_fields']:
        assert hashlib.sha256(json.dumps(values[row['path']], sort_keys=True).encode()).hexdigest() == row['value_sha256'], row['path']
    contract = (ROOT / 'pipeline/CONTRACT.md').read_text()
    displayed = {key: json.loads(value) for key, value in re.findall(r'^\| `([^`]+)` \| `([^`]+)` \|$', contract, re.M)}
    assert displayed == {key: value for key, value in values.items()
                         if isinstance(value, (int, float, bool)) and not key.startswith('implementation.')}
    extension = load_yaml(ROOT / 'pipeline/protocol_v2.4.yaml')
    assert {k: v for k, v in extension.items() if k not in ('id', 'contract', 'amendment_A2')} == {
        k: v for k, v in base.items() if k not in ('id', 'contract')}
    amendment = flat(extension['amendment_A2'])
    expected = json.loads((ROOT / 'AMENDMENT_FIELDS.json').read_text())
    assert {k: v for k, v in amendment.items() if not k.startswith('stage.')} == expected
    from integrity import verify_protocol
    verify_protocol()
    verify_protocol(yaml_path=ROOT / 'pipeline/protocol_v2.4.yaml')
    print(json.dumps(dict(pass_all=True, original_scientific_fields=len(inventory['scientific_fields']),
                          contract_typed_fields=len(displayed), extension_fields=len(expected), original_rule_preserved=True)))


if __name__ == '__main__':
    main()
