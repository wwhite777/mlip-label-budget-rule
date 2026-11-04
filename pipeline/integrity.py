"""Verify published protocol, contract and code identities before real input access."""
import argparse
import json
from pathlib import Path
from confirm_runner.common import ROOT, YAML, ProtocolError, sha256

def verify_protocol(*, yaml_path=YAML, root=ROOT):
    root=Path(root).resolve(); yaml_path=Path(yaml_path).resolve()
    pins=json.loads((root/'PROTOCOL_SHA256').read_text())
    allowed={root/k for k in pins if k.endswith('.yaml')}
    if yaml_path not in allowed:
        raise ProtocolError('unpublished specification path')
    for relative,digest in pins.items():
        if sha256(root/relative)!=digest:
            raise ProtocolError('published specification changed: '+relative)
    sources=json.loads((root/'pipeline/release_integrity.json').read_text())
    for relative,digest in sources.items():
        if sha256(root/relative)!=digest:
            raise ProtocolError('published source changed: '+relative)
    contract='CONTRACT_v2.4.md' if yaml_path.name=='protocol_v2.4.yaml' else 'CONTRACT.md'
    return dict(protocol_sha256=sha256(yaml_path),contract_sha256=sha256(root/'pipeline'/contract))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--yaml',type=Path,default=YAML);a=p.parse_args()
    verify_protocol(yaml_path=a.yaml)
    print('PASS: both published specifications, contracts and source hashes')
