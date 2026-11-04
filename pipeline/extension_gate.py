"""Public extension specification and predecessor gate."""
import json
from pathlib import Path
from confirm_runner.common import HERE, ROOT, ProtocolError, dump, sha256, load_yaml
from integrity import verify_protocol

YAML24=HERE/'protocol_v2.4.yaml'

def validate_document(doc,*,yaml_path=YAML24,verify_files=False):
    expected=load_yaml(yaml_path)
    if doc!=expected:
        raise ProtocolError('extension specification drift')
    if verify_files:
        verify_protocol(yaml_path=yaml_path)
    return True

def finish_extension(out,stage_root,*,synthetic,yaml_path=YAML24):
    from confirm_runner.stages import ORDER,require_previous
    out=Path(out).resolve()
    previous={}
    if not synthetic:
        require_previous(stage_root,'CPU-R',yaml_path=HERE/'protocol.yaml')
        previous={name:sha256(Path(stage_root)/name/'stage_manifest.json') for name in ORDER[:-1]}
    outputs={str(p.relative_to(out.parent)):sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name!='stage_manifest.json'}
    record=dict(stage='CPU-R',status='COMPLETE',synthetic_only=synthetic,yaml_sha256=sha256(yaml_path),
                previous_manifests=previous,outputs=outputs)
    dest=out/'stage_manifest.json'
    if dest.exists() and json.loads(dest.read_text())!=record:
        raise ProtocolError('extension output drift; use fresh destination')
    dump(dest,record)
    return dict(path=str(dest),sha256=sha256(dest),manifest_line='stage manifest CPU-R '+sha256(dest))
