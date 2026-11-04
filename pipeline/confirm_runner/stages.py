"""Local stage order, predecessor chain and immutable output manifests."""
from pathlib import Path
import json
from .common import ROOT, YAML, ProtocolError, dump, sha256, load_yaml

ORDER = ('CPU-P', 'judge', 'CPU-S', 'FT-P', 'FT-S', 'mechanism', 'CPU-R')


def finish_stage(stage_root, stage, paths, *, synthetic, yaml_path=YAML, status='COMPLETE', completed_units=None, incomplete_cells=None):
    root = Path(stage_root).resolve()
    files = {}
    for path in paths:
        path = Path(path).resolve()
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in candidates:
            if file.is_file() and file.name != 'stage_manifest.json':
                files[str(file.relative_to(root))] = sha256(file)
    previous_manifests = {}
    if not synthetic:
        require_previous(root, stage, yaml_path=yaml_path, allow_ft_caps=status == 'INCOMPLETE_CAP')
        previous_manifests = {name: sha256(root/name/'stage_manifest.json') for name in ORDER[:ORDER.index(stage)]}
    report = dict(previous_manifests=previous_manifests, stage=stage, status=status, synthetic_only=synthetic,
                  yaml_sha256=sha256(yaml_path), outputs=files,
                  next_stage_specification='unchanged frozen YAML; no outcome-dependent changes')
    if status == 'INCOMPLETE_CAP':
        if stage not in ('FT-P', 'FT-S') or completed_units is None or incomplete_cells is None:
            raise ProtocolError('terminal FT cap manifest requires completed units and incomplete cells')
        report.update(completed_units=completed_units, incomplete_cells=incomplete_cells,
                      F4_status='INCOMPLETE', terminal=True)
    dest = root / stage / 'stage_manifest.json'
    if dest.exists():
        if json.loads(dest.read_text()) != report:
            raise ProtocolError('completed stage output drift; use a fresh stage root')
    else:
        dump(dest, report)
    return dict(path=str(dest), sha256=sha256(dest),
                manifest_line=f'stage manifest {stage} {sha256(dest)}')



def require_previous(stage_root, stage, *, yaml_path=YAML, root=ROOT, allow_ft_caps=False):
    """Before real access, verify every predecessor manifest and output."""
    if stage not in ORDER:
        raise ProtocolError('unknown execution stage')
    stage_root = Path(stage_root).resolve()
    for previous in ORDER[:ORDER.index(stage)]:
        path = stage_root / previous / 'stage_manifest.json'
        if not path.is_file():
            raise ProtocolError(f'{stage} requires completed {previous}')
        data = json.loads(path.read_text())
        chain = {name: sha256(stage_root/name/'stage_manifest.json') for name in ORDER[:ORDER.index(previous)]}
        if data.get('previous_manifests') != chain:
            raise ProtocolError('earlier stage manifest chain changed')
        capped = (stage == 'mechanism' or allow_ft_caps) and previous in ('FT-P', 'FT-S') and data.get('status') == 'INCOMPLETE_CAP'
        if capped and (data.get('terminal') is not True or data.get('F4_status') != 'INCOMPLETE'
                       or not isinstance(data.get('completed_units'), dict) or not data.get('incomplete_cells')
                       or any(data.get('outputs', {}).get(p) != h for p,h in data['completed_units'].items())):
            raise ProtocolError('terminal FT cap manifest missing completed unit hashes or incomplete cells')
        if (data.get('stage') != previous or (data.get('status') != 'COMPLETE' and not capped) or data.get('synthetic_only') is not False
                or data.get('yaml_sha256') != sha256(yaml_path) or not data.get('outputs')):
            raise ProtocolError('earlier stage incomplete, synthetic or protocol changed')
        for relative, digest in data['outputs'].items():
            target = (stage_root / relative).resolve()
            if not target.is_relative_to(stage_root) or sha256(target) != digest:
                raise ProtocolError('earlier stage output changed')


def finish_ft_cap(stage_root, stage, ft_root, *, yaml_path=YAML, synthetic=False):
    """Record only validated completed units; leave all missing F4 cells incomplete."""
    from .ft_launcher import ft_units, unit_id, validate_done
    if stage not in ('FT-P', 'FT-S'):
        raise ProtocolError('terminal cap is only valid for FT-P/FT-S')
    root = Path(stage_root).resolve(); ft_root = Path(ft_root).resolve()
    if not ft_root.is_relative_to(root):
        raise ProtocolError('FT cap outputs must be inside stage root')
    doc = load_yaml(yaml_path); completed = {}; paths = []; missing = []
    for unit in ft_units(doc):
        if unit['design'] != stage[-1]:
            continue
        name = unit_id(unit); dest = ft_root/'units'/name; manifest = ft_root/'ids'/name/'manifest.json'
        if validate_done(dest, manifest):
            source = json.loads(manifest.read_text())
            if source.get('unit') != unit or source.get('yaml_sha256') != sha256(yaml_path) or source.get('dry_run'):
                raise ProtocolError('completed FT cap unit protocol/identity mismatch')
            if bool(source.get('synthetic_only')) != synthetic:
                raise ProtocolError('completed FT cap unit synthetic/real mismatch')
            done = dest/'DONE.json'; completed[str(done.relative_to(root))] = sha256(done)
            paths.extend([dest, manifest.parent])
        elif not unit['descriptive']:
            missing.append(dict(unit=name, design=unit['design'], checkpoint=unit['checkpoint'], n=unit['n'],
                                fold=unit['fold'], draw=unit['draw'], status='INCOMPLETE_CAP', reading=None))
    if not missing:
        raise ProtocolError('FT cap manifest requires incomplete F4 cells; finish complete FT normally')
    report_path = root/stage/'F4_incomplete.json'
    dump(report_path, dict(status='INCOMPLETE', reason='GPU_CAP', completed_units=completed, incomplete_cells=missing,
                           readings=[], primary_judgment='retained in CPU-P/judge.json'))
    return finish_stage(root, stage, paths+[report_path], synthetic=synthetic, yaml_path=yaml_path,
                        status='INCOMPLETE_CAP', completed_units=completed, incomplete_cells=missing)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terminal-ft-cap', choices=['FT-P','FT-S'], required=True)
    p.add_argument('--stage-root', type=Path, required=True)
    p.add_argument('--ft-root', type=Path, required=True)
    p.add_argument('--yaml', type=Path, default=YAML)
    p.add_argument('--out-root', type=Path, required=True)
    a=p.parse_args()
    if a.out_root.resolve() != a.stage_root.resolve()/a.terminal_ft_cap:
        p.error('--out-root must be stage-root/stage')
    marker=finish_ft_cap(a.stage_root, a.terminal_ft_cap, a.ft_root, yaml_path=a.yaml)
    print(marker['manifest_line'])


if __name__ == '__main__':
    main()
