"""Build exact shared draws, then launch finetune's unchanged training loop/backends."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np

from .common import HERE, ROOT, YAML, ProtocolError, checked, dump, load_yaml, sha256
from .config import V2Config
from .inputs import load_part, campaign_assignments
from .labels import require_protocol

PYTHON=sys.executable


def ft_units(doc):
    out=[]
    for design in ['P','S']:
        for n in doc['families']['F4']['budgets']:
            for r in doc['replicates']['ids']:
                for f in (range(doc['designs']['P']['n_folds']) if design=='P' else [0]):
                    for bb in doc['families']['F4']['checkpoints']:
                        out.append(dict(design=design,n=n,fold=f,draw=r,checkpoint=bb,descriptive=False,
                            seed=1000*(f if design=='P' else 5)+r))
    for f in range(doc['designs']['P']['n_folds']):
        for bb in doc['families']['F4']['checkpoints']:
            out.append(dict(design='P',n=10000,fold=f,draw=0,checkpoint=bb,descriptive=True,seed=1000*f))
    return out


def unit_id(unit):
    return '{design}__{checkpoint}__n{n}__f{fold}__r{draw}'.format(**unit)


def draw_indices(doc, part, unit, train=None, tests=None):
    from analysis.draws import draw_labels, inner_split, outer_train_test
    cfg=V2Config(doc); n,f,r=unit['n'],unit['fold'],unit['draw']
    if unit['design']=='P':
        draw=draw_labels(cfg,part,f,r,n); test=outer_train_test(part,f)[1]
    else:
        # Import only the unchanged sampler; do not call any PILOT data loaders.
        import importlib.util
        sys.path[:0]=[str(ROOT/'src/arms'),str(ROOT/'src/pilot')]
        spec=importlib.util.spec_from_file_location('frozen_ft_campaign',ROOT/'src/campaign/campaign_arms.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        draw=module.campaign_draw(part,train,n,r); test=tests['S-a']
    fit,valid=inner_split(cfg,part,draw,f,r)
    if len(draw)!=n or len(set(draw))!=n or set(draw)&set(test):
        raise ProtocolError('FT draw size, uniqueness or test overlap')
    if set(part.chemsys[fit])&set(part.chemsys[valid]):
        raise ProtocolError('FT inner system overlap')
    if unit['design']=='P' and set(part.chemsys[draw])&set(part.chemsys[test]):
        raise ProtocolError('FT P outer system overlap')
    return dict(train=part.ids[draw].tolist(),fit=part.ids[fit].tolist(),valid=part.ids[valid].tolist(),test=part.ids[test].tolist())


def stage_unit(output, unit, ids, *, yaml_path=YAML, dry_run=False, stage=None, stage_root=None):
    dest=Path(output)/'ids'/unit_id(unit); dest.mkdir(parents=True,exist_ok=True)
    pins={}
    for name,values in ids.items():
        path=dest/(name+'.txt'); content='\n'.join(values)+'\n'
        if path.exists() and path.read_text()!=content:
            raise ProtocolError('existing frozen ID file drift: '+str(path))
        if not path.exists(): path.write_text(content)
        pins[name]=dict(path=str(path.resolve()),sha256=sha256(path))
    manifest=dict(format='p3_ft_unit_v2.3',unit=unit,ids=pins,dry_run=dry_run,
                  stage=stage,stage_root=None if stage_root is None else str(Path(stage_root).resolve()),
                  part='PILOT' if dry_run else 'CONFIRM',yaml_sha256=sha256(yaml_path),
                  draw_rule='frozen YAML shared CPU sampler; no label-dependent sampling')
    mp=dest/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ProtocolError('existing unit manifest drift')
    if not mp.exists(): dump(mp,manifest)
    return mp


def validate_driver_run_record(doc, manifest, run_record):
    """Bind a completed driver to its exact unit, training IDs and G004 recipe."""
    if manifest.get('synthetic_only'):
        if run_record.get('synthetic_only') is not True:
            raise ProtocolError('synthetic FT run_record missing fixture marker')
        return
    dry=manifest['dry_run'];unit=manifest['unit']
    if (run_record.get('status')!=('cpu_smoke_complete' if dry else 'complete') or run_record.get('dry_run') is not dry
        or run_record.get('v2.3_unit')!=unit or run_record.get('seed')!=unit['seed']
        or run_record.get('e0_mode')!='foundation' or run_record.get('variant')!='full'
        or run_record.get('train_geometry')!='relaxed'):
        raise ProtocolError('FT completed run_record unit or recipe drift')
    ids={name:checked(spec).read_text().split() for name,spec in manifest['ids'].items()}
    if run_record.get('ids')!=dict(draw=ids['train'],fit=ids['fit'],validation=ids['valid'],test=ids['test']):
        raise ProtocolError('FT completed run_record draw/test identity drift')
    access=run_record.get('ft_label_access',{})
    if (access.get('allowed_training_ids')!=ids['train'] or access.get('training_target_ids')!=ids['train']
        or access.get('test_label_columns_opened')!=[] or set(run_record.get('target_values_eV',{}))!=set(ids['train'])):
        raise ProtocolError('FT completed run_record target scope drift')
    if any(not set(row['ids'])<=set(ids['train']) for row in access.get('trace',[]) if row.get('training_target')):
        raise ProtocolError('FT run_record records targets outside draw')
    recipe=run_record.get('recipe',{});expected=doc['ft_recipe']
    for key,value in dict(optimizer='AdamW',lr=expected['full_lr'],batch_size=expected['batch_size'],
                          max_epochs=expected['epochs_max'],patience=expected['patience'],
                          weight_decay=expected['weight_decay'],betas=expected['betas'],eps=expected['eps']).items():
        if recipe.get(key)!=value: raise ProtocolError('FT optimizer run_record drift: '+key)
    if not 0<=run_record.get('selected_epoch',-1)<=expected['epochs_max']:
        raise ProtocolError('FT selected epoch outside frozen ceiling')
    if run_record.get('inputs',{}).get('checkpoint',{}).get('sha256')!=doc['inputs']['checkpoint_inputs'][unit['checkpoint']]['sha256']:
        raise ProtocolError('FT checkpoint run_record drift')


def validate_done(unit_dir, manifest_path):
    unit_dir=Path(unit_dir); path=unit_dir/'DONE.json'
    if not path.is_file(): return False
    try:
        d=json.loads(path.read_text())
        if d['exit_code']!=0 or d['manifest_sha256']!=sha256(manifest_path): return False
        if d['status']!='COMPLETE' or not d['output_sha256']: return False
        if d['driver_sha256']!=load_yaml()['implementation']['imports']['frozen_ft_driver']['sha256']: return False
        for name,h in d['output_sha256'].items():
            p=unit_dir/d['attempt']/name
            if Path(name).name!=name or not p.is_file() or sha256(p)!=h: return False
        run_record=json.loads((unit_dir/d['attempt']/'run_record.json').read_text())
        manifest=json.loads(Path(manifest_path).read_text())
        validate_driver_run_record(load_yaml(),manifest,run_record)
        if run_record['status'] not in ('complete','cpu_smoke_complete'): return False
        for name,h in run_record['output_sha256'].items():
            if d['output_sha256'].get(name)!=h: return False
        return True
    except (KeyError,ValueError,OSError): return False


def run_unit(doc, output, unit, manifest_path, gpu, dry_run):
    dest=Path(output)/'units'/unit_id(unit); dest.mkdir(parents=True,exist_ok=True)
    if validate_done(dest,manifest_path):
        print('SKIP valid DONE '+unit_id(unit),flush=True); return True
    # Keep interrupted/invalid attempts intact; each retry gets a fresh directory and log.
    attempt='attempt_'+uuid.uuid4().hex[:12]
    run=dest/attempt
    command=[PYTHON,'-B','-u','-m','confirm_runner.ft_driver','--manifest',str(manifest_path),
             '--output',str(run),'--e0','foundation','--train-geometry','relaxed']
    if dry_run: command+=['--dry-run']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if dry_run else str(gpu),
             MPLCONFIGDIR=str(dest/'mplconfig'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    dump(dest/(attempt+'_command.json'),dict(command=command,physical_gpu=None if dry_run else gpu,device='cpu' if dry_run else 'cuda'))
    start=time.monotonic()
    with open(dest/(attempt+'.log'),'x') as log:
        code=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    dump(dest/(attempt+'_exit.json'),dict(exit_code=code,seconds=time.monotonic()-start))
    if code: raise ProtocolError('FT unit failed; attempt retained: '+str(dest/attempt))
    run_record=json.loads((run/'run_record.json').read_text())
    validate_driver_run_record(doc,json.loads(Path(manifest_path).read_text()),run_record)
    hashes={name:sha256(run/name) for name in run_record['output_sha256']}
    hashes['run_record.json']=sha256(run/'run_record.json')
    if any(hashes[name]!=value for name,value in run_record['output_sha256'].items()): raise ProtocolError('driver output hash mismatch')
    done=dict(status='COMPLETE',exit_code=0,attempt=attempt,manifest_sha256=sha256(manifest_path),
              output_sha256=hashes,driver_sha256=doc['implementation']['imports']['frozen_ft_driver']['sha256'],
              unit=unit,seconds=time.monotonic()-start,dry_run=dry_run)
    done_path=dest/'DONE.json'
    if done_path.exists():
        # Preserve earlier invalid markers; never delete an output.
        dest.joinpath(attempt+'_prior_DONE.json').write_bytes(done_path.read_bytes())
    dump(done_path,done)
    print('DONE '+unit_id(unit),flush=True)
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--yaml',type=Path,default=YAML);p.add_argument('--out-root','--output',dest='output',type=Path,required=True)
    p.add_argument('--stage',choices=['FT-P','FT-S']);p.add_argument('--stage-root',type=Path)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--stage-only',action='store_true')
    a=p.parse_args()
    if a.gpu<=0: p.error('physical GPU 0 forbidden; positive physical index required')
    if a.yaml.resolve()!=YAML.resolve(): raise ProtocolError('launcher requires the bundled specified protocol')
    doc=load_yaml(a.yaml)
    # This guard precedes metadata loading, staging, torch import, and all target I/O.
    if not a.dry_run:
        require_protocol(yaml_path=a.yaml)
        from .stages import require_previous
        if a.stage is None or a.stage_root is None: raise ProtocolError('FT requires --stage and --stage-root')
        require_previous(a.stage_root,a.stage,yaml_path=a.yaml)
    for spec in doc['implementation']['imports'].values(): checked(spec)
    a.output.mkdir(parents=True,exist_ok=True)
    units=ft_units(doc)
    if not a.dry_run: units=[u for u in units if u['design']==a.stage[-1]]
    staged=[]
    if a.dry_run:
        from .inputs import metadata_csv
        src=ROOT/'src/finetune/dryrun_ids'
        ids={name:(src/(name+'.txt')).read_text().split() for name in ['train','fit','valid','test']}
        if len(set(ids['train']+ids['test']))!=8: raise ProtocolError('exactly eight PILOT structures required')
        # Verify the exact fine-tuning ID bytes specified by the protocol.
        for name in ids:
            path=src/(name+'.txt')
            # The public protocol specifies each input hash.
            checked(doc['validation_inputs']['ft_smoke_ids'][name])
        units=[dict(design='P',n=6,fold=0,draw=0,checkpoint=bb,descriptive=True,seed=0) for bb in doc['families']['F4']['checkpoints']]
        for unit in units: staged.append((unit,stage_unit(a.output,unit,ids,yaml_path=a.yaml,dry_run=True)))
    else:
        part=load_part(doc)
        train,tests,_=campaign_assignments(doc,part)
        for unit in units:
            ids=draw_indices(doc,part,unit,train,tests)
            staged.append((unit,stage_unit(a.output,unit,ids,yaml_path=a.yaml,stage=a.stage,stage_root=a.stage_root)))
    dump(a.output/'grid.json',dict(units=[u for u,_ in staged],yaml_sha256=sha256(a.yaml),dry_run=a.dry_run))
    if a.stage_only: return 0
    skipped=0
    for unit,manifest in staged:
        skipped+=run_unit(doc,a.output,unit,manifest,a.gpu,a.dry_run)
    report=dict(status='PASS',dry_run=a.dry_run,part='PILOT' if a.dry_run else 'CONFIRM',
                unique_structures=8 if a.dry_run else doc['population']['confirm_n'],
                completed_backbones=doc['families']['F4']['checkpoints'],exit_code=0,
                units=len(staged),skipped_valid_DONE=skipped,real_label_columns_opened=[] if a.dry_run else 'training-energy IDs in unit manifests only',
                physical_gpu=None if a.dry_run else a.gpu)
    dump(a.output/'launcher_run_record.json',report)
    return 0
if __name__=='__main__': raise SystemExit(main())
