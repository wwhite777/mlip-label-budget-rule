from types import SimpleNamespace

import numpy as np
import pytest

from confirm_runner.common import load_yaml, sha256, ROOT
from confirm_runner.config import V2Config
from confirm_runner.labels import read_labels, LabelAccessError
from confirm_runner import runner
from analysis.data import build_part
from analysis.draws import draw_labels, inner_split


def fixture_input():
    rng = np.random.default_rng(2)
    ids = np.array([f'wbm-1-{i:04}' for i in range(200)])
    part = build_part('CONFIRM', ids, [f'S{i//2:03}' for i in range(200)],
                      [i//2%5 for i in range(200)], ['H']*200, dict.fromkeys(ids,'CONFIRM'))
    d = load_yaml()
    hull = rng.normal(size=(200,5))
    refs = {b: dict(r_sp=dict(e_form=hull[:,i]-2, e_hull=hull[:,i])) for i,b in enumerate(d['backbones']['order'])}
    inp = SimpleNamespace(part=part, comp=rng.random((200,4)), feats={b:rng.normal(size=(200,4)).astype('float32') for b in refs},
                          hulls=hull, refs=refs, labels=read_labels(d, part, synthetic=True), cfg=V2Config(d))
    return inp, d


def test_imports_byte_identical_and_grids():
    d = load_yaml()
    a, p, c = runner.source_imports()
    for name, module in [('arms',a),('arm_inputs',p),('campaign',c)]:
        assert sha256(module.__file__) == d['implementation']['imports'][name]['sha256']
    assert a.unit.__code__.co_filename == str(ROOT/'src/arms/run_arms.py')
    assert len(V2Config(d).head_grid('hist_gbm')) == 4


def test_exact_draws_canonical_order_and_shared_draws(tmp_path):
    inp, d = fixture_input()
    runner.setup_context(inp,d,'P',tmp_path)
    for f in range(5):
        draw,tr,va,test = runner.split(50,f,1)
        np.testing.assert_array_equal(draw, draw_labels(inp.cfg,inp.part,f,1,50))
        assert not set(inp.part.chemsys[draw]) & set(inp.part.chemsys[test])
        assert not set(inp.part.chemsys[tr]) & set(inp.part.chemsys[va])
    train=np.arange(150)
    tests={'S-a':np.arange(150,200), 'S-b':np.arange(160,200)}
    runner.setup_context(inp,d,'S',tmp_path,train,tests)
    draw=runner.split(50,0,1)[0]
    np.testing.assert_array_equal(draw, runner.CAMPAIGN.campaign_draw(inp.part,train,50,1))


def test_empty_inner_split_stops_without_redraw():
    inp,d=fixture_input()
    same_system=np.flatnonzero(inp.part.sys_index==0)
    with pytest.raises(Exception, match='empty inner split'):
        inner_split(inp.cfg, inp.part, same_system, 0, 0)


def test_single_class_fallback_and_outside_draw_guard(tmp_path):
    inp,d=fixture_input()
    runner.setup_context(inp,d,'P',tmp_path)
    draw,tr,va,test=runner.split(50,0,0)
    inp.labels.frame.loc[:,'stable']=0
    inp.labels.active_draw=set(inp.part.ids[draw])
    view=lambda idx: runner.label_frame(idx,draw).stable.to_numpy()
    res, flags=runner.ARMS.binary_select(['hist_gbm'], {'hist_gbm': inp.feats['orb']}, view,tr,va,draw,test)
    assert np.all(res['scores']['hist_gbm']==0)
    assert 'draw_single_class_constant_score' in flags
    with pytest.raises(LabelAccessError):
        view(test)
    inp.labels.active_draw=None


def test_C14_matrix_and_shared_campaign_fit():
    d=load_yaml()
    jobs=runner.jobs_for(d,'S',[300,1000],[0,1])
    assert len(jobs)==84
    assert all(f==0 for _,_,_,f,_ in jobs)
    assert {arm for arm,*_ in jobs}=={'A1','A4','A5','A5f','C1'}
    assert 100 not in d['run_matrix']['named_cpu']['budgets_by_fit_design']['S']


def test_precision_ceil_and_id_ties():
    assert runner.precision(np.array([1,0,0]), np.zeros(3),np.array(['a','b','c']))==1


def test_heldout_labels_cannot_change_fits_or_selection(tmp_path):
    inp,d=fixture_input()
    runner.setup_context(inp,d,'P',tmp_path)
    job=('A5','orb',50,0,0)
    first,_=runner.run_unit(job)
    path=tmp_path/'units/A5__orb__n50__f0__r0.npz'
    with np.load(path) as z:
        original=z['score'].copy()
    test=runner.split(50,0,0)[3]
    # Change all evaluation labels, never any member of the fixed training draw.
    inp.labels.frame.loc[inp.part.ids[test],'stable']=1-inp.labels.frame.loc[inp.part.ids[test],'stable']
    second,_=runner.run_unit(job)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['score'],original)
    assert first['chosen']==second['chosen'] and first['inner_ap']==second['inner_ap']


def test_canonical_row_order_invariant_to_source_permutation(tmp_path):
    inp,d=fixture_input()
    runner.setup_context(inp,d,'P',tmp_path)
    before=inp.part.ids[runner.split(50,2,1)[0]]
    p=inp.part
    idx=np.random.default_rng(99).permutation(p.n)
    inp.part=build_part(p.part,p.ids[idx],p.chemsys[idx],p.fold[idx],p.formula[idx],p.part_of)
    runner.setup_context(inp,d,'P',tmp_path)
    after=inp.part.ids[runner.split(50,2,1)[0]]
    np.testing.assert_array_equal(before,after)
