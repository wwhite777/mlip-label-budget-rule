from copy import deepcopy

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from confirm_runner.common import ProtocolError, load_yaml
from confirm_runner.statistics import add_maxima, read_families, judge, infer
from analysis.metrics import SortedScore
from analysis.inference import holm


def test_weighted_AP_equals_explicit_cluster_resampling():
    y = np.array([0,1,1,0,1,0])
    score = np.array([.1,.4,.4,.7,.8,.8])
    weights = np.array([2,2,0,0,3,3])
    idx = np.repeat(np.arange(6), weights)
    assert SortedScore(score, y).ap(weights) == average_precision_score(y[idx], score[idx])


def test_maxima_switch_inside_every_resample_after_mean_AP():
    d = load_yaml()
    v = {f'R0_{b}': np.array([.1,.1]) for b in d['backbones']['order']}
    v['R0_mace_medium'], v['R0_orb'] = np.array([.8,.2]), np.array([.2,.9])
    for b in d['backbones']['order']:
        v[f'A5f_hist_gbm_{b}@300'] = np.array([.1,.1])
        v[f'C1_ridge_{b}@300'] = np.array([.1,.1])
    v['A5f_hist_gbm_mace_medium@300'], v['A5f_hist_gbm_orb@300'] = np.array([.95,.3]), np.array([.4,.98])
    v['A4_hist_gbm_ensemble@300'] = np.array([.1,.1])
    v['E0_mean'] = np.array([.1,.1])
    out = add_maxima(d, v, [300])
    np.testing.assert_allclose(out['R0_star'], [.8,.9])
    np.testing.assert_allclose(out['F2_max@300'], [.95,.98])
    # max(mean_r AP) != mean_r(max AP_r); the first is the specified estimand.
    mean_AP = {'left': np.mean([.9,.1]), 'right': np.mean([.2,.8])}
    assert max(mean_AP.values()) == .5
    assert np.mean(np.max([[.9,.1],[.2,.8]], axis=0)) == pytest.approx(.85)


def test_holm_sets_are_separate_and_not_interval_decisions():
    d = load_yaml()
    pts = {f'R0_{b}':0. for b in d['backbones']['order']}; pts['R0_star']=0.
    stars = {k:np.zeros(200) for k in pts}
    for n, diff in [(300, -.1), (1000,.02), (3000,.2)]:
        pts[f'F2_max@{n}'] = diff
        stars[f'F2_max@{n}'] = np.full(200, diff)
    rows = read_families(d, pts, stars, [300,1000,3000], 'P', synthetic=True)
    assert [r['reading'] for r in rows] == ['DOES_NOT_EXCEED', 'INCONCLUSIVE', 'DESCRIPTIVE_ADDS']
    for row in rows:
        assert row['p_up_holm'] == holm([r['p_up'] for r in rows], [r['contrast'] for r in rows])[rows.index(row)]
        assert row['p_down_holm'] == holm([r['p_down'] for r in rows], [r['contrast'] for r in rows])[rows.index(row)]


def success_rows():
    from confirm_runner.statistics import contrast_specs
    doc=load_yaml();rows=[]
    for family,specs in contrast_specs(doc,[300,1000,3000,10000]).items():
        if family not in ('F1','F2'):continue
        for name,arm,bb,n,_,_ in specs:
            rows.append(dict(design='P',family=family,n=n,reading='DOES_NOT_EXCEED' if family=='F2' else 'INCONCLUSIVE',
                arm=arm,model='hist_gbm' if family=='F1' else 'maximum',checkpoint=bb,contrast=name,family_complete=True,synthetic_only=False))
    return rows


def test_judge_reads_yaml_A5f_only_success_conjunction():
    d = load_yaml()
    rows = success_rows()
    rows[5]['reading']='ADDS'  # A4 alone does not qualify.
    assert not judge(d, rows, complete=True, synthetic=True)['success']
    rows[0]['reading']='ADDS'
    assert judge(d, rows, complete=True, synthetic=True)['success']
    rows[6]['reading']='INCONCLUSIVE'
    assert not judge(d, rows, complete=True, synthetic=True)['success']
    rows[6]['reading']='DOES_NOT_EXCEED'
    rows[0]['design']='S-b'
    assert not judge(d, rows, complete=True, synthetic=True)['success']


def test_reduced_run_cannot_claim_full_success():
    result = judge(load_yaml(), success_rows()[:2], synthetic=True, complete=False)
    assert result['status'] == 'INCOMPLETE' and not result['success']
    assert not result['scientific_reading_authorized']


def test_nonfinite_contrast_has_no_reading():
    with pytest.raises(ProtocolError, match='nonfinite'):
        read_families(load_yaml(), {'R0_star': .1, 'R0_orb':.1, 'F2_max@300': .2},
                      {'R0_star': np.zeros(200), 'R0_orb':np.zeros(200), 'F2_max@300': np.full(200, np.nan)}, [300], 'P', synthetic=True)


def test_inference_restarts_seed_per_design_and_replicates_AP_not_scores(tmp_path):
    d = load_yaml()
    rng = np.random.default_rng(9)
    y = np.tile([0,1,0,0,1], 8)
    systems = np.repeat(np.arange(20), 2)
    scores = {f'R0_{b}': rng.random((1,40)) for b in d['backbones']['order']}
    scores['E0_mean'] = rng.random((1,40))
    for b in d['backbones']['order']:
        scores[f'A5f_hist_gbm_{b}@300'] = rng.random((2,40))
        scores[f'C1_ridge_{b}@300'] = rng.random((2,40))
    scores['A4_hist_gbm_ensemble@300'] = rng.random((2,40))
    for name in ['P','S-a']:
        infer(d, scores, y, systems, [300], [0,1], name, 20, tmp_path, True)
    with np.load(tmp_path/'bootstrap_P.npz') as p, np.load(tmp_path/'bootstrap_S-a.npz') as s:
        for key in p.files:
            np.testing.assert_array_equal(p[key], s[key])
    import json
    report=json.loads((tmp_path/'inference_P.json').read_text())
    key='A5f_hist_gbm_orb@300'
    assert report['points'][key]==np.mean([average_precision_score(y,row) for row in scores[key]])
    assert report['points'][key]!=average_precision_score(y,scores[key].mean(axis=0))


def test_parallel_bootstrap_has_identical_rng_and_results(tmp_path):
    d=load_yaml()
    rng=np.random.default_rng(7)
    y=np.tile([0,0,1,1],10)
    systems=np.repeat(np.arange(20),2)
    scores={f'R0_{b}':rng.random((1,40)) for b in d['backbones']['order']}
    for workers in [1,2]:
        infer(d,scores,y,systems,[],[0,1],f'workers{workers}',20,tmp_path,True,workers=workers)
    for prefix in ['bootstrap_','bootstrap_AUROC_']:
        with np.load(tmp_path/f'{prefix}workers1.npz') as a,np.load(tmp_path/f'{prefix}workers2.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key])



def test_draw_spread_is_sample_SD_of_M_r_beside_every_reading(tmp_path):
    import json
    d=load_yaml();rng=np.random.default_rng(52)
    y=np.tile([0,0,1,1],10);systems=np.repeat(np.arange(20),2)
    scores={f'R0_{b}':rng.random((1,40)) for b in d['backbones']['order']}
    scores['E0_mean']=rng.random((1,40))
    for b in d['backbones']['order']:
        for prefix in ['A5f_hist_gbm_','C1_ridge_']:
            scores[f'{prefix}{b}@300']=rng.random((5,40))
    scores['A4_hist_gbm_ensemble@300']=rng.random((5,40))
    rows,_=infer(d,scores,y,systems,[300],list(range(5)),'P',20,tmp_path,True)
    report=json.loads((tmp_path/'inference_P.json').read_text())
    for row in rows:
        spread=row['draw_spread']
        assert spread['SD_M_r']==pytest.approx(np.std(report['M_r'][spread['arm']],ddof=1))
        assert spread['reference_SD_M_r']==pytest.approx(0.)
        assert spread['SD_difference_r']==pytest.approx(spread['SD_M_r'])
        assert row['inference_scope']=='for the five frozen draws, conditional on fitted models'
