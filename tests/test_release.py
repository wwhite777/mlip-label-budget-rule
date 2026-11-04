"""Failures that would change scientific authorization or leak held-out labels."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

from confirm_runner.common import ROOT, ProtocolError, load_yaml
from confirm_runner import relaxed as stage, relaxed_statistics as stats
from confirm_runner.labels import read_labels, LabelAccessError
from extension_gate import validate_document
from integrity import verify_protocol
from synthetic_check import fixture


@pytest.fixture
def doc():
    return load_yaml(ROOT / 'pipeline/protocol_v2.4.yaml')


def point_fixture(doc, high=.6, low=.5, reference=.5):
    points = {k: reference for k in stats.references(doc)}
    for n in doc['amendment_A2']['budgets']:
        for bb in doc['backbones']['order']:
            points[stats.fitted_name('A5fR', bb, n)] = high if n == 10000 else low
        points[stats.fitted_name('A4R', 'ensemble', n)] = high if n == 10000 else low
    points = stats.add_maxima(doc, points, doc['amendment_A2']['budgets'])
    return points, {k: np.full(2000, v) for k, v in points.items()}


def readings(doc, **kw):
    points, stars = point_fixture(doc, **kw)
    return stats.read_families(doc, points, stars, doc['amendment_A2']['budgets'], 'P', synthetic=True)


def test_extension_rule_and_exact_membership(doc):
    rows = readings(doc)
    result = stats.judge(doc, rows, synthetic=True)
    assert result['success'] and not result['scientific_reading_authorized']
    assert len(rows[0]['p_up_components']) == 12
    assert rows[0]['p_up_holm'] == pytest.approx(6 / 2001)
    assert rows[-1]['p_down_holm'] == pytest.approx(3 / 2001)
    assert stats.judge(doc, rows[:-1], synthetic=True)['status'] == 'INCOMPLETE'
    assert stats.judge(doc, rows + rows[:1], synthetic=True)['status'] == 'INCOMPLETE'
    wrong = deepcopy(rows)
    wrong[0]['checkpoint'] = 'wrong'
    assert not stats.judge(doc, wrong, synthetic=True)['success']
    assert not stats.judge(doc, rows)['scientific_reading_authorized']


def test_secondary_and_energy_only_cannot_satisfy_rule(doc):
    rows = readings(doc)
    for r in rows:
        if r['arm'] == 'A5fR':
            r['reading'] = 'INCONCLUSIVE'
    assert not stats.judge(doc, rows, synthetic=True)['success']
    for r in rows:
        r['design'] = 'S-a'
    assert stats.judge(doc, rows, synthetic=True)['status'] == 'INCOMPLETE'


def test_budget_3000_is_excluded_and_low_legs_both_required(doc):
    rows = readings(doc)
    next(r for r in rows if r['n'] == 3000)['reading'] = 'INCONCLUSIVE'
    assert stats.judge(doc, rows, synthetic=True)['success']
    next(r for r in rows if r['n'] == 300)['reading'] = 'INCONCLUSIVE'
    assert not stats.judge(doc, rows, synthetic=True)['success']


def test_iut_requires_all_twelve_references(doc):
    points, stars = point_fixture(doc)
    points['R0_chgnet'] = .8
    stars['R0_chgnet'] = np.full(2000, .8)
    points = stats.add_maxima(doc, points, doc['amendment_A2']['budgets'])
    stars = stats.add_maxima(doc, stars, doc['amendment_A2']['budgets'])
    rows = stats.read_families(doc, points, stars, doc['amendment_A2']['budgets'], 'P', synthetic=True)
    assert all(r['reading'] != 'ADDS' for r in rows)
    assert all(r['down_reference'] == 'R0R_orb' and r['down_estimate'] == 0 for r in rows if r['family'] == 'F6')
    del stars['ENS']
    with pytest.raises(ProtocolError, match='incomplete twelve-reference'):
        stats.read_families(doc, points, stars, [10000], 'P', synthetic=True)


def test_missing_values_and_strict_ensemble(doc):
    score = stage.ranked_score([.1, np.nan, -.2, np.nan])
    assert score[1] == score[3] < score[0] < score[2]
    np.testing.assert_array_equal(stage.ranked_score([np.nan, np.nan]), [0., 0.])
    with pytest.raises(ProtocolError):
        stage.ranked_score([np.inf])
    inp = SimpleNamespace(hulls=np.zeros((2, 5)), relaxed=np.array([[0] * 5, [np.nan, 0, 0, 0, 0]]))
    scores = stage.reference_scores(inp, np.arange(2), doc)
    assert len(scores) == 12 and scores['ENS-R'][0, 1] < scores['ENS-R'][0, 0]


@pytest.mark.parametrize('arm', ['A4R', 'A5fR', 'A5fRS'])
def test_actual_head_cannot_use_heldout_labels(doc, tmp_path, arm):
    inp, _, _ = fixture(doc, N=500)
    inp.relaxed = inp.hulls.copy()
    inp.relaxed[0, 0] = np.nan
    job = (arm, 'ensemble' if arm == 'A4R' else 'orb', 50, 0, 0)
    name = f'{arm}__{job[1]}__n50__f0__r0.npz'
    stage.prepare_context(inp, doc, 'P', tmp_path / 'first')
    stage.run_unit(job)
    idx, first = stage.validate_unit(tmp_path / 'first/units' / name, job)
    inp.labels.frame.loc[inp.part.ids[idx], 'stable'] = 1 - inp.labels.frame.loc[inp.part.ids[idx], 'stable']
    stage.prepare_context(inp, doc, 'P', tmp_path / 'second')
    stage.run_unit(job)
    _, second = stage.validate_unit(tmp_path / 'second/units' / name, job)
    np.testing.assert_array_equal(first, second)


def test_unpublished_protocol_rejected_before_csv_io(doc, tmp_path):
    path = tmp_path / 'changed.yaml'
    path.write_text('changed')
    with patch('pandas.read_csv', side_effect=AssertionError('source opened')):
        with pytest.raises((ProtocolError, LabelAccessError), match='specification'):
            stage.read_labels(doc, SimpleNamespace(part='CONFIRM'), yaml_path=path)
        with pytest.raises(LabelAccessError, match='specification'):
            read_labels(doc, SimpleNamespace(part='CONFIRM'), yaml_path=path)


def test_extension_guard_checks_every_setting(doc):
    changed = deepcopy(doc)
    changed['amendment_A2']['inference']['margin_AP'] = .03
    with pytest.raises(ProtocolError, match='drift'):
        validate_document(changed)
    assert validate_document(doc)


def test_extension_requires_complete_predecessors(doc, tmp_path):
    with pytest.raises(ProtocolError, match='requires completed'):
        stage.authorize_real(doc, tmp_path)
    with pytest.raises(ProtocolError, match='separate'):
        stage.validate_output_root(tmp_path / 'CPU-R', tmp_path)


def test_reduced_real_mode_never_authorized(doc, tmp_path):
    with pytest.raises(ProtocolError, match='full budgets'):
        stats.infer(doc, {}, np.array([0, 1]), ['A', 'B'], [300, 10000], [0, 1], 'P', 200, tmp_path, synthetic=False)


def test_fixed_specifications_and_sources():
    assert verify_protocol()['protocol_sha256']


def test_extension_bootstrap_matches_cluster_engine_and_seed_restart(doc, tmp_path):
    from analysis.inference import bootstrap
    from analysis.metrics import SortedScore
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 60)
    systems = np.array([f'S{i // 3:02}' for i in range(60)])
    scores = {k: rng.normal(size=(1, 60)) for k in stats.references(doc)}
    for n in [300, 10000]:
        for bb in doc['backbones']['order']:
            scores[stats.fitted_name('A5fR', bb, n)] = rng.normal(size=(2, 60))
        scores[stats.fitted_name('A4R', 'ensemble', n)] = rng.normal(size=(2, 60))
    si = np.unique(systems, return_inverse=True)[1]
    expected = bootstrap({k: [SortedScore(s, y) for s in v] for k, v in scores.items()}, si, 20, 17, 20261022, True)
    for design in ('P', 'S-a'):
        stats.infer(doc, scores, y, systems, [300, 10000], [0, 1], design, 17, tmp_path, synthetic=True)
        with np.load(tmp_path / f'bootstrap_{design}.npz') as actual, np.load(tmp_path / f'bootstrap_AUROC_{design}.npz') as auc:
            for key in scores:
                np.testing.assert_array_equal(actual[key], expected[0][key])
                np.testing.assert_array_equal(auc[key], expected[1][key])


def test_source_mutation_is_rejected(tmp_path):
    import hashlib
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    folder = tmp_path / 'pipeline'
    folder.mkdir()
    protocol = folder / 'protocol.yaml'
    protocol.write_text('fixed: true\n')
    contract = folder / 'CONTRACT.md'
    contract.write_text('Fixed specification.\n')
    source = folder / 'source.py'
    source.write_text('VALUE = 1\n')
    (tmp_path / 'PROTOCOL_SHA256').write_text(json.dumps({'pipeline/protocol.yaml': digest(protocol), 'pipeline/CONTRACT.md': digest(contract)}))
    (folder / 'release_integrity.json').write_text(json.dumps({'pipeline/source.py': digest(source)}))
    assert verify_protocol(root=tmp_path, yaml_path=protocol)
    source.write_text('VALUE = 2\n')
    with pytest.raises(ProtocolError, match='source changed'):
        verify_protocol(root=tmp_path, yaml_path=protocol)


def test_predecessor_chain_accepts_good_and_rejects_drift(tmp_path):
    from confirm_runner.common import YAML, dump, sha256
    from confirm_runner.stages import ORDER, require_previous
    for stage_name in ORDER[:-1]:
        folder = tmp_path / stage_name
        folder.mkdir()
        output = folder / 'value.txt'
        output.write_text(stage_name)
        before = ORDER[:ORDER.index(stage_name)]
        dump(folder / 'stage_manifest.json', dict(stage=stage_name, status='COMPLETE', synthetic_only=False,
             yaml_sha256=sha256(YAML), previous_manifests={s: sha256(tmp_path / s / 'stage_manifest.json') for s in before},
             outputs={str(output.relative_to(tmp_path)): sha256(output)}))
    require_previous(tmp_path, 'CPU-R')
    output = tmp_path / 'mechanism/value.txt'
    output.write_text('changed')
    with pytest.raises(ProtocolError, match='output changed'):
        require_previous(tmp_path, 'CPU-R')


def test_cli_schema_uses_published_input_identities():
    schema = json.loads((ROOT / 'pipeline/protocol_schema.json').read_text())
    doc = load_yaml()
    assert all(doc[key] == value for key, value in schema.items())


@pytest.mark.parametrize('extension', [False, True])
def test_changed_summary_rejected_before_csv_parse(doc, tmp_path, extension):
    source = tmp_path / 'summary.csv'
    source.write_text('changed source bytes')
    changed = deepcopy(doc)
    changed['population']['summary'] = dict(path='summary.csv', sha256='0' * 64)
    part = SimpleNamespace(part='CONFIRM')
    if extension:
        reader, guard = stage.read_labels, 'integrity.verify_protocol'
    else:
        reader, guard = read_labels, 'confirm_runner.labels.require_protocol'
    with patch(guard), patch('pandas.read_csv', side_effect=AssertionError('parsed changed input')):
        with pytest.raises(ProtocolError, match='input hash mismatch'):
            reader(changed, part, root=tmp_path)
