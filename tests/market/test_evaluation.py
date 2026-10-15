"""Project-authored MIT synthetic fixtures, seed 294; never market evidence.

Reuses #293's real immutable corpus; edge cases are created as new artifacts.
"""
from dataclasses import replace
import json
import pytest

from tests.market.test_distributional_baselines import corpus
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig
from cognitive_runtime.adapters.finance.models.fitting import fit_baseline
from cognitive_runtime.adapters.finance.evaluation.metrics import pinball, direction_brier, score
from cognitive_runtime.adapters.finance.evaluation.bootstrap import time_block_interval
from cognitive_runtime.adapters.finance.evaluation.population import bind_population
from cognitive_runtime.adapters.finance.evaluation.calibration import fit_calibrator
from cognitive_runtime.adapters.finance.evaluation.report import make_report, seal_report, markdown, validation_objective
from cognitive_runtime.adapters.finance.evaluation.ablations import reject_leakage, run_suite, Budget, ARMS
from cognitive_runtime.adapters.finance.schemas.records import DirectionProbability
from cognitive_runtime.data.ledger import digest, canonical
from cognitive_runtime.data.raw_store import RawStore


def forecasts(c, plan, partition='test', kind='zero'):
    model = fit_baseline(c, plan, listing_id='contract:demo', config=BaselineConfig(kind=kind, text_dimensions=4))
    return model.forecast([c.feature(k) for k in getattr(plan, partition)])


def report(c, plan, predictions, **kwargs):
    return make_report(c, plan, predictions, baseline=predictions, baseline_name='zero-change',
                       population='synthetic', run_id='synthetic-294', bootstrap={'repetitions': 20}, **kwargs)


def test_hand_scores(corpus):
    assert pinball(2, 1, .25) == .25
    assert pinball(0, 1, .25) == .75
    assert direction_brier(1, .8) == pytest.approx(.04)
    assert direction_brier(0, .8) == pytest.approx(.64)
    c, plan = corpus
    f = forecasts(c, plan)[0].forecast
    s = score(f, .2)
    assert s == {'pinball': pytest.approx(.1), 'median_mae': .2, 'interval_coverage': 0., 'interval_width': 0.}
    assert score(f, 0)['interval_coverage'] == 1
    head = DirectionProbability(positive_change_probability='0.8', method='separately_calibrated_head', calibrator_id='direction-validation')
    assert score(replace(f, direction=head), .2)['direction_brier'] == pytest.approx(.04)
    for bad in (float('nan'), float('inf')):
        with pytest.raises(ValueError):
            pinball(bad, 0, .5)


def test_calibration_partition_identity_and_immutability(corpus):
    c, plan = corpus
    val, test = forecasts(c, plan, 'validation'), forecasts(c, plan)
    before = digest([p.forecast.to_dict() for p in test])
    cal = fit_calibrator(c, plan, val)
    assert cal.validation_ids == plan.validation
    assert cal.validation_labels == tuple(c.manifest['labels'][k] for k in plan.validation)
    calibrated = cal.apply(c, plan, test)
    assert all(p.forecast.calibrator_id == cal.hash for p in calibrated)
    assert digest([p.forecast.to_dict() for p in test]) == before
    assert [p.forecast_id for p in calibrated] == [p.forecast_id for p in test]
    with pytest.raises(ValueError, match='validation-only'):
        fit_calibrator(c, plan, test, partition='test')
    with pytest.raises(ValueError, match='population'):
        fit_calibrator(c, plan, test)
    for field in ('snapshot_sha256', 'label_spec_sha256', 'split_sha256', 'forecast_id'):
        with pytest.raises(ValueError, match='identity'):
            fit_calibrator(c, plan, (replace(val[0], **{field: 'f'*64}),) + val[1:])
    with pytest.raises(ValueError, match='already calibrated'):
        cal.apply(c, plan, calibrated)
    with pytest.raises(ValueError, match='model mismatch'):
        cal.apply(c, plan, forecasts(c, plan, kind='numeric'))


def test_time_blocks_are_not_independent_rows():
    a = '2025-01-01T00:00:00Z'
    b = '2025-01-02T00:00:00Z'
    result = time_block_interval([(a, 0), (a, 0), (b, 2), (b, 2)], repetitions=20)
    assert result['blocks'] == 2 and result['n'] == 4
    assert result['lower'] == 0 and result['upper'] == 2
    assert result == time_block_interval([(a, 0), (a, 0), (b, 2), (b, 2)], repetitions=20)
    assert time_block_interval([(a, 1), (a, 2)])['status'] == 'insufficient_time_blocks'
    with pytest.raises(ValueError):
        time_block_interval([], repetitions=1000000)


def test_report_pairing_lock_and_nonpromotion(corpus, tmp_path):
    c, plan = corpus
    predictions = forecasts(c, plan)
    r = report(c, plan, predictions)
    assert r['summary']['counts'] == dict(samples=3, realized=3, censored=0, abstained=0, stale=0, missing=0, scored=3)
    assert r['summary']['paired_pinball_improvement']['value'] == 0
    assert r['summary']['paired_pinball_improvement']['forecast_ids'] == [p.forecast_id for p in predictions]
    assert not r['edge_promotable'] and r['pnl'] is None
    assert r['summary']['metrics']['direction_brier']['value'] is None
    assert r['slices']['regime']['unknown']['counts']['samples'] == 3
    text = markdown(r)
    assert text.index('Coverage first') < text.index('Scores and uncertainty')
    assert 'unknown' in text and 'survivorship' in text and 'synthetic_only' in text
    path = tmp_path / 'sealed.json'
    assert seal_report(r, path) == digest(r)
    assert json.loads(path.read_text())['report']['sealed']
    with pytest.raises(FileExistsError):
        seal_report(r, path)
    with pytest.raises(ValueError, match='validation'):
        validation_objective(r, metric='pinball', predeclared_metric='pinball')
    v = report(c, plan, forecasts(c, plan, 'validation'), partition='validation')
    assert validation_objective(v, metric='pinball', predeclared_metric='pinball') >= 0
    with pytest.raises(ValueError, match='predeclared'):
        validation_objective(v, metric='pinball', predeclared_metric='median_mae')
    with pytest.raises(ValueError, match='population'):
        report(c, plan, predictions[:-1])
    with pytest.raises(ValueError, match='exactly once'):
        report(c, plan, predictions, slices={'regime': {'quiet': [plan.test[0]]}})
    with pytest.raises(ValueError, match='separate'):
        make_report(c, plan, predictions, baseline=predictions, baseline_name='zero', population='financial_market', run_id='bad')


def test_failed_abstained_and_censored_counts(corpus):
    c, plan = corpus
    # New immutable label artifact and manifest; original corpus remains unchanged.
    key = plan.test[0]
    old = c._outcome(key)
    censored = replace(old, label=None, censor_reason='missing_endpoint', anchor_to_endpoint_seconds=None)
    manifest = {**c.manifest, 'labels': dict(c.manifest['labels'])}
    manifest['labels'][key] = RawStore(c.root / 'labels').put(censored.canonical_json().encode())
    new_hash = RawStore(c.root / 'manifests').put(canonical(manifest).encode())
    from cognitive_runtime.adapters.finance.datasets.corpus import Corpus
    changed = Corpus(c.root, new_hash)
    predictions = forecasts(changed, plan)
    failed = tuple(replace(p, forecast=None, abstention_reason='model_failed') for p in predictions)
    r = report(changed, plan, failed, status='failed', failure='bounded fit failed')
    assert r['summary']['counts']['samples'] == 3
    assert r['summary']['counts']['censored'] == 1
    assert r['summary']['counts']['abstained'] == 3
    assert r['summary']['counts']['scored'] == 0
    assert r['summary']['censor_reasons'] == {'missing_endpoint': 1}
    assert c._outcome(key).label is not None


def test_oracle_control_rejected(corpus):
    c, plan = corpus
    with pytest.raises(ValueError, match='oracle'):
        reject_leakage(input_fields=('numeric_context', 'future_return'), fitted_ids=plan.train, plan=plan)
    with pytest.raises(ValueError, match='held-out'):
        reject_leakage(input_fields=('text_context',), fitted_ids=plan.test, plan=plan)
    # Concrete future label payload at the real inference boundary also fails.
    from cognitive_runtime.adapters.finance.models.distribution import observations
    feature = c.feature(plan.test[0])
    with pytest.raises(ValueError, match='feature artifact'):
        observations({**feature, 'oracle_label': c._outcome(plan.test[0]).to_dict()})


def test_equal_budget_tiny_suite(corpus):
    c, plan = corpus
    reports = run_suite(c, plan, listing_id='contract:demo', population='synthetic',
                        budget=Budget(seeds=(294,), text_dimensions=4), bootstrap={'repetitions': 20})
    assert tuple(r['ablation']['arm'] for r in reports) == ARMS
    for field in ('budget_sha256', 'labels_sha256', 'train_ids', 'usable_train_ids'):
        assert all(r['ablation'][field] == reports[0]['ablation'][field] for r in reports)
    assert all(r['partition'] == 'validation' and not r['edge_promotable'] for r in reports)
    assert reports[5]['summary']['metrics'] == reports[-1]['summary']['metrics']


def test_suite_retains_failed_arms(corpus, monkeypatch):
    c, plan = corpus
    from cognitive_runtime.adapters.finance.evaluation import ablations
    original = ablations.fit_baseline
    def fail_text(*args, **kwargs):
        if kwargs['config'].kind == 'text':
            raise ValueError('synthetic fit failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(ablations, 'fit_baseline', fail_text)
    reports = run_suite(c, plan, listing_id='contract:demo', population='synthetic',
                        budget=Budget(text_dimensions=4), bootstrap={'repetitions': 20})
    failed = next(r for r in reports if r['ablation']['arm'] == 'text')
    assert failed['status'] == 'failed' and failed['failure'] == 'synthetic fit failure'
    assert failed['summary']['counts']['abstained'] == len(plan.validation)
    assert len(reports) == len(ARMS)


def test_calibration_exact_residuals_and_changed_labels(corpus):
    c, plan = corpus
    val = forecasts(c, plan, 'validation')
    cal = fit_calibrator(c, plan, val)
    # Validation outcomes: -.04, +.04, -.04, all zero-change predictions.
    assert tuple(float(x) for x in cal.corrections) == pytest.approx((-.04, -.04, .024))
    changed = replace(cal, corpus_sha256='f'*64)
    with pytest.raises(ValueError, match='corpus/split/label'):
        changed.apply(c, plan, forecasts(c, plan))
    class MismatchedLabelView:
        def __getattr__(self, name):
            return getattr(c, name)
        def _outcome(self, key):
            return replace(c._outcome(key), spec_sha256='f'*64)
    with pytest.raises(ValueError, match='label snapshot/spec'):
        fit_calibrator(MismatchedLabelView(), plan, val)


def test_text_control_donors_remain_partition_local(corpus):
    c, plan = corpus
    from cognitive_runtime.adapters.finance.evaluation.ablations import _TextControl
    for mode in ('shuffled_text', 'lagged_text'):
        factory = lambda *args: _TextControl(*args, corpus=c, plan=plan, mode=mode, seed=294)
        m = fit_baseline(c, plan, listing_id='contract:demo', config=BaselineConfig(kind='multimodal', text_dimensions=4), model_factory=factory)
        for partition in (plan.train, plan.validation, plan.test):
            assert all(m.donors[k] is None or m.donors[k] in partition for k in partition)
            if mode == 'lagged_text':
                assert m.donors[partition[0]] is None
                assert all(int(m.donors[k]) < int(k) for k in partition[1:])
        original = [c.feature(k) for k in plan.test]
        before = digest(original)
        inputs, _ = m.inputs(original)
        baseline_inputs, _ = super(_TextControl, m).inputs(original)
        assert all(a[:4] == b[:4] for a, b in zip(inputs.features, baseline_inputs.features))
        assert digest(original) == before


def test_null_signal_cannot_promote_synthetic_edge(corpus):
    from pathlib import Path
    from cognitive_runtime.adapters.finance.datasets.corpus import Corpus
    c, plan = corpus
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/market/evaluation/null.json').read_text())
    manifest = {**c.manifest, 'labels': dict(c.manifest['labels'])}
    for key in manifest['labels']:
        outcome = c._outcome(key)
        # Null-label fixture, explicitly synthetic and separate from the original
        # label store. No claim to reconstruct the source ledger's price path.
        label = replace(outcome.label, value=fixture['target_value'], endpoint_price=outcome.label.anchor_price)
        null = replace(outcome, label=label)
        manifest['labels'][key] = RawStore(c.root / 'labels').put(null.canonical_json().encode())
    null_corpus = Corpus(c.root, RawStore(c.root / 'manifests').put(canonical(manifest).encode()))
    reports = run_suite(null_corpus, plan, listing_id='contract:demo', population='synthetic',
                       budget=Budget(text_dimensions=4), bootstrap={'repetitions': 20})
    assert all(r['summary']['paired_pinball_improvement']['value'] == 0 for r in reports)
    assert all(not r['edge_promotable'] for r in reports)
    assert all(r['summary']['metrics']['pinball']['value'] == 0 for r in reports)


def test_template_and_fixture_inventory(corpus):
    from pathlib import Path
    import hashlib
    root = Path(__file__).resolve().parents[2]
    template = json.loads((root / 'docs/market/evaluation-report.template.json').read_text())
    c, plan = corpus
    actual = report(c, plan, forecasts(c, plan))
    assert set(template) - {'template_only'} == set(actual)
    assert set(template['summary']) == set(actual['summary'])
    inventory = json.loads((root / 'tests/fixtures/market/evaluation/provenance.json').read_text())
    for relative, expected in inventory['sha256'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_lag_control_uses_instant_not_timestamp_spelling(monkeypatch):
    from types import SimpleNamespace
    from brain.cortex.model_contracts import InferenceInput
    from cognitive_runtime.adapters.finance.models.baselines import BaselineModel
    from cognitive_runtime.adapters.finance.evaluation.ablations import _TextControl
    # Valid UTC strings deliberately disagree with lexicographic time order.
    stamps = {'early': '2025-01-01T00:00:00Z', 'same': '2025-01-01T00:00:00.0Z',
              'late': '2025-01-01T00:00:00.1Z'}
    c = SimpleNamespace(samples=[SimpleNamespace(sample_id=k, decision_at=v) for k, v in stamps.items()], feature=lambda k: k)
    plan = SimpleNamespace(train=tuple(stamps), validation=(), test=())
    def init(self, *args):
        self.config = SimpleNamespace(text_dimensions=1)
    def inputs(self, keys):
        return InferenceInput('a'*64, tuple(keys), tuple((0., 0., 0., 0., float(list(stamps).index(k)), 0.) for k in keys)), ()
    monkeypatch.setattr(BaselineModel, '__init__', init)
    monkeypatch.setattr(BaselineModel, 'inputs', inputs)
    model = _TextControl(corpus=c, plan=plan, mode='lagged_text', seed=294)
    assert model.donors['early'] is None
    assert model.donors['same'] is None
    assert model.donors['late'] in ('early', 'same')


def test_inner_label_and_split_sample_bindings(corpus):
    from cognitive_runtime.adapters.finance.datasets.corpus import Corpus
    from cognitive_runtime.adapters.finance.datasets.splits import walk_forward
    c, plan = corpus
    key = plan.validation[0]
    outcome = c._outcome(key)
    foreign_labels = (
        replace(outcome.label, listing_id='foreign-contract'),
        replace(outcome.label, spec=replace(outcome.label.spec, anchor_max_age_seconds=999)),
    )
    for label in foreign_labels:
        manifest = {**c.manifest, 'labels': dict(c.manifest['labels'])}
        foreign = replace(outcome, label=label)
        manifest['labels'][key] = RawStore(c.root / 'labels').put(foreign.canonical_json().encode())
        changed = Corpus(c.root, RawStore(c.root / 'manifests').put(canonical(manifest).encode()))
        with pytest.raises(ValueError, match='inner label semantics'):
            fit_calibrator(changed, plan, forecasts(changed, plan, 'validation'))
    manifest = {**c.manifest, 'samples': [dict(s) for s in c.manifest['samples']]}
    for s in manifest['samples']:
        if s['sample_id'] == key:
            s['snapshot_sha256'] = 'f'*64
    changed = Corpus(c.root, RawStore(c.root / 'manifests').put(canonical(manifest).encode()))
    rebound = walk_forward(changed.samples, validation_at=plan.validation_at, test_at=plan.test_at)
    with pytest.raises(ValueError, match='sample/context'):
        fit_calibrator(changed, rebound, forecasts(changed, rebound, 'validation'))
    # The split validator alone cannot know that label availability was altered
    # after the declared outcome interval. Evaluation must bind that interval.
    future = replace(outcome, label=replace(outcome.label, target_available_at='2025-01-14T23:59:00Z'))
    manifest = {**c.manifest, 'labels': dict(c.manifest['labels'])}
    manifest['labels'][key] = RawStore(c.root / 'labels').put(future.canonical_json().encode())
    changed = Corpus(c.root, RawStore(c.root / 'manifests').put(canonical(manifest).encode()))
    with pytest.raises(ValueError, match='split outcome interval'):
        fit_calibrator(changed, plan, forecasts(changed, plan, 'validation'))
