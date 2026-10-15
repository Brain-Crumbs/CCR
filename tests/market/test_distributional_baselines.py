"""MIT project-authored seed 293; real ledger/corpus, no provider payloads.

Fixed tiny paths exercise bounds, pairing, missingness and train-only fitting.
They are structural acceptance evidence, not financial prediction validation.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cognitive_runtime.adapters.finance.testing.examples import examples, envelope, seal_source, source_payload
from cognitive_runtime.adapters.finance.schemas.records import Instrument, NewsEvent, PredictionContractState
from cognitive_runtime.adapters.finance.schemas.semantics import Horizon, Clock
from cognitive_runtime.adapters.finance.datasets.corpus import Corpus, Request, build_corpus
from cognitive_runtime.adapters.finance.datasets.splits import walk_forward
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig, BaselineModel, KINDS
from cognitive_runtime.adapters.finance.models.fitting import fit_baseline
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical, digest
from cognitive_runtime.data.snapshot import build_snapshot


@pytest.fixture
def corpus(tmp_path):
    base = examples()
    policy = RetentionPolicy(rights='project-authored-fixture', evidence='MIT seed 293', allow_raw=True,
                             allow_normalized=True, retain_provenance=True)
    with Ledger(tmp_path / 'ledger') as ledger:
        def append(r):
            r = seal_source(r)
            ledger.append(r, canonical(source_payload(r)).encode(), policy)
            return r
        append(replace(base['Instrument'], listing_id='contract:demo', kind='binary_contract'))
        requests = []
        for day in range(6, 18):
            at = f'2025-01-{day:02}T14:30:00Z'
            # Include exact 0/1 anchors in sealed test. No settlement used.
            price = 0.0 if day == 15 else 1.0 if day == 16 else .4
            signal = .04 if day % 2 else -.04
            for clock, value in [('14:29', .4-signal), ('14:30', price), ('14:31', min(1., max(0., price+signal)))]:
                stamp = at.replace('14:30', clock)
                append(replace(base['PredictionContractState'], **envelope(PredictionContractState, f'p-{day}-{clock}', stamp),
                    probability_price=str(value), scheduled_expiry_at='2025-02-01T00:00:00Z'))
            append(replace(base['NewsEvent'], **envelope(NewsEvent, f'n-{day}', at),
                original_published_at=at, first_seen_at=at, dedup_cluster_id=f'story-{day}',
                title='rise' if signal > 0 else 'fall', text='synthetic signal'))
            def snapshot(stamp):
                return build_snapshot(ledger, decision_at=stamp, price_seconds=180, text_seconds=180,
                    calendar_id='synthetic', calendar_version='1', calendar_sha256='a'*64, preprocessing_sha256='b'*64)
            decision, outcome = snapshot(at), snapshot(at.replace('14:30', '14:31'))
            requests.append(Request(sample_id=str(day), snapshot_sha256=decision.context.hash,
                outcome_snapshot_sha256=outcome.context.hash, listing_id='contract:demo', universe_id='synthetic-universe-v1'))
        spec = replace(base['LabelSpec'], horizon=Horizon(clock=Clock.ELAPSED, count=60, calendar_id=None, calendar_version=None))
        c = build_corpus(ledger, tmp_path / 'corpus', requests, spec=spec)
    c = Corpus(c.root, c.manifest_sha256)
    plan = walk_forward(c.samples, validation_at='2025-01-12T00:00:00Z', test_at='2025-01-15T00:00:00Z')
    assert c.population()['reasons'] == {'realized': 12}
    return c, plan


@pytest.mark.parametrize('kind', KINDS)
def test_fit_pairing_support_and_resume(corpus, tmp_path, kind):
    c, plan = corpus
    cfg = BaselineConfig(kind=kind, text_mode='tfidf', text_dimensions=8)
    model = fit_baseline(c, plan, listing_id='contract:demo', config=cfg)
    features = [c.feature(k) for k in plan.test]
    forecasts = model.forecast(features)
    zero = fit_baseline(c, plan, listing_id='contract:demo').forecast(features)
    assert [r.forecast_id for r in forecasts] == [r.forecast_id for r in zero]
    assert all(r.label_spec_sha256 == digest(c.manifest['spec']) and r.split_sha256 == plan.hash for r in forecasts)
    for result in forecasts:
        f = result.forecast
        assert f.abstention_reason is None and f.direction is None and f.calibrator_id is None
        assert f.unit.value == 'probability_price_fraction'
        values = [float(q.value) for q in f.quantiles]
        assert values == sorted(values)
        assert all(-float(f.anchor_price) <= v <= 1-float(f.anchor_price) for v in values)
        assert 'synthetic_generator_recovery_only' in f.quality
        if kind == 'zero':
            assert values == [0, 0, 0]
    assert model.state()['fit']['train_sample_ids'] == list(plan.train)
    path = tmp_path / 'state.json'
    model.save(path)
    loaded = BaselineModel.load(path, corpus_sha256=c.manifest_sha256, split_sha256=plan.hash, config=cfg)
    assert loaded.hash == model.hash
    assert loaded.forecast(features) == forecasts
    for changed in [dict(corpus_sha256='a'*64), dict(split_sha256='a'*64), dict(config=BaselineConfig(kind='zero' if kind != 'zero' else 'numeric'))]:
        args = dict(corpus_sha256=c.manifest_sha256, split_sha256=plan.hash, config=cfg); args.update(changed)
        with pytest.raises(ValueError, match='differs'):
            BaselineModel.load(path, **args)


def test_heldout_labels_never_opened(corpus, monkeypatch):
    c, plan = corpus
    original = c._outcome
    touched = []
    def read(key):
        assert key in plan.train
        touched.append(key)
        return original(key)
    monkeypatch.setattr(c, '_outcome', read)
    m = fit_baseline(c, plan, listing_id='contract:demo', config=BaselineConfig(kind='multimodal'))
    assert touched == list(plan.train)
    state = m.hash
    m.forecast([c.feature(k) for k in plan.test])
    assert m.hash == state
    with pytest.raises(ValueError):
        fit_baseline(c, replace(plan, train=plan.test, test=plan.train), listing_id='contract:demo')


def test_missing_anchor_and_stale_are_explicit(corpus):
    c, plan = corpus
    m = fit_baseline(c, plan, listing_id='contract:demo')
    f = c.feature(plan.test[0])
    from cognitive_runtime.adapters.finance.schemas.records import ContextSnapshot
    context = ContextSnapshot(**f['context'])
    f['records'] = [r for r in f['records'] if r['schema_name'] != 'PredictionContractState']
    f['context'] = replace(context, selected_record_ids=tuple(r['record_id'] for r in f['records']),
        selected_record_hashes=tuple(digest(r) for r in f['records'])).to_dict()
    result = m.forecast([f])[0]
    assert result.forecast is None and result.abstention_reason == 'missing_anchor'
    f = c.feature(plan.test[0]); context = ContextSnapshot(**f['context'])
    f['context'] = replace(context, decision_at='2025-01-15T16:30:00Z').to_dict()
    result = m.forecast([f])[0]
    assert result.abstention_reason == 'stale_anchor' and result.forecast.quantiles == ()


def test_rejects_tampering_and_training_batch(corpus, tmp_path):
    c, plan = corpus
    m = fit_baseline(c, plan, listing_id='contract:demo')
    f = c.feature(plan.test[0]); f['records'][-1]['source_id'] = 'tampered'
    with pytest.raises(ValueError):
        m.forecast([f])
    from cognitive_runtime.training.model_factory.task_contracts import TrainingBatch
    inputs, _ = m.inputs([c.feature(plan.test[0])])
    with pytest.raises(TypeError):
        m.predict(TrainingBatch(inputs, ((123.,),)))
    p = tmp_path / 'state.json'; m.save(p)
    raw = json.loads(p.read_text()); raw['state']['fit']['coefficients'][0] = 9
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='integrity'):
        BaselineModel.load(p, corpus_sha256=c.manifest_sha256, split_sha256=plan.hash, config=m.config)


@pytest.mark.parametrize('kwargs', [{'kind': 'jepa'}, {'ridge': 0}, {'text_dimensions': 4096},
    {'quantiles': [0.9, 0.1]}, {'max_train_samples': 513}, {'ridge': float('nan')}])
def test_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        BaselineConfig(**kwargs)


def test_inference_import_is_framework_and_outcome_free():
    code = '''
import sys
from cognitive_runtime.adapters.finance.models.baselines import BaselineModel
assert not any(x in sys.modules for x in ('torch', 'transformers', 'cognitive_runtime.adapters.finance.labels'))
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


def test_equity_support_and_decimal_boundary():
    from cognitive_runtime.adapters.finance.models.distribution import emit
    from cognitive_runtime.adapters.finance.schemas.semantics import Target, Unit, Mark
    from cognitive_runtime.adapters.finance.schemas.records import ContextSnapshot
    base = examples()
    feature = {'version': 1, 'sample_id': 'equity', 'context': base['ContextSnapshot'].to_dict(),
        'records': [base['PriceBar'].to_dict(), base['NewsEvent'].to_dict()], 'quality_records': [], 'universe': {}}
    spec = replace(base['LabelSpec'], target=Target.EQUITY_RETURN, unit=Unit.RETURN, mark=Mark.TRADE,
        action_policy='split_neutral_ex_dividend', horizon=Horizon(clock=Clock.SESSIONS, count=1,
        calendar_id='synthetic', calendar_version='synthetic-v1')).to_dict()
    result = emit(feature, spec, 'listing:demo', 'a'*64, 'model', [.1, .5, .9], [-2., 0., 10.])
    assert [q.value for q in result.forecast.quantiles] == ['-1', '0.0', '10.0']
    assert result.forecast.unit == Unit.RETURN and result.forecast.support_upper is None


def test_expiry_abstention_and_quantile_monotonicity(corpus):
    c, plan = corpus
    model = fit_baseline(c, plan, listing_id='contract:demo')
    f = c.feature(plan.test[0])
    from cognitive_runtime.adapters.finance.schemas.records import ContextSnapshot
    for r in f['records']:
        if r['schema_name'] == 'PredictionContractState':
            r['scheduled_expiry_at'] = '2025-01-15T14:30:30Z'
    context = ContextSnapshot(**f['context'])
    f['context'] = replace(context, selected_record_hashes=tuple(digest(r) for r in f['records'])).to_dict()
    assert model.forecast([f])[0].abstention_reason == 'closed_or_expired'


def test_inference_missing_text_mask_and_preprocessing_state(corpus):
    c, plan = corpus
    model = fit_baseline(c, plan, listing_id='contract:demo', config=BaselineConfig(kind='text', text_mode='tfidf'))
    from cognitive_runtime.adapters.finance.text.features import TextFeatures
    state = model.text.state
    restored = TextFeatures.from_state(state)
    assert restored.signature == model.text.signature
    state['idf'][0] = float('nan')
    with pytest.raises(ValueError):
        TextFeatures.from_state(state)
    from cognitive_runtime.adapters.finance.schemas.records import ContextSnapshot
    f = c.feature(plan.test[0]); context = ContextSnapshot(**f['context'])
    f['records'] = [r for r in f['records'] if r['schema_name'] != 'NewsEvent']
    f['context'] = replace(context, selected_record_ids=tuple(r['record_id'] for r in f['records']),
        selected_record_hashes=tuple(digest(r) for r in f['records'])).to_dict()
    inputs, _ = model.inputs([f])
    assert inputs.features[0] == (0.,)*model.config.text_dimensions + (1.,)
    assert model.forecast([f])[0].abstention_reason is None
