"""MIT project-authored seed 295; fake fits, real Factory/corpus/report boundaries."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from cognitive_runtime.adapters.finance.testing.factory import prepare_synthetic
from cognitive_runtime.adapters.finance.factory import market_spec, final_test
from cognitive_runtime.adapters.finance.factory_backend import MarketBackend
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig, BaselineModel
from cognitive_runtime.adapters.finance.text.features import TextFeatures
from cognitive_runtime.data.ledger import digest
from cognitive_runtime.training.model_factory.artifacts import _jsonable
from cognitive_runtime.training.model_factory.runner import run_trial
from cognitive_runtime.training.model_factory.spec import resolve
from cognitive_runtime.training.model_factory.task_contracts import TaskCancelled
from cognitive_runtime.training.model_factory.state import load_state, state_path
from cognitive_runtime.training.model_factory.checkpoint import load_factory_checkpoint
from cognitive_runtime.training.model_factory.registry import register_artifacts, RegistryError, TestBudgetExhaustedError as BudgetExhausted
from cognitive_runtime.training.model_factory.search import propose_market, run_market_search


@pytest.fixture(scope='module')
def pinned(tmp_path_factory):
    return prepare_synthetic(tmp_path_factory.mktemp('market295'))


def fake_baseline_fit(corpus, plan, *, listing_id, config=BaselineConfig()):
    text = TextFeatures(config.text_mode, config.text_dimensions).fit(corpus, plan)
    model = BaselineModel(config, corpus.manifest['spec'], listing_id, plan.hash, corpus.manifest_sha256, text.state)
    width = config.text_dimensions + (1 if config.kind == 'text' else 5) if config.kind in ('text', 'multimodal') else 4
    model._fit = {'mean': [0.] * width, 'scale': [1.] * width, 'coefficients': [0.] * (width + 1),
        'residual_quantiles': [0.] * len(config.quantiles), 'volatility_floor': .01,
        'train_sample_ids': list(plan.train), 'censored_train_sample_ids': [],
        'training_target_sha256': digest({'fake_seed': 295}), 'algorithm': 'fake-zero-v1', 'calibration': 'none'}
    return model

@pytest.fixture(autouse=True)
def fake_fit(monkeypatch):
    monkeypatch.setattr('cognitive_runtime.adapters.finance.factory_backend.fit_baseline', fake_baseline_fit)


def launch(pinned, tmp_path, **kwargs):
    corpus, plan = pinned
    return run_trial(market_spec(corpus, plan), root=tmp_path, corpus_root=corpus.root.parent, **kwargs)


def manifest(result):
    return json.loads((result.directory / 'run_manifest.json').read_text())


def test_lifecycle_reproducible_and_private(pinned, tmp_path):
    a, b = launch(pinned, tmp_path), launch(pinned, tmp_path)
    ma, mb = manifest(a), manifest(b)
    assert ma['status'] == 'completed' and not ma['partial']
    assert ma['reproducibility_sha256'] == mb['reproducibility_sha256']
    assert a.checkpoint_sha256 == b.checkpoint_sha256
    assert a.evaluation == b.evaluation
    assert ma['attempt_id'] != mb['attempt_id'] and a.run_id != b.run_id
    assert all(ma['hashes'].values())
    assert ma['git_commit'] and ma['environment_lock']['packages']['numpy']
    assert ma['seed'] == 295 and ma['budget']['provider_usd'] == 0
    assert ma['budget']['process_peak_rss_bytes'] > 0
    assert ma['state_components']['optimizer'] is None
    assert ma['readiness'] == 'synthetic_only' and not ma['edge_promotable']
    assert a.evaluation['partition'] == 'validation' and not a.evaluation['sealed']
    for ref in ma['artifacts']:
        path = a.directory / ref['path']
        assert path.stat().st_size == ref['size_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
        assert 'synthetic signal' not in path.read_text()
    assert not list(a.directory.rglob('sealed-test.json'))
    restored = load_factory_checkpoint(a.checkpoint_path).model['baseline']
    assert restored.hash == a.training_stats['model_sha256']


def test_seed_and_configuration_hashes(pinned, tmp_path):
    c, p = pinned
    a = launch(pinned, tmp_path)
    b = run_trial(market_spec(c, p, seed=296), root=tmp_path, corpus_root=c.root.parent)
    assert manifest(a)['hashes']['config'] != manifest(b)['hashes']['config']
    changed_c, changed_p = prepare_synthetic(tmp_path / 'changed', seed=296)
    assert changed_c.manifest_sha256 != c.manifest_sha256
    assert changed_p.hash != p.hash  # split binds changed snapshot evidence


@pytest.mark.parametrize('stage', ['prepare', 'fit', 'export'])
def test_failure_partial_manifest_redacts_message(pinned, tmp_path, monkeypatch, stage):
    def fail(*a, **kw):
        raise RuntimeError('SECRET-credential protected text')
    monkeypatch.setattr(MarketBackend, stage, fail)
    with pytest.raises(RuntimeError):
        launch(pinned, tmp_path, run_id='failure')
    directory = tmp_path / 'MarketSynthetic/failure'
    paths = list(directory.glob('attempts/*/run_manifest.json'))
    assert len(paths) == 1
    doc = json.loads(paths[0].read_text())
    assert doc['partial'] and doc['status'] == 'failed' and not doc['promotable']
    assert 'SECRET' not in ''.join(p.read_text() for p in directory.rglob('*.json'))
    assert load_state(state_path(directory)).state == 'failed'
    with pytest.raises(FileExistsError):
        launch(pinned, tmp_path, run_id='failure')


@pytest.mark.parametrize('outcome', ['cancelled', 'budget_exceeded'])
def test_cancel_and_budget_visible(pinned, tmp_path, monkeypatch, outcome):
    from cognitive_runtime.training.model_factory.task_runner import TaskBudgetExceeded
    def stop(*args, **kw):
        raise TaskCancelled('cancel') if outcome == 'cancelled' else TaskBudgetExceeded('limit')
    monkeypatch.setattr(MarketBackend, 'fit', stop)
    result = launch(pinned, tmp_path)
    doc = manifest(result)
    assert result.state == outcome and doc['partial'] and not doc['promotable']
    assert doc['budget']['elapsed_seconds'] >= 0
    assert load_factory_checkpoint(result.checkpoint_path).model['baseline'] is None


def test_rights_and_export_boundaries(pinned, tmp_path, monkeypatch):
    prepare = MarketBackend.prepare
    def unknown(self, *args, **kw):
        prepared = prepare(self, *args, **kw)
        self.rights = [json.dumps({'rights': 'unknown', 'license': 'unknown', 'synthetic': True})]
        return prepared
    monkeypatch.setattr(MarketBackend, 'prepare', unknown)
    with pytest.raises(RegistryError, match='rights_blocked'):
        launch(pinned, tmp_path)
    assert not list(tmp_path.rglob('artifact_registry.json'))
    (tmp_path / 'large').write_bytes(b'1234')
    with pytest.raises(RegistryError, match='budget'):
        register_artifacts(tmp_path, ['large'], rights_cleared=True, max_bytes=3)
    with pytest.raises(RegistryError, match='escapes'):
        register_artifacts(tmp_path, ['../secret'], rights_cleared=True, max_bytes=100)
    (tmp_path / 'link').symlink_to(tmp_path / 'large')
    with pytest.raises(RegistryError, match='symlink'):
        register_artifacts(tmp_path, ['link'], rights_cleared=True, max_bytes=100)


def test_clone_preserves_parent_and_rejects_changed_pins(pinned, tmp_path):
    c, p = pinned
    parent = launch(pinned, tmp_path)
    original = Path(parent.checkpoint_path).read_bytes()
    raw = _jsonable(market_spec(c, p).to_dict())
    raw.update(mode='clone', parent={'run_id': parent.run_id, 'sha256': parent.checkpoint_sha256,
        'checkpoint': str(Path(parent.checkpoint_path).relative_to(tmp_path / 'MarketSynthetic' / parent.run_id))})
    child = run_trial(raw, root=tmp_path, corpus_root=c.root.parent)
    assert child.checkpoint_sha256 == parent.checkpoint_sha256
    assert Path(parent.checkpoint_path).read_bytes() == original
    for changed in ('model', 'data'):
        value = json.loads(json.dumps(raw))
        if changed == 'model':
            value['model']['baseline']['text_dimensions'] = 9
        else:
            value['data']['corpus_sha256'] = 'a' * 64
        with pytest.raises(ValueError, match='differs|incompatible'):
            run_trial(value, root=tmp_path, corpus_root=c.root.parent)


def test_search_deduplicates_and_cannot_read_test(pinned, tmp_path, monkeypatch):
    from cognitive_runtime.adapters.finance.datasets.corpus import Corpus
    c, p = pinned
    read = Corpus._outcome
    def forbid_test(self, key):
        assert key not in p.test
        return read(self, key)
    monkeypatch.setattr(Corpus, '_outcome', forbid_test)
    base = market_spec(c, p, baseline=BaselineConfig(kind='text', text_dimensions=8))
    proposals = propose_market(base, 2, 295)
    assert proposals == propose_market(base, 2, 295)
    assert all(s.data == base.data and s.training == base.training and s.evaluation == base.evaluation for s in proposals)
    results = run_market_search([proposals[0], proposals[0]], root=tmp_path, corpus_root=c.root.parent)
    assert results[0] == results[1] and results[0]['status'] == 'completed'
    assert results[0]['score'] is not None
    assert len(list(tmp_path.rglob('run_manifest.json'))) == 1
    assert run_market_search([proposals[0]], root=tmp_path, corpus_root=c.root.parent)[0] == results[0]


def test_final_test_is_explicit_once_per_protocol(pinned, tmp_path):
    from cognitive_runtime.adapters.finance.evaluation.report import validation_objective
    c, p = pinned
    result = launch(pinned, tmp_path)
    report_path = final_test(result, root=tmp_path / 'MarketSynthetic', corpus_root=c.root.parent)
    report = json.loads(report_path.read_text())['report']
    assert report['sealed'] and report['partition'] == 'test'
    with pytest.raises(ValueError, match='validation'):
        validation_objective(report, metric='pinball', predeclared_metric='pinball')
    other = launch(pinned, tmp_path)
    for candidate in (result, other):
        with pytest.raises(BudgetExhausted):
            final_test(candidate, root=tmp_path / 'MarketSynthetic', corpus_root=c.root.parent)


def test_interrupted_resume_keeps_attempt_evidence(pinned, tmp_path):
    import subprocess
    import sys
    from cognitive_runtime.training.model_factory.state import heartbeat_path
    c, p = pinned
    raw = _jsonable(market_spec(c, p).to_dict())
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(raw))
    code = """
import json, os, sys
from tests.test_model_factory_market import fake_baseline_fit
import cognitive_runtime.adapters.finance.factory_backend as fb
from cognitive_runtime.training.model_factory.runner import run_trial
fb.fit_baseline = fake_baseline_fit
fit = fb.MarketBackend.fit
def interrupt(self, model, data, configuration, control, *, resume):
    fit(self, model, data, configuration, control, resume=resume)
    control.checkpoint()
    os._exit(23)
fb.MarketBackend.fit = interrupt
run_trial(json.load(open(sys.argv[1])), root=sys.argv[2], corpus_root=sys.argv[3], run_id='interrupted')
"""
    child = subprocess.run([sys.executable, '-c', code, str(path), str(tmp_path), str(c.root.parent)], timeout=20)
    assert child.returncode == 23
    directory = tmp_path / 'MarketSynthetic/interrupted'
    attempt = next((directory / 'attempts').iterdir())
    before = {str(x): x.read_bytes() for x in attempt.rglob('*.json')}
    heartbeat = json.loads(heartbeat_path(directory).read_text())
    heartbeat['heartbeat_seconds'] = 0
    heartbeat_path(directory).write_text(json.dumps(heartbeat))
    checkpoint = attempt / 'checkpoints/last.json'
    raw.update(mode='resume', parent={'run_id': 'interrupted',
        'checkpoint': str(checkpoint.relative_to(directory)),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    resumed = run_trial(raw, root=tmp_path, corpus_root=c.root.parent)
    assert resumed.state == 'completed' and resumed.run_id == 'interrupted'
    assert len(list((directory / 'attempts').iterdir())) == 2
    assert {str(x): x.read_bytes() for x in attempt.rglob('*.json')} == before
    assert resumed.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
