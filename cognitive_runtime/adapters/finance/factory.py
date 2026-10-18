"""Operator helpers for the shared Factory; no independent trainer or state loop."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from cognitive_runtime.data.ledger import digest
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json, _jsonable
from cognitive_runtime.training.model_factory.runner import run_trial
from cognitive_runtime.training.model_factory.spec import resolve, TASK_DOCUMENT_FORMAT
from cognitive_runtime.training.model_factory.registry import (
    promote, record_test_use, reserve_test_protocol)
from cognitive_runtime.training.model_factory.checkpoint import load_factory_checkpoint
from .factory_backend import MarketBackend
from .models.baselines import BaselineConfig
from .evaluation.report import seal_report, validation_objective


def market_spec(corpus, plan, *, seed=295, baseline=None, organism='MarketSynthetic', seconds=20):
    return resolve({'format': TASK_DOCUMENT_FORMAT, 'organism': organism,
        'data': {'corpus_id': corpus.root.name, 'corpus_sha256': corpus.manifest_sha256,
                 'split': plan.to_dict(), 'listing_id': 'contract:demo'},
        'model': {'backend': 'market-baseline', 'baseline': asdict(baseline or BaselineConfig())},
        'training': {'max_training_seconds': seconds, 'determinism_policy': {'seed': seed}},
        'evaluation': {'selection_metric': 'pinball'}})


def final_test(result, *, corpus_root, root):
    """Explicit one-use reporting; never called by run_trial or search.

    Consume both the existing candidate ledger and a data/split-wide reservation
    before reading test labels. Retrying a failed report requires a new protocol,
    not deleting these records. Arbitrary filesystem writers are outside this guard.
    """
    directory = Path(root) / result.run_id
    attempt = Path(result.directory).resolve()
    if not attempt.is_relative_to((directory / 'attempts').resolve()):
        raise ValueError('result attempt does not belong to run')
    manifest = json.loads((attempt / 'run_manifest.json').read_text())
    if manifest['run_id'] != result.run_id or manifest['status'] != 'completed':
        raise ValueError('completed market manifest required')
    for ref in manifest['artifacts']:
        artifact = attempt / ref['path']
        if not artifact.resolve().is_relative_to(attempt) or hashlib.sha256(artifact.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('artifact evidence differs')
    checkpoint = attempt / 'checkpoints/last.json'
    if Path(result.checkpoint_path).resolve() != checkpoint or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result.checkpoint_sha256:
        raise ValueError('checkpoint differs from selected result')
    spec = resolve(json.loads((directory / 'trial_spec.json').read_text()))
    backend = MarketBackend()
    backend.validate(spec)
    backend.spec = spec
    backend.prepare(spec, corpus_root=corpus_root)
    evaluation = json.loads((attempt / 'metrics/validation.json').read_text())
    score = validation_objective(evaluation, metric='pinball', predeclared_metric='pinball')
    slot = dict(family='market-price-change-v1', tier='fast', objective='validation.pinball')
    promote(root, **slot, run_id=result.run_id, checkpoint_path=result.checkpoint_path,
            checkpoint_sha256=result.checkpoint_sha256, metrics={'pinball': score},
            as_leading=False, reason='synthetic-only final reporting candidate')
    protocol = digest({'label_spec': backend.corpus.manifest['spec'],
        'heldout': {k: {'feature': backend.corpus.manifest['features'][k],
                        'label': backend.corpus.manifest['labels'][k]} for k in backend.plan.test}})
    reserve_test_protocol(Path(root).parent, protocol_sha256=protocol, run_id=result.run_id)
    record_test_use(root, **slot, run_id=result.run_id, max_uses=1)
    model = load_factory_checkpoint(result.checkpoint_path).model['baseline']
    report = backend.report(model, 'test')
    path = result.directory / 'metrics' / 'sealed-test.json'
    seal_report(report, path)
    return path
