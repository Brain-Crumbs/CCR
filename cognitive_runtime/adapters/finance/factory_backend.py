"""Synthetic-only market lifecycle through the existing neutral Factory runner.

No providers, downloads or optional model imports. Closed-form baseline state
has no optimizer/scheduler/EMA; these are explicitly inapplicable, not omitted
neural state. Private corpora are never part of the export inventory.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys

from brain.cortex.model_contracts import InferenceInput, require_inference_input
from cognitive_runtime.data.ledger import digest
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json, _jsonable
from cognitive_runtime.training.model_factory.task_contracts import (
    TaskIdentity, TaskDataContract, ModelDefinition, PreparedTask, TrainingBatch, EvaluationBatch)
from .models.baselines import BaselineConfig, BaselineModel, numeric
from .models.fitting import fit_baseline
from .datasets.corpus import Corpus
from .datasets.splits import SplitPlan, validate_split
from .evaluation.report import make_report, validation_objective

IDENTITY = TaskIdentity('market', 'price-change', 'market-baseline')


def execution_identity():
    """Hash executed project sources and installed versions, including dirty code."""
    from importlib import metadata
    import subprocess
    root = Path(__file__).resolve().parents[3]
    files = {}
    for folder in ('cognitive_runtime', 'brain'):
        for path in sorted((root / folder).rglob('*.py')):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=5)
    return {'source_sha256': digest(files), 'commit': commit.stdout.strip() if commit.returncode == 0 else None,
            'packages': dict(sorted(versions.items())), 'python': platform.python_version(),
            'machine': platform.machine(), 'implementation': platform.python_implementation()}


class MarketBackend:
    identity = IDENTITY
    capabilities = frozenset({'prepare', 'fit', 'predict', 'evaluate', 'save', 'load',
                              'resume', 'clone', 'manifest', 'export'})

    def validate(self, spec):
        if set(spec.data) != {'corpus_id', 'corpus_sha256', 'split', 'listing_id'}:
            raise ValueError('market data requires exact corpus, split and listing pins')
        from .schemas.base import sha256
        sha256(spec.data['corpus_sha256'])
        SplitPlan(**_jsonable(spec.data['split']))
        if set(spec.model) != {'backend', 'baseline'}:
            raise ValueError('market model requires baseline configuration only')
        BaselineConfig(**_jsonable(spec.model['baseline']))
        if set(spec.training) != {'device', 'precision', 'determinism_policy', 'max_training_seconds'}:
            raise ValueError('unsupported market training setting')
        if dict(spec.training['determinism_policy']).keys() != {'seed'} or type(spec.training['determinism_policy']['seed']) is not int:
            raise ValueError('integer seed required')
        if dict(spec.evaluation) != {'selection_metric': 'pinball'}:
            raise ValueError('market selection requires validation pinball only')
        for value in (spec.data['corpus_id'], spec.data['listing_id']):
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError('bounded identifiers required')
        if Path(spec.data['corpus_id']).name != spec.data['corpus_id'] or spec.data['corpus_id'] in ('.', '..'):
            raise ValueError('corpus_id must be a single safe path component')

    def describe(self, spec):
        identity = self.identity.to_dict()
        schema = digest({'version': 1, 'baseline': _jsonable(spec.model['baseline'])})
        definition = ModelDefinition(identity, _jsonable(spec.model), schema)
        data = TaskDataContract(identity, spec.data['corpus_id'], schema, _jsonable(spec.data))
        return definition, data

    def begin(self, spec, artifacts, *, run_directory, attempt_id):
        self.spec, self.run_directory, self.attempt_id = spec, run_directory, attempt_id
        self.execution = execution_identity()
        self.stage = 'prepare'
        self.corpus = self.plan = None
        self.rights = None
        self.exported = []
        self.finish(artifacts, None, status='running', budget={'elapsed_seconds': 0, 'enforcement': 'cooperative'})

    def prepare(self, spec, *, corpus_root):
        if corpus_root is None:
            raise ValueError('explicit private corpus_root required')
        self.corpus = Corpus(Path(corpus_root) / spec.data['corpus_id'], spec.data['corpus_sha256'])
        self.plan = SplitPlan(**_jsonable(spec.data['split']))
        validate_split(self.corpus.samples, self.plan)
        if len(self.corpus.samples) > 512:
            raise ValueError('market corpus exceeds bounded 512-sample reference budget')
        rights = []
        for key in self.corpus.manifest['features']:
            feature = self.corpus.feature(key)
            if not feature['context']['provenance']['synthetic']:
                raise ValueError('Factory market backend is synthetic_only')
            for record in [*feature['records'], *feature['quality_records']]:
                provenance = record['provenance']
                if not provenance['synthetic']:
                    raise ValueError('Factory market backend is synthetic_only')
                rights.append({k: provenance[k] for k in ('rights', 'license', 'synthetic')})
        self.rights = sorted({json.dumps(r, sort_keys=True) for r in rights})
        definition, data = self.describe(spec)
        def inputs(ids):
            # IDs are lookup capabilities for the fixed feature store. Only
            # model.inputs' numeric InferenceInput reaches model.predict.
            return InferenceInput(definition.feature_schema, tuple(ids), tuple(
                tuple(numeric(self.corpus.feature(k), self.corpus.manifest['spec'], spec.data['listing_id'])[0]) for k in ids))
        return PreparedTask(data, definition, TrainingBatch(inputs(self.plan.train), tuple(() for _ in self.plan.train)),
                            EvaluationBatch(inputs(self.plan.validation), tuple(() for _ in self.plan.validation)))

    def build(self, definition):
        return {'baseline': None, 'definition': _jsonable(definition.to_dict())}

    def fit(self, model, data, configuration, control, *, resume):
        self.stage = 'fit'
        control.check()
        if model['baseline'] is None:
            model['baseline'] = fit_baseline(self.corpus, self.plan, listing_id=self.spec.data['listing_id'],
                config=BaselineConfig(**_jsonable(self.spec.model['baseline'])))
        control.check()
        return {'algorithm': 'closed_form', 'fit_rows': len(model['baseline']._fit['train_sample_ids']),
                'model_sha256': model['baseline'].hash, 'optimizer': None, 'scheduler': None, 'ema': None,
                'cursor': 'fit_complete', 'seed': configuration['determinism_policy']['seed']}

    def predict(self, model, inputs):
        self.stage = 'evaluate'
        require_inference_input(inputs)
        if inputs.sample_ids != tuple(self.plan.validation):
            raise ValueError('Factory prediction accepts validation partition only')
        self.model = model['baseline']
        return self.model.forecast([self.corpus.feature(k) for k in inputs.sample_ids])

    def report(self, model, partition):
        features = [self.corpus.feature(k) for k in getattr(self.plan, partition)]
        predictions = model.forecast(features)
        zero = fit_baseline(self.corpus, self.plan, listing_id=self.spec.data['listing_id'],
            config=BaselineConfig(kind='zero', quantiles=model.config.quantiles))
        return make_report(self.corpus, self.plan, predictions, baseline=zero.forecast(features),
            population='synthetic', baseline_name='zero-change:' + zero.hash,
            run_id=model.hash, partition=partition, pretraining_cutoff='not_applicable_non_pretrained',
            bootstrap={'repetitions': 20, 'seed': self.spec.training['determinism_policy']['seed']})

    def evaluate(self, predictions, data):
        report = self.report(self.model, 'validation')
        if digest([asdict(r) for r in predictions]) != report['identities']['prediction_sha256']:
            raise ValueError('evaluation predictions differ')
        validation_objective(report, metric='pinball', predeclared_metric='pinball')
        return report

    def save(self, model):
        return {'version': 1, 'definition': model['definition'],
                'baseline': None if model['baseline'] is None else model['baseline'].state(),
                'optimizer': None, 'scheduler': None, 'ema': None,
                'rng': 'not_applicable_closed_form', 'cursor': 'unfitted' if model['baseline'] is None else 'fit_complete'}

    def load(self, definition, state, *, resume):
        if state['version'] != 1 or state['definition'] != _jsonable(definition.to_dict()):
            raise ValueError('market checkpoint definition differs')
        result = self.build(definition)
        raw = state['baseline']
        if raw is not None:
            result['baseline'] = BaselineModel.from_state(raw,
                corpus_sha256=raw['corpus_sha256'], split_sha256=raw['split_sha256'],
                config=BaselineConfig(**_jsonable(definition.model['baseline'])))
        return result

    def export(self, artifacts, model, evaluation, control):
        from cognitive_runtime.training.model_factory.registry import register_artifacts
        self.stage = 'export'
        control.check()
        cleared = self.rights and all(json.loads(r) == {'rights': 'project-authored-fixture', 'license': 'MIT', 'synthetic': True} for r in self.rights)
        self.exported = register_artifacts(artifacts.directory,
            ('checkpoints/last.json', 'checkpoints/last.json.json', 'metrics/validation.json'),
            rights_cleared=bool(cleared), max_bytes=1024 * 1024)

    def finish(self, artifacts, model, *, status, budget):
        execution = json.loads((self.run_directory / 'execution.json').read_text())
        baseline = None if model is None else model['baseline']
        config = _jsonable(self.spec.to_dict())
        # Environment lock is actual installed versions, not a claim of a staged
        # dependency lockfile. Volatile run IDs/timing live outside identity.
        environment = self.execution
        identities = {'code': self.execution['source_sha256'], 'config': digest(config),
            'environment': digest(environment), 'data': self.spec.data['corpus_sha256'],
            'split': digest(_jsonable(self.spec.data['split'])),
            'rights': None if self.rights is None else digest(self.rights),
            'schema': digest({'manifest': 'market-factory-run-v1', 'task': self.identity.to_dict()}),
            'calendar': None if self.corpus is None else digest(self.corpus.manifest['calendar']),
            'label': None if self.corpus is None else digest(self.corpus.manifest['spec']),
            'feature': None if baseline is None else baseline.feature_schema,
            'model': None if baseline is None else baseline.hash,
            'tokenizer': digest({'algorithm': 'builtin_hashing_regex_v1', 'optional_encoder': None}),
            'calibrator': digest({'calibration': 'none'})}
        manifest = {'format': 'market-factory-run-v1', 'task_identity': self.identity.to_dict(),
            'run_id': artifacts.run_id, 'attempt_id': self.attempt_id,
            'status': status, 'failure': None if status in ('completed', 'running') else 'task_' + status,
            'stage': self.stage, 'partial': status != 'completed', 'readiness': 'synthetic_only',
            'promotable': status == 'completed', 'edge_promotable': False,
            'git_commit': execution['git_commit'], 'dirty_tree': execution['dirty_tree'],
            'resolved_config': config, 'effective_config': config,
            'environment_lock': environment, 'seed': config['training']['determinism_policy']['seed'],
            'hardware': {'machine': platform.machine(), 'processor': platform.processor(), 'device': 'cpu'},
            'hashes': identities, 'reproducibility_sha256': digest(identities),
            'tolerances': {'same_environment': 'exact', 'cross_environment': 'not_guaranteed'},
            'budget': {**budget, 'provider_usd': 0,
                'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                'memory_enforcement': 'measurement_only'},
            'state_components': {'optimizer': None, 'scheduler': None, 'ema': None,
                'rng': 'not_applicable_closed_form', 'cursor': 'unfitted' if baseline is None else 'fit_complete'},
            'artifacts': self.exported}
        atomic_write_json(artifacts.directory / 'run_manifest.json', manifest)
