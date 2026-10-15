"""Small equal-budget offline baseline suite, with isolated text-vector controls.

No context artifacts are rewritten. Shuffling deliberately destroys association
within each partition and is a diagnostic, not a causal deployed predictor.
"""
from dataclasses import dataclass, asdict, replace
import random
from ..models.baselines import BaselineConfig, BaselineModel
from ..models.fitting import fit_baseline
from ..models.distribution import ForecastResult
from ..datasets.splits import validate_split
from ..schemas.base import timestamp
from cognitive_runtime.data.ledger import digest
from .report import make_report

ARMS = ('zero', 'unconditional', 'volatility', 'numeric', 'text', 'multimodal',
        'shuffled_text', 'lagged_text', 'non_pretrained')
# Latent-only/combined registration deliberately waits for #296.


@dataclass(frozen=True)
class Budget:
    seeds: tuple = (294,)
    max_train_samples: int = 32
    text_dimensions: int = 8
    ridge: float = 1.0
    max_population: int = 128

    def __post_init__(self):
        if not self.seeds or len(self.seeds) > 3 or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int for s in self.seeds):
            raise ValueError('one to three unique integer seeds required')
        if type(self.max_population) is not int or not 1 <= self.max_population <= 512:
            raise ValueError('bounded population required')
        BaselineConfig(max_train_samples=self.max_train_samples, text_dimensions=self.text_dimensions, ridge=self.ridge)


class _TextControl(BaselineModel):
    """Train and evaluate the same partition-local text-vector permutation."""
    def __init__(self, *args, corpus, plan, mode, seed):
        super().__init__(*args)
        self.control = {'mode': mode, 'seed': seed}
        self.vectors = {}
        self.donors = {}
        samples = {s.sample_id: s for s in corpus.samples}
        rng = random.Random(seed)
        for partition in ('train', 'validation', 'test'):
            ids = sorted(getattr(plan, partition), key=lambda k: (timestamp(samples[k].decision_at), k))
            inputs, _ = super().inputs([corpus.feature(k) for k in ids])
            text = {key: row[4:] for key, row in zip(ids, inputs.features)}
            donors = list(ids)
            if mode == 'shuffled_text':
                rng.shuffle(donors)
            elif mode == 'lagged_text':
                # Only a strictly earlier decision can donate text. Equal-time
                # records cannot silently become each other's past.
                donors = [next((other for other in reversed(ids[:i]) if timestamp(samples[other].decision_at) < timestamp(samples[key].decision_at)), None)
                          for i, key in enumerate(ids)]
            else:
                raise ValueError('unknown text control')
            for key, donor in zip(ids, donors):
                self.donors[key] = donor
                self.vectors[key] = text[donor] if donor is not None else (0.,)*self.config.text_dimensions + (1.,)

    def inputs(self, features):
        inputs, reasons = super().inputs(features)
        rows = tuple(row[:4] + tuple(self.vectors[key]) for key, row in zip(inputs.sample_ids, inputs.features))
        return replace(inputs, features=rows), reasons

    def state(self):
        return {**super().state(), 'evaluation_text_control': self.control,
                'donor_sha256': digest(self.donors)}


def reject_leakage(*, input_fields, fitted_ids, plan):
    """Explicit controls reject oracle/target channels and held-out fitting.

    This checks the declared interface, not arbitrary model memorization. Actual
    feature schema/context clock checks also run in bind_population.
    """
    allowed = {'numeric_context', 'text_context', 'missingness'}
    if not set(input_fields) <= allowed:
        raise ValueError('oracle/outcome or unknown input channel rejected')
    if not set(fitted_ids) <= set(plan.train):
        raise ValueError('held-out fitting leakage rejected')


def run_suite(corpus, plan, *, listing_id, population, budget=Budget(), partition='validation', bootstrap=None):
    """Explicit invocation only; does not run a campaign, download, or tune.

    All arms use the same rows, labels, seeds, ridge and maximum fit budget.
    Closed-form baselines have no step budget; actual algorithm and usable fit
    count are recorded. Non-pretrained is the explicit hashing multimodal arm.
    """
    validate_split(corpus.samples, plan)
    if len(corpus.samples) > budget.max_population or len(plan.train) > budget.max_train_samples:
        raise ValueError('equal-budget population limit exceeded')
    if partition not in ('validation', 'test'):
        raise ValueError('held-out partition required')
    reject_leakage(input_fields=('numeric_context', 'text_context', 'missingness'), fitted_ids=plan.train, plan=plan)
    features = tuple(corpus.feature(k) for k in getattr(plan, partition))
    outputs = []
    for seed in budget.seeds:
        baseline = None
        for arm in ARMS:
            kind = 'multimodal' if arm in ('non_pretrained', 'shuffled_text', 'lagged_text') else arm
            config = BaselineConfig(kind=kind, text_mode='hashing', text_dimensions=budget.text_dimensions,
                                    max_train_samples=budget.max_train_samples, ridge=budget.ridge)
            factory = BaselineModel
            if arm in ('shuffled_text', 'lagged_text'):
                factory = lambda *args: _TextControl(*args, corpus=corpus, plan=plan, mode=arm, seed=seed)
            model = None
            failure = None
            try:
                model = fit_baseline(corpus, plan, listing_id=listing_id, config=config, model_factory=factory)
                forecasts = model.forecast(features)
            except ValueError as exc:
                # Keep every requested ID on bounded/data failure. No hidden
                # dropping of unsuccessful arms from comparison reports.
                failure = str(exc)
                forecasts = tuple(ForecastResult(
                    digest({'sample': f['sample_id'], 'snapshot': digest(f['context']), 'listing': listing_id,
                            'label_spec': digest(corpus.manifest['spec']), 'split': plan.hash}),
                    f['sample_id'], digest(f['context']), digest(corpus.manifest['spec']), plan.hash,
                    None, 'model_failed') for f in features)
            if arm == 'zero':
                baseline = forecasts
            report = make_report(corpus, plan, forecasts, baseline=baseline, population=population,
                baseline_name='zero-change:' + baseline[0].forecast.model_id if baseline and baseline[0].forecast else 'zero-change',
                run_id=f'{arm}:seed={seed}', partition=partition, status='failed' if failure else 'completed', failure=failure, pretraining_cutoff='not_applicable_non_pretrained', bootstrap=bootstrap,
                caveats=('All arms use non-pretrained hashing; non_pretrained duplicates multimodal as an explicit contamination reference.',
                         'Shuffled text is a partition-local diagnostic; lagged text uses strictly earlier decisions in the same partition.'))
            report['ablation'] = {'arm': arm, 'seed': seed, 'budget': asdict(budget), 'budget_sha256': digest(asdict(budget)),
                'algorithm': 'closed_form_ridge_unpenalized_intercept_v1', 'train_ids': list(plan.train),
                'usable_train_ids': model.state()['fit']['train_sample_ids'] if model is not None and model._fit is not None else [], 'model_sha256': model.hash if model is not None and model._fit is not None else None,
                'labels_sha256': digest(corpus.manifest['labels']), 'control': getattr(model, 'control', None)}
            outputs.append(report)
    return tuple(outputs)
