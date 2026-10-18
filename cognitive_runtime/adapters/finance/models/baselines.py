"""Bounded NumPy reference baselines implementing the neutral PredictiveModel.

Fit is deliberately separate from inference. Shared Factory campaign/reporting
integration is #295; this backend consumes the existing neutral InferenceInput.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import numpy as np

from brain.cortex.model_contracts import InferenceInput, require_inference_input
from cognitive_runtime.data.ledger import digest
from ..schemas.semantics import Target
from ..text.features import TextFeatures
from .distribution import anchor, emit

KINDS = ('zero', 'unconditional', 'volatility', 'numeric', 'text', 'multimodal')


@dataclass(frozen=True)
class BaselineConfig:
    kind: str = 'zero'
    ridge: float = 1.0
    text_mode: str = 'hashing'
    text_dimensions: int = 16
    quantiles: tuple = (0.1, 0.5, 0.9)
    max_train_samples: int = 512

    def __post_init__(self):
        object.__setattr__(self, 'quantiles', tuple(self.quantiles))
        if self.kind not in KINDS or type(self.ridge) not in (float, int) or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('unknown baseline or invalid positive ridge')
        if self.text_mode not in ('hashing', 'tfidf') or type(self.text_dimensions) is not int or not 1 <= self.text_dimensions <= 64:
            raise ValueError('bounded hashing/tfidf dimensions required')
        if type(self.max_train_samples) is not int or not 1 <= self.max_train_samples <= 512:
            raise ValueError('train sample budget must be 1..512')
        if not self.quantiles or len(self.quantiles) > 19 or any(type(q) not in (float, int) or not math.isfinite(q) or not 0 < q < 1 for q in self.quantiles) or list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError('increasing quantile levels required')


def numeric(feature, spec, listing):
    from .distribution import observations
    from ..schemas.records import PriceBar, PredictionContractState
    from ..schemas.base import timestamp
    context, record, price, at, reason = anchor(feature, spec, listing)
    if record is None:
        return [0., 0., 0., 1.], reason
    # Only selected context prices, same feed/mark/currency and representation.
    _, records = observations(feature)
    if type(record) is PriceBar:
        rows = [(r.end_at, float(r.close)) for r in records if type(r) is PriceBar and
                (r.listing_id, r.feed, r.mark, r.currency, r.venue, r.adjustment, r.calendar_version) ==
                (record.listing_id, record.feed, record.mark, record.currency, record.venue, record.adjustment, record.calendar_version)
                and not (r.stale or r.missing or r.filled) and float(r.close) > 0]
    else:
        rows = [(r.event_at, float(r.probability_price)) for r in records if type(r) is PredictionContractState and
                (r.listing_id, r.source_id, r.quote_basis, r.currency, r.market_id, r.token_id, r.side, r.question_version) ==
                (record.listing_id, record.source_id, record.quote_basis, record.currency, record.market_id, record.token_id, record.side, record.question_version)
                and r.lifecycle == 'open' and r.probability_price is not None]
    prices = np.asarray([p for _, p in sorted(rows)])
    changes = np.diff(prices)
    if spec['target'] == Target.EQUITY_RETURN and len(changes):
        changes = changes / prices[:-1]
    momentum = float(changes[-1]) if len(changes) else 0.
    volatility = float(np.sqrt(np.mean(changes**2))) if len(changes) else 0.
    age = (timestamp(context.decision_at) - timestamp(at)).total_seconds() / max(1, spec['anchor_max_age_seconds'])
    return [momentum, volatility, age, float(len(changes) == 0)], reason


class BaselineModel:
    def __init__(self, config, spec, listing_id, split_sha256, corpus_sha256, text_state):
        self.config = config
        self.spec = json.loads(json.dumps(spec))
        self.listing_id, self.split_sha256, self.corpus_sha256 = listing_id, split_sha256, corpus_sha256
        self.text = TextFeatures.from_state(text_state)
        if (text_state['corpus_sha256'], text_state['split_sha256']) != (corpus_sha256, split_sha256):
            raise ValueError('text preprocessing corpus/split differs')
        self._fit = None

    @property
    def feature_schema(self):
        return digest({'version': 1, 'config': asdict(self.config), 'spec': self.spec,
                       'listing': self.listing_id, 'text': self.text.signature})

    def inputs(self, features):
        rows, ids, reasons = [], [], []
        for feature in features:
            nums, reason = numeric(feature, self.spec, self.listing_id)
            text = self.text.transform(feature)
            vector = np.mean(text.values, axis=0).tolist() if text.values else [0.] * self.config.text_dimensions
            if self.config.kind == 'text':
                row = vector + [float(not text.record_ids)]
            elif self.config.kind == 'multimodal':
                row = nums + vector + [float(not text.record_ids)]
            else:
                row = nums
            rows.append(tuple(row)); ids.append(feature['sample_id']); reasons.append(reason)
        return InferenceInput(self.feature_schema, tuple(ids), tuple(rows)), tuple(reasons)

    def predict(self, inputs):
        require_inference_input(inputs)
        if self._fit is None:
            raise ValueError('model has not been fitted')
        if inputs.feature_schema != self.feature_schema:
            raise ValueError('feature/preprocessing schema differs')
        width = len(self._fit['mean'])
        if not inputs.features:
            return ()
        x = np.asarray(inputs.features, dtype=float)
        if x.shape[1] != width:
            raise ValueError('feature width differs')
        z = (x - self._fit['mean']) / self._fit['scale']
        center = np.asarray(self._fit['coefficients'])[0] + z @ np.asarray(self._fit['coefficients'])[1:]
        residuals = np.asarray(self._fit['residual_quantiles'])
        if self.config.kind == 'volatility':
            # Standardized residuals fitted using training context RMS only.
            scales = np.maximum(x[:, 1], self._fit['volatility_floor'])
            out = center[:, None] + scales[:, None] * residuals
        else:
            out = center[:, None] + residuals
        if not np.isfinite(out).all():
            raise ValueError('nonfinite prediction')
        return tuple(tuple(float(v) for v in row) for row in out)

    def forecast(self, features):
        features = tuple(features)
        inputs, reasons = self.inputs(features)
        values = self.predict(inputs)
        model_id = self.hash
        return tuple(emit(f, self.spec, self.listing_id, self.split_sha256, model_id,
                         self.config.quantiles, row, reason=reason)
                     for f, row, reason in zip(features, values, reasons))

    def state(self):
        if self._fit is None:
            raise ValueError('model has not been fitted')
        return json.loads(json.dumps({'version': 1, 'config': asdict(self.config), 'spec': self.spec,
            'listing_id': self.listing_id, 'split_sha256': self.split_sha256, 'corpus_sha256': self.corpus_sha256,
            'text': self.text.state, 'fit': self._fit, 'calibrator': None,
            'head': 'clipped_change_train_residual_quantiles_v1'}, allow_nan=False))

    @property
    def hash(self):
        return digest(self.state())

    def save(self, path):
        from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
        state = self.state()
        atomic_write_json(path, {'sha256': digest(state), 'state': state})

    @classmethod
    def load(cls, path, *, corpus_sha256, split_sha256, config):
        raw = json.loads(Path(path).read_text())
        state = raw['state']
        if digest(state) != raw['sha256']:
            raise ValueError('checkpoint integrity mismatch')
        return cls.from_state(state, corpus_sha256=corpus_sha256, split_sha256=split_sha256, config=config)

    @classmethod
    def from_state(cls, state, *, corpus_sha256, split_sha256, config):
        state = json.loads(json.dumps(state, allow_nan=False))
        if state['version'] != 1:
            raise ValueError('checkpoint integrity/version mismatch')
        if state['corpus_sha256'] != corpus_sha256 or state['split_sha256'] != split_sha256 or BaselineConfig(**state['config']) != config:
            raise ValueError('resume corpus/split/effective configuration differs')
        if state['calibrator'] is not None or state['head'] != 'clipped_change_train_residual_quantiles_v1':
            raise ValueError('unsupported head/calibrator')
        model = cls(config, state['spec'], state['listing_id'], split_sha256, corpus_sha256, state['text'])
        model._fit = state['fit']
        width = config.text_dimensions + 1 if config.kind == 'text' else config.text_dimensions + 5 if config.kind == 'multimodal' else 4
        for key, size in [('mean', width), ('scale', width), ('coefficients', width+1), ('residual_quantiles', len(config.quantiles))]:
            values = np.asarray(model._fit[key], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError('invalid fitted state')
        if not math.isfinite(model._fit['volatility_floor']) or model._fit['volatility_floor'] <= 0:
            raise ValueError('invalid volatility floor')
        if not set(model._fit['train_sample_ids']) <= set(model.text.state['train_sample_ids']):
            raise ValueError('fit/preprocessing training membership differs')
        if min(model._fit['scale']) <= 0 or model._fit['residual_quantiles'] != sorted(model._fit['residual_quantiles']):
            raise ValueError('invalid scaler/quantiles')
        return model
