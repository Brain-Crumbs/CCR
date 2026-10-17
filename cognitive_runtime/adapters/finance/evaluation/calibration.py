"""Validation-only empirical residual quantile correction; no coverage guarantee."""
from dataclasses import dataclass, replace, asdict
from decimal import Decimal, localcontext
import numpy as np
from cognitive_runtime.data.ledger import digest
from ..schemas.records import Quantile
from .population import bind_population


@dataclass(frozen=True)
class QuantileCalibrator:
    corpus_sha256: str
    split_sha256: str
    label_spec_sha256: str
    model_id: str
    levels: tuple
    corrections: tuple
    validation_ids: tuple
    validation_labels: tuple

    @property
    def hash(self):
        return digest(asdict(self))

    def apply(self, corpus, plan, predictions, *, partition='test'):
        if (corpus.manifest_sha256, plan.hash, digest(corpus.manifest['spec'])) != (
                self.corpus_sha256, self.split_sha256, self.label_spec_sha256):
            raise ValueError('calibrator corpus/split/label mismatch')
        rows = bind_population(corpus, plan, predictions, partition=partition)
        output = []
        for row in rows:
            p, f = row.result, row.result.forecast
            if f is not None and f.model_id != self.model_id:
                raise ValueError('calibrator model mismatch')
            if p.abstention_reason:
                output.append(p)
                continue
            if f.calibrator_id is not None or tuple(q.level for q in f.quantiles) != self.levels:
                raise ValueError('already calibrated or quantile grid differs')
            with localcontext() as ctx:
                ctx.prec = 260
                values = sorted(Decimal(q.value) + Decimal(c) for q, c in zip(f.quantiles, self.corrections))
                lower = Decimal(f.support_lower)
                values = [max(lower, v) if f.support_upper is None else max(lower, min(Decimal(f.support_upper), v)) for v in values]
            calibrated = replace(f, quantiles=tuple(Quantile(level=q, value=format(v, 'f')) for q, v in zip(self.levels, values)),
                calibrator_id=self.hash, quality=tuple(q for q in f.quality if q != 'uncalibrated_train_residual_quantiles') +
                ('validation_residual_correction_no_coverage_guarantee',))
            output.append(replace(p, forecast=calibrated))
        return tuple(output)


def fit_calibrator(corpus, plan, predictions, *, partition='validation'):
    if partition != 'validation':
        raise ValueError('calibration fitting is validation-only')
    rows = bind_population(corpus, plan, predictions, partition=partition)
    usable = [r for r in rows if r.outcome.label is not None and not r.result.abstention_reason]
    if not usable:
        raise ValueError('no usable validation labels')
    forecasts = [r.result.forecast for r in usable]
    levels = tuple(q.level for q in forecasts[0].quantiles)
    model_id = forecasts[0].model_id
    if any(f.calibrator_id is not None or f.model_id != model_id or tuple(q.level for q in f.quantiles) != levels for f in forecasts):
        raise ValueError('calibration requires one uncalibrated model/quantile grid')
    corrections = tuple(format(float(np.quantile([float(r.outcome.label.value)-float(r.result.forecast.quantiles[i].value)
        for r in usable], float(q))), '.17f') for i, q in enumerate(levels))
    return QuantileCalibrator(corpus.manifest_sha256, plan.hash, digest(corpus.manifest['spec']), model_id,
        levels, corrections, tuple(r.result.sample_id for r in rows), tuple(r.label_sha256 for r in rows))
