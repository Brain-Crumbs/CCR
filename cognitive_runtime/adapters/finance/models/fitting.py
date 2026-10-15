"""Training-only baseline fit; only validated train labels are opened here."""
import numpy as np

from .baselines import BaselineConfig, BaselineModel
from ..datasets.splits import validate_split
from ..text.features import TextFeatures
from cognitive_runtime.data.ledger import digest


def fit_baseline(corpus, plan, *, listing_id, config=BaselineConfig(), model_factory=BaselineModel):
    """Fit train outcomes; model_factory is a trusted offline ablation hook.

    The default factory preserves ordinary baseline state and behavior. Custom
    evaluation control models require their own lifecycle/serialization contract.
    """
    validate_split(corpus.samples, plan)
    if not plan.train or len(plan.train) > config.max_train_samples:
        raise ValueError('empty or over-budget training partition')
    spec = corpus.manifest['spec']
    # Validate the full label specification only in this training module.
    from ..labels import LabelSpec
    LabelSpec(**spec)
    text = TextFeatures(config.text_mode, config.text_dimensions).fit(corpus, plan)
    model = model_factory(config, spec, listing_id, plan.hash, corpus.manifest_sha256, text.state)
    rows, targets, ids, censored = [], [], [], []
    for feature, outcome, _ in corpus.training(plan):
        if outcome.listing_id != listing_id or outcome.spec_sha256 != digest(spec):
            raise ValueError('training listing/label spec differs')
        inputs, reasons = model.inputs((feature,))
        if outcome.label is None or reasons[0] is not None:
            censored.append(feature['sample_id'])
            continue
        label = outcome.label
        from .distribution import anchor
        context, record, price, at, _ = anchor(feature, spec, listing_id)
        if (outcome.snapshot_sha256, label.anchor_record_id, label.anchor_price, label.anchor_at) != (context.hash, record.record_id, price, at):
            raise ValueError('training label/context anchor differs')
        rows.append(inputs.features[0]); targets.append(float(label.value)); ids.append(feature['sample_id'])
    if not rows:
        raise ValueError('no usable training labels')
    x, y = np.asarray(rows), np.asarray(targets)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite training target')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), z))
    coefficients = np.zeros(design.shape[1])
    if config.kind in ('numeric', 'text', 'multimodal'):
        penalty = np.eye(design.shape[1]) * config.ridge
        penalty[0, 0] = 0.
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    elif config.kind == 'unconditional':
        coefficients[0] = y.mean()
    # Volatility baseline is zero-centered; unconditional is empirical returns.
    residual = y - design @ coefficients
    floor = max(float(np.sqrt(np.mean(y*y))), 1e-8)
    if config.kind == 'volatility':
        residual = residual / np.maximum(x[:, 1], floor)
    quantiles = np.zeros(len(config.quantiles)) if config.kind == 'zero' else np.quantile(residual, config.quantiles, method='linear')
    model._fit = {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': coefficients.tolist(),
        'residual_quantiles': quantiles.tolist(), 'volatility_floor': floor, 'train_sample_ids': ids,
        'censored_train_sample_ids': censored, 'training_target_sha256': digest({'ids': ids, 'targets': targets}),
        'algorithm': 'closed_form_ridge_unpenalized_intercept_v1', 'calibration': 'none'}
    if not np.isfinite(coefficients).all() or not np.isfinite(quantiles).all():
        raise ValueError('nonfinite fitted model')
    return model
