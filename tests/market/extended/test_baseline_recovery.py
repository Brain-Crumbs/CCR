"""Opt-in tiny CPU generator recovery. Seed 293; no market performance claim."""
import numpy as np
from tests.market.test_distributional_baselines import corpus
from cognitive_runtime.adapters.finance.models.baselines import BaselineConfig
from cognitive_runtime.adapters.finance.models.fitting import fit_baseline


def test_signal_recovery_and_zero_reference_tiny_cpu(corpus):
    c, plan = corpus
    # Validation only; sealed boundary test anchors are not tuning feedback.
    features = [c.feature(k) for k in plan.validation]
    labels = np.array([outcome.label.value for _, outcome in c.evaluation(plan, 'validation')], dtype=float)
    errors = {}
    for kind in ('zero', 'numeric', 'text', 'multimodal'):
        model = fit_baseline(c, plan, listing_id='contract:demo', config=BaselineConfig(kind=kind, ridge=.1))
        medians = np.array([float(r.forecast.quantiles[1].value) for r in model.forecast(features)])
        errors[kind] = np.abs(medians-labels).mean()
    assert all(errors[k] < errors['zero'] for k in ('numeric', 'text', 'multimodal'))
    # Constant-zero reference; this is not a null-signal dataset experiment.
    zero = fit_baseline(c, plan, listing_id='contract:demo')
    assert all(float(q.value) == 0 for r in zero.forecast(features) for q in r.forecast.quantiles)
