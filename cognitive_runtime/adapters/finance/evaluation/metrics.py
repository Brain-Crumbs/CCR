"""Small, explicit proper scores. Quantiles are not a full CDF."""
import math


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite metric input')
    return value


def pinball(value, prediction, level):
    value, prediction, level = map(finite, (value, prediction, level))
    if not 0 < level < 1:
        raise ValueError('quantile level must be in (0,1)')
    error = value - prediction
    return max(level * error, (level - 1) * error)


def direction_brier(value, probability):
    value, probability = map(finite, (value, probability))
    if not 0 <= probability <= 1:
        raise ValueError('direction probability must be in [0,1]')
    return (probability - float(value > 0)) ** 2


def score(forecast, value):
    if forecast.abstention_reason:
        raise ValueError('cannot score abstention')
    value = finite(value)
    qs = [(finite(q.level), finite(q.value)) for q in forecast.quantiles]
    scores = {'pinball': sum(pinball(value, v, q) for q, v in qs) / len(qs)}
    # Do not invent a mean/median or a direction head from three quantiles.
    median = next((v for q, v in qs if q == .5), None)
    if median is not None:
        scores['median_mae'] = abs(value - median)
    if len(qs) >= 2:
        scores.update(interval_coverage=float(qs[0][1] <= value <= qs[-1][1]),
                      interval_width=qs[-1][1] - qs[0][1])
    if forecast.direction is not None:
        scores['direction_brier'] = direction_brier(value, forecast.direction.positive_change_probability)
    return scores
