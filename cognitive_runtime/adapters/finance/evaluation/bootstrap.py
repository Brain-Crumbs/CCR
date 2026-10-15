"""Seeded calendar-time block bootstrap, sharing draws across paired scores."""
import math
import random
from .metrics import finite
from ..schemas.base import timestamp


def time_block_interval(observations, *, block_seconds=86400, repetitions=200, seed=0, confidence=.95):
    """Observations are (UTC decision_at, score); equal timestamps stay together.

    Nonoverlapping UTC-epoch blocks are resampled as clusters. Sparse/one-block
    evidence is explicitly unavailable, never an invented zero-width interval.
    """
    if type(block_seconds) is not int or block_seconds < 1 or type(repetitions) is not int or not 20 <= repetitions <= 2000:
        raise ValueError('invalid bounded bootstrap policy')
    if type(seed) is not int or not 0 < confidence < 1:
        raise ValueError('invalid seed/confidence')
    observations = tuple(observations)
    if len(observations) > 10000:
        raise ValueError('bootstrap sample budget exceeded')
    blocks = {}
    for at, value in observations:
        key = math.floor(timestamp(at).timestamp() / block_seconds)
        blocks.setdefault(key, []).append(finite(value))
    groups = [blocks[k] for k in sorted(blocks)]
    common = dict(method='nonoverlapping_utc_time_block_percentile', block_seconds=block_seconds,
                  repetitions=repetitions, seed=seed, confidence=confidence, n=len(observations), blocks=len(groups))
    if len(groups) < 2:
        return {**common, 'status': 'insufficient_time_blocks', 'lower': None, 'upper': None}
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        drawn = [v for _ in groups for v in rng.choice(groups)]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    tail = (1-confidence)/2
    return {**common, 'status': 'estimated', 'lower': means[int(tail*(repetitions-1))],
            'upper': means[math.ceil((1-tail)*(repetitions-1))]}
