"""Opt-in generated interval coverage, MIT project-authored seed 291."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import random

from cognitive_runtime.adapters.finance.datasets.calendars import utc
from cognitive_runtime.adapters.finance.datasets.splits import Sample, walk_forward, validate_split


def test_generated_overlap_corpus_and_order_invariance():
    rng = random.Random(291)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    samples = []
    for i in range(300):
        at = start + timedelta(hours=i)
        samples.append(Sample(sample_id=str(i), snapshot_sha256='a'*64, decision_at=utc(at),
            context_start_at=utc(at - timedelta(hours=rng.randint(1,12))),
            label_end_at=utc(at + timedelta(hours=rng.randint(1,8))),
            future_end_at=utc(at + timedelta(hours=rng.randint(1,16))),
            issuer_ids=(str(i % 31),), event_families=(str(i % 41),)))
    args = dict(validation_at=utc(start + timedelta(hours=100)), test_at=utc(start + timedelta(hours=200)))
    for strict in (False,True):
        for group in ('temporal','issuer','event_family'):
            plan = walk_forward(samples, **args, strict_context=strict, group=group)
            validate_split(samples, plan)
            rng.shuffle(samples)
            assert walk_forward(samples, **args, strict_context=strict, group=group) == plan
            assert len(plan.train)+len(plan.validation)+len(plan.test)+len(plan.purged) == 300
