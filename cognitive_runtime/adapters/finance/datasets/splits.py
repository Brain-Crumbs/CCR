"""Walk-forward partitions with actual-interval purging and group contamination checks."""
from dataclasses import dataclass
from datetime import timedelta

from ..schemas.base import Strict, timestamp, sha256


@dataclass(frozen=True, kw_only=True)
class Sample(Strict):
    sample_id: str
    snapshot_sha256: str
    decision_at: str
    context_start_at: str
    label_end_at: str
    future_end_at: str | None = None
    issuer_ids: tuple[str, ...] = ()
    event_families: tuple[str, ...] = ()
    dedup_keys: tuple[str, ...] = ()
    source_keys: tuple[str, ...] = ()

    def validate(self):
        sha256(self.snapshot_sha256)
        if not timestamp(self.context_start_at) <= timestamp(self.decision_at) < timestamp(self.label_end_at):
            raise ValueError('invalid context/label interval')
        if self.future_end_at and timestamp(self.future_end_at) <= timestamp(self.decision_at):
            raise ValueError('future interval must follow decision')

    @property
    def end(self):
        return max(timestamp(self.label_end_at), timestamp(self.future_end_at or self.label_end_at))


def _overlap(a, b, strict_context):
    start_a = timestamp(a.context_start_at if strict_context else a.decision_at)
    start_b = timestamp(b.context_start_at if strict_context else b.decision_at)
    # Closed intervals deliberately purge an observation shared at a boundary.
    return start_a <= b.end and start_b <= a.end


def _conflicts(a, b, group):
    fields = ['dedup_keys', 'source_keys']
    if group == 'issuer':
        fields.append('issuer_ids')
    elif group == 'event_family':
        fields.append('event_families')
    return [name for name in fields if set(getattr(a, name)) & set(getattr(b, name))]


@dataclass(frozen=True, kw_only=True)
class SplitPlan(Strict):
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    purged: tuple[tuple[str, ...], ...]
    strict_context: bool
    group: str
    validation_at: str
    test_at: str
    embargo_seconds: int
    samples_sha256: str

    def validate(self):
        if self.group not in ('temporal', 'issuer', 'event_family') or self.embargo_seconds < 0:
            raise ValueError('invalid split policy')
        if timestamp(self.validation_at) >= timestamp(self.test_at):
            raise ValueError('validation must precede test')
        sha256(self.samples_sha256)
        ids = self.train + self.validation + self.test + tuple(x[0] for x in self.purged)
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate split membership')


def _sample_hash(samples):
    from cognitive_runtime.data.ledger import digest
    return digest([s.to_dict() for s in sorted(samples, key=lambda s: s.sample_id)])


def validate_split(samples, plan):
    samples = tuple(samples)
    if _sample_hash(samples) != plan.samples_sha256:
        raise ValueError('split/sample hash mismatch')
    by_id = {s.sample_id: s for s in samples}
    population = plan.train + plan.validation + plan.test + tuple(x[0] for x in plan.purged)
    if len(by_id) != len(samples) or set(population) != set(by_id):
        raise ValueError('split must account for entire population')
    groups = [plan.train, plan.validation, plan.test]
    for index, ids in enumerate(groups):
        for key in ids:
            s = by_id[key]
            at = timestamp(s.decision_at)
            expected = 0 if at < timestamp(plan.validation_at) else 1 if at < timestamp(plan.test_at) else 2
            if expected != index:
                raise ValueError('walk-forward membership mismatch')
            if plan.group == 'issuer' and not s.issuer_ids or plan.group == 'event_family' and not s.event_families:
                raise ValueError('group identity missing')
        for held in groups[index + 1:]:
            for a in (by_id[key] for key in ids):
                for b in (by_id[key] for key in held):
                    if _overlap(a, b, plan.strict_context):
                        raise ValueError('label/future/context interval leakage')
                    if a.end + timedelta(seconds=plan.embargo_seconds) >= timestamp(b.context_start_at if plan.strict_context else b.decision_at):
                        raise ValueError('boundary embargo violation')
                    if _conflicts(a, b, plan.group):
                        raise ValueError('source/dedup/group contamination')
    return plan


def walk_forward(samples, *, validation_at, test_at, strict_context=False, group='temporal', embargo_seconds=0):
    samples = tuple(samples)
    if group not in ('temporal', 'issuer', 'event_family') or type(embargo_seconds) is not int or embargo_seconds < 0:
        raise ValueError('invalid split policy')
    if timestamp(validation_at) >= timestamp(test_at):
        raise ValueError('unordered split boundaries')
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError('duplicate sample ID')
    groups = [[], [], []]
    for sample in sorted(samples, key=lambda s: (timestamp(s.decision_at), s.sample_id)):
        if group == 'issuer' and not sample.issuer_ids or group == 'event_family' and not sample.event_families:
            raise ValueError('group identity missing')
        at = timestamp(sample.decision_at)
        index = 0 if at < timestamp(validation_at) else 1 if at < timestamp(test_at) else 2
        groups[index].append(sample)
    purged = []
    # Sealed test has precedence; purge validation against test, then training
    # against the original held-out populations (including purged validation).
    originals = [tuple(g) for g in groups]
    for i in (1, 0):
        held = [s for g in originals[i + 1:] for s in g]
        keep = []
        for sample in groups[i]:
            reasons = set()
            for other in held:
                if _overlap(sample, other, strict_context):
                    reasons.add('overlapping_interval')
                start = timestamp(other.context_start_at if strict_context else other.decision_at)
                if sample.end + timedelta(seconds=embargo_seconds) >= start:
                    reasons.add('actual_interval_embargo')
                reasons.update(_conflicts(sample, other, group))
            if reasons:
                purged.append((sample.sample_id, ','.join(sorted(reasons))))
            else:
                keep.append(sample)
        groups[i] = keep
    plan = SplitPlan(train=tuple(s.sample_id for s in groups[0]), validation=tuple(s.sample_id for s in groups[1]),
        test=tuple(s.sample_id for s in groups[2]), purged=tuple(sorted(purged)), strict_context=strict_context,
        group=group, validation_at=validation_at, test_at=test_at, embargo_seconds=embargo_seconds,
        samples_sha256=_sample_hash(samples))
    return validate_split(samples, plan)
