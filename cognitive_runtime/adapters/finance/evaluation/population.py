"""Bind immutable forecast results to exact label artifacts and held-out IDs."""
from dataclasses import dataclass
from ..datasets.splits import validate_split
from ..models.distribution import ForecastResult, observations, anchor
from ..schemas.base import timestamp
from cognitive_runtime.data.ledger import digest


@dataclass(frozen=True)
class EvaluationRow:
    result: ForecastResult
    outcome: object
    label_sha256: str
    decision_at: str
    source: str
    instrument: str
    synthetic: bool
    stale: bool
    missing: bool
    issuer_ids: tuple
    event_families: tuple


def bind_population(corpus, plan, predictions, *, partition='test'):
    validate_split(corpus.samples, plan)
    if partition not in ('validation', 'test'):
        raise ValueError('evaluation requires validation or test')
    predictions = tuple(predictions)
    keys = getattr(plan, partition)
    if len(predictions) != len(keys) or {p.sample_id for p in predictions} != set(keys):
        raise ValueError('predictions must retain exact held-out population including abstentions')
    by_id = {p.sample_id: p for p in predictions}
    samples = {s.sample_id: s for s in corpus.samples}
    spec = corpus.manifest['spec']
    rows = []
    for key in keys:
        p = by_id[key]
        if type(p) is not ForecastResult:
            raise TypeError('exact ForecastResult required')
        feature = corpus.feature(key)
        context, _ = observations(feature)  # rejects future/outcome fields at inference boundary
        sample = samples[key]
        if feature['sample_id'] != key or sample.snapshot_sha256 != context.hash or timestamp(sample.decision_at) != timestamp(context.decision_at):
            raise ValueError('split sample/context identity mismatch')
        outcome = corpus._outcome(key)  # labels only, never future training targets
        expected = digest({'sample': key, 'snapshot': context.hash, 'listing': outcome.listing_id,
                           'label_spec': digest(spec), 'split': plan.hash})
        if (p.forecast_id, p.snapshot_sha256, p.label_spec_sha256, p.split_sha256) != (expected, context.hash, digest(spec), plan.hash):
            raise ValueError('forecast snapshot/label/split identity mismatch')
        if (outcome.snapshot_sha256, outcome.spec_sha256, outcome.decision_at) != (context.hash, digest(spec), context.decision_at):
            raise ValueError('label snapshot/spec identity mismatch')
        if outcome.hash != corpus.manifest['labels'][key]:
            raise ValueError('label artifact hash mismatch')
        if outcome.label is not None:
            label = outcome.label
            if (label.spec.hash, label.listing_id, label.decision_at, label.nominal_endpoint_at) != (
                    digest(spec), outcome.listing_id, outcome.decision_at, outcome.nominal_endpoint_at):
                raise ValueError('inner label semantics mismatch')
            if max(timestamp(label.actual_endpoint_at), timestamp(label.target_available_at)) > sample.end:
                raise ValueError('label exceeds split outcome interval')
            if label.provenance.synthetic != context.provenance.synthetic:
                raise ValueError('label/context synthetic provenance differs')
        f = p.forecast
        _, record, price, at, reason = anchor(feature, spec, outcome.listing_id)
        if outcome.label and (record is None or (outcome.label.anchor_record_id, outcome.label.anchor_price, outcome.label.anchor_at) != (record.record_id, price, at)):
            raise ValueError('label anchor mismatch')
        if f is None:
            if not p.abstention_reason:
                raise ValueError('missing forecast must abstain explicitly')
        else:
            if (f.snapshot_id, f.listing_id, f.decision_at, f.target, f.unit, f.horizon.to_dict(), f.feed, f.mark, f.currency) != (
                    context.record_id, outcome.listing_id, context.decision_at, spec['target'], spec['unit'], spec['horizon'], spec['feed'], spec['mark'], spec['currency']):
                raise ValueError('forecast financial semantics mismatch')
            if f.abstention_reason != p.abstention_reason:
                raise ValueError('inconsistent abstention')
            if record is None or (f.anchor_record_id, f.anchor_price, f.anchor_at) != (record.record_id, price, at):
                raise ValueError('forecast anchor mismatch')
            if reason and not p.abstention_reason:
                raise ValueError('unusable context must abstain')
        rows.append(EvaluationRow(p, outcome, outcome.hash, context.decision_at, spec['feed'], outcome.listing_id,
            context.provenance.synthetic, bool(context.stale_record_ids) or reason == 'stale_anchor',
            bool(context.missing_record_ids) or record is None, sample.issuer_ids, sample.event_families))
    return tuple(rows)
