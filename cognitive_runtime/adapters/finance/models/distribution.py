"""Inference-only bounded quantiles. Never reads labels or terminal outcomes."""
from dataclasses import dataclass
from decimal import Decimal, localcontext

from ..schemas.base import timestamp
from ..schemas.records import (ContextSnapshot, PriceBar, PredictionContractState,
                              ForecastDistribution, Quantile)
from ..schemas.semantics import Target, Unit, Horizon, Mark, support
from ..inference import OBSERVATIONS, validate_context
from cognitive_runtime.data.ledger import digest


def observations(feature):
    if set(feature) != {'version', 'sample_id', 'context', 'universe', 'records', 'quality_records'} or feature['version'] != 1:
        raise ValueError('expected corpus feature artifact')
    context = ContextSnapshot(**feature['context'])
    types = {t.__name__: t for t in OBSERVATIONS}
    records = tuple(types[r['schema_name']](**r) for r in feature['records'])
    validate_context(context, records)
    return context, records


def anchor(feature, spec, listing_id):
    context, records = observations(feature)
    equity = spec['target'] == Target.EQUITY_RETURN
    candidates = [r for r in records if (
        type(r) is PriceBar and equity and r.feed == spec['feed'] and r.mark == spec['mark'] and r.adjustment == 'raw'
        or type(r) is PredictionContractState and not equity and r.source_id == spec['feed'] and r.quote_basis == spec['mark'])
        and r.listing_id == listing_id and r.currency == spec['currency']]
    # A quarantined newer observation must never resurrect a convenient old one.
    quality = [r for r in feature['quality_records'] if r.get('listing_id') == listing_id
               and r.get('currency') == spec['currency'] and (
                   equity and r.get('schema_name') == 'PriceBar' and r.get('feed') == spec['feed']
                   and r.get('mark') == spec['mark'] and r.get('adjustment') == 'raw'
                   or not equity and r.get('schema_name') == 'PredictionContractState'
                   and r.get('source_id') == spec['feed'] and r.get('quote_basis') == spec['mark'])]
    clock = lambda r: r.end_at if type(r) is PriceBar else r.event_at
    if not candidates:
        return context, None, None, None, 'missing_anchor'
    latest = max(timestamp(clock(r)) for r in candidates)
    selected = [r for r in candidates if timestamp(clock(r)) == latest]
    if len(selected) != 1:
        return context, None, None, None, 'ambiguous_anchor'
    record = selected[0]
    value = record.close if equity else record.probability_price
    if value is None or equity and Decimal(value) <= 0:
        return context, None, None, None, 'invalid_anchor'
    reason = None
    if any(timestamp(r.get('end_at', r['event_at'])) >= latest for r in quality):
        reason = 'quarantined_anchor'
    elif record.record_id in context.stale_record_ids + context.missing_record_ids:
        reason = 'invalid_anchor'
    elif (timestamp(context.decision_at) - latest).total_seconds() > spec['anchor_max_age_seconds']:
        reason = 'stale_anchor'
    elif not equity and (record.lifecycle != 'open' or (timestamp(record.scheduled_expiry_at) - timestamp(context.decision_at)).total_seconds() <= spec['horizon']['count']):
        reason = 'closed_or_expired'
    return context, record, value, clock(record), reason


@dataclass(frozen=True)
class ForecastResult:
    """No-anchor samples retain identity without fabricating a schema anchor."""
    forecast_id: str
    sample_id: str
    snapshot_sha256: str
    label_spec_sha256: str
    split_sha256: str
    forecast: ForecastDistribution | None
    abstention_reason: str | None


def emit(feature, spec, listing_id, split_sha256, model_id, levels, values, *, reason=None):
    context, record, price, at, anchor_reason = anchor(feature, spec, listing_id)
    key = digest({'sample': feature['sample_id'], 'snapshot': context.hash, 'listing': listing_id,
                  'label_spec': digest(spec), 'split': split_sha256})
    reason = anchor_reason or reason
    if record is None:
        return ForecastResult(key, feature['sample_id'], context.hash, digest(spec), split_sha256, None, reason)
    target = Target(spec['target'])
    lower, upper = support(target, price)
    # Clamp in Decimal after float serialization: binary rounding cannot exceed
    # exact support at tiny anchors or anchors 0/1.
    with localcontext() as ctx:
        ctx.prec = 260
        qs = tuple(Quantile(level=str(q), value=format(max(lower, min(upper, Decimal(str(v)))
            if upper is not None else Decimal(str(v))), 'f')) for q, v in zip(levels, values)) if reason is None else ()
    env = context.to_dict()
    env = {k: env[k] for k in ('schema_version', 'revision', 'supersedes_record_id', 'event_at', 'available_at',
        'observed_at', 'ingested_at', 'availability_basis', 'availability_confidence', 'original_timezone',
        'timestamp_precision', 'provenance')}
    env.update(schema_name='ForecastDistribution', record_id='forecast:' + key, source_id='baseline', source_record_id=key,
               revision=1, supersedes_record_id=None)
    # Derived provenance binds inputs/model, not a fictitious provider payload.
    env['provenance'] = {**env['provenance'], 'raw_sha256': digest({'snapshot': context.hash, 'model': model_id}),
                         'adapter_version': 'distributional-baseline-v1'}
    result = ForecastDistribution(**env, snapshot_id=context.record_id, listing_id=listing_id,
        decision_at=context.decision_at, target=target, unit=Unit(spec['unit']), horizon=Horizon(**spec['horizon']),
        anchor_price=price, anchor_at=at, anchor_record_id=record.record_id, currency=spec['currency'], feed=spec['feed'],
        mark=Mark(spec['mark']), support_lower=format(lower, 'f'), support_upper=None if upper is None else format(upper, 'f'),
        quantiles=qs, direction=None, model_id=model_id, calibrator_id=None, abstention_reason=reason,
        quality=('baseline', 'synthetic_generator_recovery_only' if context.provenance.synthetic else 'unvalidated_research',
                 'uncalibrated_train_residual_quantiles'))
    return ForecastResult(key, feature['sample_id'], context.hash, digest(spec), split_sha256, result, reason)
