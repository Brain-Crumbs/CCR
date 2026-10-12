"""Outcome-only derivation. Never imported by snapshot or inference code."""
from dataclasses import dataclass
from decimal import Decimal, localcontext

from ..labels import LabelSpec, RealizedLabel
from ..schemas.base import Strict, timestamp
from ..schemas.records import PriceBar, PredictionContractState, CorporateAction, Instrument
from ..schemas.semantics import Target, price_change
from .calendars import elapsed_endpoint
from .universe import verify_bundle


@dataclass(frozen=True, kw_only=True)
class LabelOutcome(Strict):
    """Retain every requested sample, including those without any usable anchor."""
    snapshot_sha256: str
    outcome_snapshot_sha256: str
    spec_sha256: str
    listing_id: str
    decision_at: str
    nominal_endpoint_at: str
    anchor_observed_at: str | None
    anchor_to_endpoint_seconds: int | None
    label: RealizedLabel | None
    censor_reason: str | None

    def validate(self):
        if (self.label is None) == (self.censor_reason is None):
            raise ValueError('exactly one realized label or censor reason required')
        if self.label is not None and self.label.censor_reason is not None:
            raise ValueError('censoring belongs in outcome population')


def _clock(record):
    return record.end_at if type(record) is PriceBar else record.event_at


def _basis(record):
    if type(record) is PriceBar:
        return (record.source_id, record.listing_id, record.venue, record.feed, record.currency,
                record.mark, record.adjustment, record.calendar_version,
                (timestamp(record.end_at) - timestamp(record.start_at)).total_seconds())
    return (record.source_id, record.listing_id, record.market_id, record.token_id, record.side,
            record.question_version, record.payout_scale, record.currency, record.quote_basis)


def _matches(record, listing, spec):
    if spec.target == Target.EQUITY_RETURN:
        return (type(record) is PriceBar and record.listing_id == listing and record.feed == spec.feed
                and record.mark == spec.mark and record.currency == spec.currency and record.adjustment == 'raw')
    return (type(record) is PredictionContractState and record.listing_id == listing
            and record.source_id == spec.feed and record.quote_basis == spec.mark and record.currency == spec.currency)


def _usable(record):
    if type(record) is PriceBar:
        return record.complete and not (record.stale or record.missing or record.filled) and Decimal(record.close) > 0
    return record.lifecycle == 'open' and record.probability_price is not None


def _price(record):
    return record.close if type(record) is PriceBar else record.probability_price


def derive_label(decision, outcomes, *, listing_id, spec, calendar=None):
    """Outcomes must be a separately frozen snapshot with enough history to
    cover anchors, endpoints, actions and lifecycle events. Late revisions are
    resolved by that snapshot's declared as-of policy, not ingestion order.
    Fresh anchors are mandatory; there is no implicit last-known-price task.
    """
    if type(spec) is not LabelSpec or spec.endpoint_policy != 'first_at_or_after':
        raise ValueError('dataset v1 requires first-at/after LabelSpec')
    verify_bundle(decision)
    verify_bundle(outcomes)
    at = decision.context.decision_at
    if spec.target == Target.EQUITY_RETURN:
        if calendar is None:
            raise ValueError('equity calendar required')
        for bundle in (decision, outcomes):
            if bundle.manifest['calendar'] != [calendar.calendar_id, calendar.version, calendar.hash]:
                raise ValueError('snapshot/calendar hash mismatch')
        nominal = calendar.endpoint(at, spec.horizon)
    else:
        nominal = elapsed_endpoint(at, spec.horizon)
    if timestamp(outcomes.context.decision_at) < timestamp(nominal):
        raise ValueError('outcome snapshot precedes horizon')
    from datetime import timedelta
    # Do not silently miss earlier endpoints/actions because the outcome
    # snapshot used a short context window.
    history_start = timestamp(outcomes.context.decision_at) - timedelta(seconds=outcomes.context.price_context.count)
    if history_start > timestamp(at) - timedelta(seconds=spec.anchor_max_age_seconds):
        raise ValueError('outcome context does not cover label interval')
    common = dict(snapshot_sha256=decision.context.hash, outcome_snapshot_sha256=outcomes.context.hash,
                  spec_sha256=spec.hash, listing_id=listing_id, decision_at=at, nominal_endpoint_at=nominal)
    def censored(reason, anchor=None):
        return LabelOutcome(**common, anchor_observed_at=None if anchor is None else anchor.observed_at,
                            anchor_to_endpoint_seconds=None, label=None, censor_reason=reason)
    candidates = [r for r in decision.records + decision.quality_records if _matches(r, listing_id, spec)
                  and timestamp(_clock(r)) <= timestamp(at)]
    if not candidates:
        return censored('missing_anchor')
    latest = max(timestamp(_clock(r)) for r in candidates)
    anchors = [r for r in candidates if timestamp(_clock(r)) == latest]
    if len({_basis(r) for r in anchors}) != 1:
        return censored('ambiguous_anchor_basis')
    anchor = max(anchors, key=lambda r: (r.revision, timestamp(r.available_at), timestamp(r.observed_at), r.record_id, r.hash))
    if not _usable(anchor):
        return censored('invalid_anchor', anchor)
    if (timestamp(at) - timestamp(_clock(anchor))).total_seconds() > spec.anchor_max_age_seconds:
        return censored('stale_anchor', anchor)
    if calendar is not None and anchor.calendar_version != calendar.version:
        return censored('anchor_calendar_mismatch', anchor)
    if calendar is not None and not calendar.is_price_time(_clock(anchor)):
        return censored('anchor_outside_session', anchor)
    records = outcomes.records + outcomes.quality_records
    candidates = [r for r in records if _matches(r, listing_id, spec) and _basis(r) == _basis(anchor)
                  and timestamp(_clock(r)) >= timestamp(nominal)]
    # Select first observation before checking quality. Skipping a known bad
    # endpoint would quietly choose a later, more convenient outcome.
    if not candidates:
        return censored('missing_endpoint', anchor)
    endpoint = min(candidates, key=lambda r: (timestamp(_clock(r)), r.record_id, r.hash))
    if (timestamp(_clock(endpoint)) - timestamp(nominal)).total_seconds() > spec.endpoint_tolerance_seconds:
        return censored('endpoint_outside_tolerance', anchor)
    if not _usable(endpoint):
        return censored('invalid_endpoint', anchor)
    if calendar is not None and not calendar.is_price_time(_clock(endpoint)):
        return censored('endpoint_outside_session', anchor)
    actions = []
    if spec.target == Target.CONTRACT_CHANGE:
        lifecycle = [r for r in records if type(r) is PredictionContractState
                     and (r.market_id, r.token_id) == (anchor.market_id, anchor.token_id)
                     and timestamp(_clock(anchor)) <= timestamp(r.event_at) <= timestamp(_clock(endpoint))]
        if timestamp(anchor.scheduled_expiry_at) <= timestamp(_clock(endpoint)) or any(
                r.lifecycle != 'open' or timestamp(r.scheduled_expiry_at) <= timestamp(_clock(endpoint)) for r in lifecycle):
            return censored('closed_expired_or_halted', anchor)
    else:
        identities = [r for r in decision.records if type(r) is Instrument and r.listing_id == listing_id]
        if len(identities) != 1:
            return censored('missing_or_ambiguous_identity', anchor)
        security = identities[0].security_id
        actions = [r for r in records if type(r) is CorporateAction and r.security_id == security
                   and timestamp(_clock(anchor)) < timestamp(r.effective_at) <= timestamp(_clock(endpoint))]
        if any(r.kind in ('halt', 'delisting', 'successor') for r in actions):
            return censored('halt_delisting_or_successor', anchor)
    splits = sorted((r for r in actions if r.kind == 'split'), key=lambda r: r.record_id)
    if len({r.effective_at for r in splits}) != len(splits):
        return censored('ambiguous_split_actions', anchor)
    with localcontext() as ctx:
        ctx.prec = 260
        multiplier = Decimal(1)
        for action in splits:
            multiplier *= Decimal(action.split_ratio)
        adjusted = Decimal(_price(endpoint)) * multiplier
    value = price_change(spec.target, _price(anchor), format(adjusted, 'f'))
    available = max([timestamp(r.available_at) for r in (endpoint, *splits)] + [timestamp(_clock(endpoint))])
    from .calendars import utc
    # Derived record envelope binds exact input snapshots and spec, not fabricated
    # provider raw content. Minimal provenance remains separate from raw storage.
    from cognitive_runtime.data.ledger import digest
    from ..schemas.base import Provenance
    identity = digest({**common, 'anchor': anchor.hash, 'endpoint': endpoint.hash, 'actions': [r.hash for r in splits]})
    label = RealizedLabel(schema_name='RealizedLabel', schema_version='1.0.0', record_id='label:' + identity,
        source_id='pit-label-v1', source_record_id=identity, revision=1, supersedes_record_id=None,
        event_at=utc(available), available_at=outcomes.context.observed_at, observed_at=outcomes.context.observed_at,
        ingested_at=outcomes.context.ingested_at, availability_basis=outcomes.context.availability_basis,
        availability_confidence='1', original_timezone='UTC', timestamp_precision='microsecond',
        provenance=Provenance(raw_sha256=identity, adapter_version='pit-label-v1', rights='derived-provenance',
            license='operator-policy', synthetic=decision.context.provenance.synthetic and outcomes.context.provenance.synthetic,
            untrusted_source_text=False),
        spec=spec, listing_id=listing_id, decision_at=at, anchor_record_id=anchor.record_id,
        endpoint_record_id=endpoint.record_id, action_record_ids=tuple(r.record_id for r in splits),
        anchor_price=_price(anchor), endpoint_price=_price(endpoint), anchor_at=_clock(anchor), nominal_endpoint_at=nominal,
        actual_endpoint_at=_clock(endpoint), anchor_currency=spec.currency, endpoint_currency=spec.currency,
        anchor_feed=spec.feed, endpoint_feed=spec.feed, anchor_mark=spec.mark, endpoint_mark=spec.mark,
        split_multiplier=format(multiplier, 'f'), value=format(value, 'f'), target_available_at=utc(available), censor_reason=None)
    return LabelOutcome(**common, anchor_observed_at=anchor.observed_at,
        anchor_to_endpoint_seconds=int((timestamp(_clock(endpoint)) - timestamp(_clock(anchor))).total_seconds()),
        label=label, censor_reason=None)
