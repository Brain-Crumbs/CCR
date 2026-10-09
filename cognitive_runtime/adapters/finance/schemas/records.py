"""Market observations, context and forecast records (inference-safe imports)."""
from dataclasses import dataclass
from decimal import Decimal
from .base import Record, Strict, decimal, timestamp, sha256
from .semantics import Target, Unit, Mark, Horizon, validate_target, support


def interval(start, end):
    if timestamp(start) >= timestamp(end):
        raise ValueError('interval must be nonempty [start,end)')


def currency(value):
    if len(value) != 3 or not value.isascii() or not value.isupper() or not value.isalpha():
        raise ValueError('currency must be three uppercase ASCII letters')


def unique(values):
    if len(values) != len(set(values)):
        raise ValueError('duplicate identities')


@dataclass(frozen=True, kw_only=True)
class Instrument(Record):
    issuer_id: str
    security_id: str
    listing_id: str
    share_class: str
    venue: str
    currency: str
    kind: str
    successor_security_id: str | None
    external_ids: tuple[str, ...]
    effective_from: str
    effective_to: str | None
    universe_id: str

    def validate(self):
        super().validate()
        currency(self.currency)
        if self.kind not in ('equity', 'binary_contract'):
            raise ValueError('unsupported instrument kind')
        timestamp(self.effective_from)
        if self.effective_to:
            interval(self.effective_from, self.effective_to)
        unique(self.external_ids)
        if any(':' not in item for item in self.external_ids):
            raise ValueError('external IDs require namespaces')


@dataclass(frozen=True, kw_only=True)
class InstrumentAlias(Record):
    alias: str
    namespace: str
    security_id: str
    listing_id: str
    effective_from: str
    effective_to: str | None

    def validate(self):
        super().validate()
        timestamp(self.effective_from)
        if self.effective_to:
            interval(self.effective_from, self.effective_to)


@dataclass(frozen=True, kw_only=True)
class PriceBar(Record):
    listing_id: str
    venue: str
    feed: str
    currency: str
    mark: Mark
    start_at: str
    end_at: str
    session_id: str
    calendar_version: str
    adjustment: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    volume_unit: str
    complete: bool
    stale: bool
    missing: bool
    filled: bool

    def validate(self):
        super().validate()
        currency(self.currency)
        interval(self.start_at, self.end_at)
        o, h, lo, c, v = map(decimal, (self.open, self.high, self.low, self.close, self.volume))
        if min(o, h, lo, c, v) < 0 or not lo <= min(o, c) <= max(o, c) <= h:
            raise ValueError('invalid OHLC/volume')
        if self.adjustment not in ('raw', 'split_adjusted') or self.volume_unit not in ('shares', 'contracts'):
            raise ValueError('unknown price/volume basis')
        if self.complete and (self.available_at is None or timestamp(self.available_at) < timestamp(self.end_at)):
            raise ValueError('complete bar cannot be available before completion')
        if self.complete and (self.missing or self.filled):
            raise ValueError('missing or filled bar cannot be complete')


@dataclass(frozen=True, kw_only=True)
class Quote(Record):
    listing_id: str
    venue: str
    feed: str
    currency: str
    bid: str | None
    ask: str | None
    stale: bool
    missing: bool
    crossed: bool

    def validate(self):
        super().validate()
        currency(self.currency)
        values = [decimal(x) for x in (self.bid, self.ask) if x is not None]
        if any(x < 0 for x in values):
            raise ValueError('negative quote')
        if self.missing != (self.bid is None or self.ask is None):
            raise ValueError('quote missing mask differs')
        actual = self.bid is not None and self.ask is not None and decimal(self.bid) > decimal(self.ask)
        if actual != self.crossed:
            raise ValueError('crossed quote must be explicitly flagged')


@dataclass(frozen=True, kw_only=True)
class CorporateAction(Record):
    security_id: str
    kind: str
    announced_at: str
    effective_at: str
    ex_at: str | None
    record_at: str | None
    payment_at: str | None
    split_ratio: str | None
    cash: str | None
    currency: str | None
    successor_security_id: str | None
    evidence_record_ids: tuple[str, ...]

    def validate(self):
        super().validate()
        for value in (self.announced_at, self.effective_at, self.ex_at, self.record_at, self.payment_at):
            if value is not None:
                timestamp(value)
        if self.kind not in ('split', 'dividend', 'delisting', 'successor', 'halt'):
            raise ValueError('unsupported corporate action')
        if self.kind == 'split':
            if self.split_ratio is None or decimal(self.split_ratio) <= 0 or self.cash is not None:
                raise ValueError('split requires positive ratio and no cash')
        elif self.split_ratio is not None:
            raise ValueError('ratio applies only to splits')
        if self.kind == 'dividend':
            if self.cash is None or decimal(self.cash) < 0 or self.currency is None:
                raise ValueError('dividend requires cash and currency')
        elif self.cash is not None:
            raise ValueError('cash applies only to dividends')
        if self.currency is not None:
            currency(self.currency)
        if self.kind == 'successor' and self.successor_security_id is None:
            raise ValueError('successor identity required')
        if not self.evidence_record_ids:
            raise ValueError('corporate action requires evidence')


@dataclass(frozen=True, kw_only=True)
class EntityEvidence(Strict):
    issuer_id: str
    instrument_ids: tuple[str, ...]
    confidence: str
    ambiguous: bool
    supporting_span: str

    def validate(self):
        if not 0 <= decimal(self.confidence) <= 1:
            raise ValueError('entity confidence outside [0,1]')


@dataclass(frozen=True, kw_only=True)
class NewsEvent(Record):
    original_published_at: str
    first_seen_at: str
    title: str
    text: str | None
    links: tuple[str, ...]
    entities: tuple[EntityEvidence, ...]
    dedup_cluster_id: str
    deleted: bool
    extraction_model_id: str | None

    def validate(self):
        super().validate()
        timestamp(self.original_published_at)
        if timestamp(self.first_seen_at) != timestamp(self.observed_at):
            raise ValueError('first_seen must equal immutable observation clock')
        if not self.provenance.untrusted_source_text:
            raise ValueError('news text must be marked untrusted')
        if self.deleted and self.text is not None:
            raise ValueError('deletion must not carry text')


@dataclass(frozen=True, kw_only=True)
class PredictionContractState(Record):
    market_id: str
    token_id: str
    listing_id: str
    side: str
    question: str
    question_version: str
    payout_scale: str
    currency: str
    quote_basis: Mark
    lifecycle: str
    scheduled_expiry_at: str
    probability_price: str | None

    def validate(self):
        super().validate()
        currency(self.currency)
        timestamp(self.scheduled_expiry_at)
        if self.side not in ('YES', 'NO') or self.lifecycle not in ('open', 'halted', 'closed'):
            raise ValueError('unsupported contract side/lifecycle')
        if decimal(self.payout_scale) <= 0:
            raise ValueError('payout scale must be positive')
        if self.probability_price is not None and not 0 <= decimal(self.probability_price) <= 1:
            raise ValueError('probability-price outside [0,1]')


@dataclass(frozen=True, kw_only=True)
class ContextWindow(Strict):
    clock: str
    count: int
    calendar_id: str | None
    calendar_version: str | None

    def validate(self):
        if self.clock not in ('utc_elapsed_seconds', 'exchange_sessions') or self.count <= 0:
            raise ValueError('explicit positive context clock required')
        if self.clock == 'exchange_sessions':
            if not self.calendar_id or not self.calendar_version:
                raise ValueError('session context needs a versioned calendar')
        elif self.calendar_id is not None or self.calendar_version is not None:
            raise ValueError('elapsed context has no exchange calendar')


@dataclass(frozen=True, kw_only=True)
class ContextSnapshot(Record):
    decision_at: str
    as_of_mode: str
    selected_record_ids: tuple[str, ...]
    selected_record_hashes: tuple[str, ...]
    price_context: ContextWindow
    text_context: ContextWindow
    missing_record_ids: tuple[str, ...]
    stale_record_ids: tuple[str, ...]
    feature_schema: str
    preprocessing_sha256: str
    manifest_sha256: str
    caveats: tuple[str, ...]

    def validate(self):
        super().validate()
        timestamp(self.decision_at)
        if self.as_of_mode not in ('strict_replay', 'historical_source_as_of'):
            raise ValueError('explicit as-of mode required')
        if self.as_of_mode == 'historical_source_as_of' and not self.caveats:
            raise ValueError('historical reconstruction requires caveats')
        unique(self.selected_record_ids)
        if len(self.selected_record_ids) != len(self.selected_record_hashes):
            raise ValueError('selected record IDs/hashes must align')
        for value in (*self.selected_record_hashes, self.preprocessing_sha256, self.manifest_sha256):
            sha256(value)
        if not set(self.stale_record_ids + self.missing_record_ids) <= set(self.selected_record_ids):
            raise ValueError('quality masks must refer to selected records')


@dataclass(frozen=True, kw_only=True)
class Quantile(Strict):
    level: str
    value: str

    def validate(self):
        if not 0 < decimal(self.level) < 1:
            raise ValueError('quantile level outside (0,1)')
        decimal(self.value)


@dataclass(frozen=True, kw_only=True)
class DirectionProbability(Strict):
    positive_change_probability: str
    method: str
    calibrator_id: str

    def validate(self):
        if self.method != 'separately_calibrated_head':
            raise ValueError('quantiles do not determine direction probability')
        if not 0 <= decimal(self.positive_change_probability) <= 1:
            raise ValueError('direction probability outside [0,1]')


@dataclass(frozen=True, kw_only=True)
class ForecastDistribution(Record):
    snapshot_id: str
    listing_id: str
    decision_at: str
    target: Target
    unit: Unit
    horizon: Horizon
    anchor_price: str
    anchor_at: str
    anchor_record_id: str
    currency: str
    feed: str
    mark: Mark
    support_lower: str
    support_upper: str | None
    quantiles: tuple[Quantile, ...]
    direction: DirectionProbability | None
    model_id: str
    calibrator_id: str | None
    abstention_reason: str | None
    quality: tuple[str, ...]

    def validate(self):
        super().validate()
        currency(self.currency)
        validate_target(self.target, self.unit, self.horizon)
        if timestamp(self.anchor_at) > timestamp(self.decision_at):
            raise ValueError('anchor must not follow decision')
        lower, upper = support(self.target, self.anchor_price)
        if decimal(self.support_lower) != lower or (None if self.support_upper is None else decimal(self.support_upper)) != upper:
            raise ValueError('support must match target and anchor')
        levels = [decimal(q.level) for q in self.quantiles]
        values = [decimal(q.value) for q in self.quantiles]
        if levels != sorted(set(levels)) or values != sorted(values):
            raise ValueError('quantiles must have increasing levels and monotone values')
        if any(v < lower or (upper is not None and v > upper) for v in values):
            raise ValueError('quantile outside anchor-dependent support')
        if self.abstention_reason is None:
            if not self.quantiles:
                raise ValueError('forecast requires quantiles or explicit abstention')
        elif self.quantiles or self.direction is not None:
            raise ValueError('abstained forecast cannot carry outputs')
        if not self.quality:
            raise ValueError('explicit quality required')


@dataclass(frozen=True, kw_only=True)
class ProviderCapabilityManifest(Record):
    provider: str
    status: str
    authentication: str
    free_entitlement: str
    usage_rights: str
    access: str
    endpoint_allowlist: tuple[str, ...]
    requests_per_minute: int | None
    latency_seconds: int | None
    timestamp_quality: str
    revision_quality: str
    history_quality: str
    identity_quality: str
    evidence_at: str
    evidence_urls: tuple[str, ...]
    provider_budget_usd: str

    def validate(self):
        super().validate()
        if self.status not in ('docs_only', 'needs_user_key', 'rights_blocked', 'access_denied', 'synthetic_only', 'validated_live'):
            raise ValueError('unsupported capability status')
        timestamp(self.evidence_at)
        if decimal(self.provider_budget_usd) != 0:
            raise ValueError('provider budget must remain zero')
        for value in (self.requests_per_minute, self.latency_seconds):
            if value is not None and value < 0:
                raise ValueError('negative limit/latency')
        if self.status == 'validated_live' and not self.evidence_urls:
            raise ValueError('live validation requires dated evidence')


@dataclass(frozen=True, kw_only=True)
class RunManifest(Record):
    code_sha256: str
    config_sha256: str
    environment_sha256: str
    data_sha256: str
    model_sha256: str
    calendar_sha256: str
    split_sha256: str
    rights_sha256: str
    status: str
    failure: str | None
    provider_budget_usd: str
    runtime_seconds: str

    def validate(self):
        super().validate()
        for name in ('code', 'config', 'environment', 'data', 'model', 'calendar', 'split', 'rights'):
            sha256(getattr(self, name + '_sha256'))
        if self.status not in ('completed', 'failed', 'cancelled', 'budget_exceeded'):
            raise ValueError('unsupported run state')
        if (self.status == 'completed') != (self.failure is None):
            raise ValueError('failure reason/state mismatch')
        if decimal(self.provider_budget_usd) != 0 or decimal(self.runtime_seconds) < 0:
            raise ValueError('invalid budget/runtime')


@dataclass(frozen=True, kw_only=True)
class Metric(Strict):
    name: str
    value: str
    unit: str

    def validate(self):
        decimal(self.value)


@dataclass(frozen=True, kw_only=True)
class EvaluationReport(Record):
    run_id: str
    run_sha256: str
    dataset_kind: str
    forecast_ids: tuple[str, ...]
    metrics: tuple[Metric, ...]
    total_count: int
    evaluated_count: int
    censored_count: int
    abstained_count: int
    stale_count: int
    caveats: tuple[str, ...]

    def validate(self):
        super().validate()
        sha256(self.run_sha256)
        unique(self.forecast_ids)
        if self.dataset_kind not in ('synthetic', 'play_money', 'financial'):
            raise ValueError('datasets must be reported separately')
        counts = (self.total_count, self.evaluated_count, self.censored_count, self.abstained_count, self.stale_count)
        if min(counts) < 0 or self.evaluated_count + self.censored_count + self.abstained_count != self.total_count:
            raise ValueError('coverage counts do not partition population')
        if self.stale_count > self.total_count or len(self.forecast_ids) != self.evaluated_count:
            raise ValueError('forecast identities/coverage differ')
        unique(tuple(m.name for m in self.metrics))
