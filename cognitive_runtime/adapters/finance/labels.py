"""Label/training-only records. Inference modules must never import this module.

No settlement price is an endpoint for the quoted-price-change task.
"""
from dataclasses import dataclass
from .schemas.base import Record, decimal, timestamp
from .schemas.records import currency
from .schemas.semantics import Target, Unit, Horizon, Mark, validate_target, support, price_change


@dataclass(frozen=True, kw_only=True)
class LabelSpec(Record):
    target: Target
    unit: Unit
    horizon: Horizon
    currency: str
    feed: str
    mark: Mark
    anchor_policy: str
    anchor_max_age_seconds: int
    endpoint_policy: str
    endpoint_tolerance_seconds: int
    action_policy: str
    terminal_policy: str

    def validate(self):
        super().validate()
        currency(self.currency)
        validate_target(self.target, self.unit, self.horizon)
        if self.anchor_policy != 'last_eligible_at_or_before_decision':
            raise ValueError('unsupported anchor policy')
        if self.endpoint_policy not in ('first_at_or_after', 'last_at_or_before'):
            raise ValueError('explicit endpoint policy required')
        if min(self.anchor_max_age_seconds, self.endpoint_tolerance_seconds) < 0:
            raise ValueError('negative anchor/endpoint tolerance')
        expected = 'split_neutral_ex_dividend' if self.target == Target.EQUITY_RETURN else 'none'
        if self.action_policy != expected or self.terminal_policy != 'censor_without_preterminal_market_price':
            raise ValueError('unsupported action/terminal policy')


@dataclass(frozen=True, kw_only=True)
class RealizedLabel(Record):
    spec: LabelSpec
    listing_id: str
    decision_at: str
    anchor_record_id: str
    endpoint_record_id: str | None
    action_record_ids: tuple[str, ...]
    anchor_price: str
    endpoint_price: str | None
    anchor_at: str
    nominal_endpoint_at: str
    actual_endpoint_at: str | None
    anchor_currency: str
    endpoint_currency: str
    anchor_feed: str
    endpoint_feed: str
    anchor_mark: Mark
    endpoint_mark: Mark
    split_multiplier: str
    value: str | None
    target_available_at: str | None
    censor_reason: str | None

    def validate(self):
        super().validate()
        for value in (self.anchor_currency, self.endpoint_currency):
            if value != self.spec.currency:
                raise ValueError('mismatched currency')
        if self.anchor_feed != self.spec.feed or self.endpoint_feed != self.spec.feed:
            raise ValueError('mismatched feed')
        if self.anchor_mark != self.spec.mark or self.endpoint_mark != self.spec.mark:
            raise ValueError('mismatched mark')
        decision, anchor, nominal = map(timestamp, (self.decision_at, self.anchor_at, self.nominal_endpoint_at))
        if not 0 <= (decision - anchor).total_seconds() <= self.spec.anchor_max_age_seconds:
            raise ValueError('stale/future anchor')
        if nominal <= decision:
            raise ValueError('target endpoint must follow decision')
        if self.spec.horizon.clock.value == 'utc_elapsed_seconds' and (nominal - decision).total_seconds() != self.spec.horizon.count:
            raise ValueError('elapsed horizon must originate at decision')
        support(self.spec.target, self.anchor_price)
        multiplier = decimal(self.split_multiplier)
        if multiplier <= 0 or (self.spec.target == Target.CONTRACT_CHANGE and multiplier != 1):
            raise ValueError('invalid split multiplier')
        if multiplier != 1 and not self.action_record_ids:
            raise ValueError('split adjustment requires exact action evidence')
        if self.censor_reason is not None:
            if any(x is not None for x in (self.value, self.endpoint_price, self.actual_endpoint_at, self.endpoint_record_id, self.target_available_at)):
                raise ValueError('censored label cannot manufacture a price or return')
            return
        if any(x is None for x in (self.value, self.endpoint_price, self.actual_endpoint_at, self.endpoint_record_id, self.target_available_at)):
            raise ValueError('realized label requires endpoint and availability evidence')
        actual, available = timestamp(self.actual_endpoint_at), timestamp(self.target_available_at)
        if actual <= decision or available < actual:
            raise ValueError('invalid target/availability clocks')
        delta = (actual - nominal).total_seconds()
        if abs(delta) > self.spec.endpoint_tolerance_seconds:
            raise ValueError('endpoint exceeds tolerance')
        if (self.spec.endpoint_policy == 'first_at_or_after' and delta < 0) or (self.spec.endpoint_policy == 'last_at_or_before' and delta > 0):
            raise ValueError('endpoint violates selection policy')
        from decimal import localcontext
        with localcontext() as ctx:
            ctx.prec = 80
            endpoint = decimal(self.endpoint_price) * multiplier
        expected = price_change(self.spec.target, self.anchor_price, format(endpoint, 'f'))
        if decimal(self.value) != expected:
            raise ValueError('target value differs from decimal-safe derivation')


@dataclass(frozen=True, kw_only=True)
class ContractTerminalMetadata(Record):
    market_id: str
    token_id: str
    resolved_at: str
    actual_closed_at: str
    payout: str
    currency: str
    outcome: str

    def validate(self):
        super().validate()
        currency(self.currency)
        if timestamp(self.resolved_at) < timestamp(self.actual_closed_at):
            raise ValueError('resolution precedes closure')
        if not 0 <= decimal(self.payout) <= 1 or self.outcome not in ('YES', 'NO', 'VOID'):
            raise ValueError('invalid terminal outcome')


@dataclass(frozen=True, kw_only=True)
class FutureTarget(Record):
    split: str
    context_snapshot_id: str
    window_start_at: str
    window_end_at: str
    future_record_ids: tuple[str, ...]
    stop_gradient: bool

    def validate(self):
        super().validate()
        if self.split != 'train' or not self.stop_gradient:
            raise ValueError('future targets are train-only and stop-gradient')
        if timestamp(self.window_start_at) >= timestamp(self.window_end_at):
            raise ValueError('invalid future window')
        if not self.future_record_ids:
            raise ValueError('future target requires input record identities')
