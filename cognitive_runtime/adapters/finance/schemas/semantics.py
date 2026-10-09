"""Inference-safe target definitions; no realized values or terminal outcomes."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from .base import Strict, decimal


class Target(str, Enum):
    EQUITY_RETURN = 'equity_simple_return'
    CONTRACT_CHANGE = 'contract_absolute_probability_price_change'


class Unit(str, Enum):
    RETURN = 'return_fraction'
    PRICE_CHANGE = 'probability_price_fraction'


class Clock(str, Enum):
    TRADING_HOURS = 'exchange_trading_hours'
    SESSIONS = 'exchange_sessions'
    ELAPSED = 'utc_elapsed_seconds'


class Mark(str, Enum):
    TRADE = 'trade'
    BID = 'bid'
    ASK = 'ask'
    MID = 'mid'


@dataclass(frozen=True, kw_only=True)
class Horizon(Strict):
    clock: Clock
    count: int
    calendar_id: str | None
    calendar_version: str | None
    origin: str = 'decision_at'

    def validate(self):
        if self.origin != 'decision_at' or self.count <= 0:
            raise ValueError('horizon must originate at decision_at with positive count')
        if self.clock == Clock.ELAPSED:
            if self.calendar_id is not None or self.calendar_version is not None:
                raise ValueError('elapsed horizon must not imply an exchange calendar')
        elif not self.calendar_id or not self.calendar_version:
            raise ValueError('exchange horizon requires versioned calendar')
        elif self.count not in ({1, 4} if self.clock == Clock.TRADING_HOURS else {1, 5}):
            raise ValueError('unsupported initial equity horizon')


def validate_target(target, unit, horizon):
    expected = Unit.RETURN if target == Target.EQUITY_RETURN else Unit.PRICE_CHANGE
    if unit != expected:
        raise ValueError('target and units differ')
    if (target == Target.CONTRACT_CHANGE) != (horizon.clock == Clock.ELAPSED):
        raise ValueError('target and horizon clock differ')


def support(target, anchor):
    anchor = decimal(anchor)
    if target == Target.EQUITY_RETURN:
        if anchor <= 0:
            raise ValueError('equity anchor must be positive')
        return Decimal('-1'), None
    if not 0 <= anchor <= 1:
        raise ValueError('contract anchor outside [0,1]')
    with localcontext() as ctx:
        ctx.prec = 80
        return -anchor, Decimal(1) - anchor


def price_change(target: Target, anchor: str, endpoint: str) -> Decimal:
    """34 significant digits, ROUND_HALF_EVEN, independent of ambient context."""
    if type(target) is not Target:
        raise TypeError('target must be a Target enum')
    support(target, anchor)
    start, end = decimal(anchor), decimal(endpoint)
    if end < 0 or (target == Target.CONTRACT_CHANGE and end > 1):
        raise ValueError('endpoint outside price support')
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.rounding = 'ROUND_HALF_EVEN'
        return end / start - 1 if target == Target.EQUITY_RETURN else end - start
