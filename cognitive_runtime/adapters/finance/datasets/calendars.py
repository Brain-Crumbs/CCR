"""Versioned explicit UTC schedules; no inferred exchange history or downloads."""
from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

from ..schemas.base import Strict, timestamp
from ..schemas.semantics import Clock, Horizon


def utc(value):
    return value.isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True, kw_only=True)
class Session(Strict):
    session_id: str
    open_at: str
    close_at: str
    halts: tuple[tuple[str, ...], ...] = ()

    def validate(self):
        start, end = timestamp(self.open_at), timestamp(self.close_at)
        if start >= end:
            raise ValueError('empty session')
        previous = start
        for halt in self.halts:
            if len(halt) != 2:
                raise ValueError('halt needs start/end')
            a, b = map(timestamp, halt)
            if not previous <= a < b <= end:
                raise ValueError('unordered/overlapping/outside-session halt')
            previous = b

    def segments(self):
        start = timestamp(self.open_at)
        result = []
        for a, b in self.halts:
            a, b = timestamp(a), timestamp(b)
            if start < a:
                result.append((start, a))
            start = b
        if start < timestamp(self.close_at):
            result.append((start, timestamp(self.close_at)))
        return result


@dataclass(frozen=True, kw_only=True)
class Calendar(Strict):
    calendar_id: str
    version: str
    timezone: str
    coverage_start_at: str
    coverage_end_at: str
    sessions: tuple[Session, ...]
    closed_dates: tuple[str, ...]
    evidence: str
    edge_policy: str = 'roll_to_next_open'

    def validate(self):
        ZoneInfo(self.timezone)
        start, end = timestamp(self.coverage_start_at), timestamp(self.coverage_end_at)
        if start >= end or self.edge_policy not in ('roll_to_next_open', 'reject_closed'):
            raise ValueError('invalid coverage/edge policy')
        previous = start
        ids = set()
        for session in self.sessions:
            if not previous <= timestamp(session.open_at) < timestamp(session.close_at) <= end:
                raise ValueError('unordered/overlapping/outside-coverage session')
            if session.session_id in ids:
                raise ValueError('duplicate session')
            ids.add(session.session_id)
            previous = timestamp(session.close_at)
        # Every local date in the coverage is accounted for explicitly. This
        # prevents an accidentally omitted weekday being treated as a holiday.
        from datetime import date
        days = {date.fromisoformat(day) for day in self.closed_dates}
        if len(days) != len(self.closed_dates):
            raise ValueError('duplicate closed date')
        open_days = {timestamp(s.open_at).astimezone(ZoneInfo(self.timezone)).date() for s in self.sessions}
        if days & open_days:
            raise ValueError('date both closed and open')
        day = start.astimezone(ZoneInfo(self.timezone)).date()
        last = (end - timedelta(microseconds=1)).astimezone(ZoneInfo(self.timezone)).date()
        while day <= last:
            if day not in days | open_days:
                raise ValueError('unaccounted calendar date')
            day += timedelta(days=1)

    def is_price_time(self, at):
        """Bar completion at session close is valid; inside a halt is not."""
        at = timestamp(at)
        return any(timestamp(s.open_at) <= at <= timestamp(s.close_at)
                   and not any(timestamp(a) <= at < timestamp(b) for a, b in s.halts)
                   for s in self.sessions)

    def endpoint(self, decision_at, horizon):
        if type(horizon) is not Horizon:
            raise TypeError('typed horizon required')
        decision = timestamp(decision_at)
        if horizon.clock == Clock.ELAPSED:
            raise ValueError('24/7 elapsed horizons use elapsed_endpoint')
        if (horizon.calendar_id, horizon.calendar_version) != (self.calendar_id, self.version):
            raise ValueError('calendar identity/version mismatch')
        if not timestamp(self.coverage_start_at) <= decision < timestamp(self.coverage_end_at):
            raise ValueError('calendar coverage unavailable')
        active = any(a <= decision < b for s in self.sessions for a, b in s.segments())
        if not active and self.edge_policy == 'reject_closed':
            raise ValueError('decision outside trading segment')
        if horizon.clock == Clock.SESSIONS:
            # nth scheduled close strictly after decision; closed decisions roll
            # to the next session, never reinterpret sessions as 24-hour days.
            closes = [timestamp(s.close_at) for s in self.sessions if timestamp(s.close_at) > decision]
            if len(closes) >= horizon.count:
                return utc(closes[horizon.count - 1])
        else:
            remaining = timedelta(hours=horizon.count)
            for s in self.sessions:
                for a, b in s.segments():
                    a = max(a, decision)
                    if a >= b:
                        continue
                    if remaining <= b - a:
                        return utc(a + remaining)
                    remaining -= b - a
        raise ValueError('calendar horizon exceeds coverage')


def elapsed_endpoint(decision_at, horizon):
    if type(horizon) is not Horizon or horizon.clock != Clock.ELAPSED:
        raise ValueError('explicit UTC elapsed horizon required')
    return utc(timestamp(decision_at) + timedelta(seconds=horizon.count))
