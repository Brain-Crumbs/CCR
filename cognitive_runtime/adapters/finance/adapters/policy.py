"""Fail-closed USD-zero, historical-bars-only policy and shared rate budget."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time
import sqlite3

from cognitive_runtime.adapters.finance.schemas.base import timestamp
from cognitive_runtime.data.raw_store import private_path

BARS_URL = 'https://data.alpaca.markets/v2/stocks/bars'


def utcnow():
    return datetime.now(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def interval_delta(timeframe):
    match = re.fullmatch(r'([1-9][0-9]?)(Min|Hour)', timeframe)
    if match:
        n, unit = int(match[1]), match[2]
        if n <= (59 if unit == 'Min' else 23):
            return timedelta(**{'minutes' if unit == 'Min' else 'hours': n})
    if timeframe == '1Day':
        return None  # calendar day in America/New_York, not 24 elapsed hours
    raise ValueError('unsupported timeframe')


@dataclass(frozen=True)
class FreeOnlyPolicy:
    provider: str = 'alpaca_basic'
    budget_usd: str = '0'
    cost_per_request_usd: str = '0'
    endpoint: str = BARS_URL
    method: str = 'GET'
    feeds: tuple = ('iex', 'sip')
    adjustment: str = 'raw'
    requests_per_minute: int = 200
    max_requests: int = 100
    timeout_seconds: int = 10
    max_response_bytes: int = 8 * 1024 * 1024
    allow_trial: bool = False
    allow_upgrade: bool = False
    allow_fallback: bool = False

    def __post_init__(self):
        if (self.provider != 'alpaca_basic' or self.endpoint != BARS_URL or self.method != 'GET'
                or self.adjustment != 'raw' or self.feeds != ('iex', 'sip')):
            raise ValueError('provider endpoint/feed policy rejected')
        for value in (self.budget_usd, self.cost_per_request_usd):
            if type(value) is not str or not Decimal(value).is_finite() or Decimal(value) != 0:
                raise ValueError('provider expenditure must be USD zero')
        if any(type(v) is not bool or v for v in (self.allow_trial, self.allow_upgrade, self.allow_fallback)):
            raise ValueError('trial, upgrade and fallback forbidden')
        for v, maximum in ((self.requests_per_minute, 200), (self.max_requests, 10000),
                           (self.timeout_seconds, 30), (self.max_response_bytes, 8 * 1024 * 1024)):
            if type(v) is not int or not 1 <= v <= maximum:
                raise ValueError('invalid request/resource budget')

    @property
    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        if 'feeds' in data:
            data['feeds'] = tuple(data['feeds'])
        return cls(**data)  # unknown fields, including credential fields, fail closed

    def check(self, url, params, *, now):
        if url != BARS_URL or type(params) is not dict:
            raise ValueError('endpoint rejected')
        required = {'symbols', 'timeframe', 'start', 'end', 'feed', 'adjustment', 'limit', 'sort', 'asof', 'currency'}
        if not required <= params.keys() or params.keys() - required - {'page_token'}:
            raise ValueError('query allowlist rejected')
        if (params['feed'] not in self.feeds or params['adjustment'] != 'raw'
                or params['sort'] != 'asc' or params['asof'] != '-' or params['currency'] != 'USD'):
            raise ValueError('query policy rejected')
        symbols = params['symbols'].split(',')
        if len(symbols) > 100 or len(set(symbols)) != len(symbols) or any(not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,14}', s) for s in symbols):
            raise ValueError('invalid symbols')
        interval_delta(params['timeframe'])
        start, end = timestamp(params['start']), timestamp(params['end'])
        if start >= end or end > now or start < datetime(2016, 1, 1, tzinfo=timezone.utc):
            raise ValueError('invalid history interval')
        if params['feed'] == 'sip' and end > now - timedelta(minutes=15):
            raise ValueError('SIP end must be at least 15 minutes old')
        if type(params['limit']) is not int or not 1 <= params['limit'] <= 10000:
            raise ValueError('invalid combined page limit')
        if 'page_token' in params and (type(params['page_token']) is not str or not 1 <= len(params['page_token']) <= 4096):
            raise ValueError('invalid pagination token')


class GlobalRateBudget:
    """Cross-instance/process rolling minute; fail fast, no retries or sleeping.

    One private shared file per OS user, across all credentials and feeds.
    Multiple machines must use a single collector; this is not distributed quota.
    Wall-clock rollback fails closed. Corrupt state fails closed.
    """
    def __init__(self, root=None, *, clock=time.time):
        self.root = Path(root) if root is not None else Path.home() / '.local/state/ccr/alpaca-budget'
        self.clock = clock

    def acquire(self, limit=200):
        root = private_path(self.root)
        path = root / 'requests.sqlite'
        with sqlite3.connect(path, timeout=5) as db:
            os.chmod(path, 0o600)
            db.execute('CREATE TABLE IF NOT EXISTS requests (at REAL NOT NULL)')
            db.execute('BEGIN IMMEDIATE')
            now = self.clock()
            if db.execute('SELECT 1 FROM requests WHERE at > ? OR at < 0', (now,)).fetchone():
                raise ValueError('clock rollback or invalid rate state')
            db.execute('DELETE FROM requests WHERE at <= ?', (now - 60,))
            if db.execute('SELECT count(*) FROM requests').fetchone()[0] >= min(limit, 200):
                return False
            db.execute('INSERT INTO requests VALUES (?)', (now,))
            return True
