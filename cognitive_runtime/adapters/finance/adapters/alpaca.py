"""Alpaca Basic REST bars. Offline fixtures are the only validated capability.

No network work occurs until explicit fetch(live=True) with existing keys,
free-entitlement evidence and a permitted retention policy. No trading SDK.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
import hashlib
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener
from zoneinfo import ZoneInfo

from cognitive_runtime.adapters.finance.schemas.base import AvailabilityBasis, Provenance, timestamp
from cognitive_runtime.adapters.finance.schemas.records import PriceBar, ProviderCapabilityManifest
from cognitive_runtime.adapters.finance.schemas.semantics import Mark
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, digest
from .base import Response, FetchResult
from .policy import BARS_URL, FreeOnlyPolicy, GlobalRateBudget, interval_delta, iso, utcnow

VERSION = 'alpaca-basic-v1'
EVIDENCE_AT = '2026-11-05T00:00:00Z'
EVIDENCE = (
    'https://docs.alpaca.markets/us/docs/about-market-data-api',
    'https://docs.alpaca.markets/us/docs/market-data-faq',
    'https://docs.alpaca.markets/us/reference/stockbars',
    'https://files.alpaca.markets/disclosures/library/AcctAppMarginAndCustAgmt.pdf',
)


@dataclass(frozen=True)
class Credentials:
    key: str = field(repr=False)
    secret: str = field(repr=False)

    @classmethod
    def from_environment(cls):
        key, secret = os.environ.get('APCA_API_KEY_ID'), os.environ.get('APCA_API_SECRET_KEY')
        if not key or not secret or any(ord(c) < 33 or ord(c) > 126 for c in key + secret):
            return None
        return cls(key, secret)

    def headers(self):
        return {'APCA-API-KEY-ID': self.key, 'APCA-API-SECRET-KEY': self.secret}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPTransport:
    """No redirects, SDK retries or alternate URLs. TLS/proxy defaults retained."""
    def __init__(self, policy, *, now=utcnow):
        self.policy, self.now = policy, now

    def get(self, url, *, params, headers, timeout):
        self.policy.check(url, params, now=self.now())
        if set(headers) != {'APCA-API-KEY-ID', 'APCA-API-SECRET-KEY'}:
            raise ValueError('authentication headers rejected')
        request = Request(url + '?' + urlencode(params), headers=headers, method='GET')
        try:
            with build_opener(NoRedirect()).open(request, timeout=min(timeout, self.policy.timeout_seconds)) as response:
                body = response.read(self.policy.max_response_bytes + 1)
                if len(body) > self.policy.max_response_bytes:
                    raise ValueError('response limit')
                return Response(response.status, body)
        except HTTPError as error:
            code = error.code
            error.close()
            return Response(code, b'')  # never persist provider error text


@dataclass(frozen=True)
class ListingBinding:
    """Operator-verified PIT identity, scoped to the complete query interval.

    Split queries at ticker reuse/identity boundaries; never infer an ID from a
    ticker. asof='-' disables the provider's implicit present-day rename mapping.
    """
    listing_id: str
    effective_from: str
    effective_to: str
    evidence: str

    def covers(self, start, end):
        return (bool(self.listing_id and self.evidence) and timestamp(self.effective_from) <= start
                and end < timestamp(self.effective_to))


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate response key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite response number')
    return json.loads(body, parse_float=Decimal, object_pairs_hook=unique, parse_constant=invalid)


def _decimal(value):
    if type(value) not in (int, str, Decimal):
        raise ValueError('missing or invalid numeric field')
    d = Decimal(value)
    if not d.is_finite() or len(d.as_tuple().digits) > 128 or abs(d.as_tuple().exponent) > 128:
        raise ValueError('numeric field out of bounds')
    return format(d, 'f')


def _bar_time(value):
    # Alpaca can emit nanosecond RFC3339. Accept only exactly representable values;
    # never silently truncate a timestamp beyond the schema's microsecond limit.
    if type(value) is str and value.endswith('Z') and '.' in value:
        prefix, fraction = value[:-1].split('.')
        if len(fraction) > 6 and set(fraction[6:]) <= {'0'}:
            value = prefix + '.' + fraction[:6] + 'Z'
    return timestamp(value)


class AlpacaAdapter:
    def __init__(self, *, policy=None, transport=None, rate_budget=None, now=utcnow):
        self.policy = policy or FreeOnlyPolicy()
        self.now = now
        self.transport = transport or HTTPTransport(self.policy, now=now)
        self.rate_budget = rate_budget or GlobalRateBudget()

    def capability(self, *, synthetic=False, credentials=False, rights=False, entitlement=False, denied=False):
        status = ('synthetic_only' if synthetic else 'needs_user_key' if not credentials else
                  'rights_blocked' if not rights else 'access_denied' if denied else 'docs_only')
        source = {'adapter': VERSION, 'evidence': EVIDENCE, 'config': self.policy.hash,
                  'synthetic': synthetic, 'credentials': credentials, 'rights': rights,
                  'entitlement': entitlement, 'denied': denied}
        return ProviderCapabilityManifest(
            schema_name='ProviderCapabilityManifest', schema_version='1.0.0',
            record_id='alpaca-capability:' + digest(source), source_id='alpaca_basic',
            source_record_id='capability', revision=1, supersedes_record_id=None,
            event_at=EVIDENCE_AT, available_at=EVIDENCE_AT, observed_at=EVIDENCE_AT, ingested_at=EVIDENCE_AT,
            availability_basis=AvailabilityBasis.EVIDENCED, availability_confidence='1',
            original_timezone='UTC', timestamp_precision='second',
            provenance=Provenance(raw_sha256=digest(source), adapter_version=VERSION,
                                  rights='public-documentation-reference', license='reference-only',
                                  synthetic=synthetic, untrusted_source_text=False),
            provider='alpaca_basic', status=status,
            authentication='synthetic' if synthetic else 'existing_environment' if credentials else 'missing',
            free_entitlement='operator_attested' if entitlement else 'unverified',
            usage_rights='operator_attested' if rights else 'unverified',
            access='denied' if denied else 'offline_fixture' if synthetic else 'live_readiness_pending',
            endpoint_allowlist=('GET ' + BARS_URL,), requests_per_minute=self.policy.requests_per_minute,
            latency_seconds=None, timestamp_quality='bar-start; completion checked; observation proxy',
            revision_quality='corrections appended when observed; no historical revision guarantee',
            history_quality='documented since 2016; coverage unverified',
            identity_quality='explicit PIT listing bindings required; no rename fallback',
            evidence_at=EVIDENCE_AT, evidence_urls=EVIDENCE, provider_budget_usd='0')

    def fetch(self, ledger: Ledger, *, symbols: dict[str, ListingBinding], feed: str,
              timeframe: str, start: str, end: str, retention: RetentionPolicy | None = None,
              entitlement_evidence: str | None = None, limit=1000, live=False):
        """Append accepted bars; return safe counts even after a partial failure.

        Synthetic transports require live=False and never see environment keys.
        Live execution requires the built-in constrained HTTP transport. Success
        never changes readiness to validated_live; a separate opt-in smoke owns it.
        """
        synthetic = not live
        credentials = Credentials.from_environment() if live else None
        rights = (type(retention) is RetentionPolicy and retention.allow_normalized
                  and retention.retain_provenance and bool(retention.rights and retention.evidence)
                  and (retention.expires_at is None or timestamp(retention.expires_at) > self.now()))
        capability = self.capability(synthetic=synthetic, credentials=credentials is not None,
                                     rights=rights, entitlement=bool(entitlement_evidence))
        result = FetchResult(feed, self.policy.hash, capability.hash, readiness=capability.status,
                             by_symbol={s: 0 for s in symbols})
        params = dict(symbols=','.join(sorted(symbols)), feed=feed, timeframe=timeframe,
                      start=start, end=end, adjustment='raw', limit=limit, sort='asc', asof='-', currency='USD')
        try:
            self.policy.check(BARS_URL, params, now=self.now())
            if any(type(b) is not ListingBinding or not b.covers(timestamp(start), timestamp(end)) for b in symbols.values()):
                raise ValueError('missing PIT identity evidence')
        except (ValueError, TypeError, AttributeError):
            result.status = 'policy_rejected'
            return result
        if live and credentials is None:
            result.status = 'needs_user_key'
            return result
        if not rights:
            result.status, result.readiness = 'rights_blocked', 'rights_blocked'
            return result
        if live and not entitlement_evidence:
            result.status = 'entitlement_unverified'
            return result
        if (live and type(self.transport) is not HTTPTransport) or (not live and isinstance(self.transport, HTTPTransport)):
            result.status = 'transport_mode_rejected'
            return result
        # Read the ledger before transport; failure is explicit, never a refetch fallback.
        try:
            prior = {}
            for record in ledger.records():
                if type(record) is PriceBar and record.source_id == ('alpaca_synthetic' if synthetic else 'alpaca_basic'):
                    key = record.source_record_id
                    if key not in prior or prior[key].revision < record.revision:
                        prior[key] = record
        except Exception:
            result.status = 'storage_error'
            return result
        tokens = set()
        while result.requests < self.policy.max_requests:
            if retention.expires_at and timestamp(retention.expires_at) <= self.now():
                result.status, result.readiness = 'rights_blocked', 'rights_blocked'
                return result
            try:
                self.policy.check(BARS_URL, params, now=self.now())
                if not self.rate_budget.acquire(self.policy.requests_per_minute):
                    result.status = 'rate_limited'
                    return result
                result.requests += 1
                response = self.transport.get(BARS_URL, params=dict(params),
                                              headers=credentials.headers() if credentials else {},
                                              timeout=self.policy.timeout_seconds)
            except TimeoutError:
                result.status = 'timeout'
                return result
            except URLError as error:
                result.status = 'timeout' if isinstance(error.reason, TimeoutError) else 'transport_error'
                return result
            except Exception:
                result.status = 'transport_error'
                return result
            result.http_status = response.status
            if response.status != 200:
                result.status = ('access_denied' if response.status in (401, 403) else
                                 'rate_limited' if response.status == 429 else
                                 'request_rejected' if response.status in (400, 422) else 'http_error')
                if response.status in (401, 403):
                    denied = self.capability(synthetic=synthetic, credentials=credentials is not None,
                                             rights=rights, entitlement=bool(entitlement_evidence), denied=True)
                    result.readiness, result.capability_sha256 = denied.status, denied.hash
                return result
            try:
                if type(response.body) is not bytes or len(response.body) > self.policy.max_response_bytes:
                    raise ValueError('response limit')
                if credentials and any(secret.encode() in response.body for secret in (credentials.key, credentials.secret)):
                    raise ValueError('reflected credential')
                data = _json(response.body)
                if type(data) is not dict or type(data.get('bars')) is not dict or 'next_page_token' not in data:
                    raise ValueError('malformed page')
                token = data['next_page_token']
                if token is not None and (type(token) is not str or not 1 <= len(token) <= 4096):
                    raise ValueError('invalid page token')
                if any(type(bars) is not list for bars in data['bars'].values()):
                    raise ValueError('malformed bars')
            except Exception:
                result.status = 'invalid_response'
                return result
            result.pages += 1
            raw_hash = hashlib.sha256(response.body).hexdigest()
            result.response_sha256.append(raw_hash)
            observed = self.now()
            for symbol, bars in data['bars'].items():
                for bar in bars:
                    result.received += 1
                    try:
                        if symbol not in symbols:
                            raise ValueError('unrequested symbol')
                        record = self._normalize(bar, symbol, symbols[symbol], params, observed,
                                                 raw_hash, retention.rights, synthetic, prior)
                    except Exception:
                        result.reject('invalid_or_incomplete_bar')
                        continue
                    old = prior.get(record.source_record_id)
                    # First observation and original page hash survive exact replay.
                    if old is not None and record is old:
                        result.reused += 1
                    else:
                        try:
                            ledger.append(record, response.body, retention)
                        except Exception:
                            result.reject('storage_error')
                            continue
                        prior[record.source_record_id] = record
                        result.written += 1
                    result.record_ids.append(record.record_id)
                    result.by_symbol[symbol] += 1
            if token is None:
                result.drained = True
                result.missing_symbols = [s for s, count in result.by_symbol.items() if count == 0]
                result.status = ('partial_data' if result.rejected else 'empty' if not result.received
                                 else 'partial_data' if result.missing_symbols else 'ok')
                return result
            if token in tokens:
                result.status = 'pagination_loop'
                return result
            tokens.add(token)
            params['page_token'] = token
        result.status = 'request_budget_exhausted'
        return result

    def _normalize(self, bar, symbol, binding, params, observed, raw_hash, rights, synthetic, prior):
        if type(bar) is not dict:
            raise ValueError('invalid bar')
        start = _bar_time(bar['t'])
        if not timestamp(params['start']) <= start <= timestamp(params['end']):
            raise ValueError('bar outside requested interval')
        delta = interval_delta(params['timeframe'])
        local = start.astimezone(ZoneInfo('America/New_York'))
        if delta is None and (local.hour, local.minute, local.second, local.microsecond) != (0, 0, 0, 0):
            raise ValueError('daily bar must start at local midnight')
        end = start + delta if delta else (local + timedelta(days=1)).astimezone(start.tzinfo)
        if end > observed or (params['feed'] == 'sip' and end > observed - timedelta(minutes=15)):
            raise ValueError('incomplete or not yet delayed bar')
        if end > timestamp(binding.effective_to):
            raise ValueError('bar crosses identity boundary')
        values = tuple(_decimal(bar[key]) for key in ('o', 'h', 'l', 'c', 'v'))
        identity = digest([synthetic, binding.listing_id, symbol, params['feed'], params['timeframe'], iso(start), 'raw'])
        old = prior.get(identity)
        if old and values == (old.open, old.high, old.low, old.close, old.volume) and rights == old.provenance.rights:
            return old
        revision = old.revision + 1 if old else 1
        return PriceBar(
            schema_name='PriceBar', schema_version='1.0.0', record_id=f'alpaca:{identity}:r{revision}',
            source_id='alpaca_synthetic' if synthetic else 'alpaca_basic', source_record_id=identity,
            revision=revision, supersedes_record_id=old.record_id if old else None,
            event_at=iso(start), available_at=iso(observed), observed_at=iso(observed), ingested_at=iso(self.now()),
            availability_basis=AvailabilityBasis.OBSERVED, availability_confidence='1',
            original_timezone='UTC', timestamp_precision='microsecond',
            provenance=Provenance(raw_sha256=raw_hash, adapter_version=VERSION, rights=rights,
                                  license='operator-retention-policy', synthetic=synthetic, untrusted_source_text=True),
            listing_id=binding.listing_id, venue='IEX' if params['feed'] == 'iex' else 'US-consolidated',
            feed=params['feed'], currency='USD', mark=Mark.TRADE, start_at=iso(start), end_at=iso(end),
            session_id=local.date().isoformat(), calendar_version='alpaca-aggregation-v1', adjustment='raw',
            open=values[0], high=values[1], low=values[2], close=values[3], volume=values[4], volume_unit='shares',
            complete=True, stale=False, missing=False, filled=False)
