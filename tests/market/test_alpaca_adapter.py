"""Project-authored MIT synthetic HTTP fixtures, seed 288. Never provider calls."""
from dataclasses import asdict, replace
from datetime import timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognitive_runtime.adapters.finance.adapters.alpaca import (
    AlpacaAdapter, Credentials, HTTPTransport, ListingBinding, NoRedirect, _json,
)
from cognitive_runtime.adapters.finance.adapters.base import Response
from cognitive_runtime.adapters.finance.adapters.policy import (
    BARS_URL, FreeOnlyPolicy, GlobalRateBudget,
)
from cognitive_runtime.adapters.finance.schemas.base import timestamp
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy

NOW = timestamp('2025-01-06T16:00:00Z')
FIXTURE = Path(__file__).parents[1] / 'fixtures/market/alpaca'
RETENTION = RetentionPolicy(rights='project-authored-fixture', evidence='MIT; synthetic seed 288',
                            allow_raw=True, allow_normalized=True, retain_provenance=True)
BINDINGS = {s: ListingBinding('listing:synthetic:' + s, '2025-01-01T00:00:00Z',
                             '2025-02-01T00:00:00Z', 'synthetic PIT fixture') for s in ('DEMOA', 'DEMOB')}


def pages():
    # Parse and re-encode without ever converting the high-precision number to float.
    raw = FIXTURE.joinpath('pages.json').read_text()
    values = json.loads(raw, parse_float=str)
    return [Response(200, json.dumps(v).replace('"10.123456789012345678"', '10.123456789012345678').encode()) for v in values]


class FakeTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def setup(tmp_path):
    with Ledger(tmp_path / 'ledger') as ledger:
        yield ledger, GlobalRateBudget(tmp_path / 'rate', clock=lambda: NOW.timestamp())


def fetch(setup, responses=None, **kwargs):
    ledger, rate = setup
    transport = FakeTransport(pages() if responses is None else responses)
    policy = kwargs.pop('policy', FreeOnlyPolicy())
    adapter = AlpacaAdapter(policy=policy, transport=transport, rate_budget=rate, now=lambda: NOW)
    args = dict(symbols=BINDINGS, feed='iex', timeframe='1Min', start='2025-01-06T14:30:00Z',
                end='2025-01-06T15:00:00Z', retention=RETENTION, limit=2)
    args.update(kwargs)
    return adapter.fetch(ledger, **args), transport, adapter


def test_fixture_hash_and_config():
    manifest = json.loads((FIXTURE / 'provenance.json').read_text())
    assert manifest['synthetic'] and manifest['seed'] == 288 and manifest['license'] == 'MIT'
    assert hashlib.sha256((FIXTURE / 'pages.json').read_bytes()).hexdigest() == manifest['files']['pages.json']['sha256']
    assert FreeOnlyPolicy.load('config/providers/free-only.yaml') == FreeOnlyPolicy()


def test_pagination_decimal_ledger_and_replay(setup):
    result, transport, adapter = fetch(setup)
    assert (result.status, result.readiness, result.requests, result.pages, result.received, result.written) == ('ok', 'synthetic_only', 2, 2, 3, 3)
    assert result.drained and result.by_symbol == {'DEMOA': 2, 'DEMOB': 1}
    assert transport.calls[1][1]['params']['page_token'] == 'synthetic-page-2'
    for url, call in transport.calls:
        assert url == BARS_URL and call['headers'] == {}
        assert call['params']['feed'] == 'iex' and call['params']['adjustment'] == 'raw'
        assert call['params']['asof'] == '-' and call['params']['limit'] == 2
    first = setup[0].get(result.record_ids[0])
    assert first.open == '10.123456789012345678'
    assert first.feed == 'iex' and first.end_at == '2025-01-06T14:31:00Z'
    assert first.available_at == first.observed_at == '2025-01-06T16:00:00Z'
    assert first.availability_basis.value == 'collector_observation_proxy'
    assert hashlib.sha256(setup[0].raw_payload(first.record_id)).hexdigest() == first.provenance.raw_sha256
    replay, _, _ = fetch(setup)
    assert replay.written == 0 and replay.reused == 3 and replay.record_ids == result.record_ids
    assert setup[0].get(first.record_id) == first
    sip, _, _ = fetch(setup, feed='sip')
    assert sip.written == 3 and not set(sip.record_ids) & set(result.record_ids)
    assert adapter.capability(synthetic=True).status == 'synthetic_only'


@pytest.mark.parametrize('change', [
    {'budget_usd':'0.01'}, {'cost_per_request_usd':'1'}, {'budget_usd':'NaN'}, {'budget_usd':0},
    {'provider':'premium'}, {'endpoint':'https://api.alpaca.markets/v2/orders'}, {'method':'POST'},
    {'feeds':('sip',)}, {'adjustment':'all'}, {'requests_per_minute':201}, {'max_requests':True},
    {'allow_trial':True}, {'allow_upgrade':True}, {'allow_fallback':True}, {'allow_trial':1},
])
def test_policy_config_rejects_before_transport(change):
    with pytest.raises((ValueError, TypeError)):
        FreeOnlyPolicy(**change)


@pytest.mark.parametrize('url', [
    'http://data.alpaca.markets/v2/stocks/bars', 'https://api.alpaca.markets/v2/orders',
    'https://paper-api.alpaca.markets/v2/orders', BARS_URL + '/latest', BARS_URL + '?feed=sip',
    BARS_URL + '/', 'https://data.alpaca.markets:443/v2/stocks/bars',
    'https://data.alpaca.markets.evil/v2/stocks/bars', 'https://user@data.alpaca.markets/v2/stocks/bars',
    'https://data.alpaca.markets/v2/stocks/snapshots', 'https://data.alpaca.markets/v2/../v2/stocks/bars',
])
def test_endpoint_allowlist_exhaustive(url):
    with pytest.raises(ValueError):
        FreeOnlyPolicy().check(url, {}, now=NOW)


@pytest.mark.parametrize('change', [
    {'feed':'boats'}, {'feed':'otc'}, {'feed':None}, {'timeframe':'1Week'},
    {'timeframe':'60Min'}, {'timeframe':'24Hour'}, {'limit':0}, {'limit':10001},
    {'end':'2025-01-06T16:00:01Z'}, {'start':'2015-01-01T00:00:00Z'},
    {'symbols':{'DEMOA&feed=sip':BINDINGS['DEMOA']}},
    {'symbols':{'DEMOA':replace(BINDINGS['DEMOA'], evidence='')}},
    {'symbols':{'DEMOA':replace(BINDINGS['DEMOA'], effective_to='2025-01-06T14:40:00Z')}},
])
def test_request_rejected_without_transport(setup, change):
    result, transport, _ = fetch(setup, **change)
    assert result.status == 'policy_rejected' and transport.calls == []


def test_sip_delay_boundary_and_missing_parameters(setup):
    result, _, _ = fetch(setup, feed='sip', end='2025-01-06T15:45:00Z')
    assert result.status == 'ok'
    result, transport, _ = fetch(setup, feed='sip', end='2025-01-06T15:45:00.000001Z')
    assert result.status == 'policy_rejected' and not transport.calls
    _, t, _ = fetch(setup)
    params = t.calls[0][1]['params']
    for key in params:
        with pytest.raises(ValueError):
            FreeOnlyPolicy().check(BARS_URL, {k:v for k,v in params.items() if k != key}, now=NOW)
    for key, value in [('feed',''), ('adjustment','all'), ('asof','2025-01-06'), ('currency','EUR'), ('sort','desc'), ('extra','x')]:
        with pytest.raises(ValueError):
            FreeOnlyPolicy().check(BARS_URL, {**params, key:value}, now=NOW)


@pytest.mark.parametrize('response,status', [
    (Response(429,b'secret'), 'rate_limited'), (Response(401,b'secret'), 'access_denied'),
    (Response(403,b'secret'), 'access_denied'), (Response(422,b'secret'), 'request_rejected'),
    (Response(500,b'secret'), 'http_error'), (Response(302,b'secret'), 'http_error'),
    (TimeoutError('secret'), 'timeout'), (RuntimeError('secret'), 'transport_error'),
    (Response(200,b'{bad secret'), 'invalid_response'),
])
def test_failure_preserves_partial_counts_and_redacts(setup, response, status):
    result, transport, _ = fetch(setup, [pages()[0], response])
    assert result.status == status and result.written == result.received == 2
    assert result.pages == 1 and result.requests == 2 and not result.drained
    assert result.by_symbol == {'DEMOA':2, 'DEMOB':0}
    assert 'secret' not in json.dumps(asdict(result))
    assert len(transport.calls) == 2  # no retries/fallbacks


@pytest.mark.parametrize('body', [b'null', b'{}', b'{"bars":null,"next_page_token":null}',
    b'{"bars":{"DEMOA":null},"next_page_token":null}',
    b'{"bars":{},"next_page_token":3}', b'{"bars":{},"bars":{},"next_page_token":null}'])
def test_malformed_page_is_not_empty_success(setup, body):
    result, _, _ = fetch(setup, [Response(200,body)])
    assert result.status == 'invalid_response' and not result.drained and result.received == 0


@pytest.mark.parametrize('change', [
    {'o':None}, {'c':'NaN'}, {'v':True}, {'h':1}, {'t':'2025-01-06T14:30:00.123456789Z'},
    {'t':'2025-01-07T14:30:00Z'}, {'t':'2025-01-06T14:29:00Z'}, {'o':'1e100000000'},
])
def test_bad_rows_preserve_good_rows_and_denominator(setup, change):
    page = json.loads(pages()[0].body)
    page['bars']['DEMOA'][0].update(change)
    page['next_page_token'] = None
    result, _, _ = fetch(setup, [Response(200,json.dumps(page).encode())])
    assert (result.status,result.received,result.written,result.rejected) == ('partial_data',2,1,1)
    assert result.drained


def test_incomplete_bars_and_empty_coverage(setup):
    page = json.loads(pages()[1].body)
    page['bars']['DEMOB'][0]['t'] = '2025-01-06T16:00:00Z'
    result, _, _ = fetch(setup, [Response(200,json.dumps(page).encode())], end='2025-01-06T16:00:00Z')
    assert result.rejected == 1 and result.written == 0
    result, _, _ = fetch(setup, [Response(200,b'{"bars":{},"next_page_token":null}')])
    assert result.status == 'empty' and result.drained and result.by_symbol == {'DEMOA':0,'DEMOB':0}


def test_short_pages_and_token_loop_and_budget(setup):
    first = Response(200,b'{"bars":{},"next_page_token":"next"}')
    result, t, _ = fetch(setup,[first,pages()[1]])
    assert result.status == 'partial_data' and result.requests == 2 and result.received == 1
    assert result.missing_symbols == ['DEMOA']
    result, _, _ = fetch(setup,[first,first])
    assert result.status == 'pagination_loop' and not result.drained
    result, t, _ = fetch(setup,policy=FreeOnlyPolicy(max_requests=1))
    assert result.status == 'request_budget_exhausted' and result.written == 2 and len(t.calls) == 1


def test_global_rate_cross_instances_and_rollback(tmp_path):
    first = GlobalRateBudget(tmp_path / 'rate',clock=lambda:1000)
    second = GlobalRateBudget(tmp_path / 'rate',clock=lambda:1000)
    for _ in range(100):
        assert first.acquire() and second.acquire()
    assert not second.acquire()
    assert GlobalRateBudget(tmp_path / 'rate',clock=lambda:1060).acquire()
    with pytest.raises(ValueError):
        first.acquire()


def test_rate_limit_preserves_counts(setup):
    result, t, _ = fetch(setup,policy=FreeOnlyPolicy(requests_per_minute=1))
    assert result.status == 'rate_limited' and result.requests == 1 and result.written == 2
    assert len(t.calls) == 1


def test_credentials_rights_and_opt_in_guards(setup, monkeypatch):
    monkeypatch.delenv('APCA_API_KEY_ID',raising=False)
    monkeypatch.delenv('APCA_API_SECRET_KEY',raising=False)
    result, t, adapter = fetch(setup,live=True)
    assert result.status == result.readiness == 'needs_user_key' and not t.calls
    monkeypatch.setenv('APCA_API_KEY_ID','fake-private-key')
    monkeypatch.setenv('APCA_API_SECRET_KEY','fake-private-secret')
    assert 'fake-private' not in repr(Credentials.from_environment())
    result,t,_ = fetch(setup,live=True,retention=None)
    assert result.status == 'rights_blocked' and not t.calls
    result,t,_ = fetch(setup,live=True)
    assert result.status == 'entitlement_unverified' and not t.calls
    result,t,_ = fetch(setup,live=True,entitlement_evidence='synthetic evidence')
    assert result.status == 'transport_mode_rejected' and not t.calls
    result,t,_ = fetch(setup)
    assert result.status == 'ok' and all(call['headers'] == {} for _,call in t.calls)
    assert adapter.capability(credentials=True,rights=True,entitlement=True).status == 'docs_only'
    assert adapter.capability(credentials=True,rights=True,denied=True).status == 'access_denied'
    monkeypatch.setenv('APCA_API_SECRET_KEY','bad\nheader')
    assert Credentials.from_environment() is None


def test_http_transport_with_mock_opener_only(setup, monkeypatch):
    monkeypatch.setenv('APCA_API_KEY_ID','fake-private-key')
    monkeypatch.setenv('APCA_API_SECRET_KEY','fake-private-secret')
    ledger, rate = setup
    adapter = AlpacaAdapter(rate_budget=rate, now=lambda:NOW)
    bodies = iter(pages())
    requests = []
    class Reply:
        status = 200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,n): return next(bodies).body
    def open_request(req,timeout):
        requests.append(req)
        assert req.method == 'GET' and timeout == 10
        assert req.get_header('Apca-api-key-id') == 'fake-private-key'
        return Reply()
    with patch('cognitive_runtime.adapters.finance.adapters.alpaca.build_opener',return_value=SimpleNamespace(open=open_request)):
        result = adapter.fetch(ledger,symbols=BINDINGS,feed='iex',timeframe='1Min',start='2025-01-06T14:30:00Z',
                               end='2025-01-06T15:00:00Z',retention=RETENTION,live=True,entitlement_evidence='synthetic test')
    assert result.written == 3 and result.readiness == 'docs_only'
    assert len(requests) == 2 and 'fake-private' not in repr(result)
    assert NoRedirect().redirect_request(None,None,302,'',{},'https://evil.test') is None


def test_corrections_append_and_storage_failure_counted(setup):
    result,_,_ = fetch(setup)
    old = setup[0].get(result.record_ids[0])
    page = json.loads(pages()[0].body,parse_float=str)
    page['bars']['DEMOA'][0]['c'] = '10.6'
    page['next_page_token'] = None
    result,_,_ = fetch(setup,[Response(200,json.dumps(page).encode())])
    new = setup[0].get(result.record_ids[0])
    assert new.revision == 2 and new.supersedes_record_id == old.record_id
    assert setup[0].get(old.record_id) == old
    with patch.object(setup[0],'append',side_effect=OSError('private secret')):
        result,_,_ = fetch(setup,feed='sip')
    assert result.status == 'partial_data' and result.rejected == 3 and result.written == 0
    assert result.reasons == {'storage_error':3}
    assert 'private' not in repr(result)


def test_daily_dst_completion_uses_local_calendar(setup):
    ledger, rate = setup
    now = timestamp('2025-03-10T12:00:00Z')
    body = b'{"bars":{"DEMOA":[{"t":"2025-03-09T05:00:00Z","o":10,"h":11,"l":9,"c":10,"v":1}]},"next_page_token":null}'
    adapter = AlpacaAdapter(transport=FakeTransport([Response(200,body)]),rate_budget=rate,now=lambda:now)
    binding = ListingBinding('listing:demo','2025-01-01T00:00:00Z','2025-04-01T00:00:00Z','synthetic')
    result=adapter.fetch(ledger,symbols={'DEMOA':binding},feed='iex',timeframe='1Day',
                         start='2025-03-09T05:00:00Z',end='2025-03-10T05:00:00Z',retention=RETENTION)
    record=ledger.get(result.record_ids[0])
    assert timestamp(record.end_at)-timestamp(record.start_at) == timedelta(hours=23)


def test_response_bound_reflection_and_redirect_errors(setup, monkeypatch):
    from urllib.error import HTTPError, URLError
    import io
    monkeypatch.setenv('APCA_API_KEY_ID','fake-private-key')
    monkeypatch.setenv('APCA_API_SECRET_KEY','fake-private-secret')
    ledger, rate = setup
    args=dict(symbols=BINDINGS,feed='iex',timeframe='1Min',start='2025-01-06T14:30:00Z',
              end='2025-01-06T15:00:00Z',retention=RETENTION,live=True,entitlement_evidence='synthetic')
    adapter = AlpacaAdapter(rate_budget=rate,now=lambda:NOW)
    with patch.object(HTTPTransport,'get',return_value=Response(200,b'fake-private-secret')):
        assert adapter.fetch(ledger,**args).status == 'invalid_response'
    assert ledger.records() == ()
    oversized,_,_ = fetch(setup,[Response(200,b' ' * 100)],policy=FreeOnlyPolicy(max_response_bytes=50))
    assert oversized.status == 'invalid_response'
    error = HTTPError(BARS_URL,302,'secret',{},io.BytesIO(b'fake-private-secret'))
    with patch('cognitive_runtime.adapters.finance.adapters.alpaca.build_opener',return_value=SimpleNamespace(open=lambda *a,**k: (_ for _ in ()).throw(error))):
        result = adapter.fetch(ledger,**args)
    assert result.status == 'http_error' and 'fake-private' not in repr(result)
    result,_,_ = fetch(setup,[URLError(TimeoutError('private'))])
    assert result.status == 'timeout'


def test_synthetic_and_live_namespaces_do_not_collide(setup, monkeypatch):
    synthetic,_,_ = fetch(setup)
    monkeypatch.setenv('APCA_API_KEY_ID','fake-private-key')
    monkeypatch.setenv('APCA_API_SECRET_KEY','fake-private-secret')
    adapter=AlpacaAdapter(rate_budget=setup[1],now=lambda:NOW)
    with patch.object(HTTPTransport,'get',side_effect=pages()):
        live=adapter.fetch(setup[0],symbols=BINDINGS,feed='iex',timeframe='1Min',start='2025-01-06T14:30:00Z',
                           end='2025-01-06T15:00:00Z',retention=RETENTION,live=True,entitlement_evidence='synthetic')
    assert live.written == 3 and not set(synthetic.record_ids) & set(live.record_ids)


def test_rate_budget_survives_subprocess(tmp_path):
    import subprocess
    import sys
    root = tmp_path / 'rate'
    assert GlobalRateBudget(root,clock=lambda:1000).acquire(1)
    result = subprocess.run([sys.executable,'-c',
        'import sys; from cognitive_runtime.adapters.finance.adapters.policy import GlobalRateBudget; '
        'assert not GlobalRateBudget(sys.argv[1],clock=lambda:1000).acquire(1)',str(root)],
        timeout=5,capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
