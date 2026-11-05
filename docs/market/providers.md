# Free-only equities provider (#288)

The default executable path remains synthetic. Alpaca Basic REST integration is
implemented and tested with project-authored HTTP fixtures; **live readiness is
pending**. No account, key, rights or coverage has been validated by those tests.
Related architecture: [#283](https://github.com/Brain-Crumbs/CCR/issues/283);
ledger dependency: [#287](https://github.com/Brain-Crumbs/CCR/issues/287).

## Policy and source evidence

`config/providers/free-only.yaml` is executable, strictly loaded policy. Both the
total budget and per-request cost must be USD 0. The sole operation is GET
`https://data.alpaca.markets/v2/stocks/bars`; the URL is exact, redirects are refused,
and all query fields are allowlisted. Every page pins `feed=iex|sip`,
`adjustment=raw`, symbols, timeframe, start/end, USD, ascending order and `asof=-`.
No trading, accounts/assets discovery, latest/snapshot endpoints, trial, upgrade,
payment setup, fallback feed/provider or SDK retry is available. Unknown config
fields fail closed. Twelve Data is a future optional adapter, never a fallback.

Official documentation checked 2026-11-05:

- [Plans/authentication](https://docs.alpaca.markets/us/docs/about-market-data-api):
  Basic is free; IEX live coverage, history since 2016, 200 historical calls/minute;
  30 websocket symbols if streaming is ever added. This adapter does not stream.
- [Feed restrictions](https://docs.alpaca.markets/us/docs/market-data-faq): SIP
  historical end must be at least 15 minutes old; latest SIP requires a paid plan.
- [Historical bars](https://docs.alpaca.markets/us/reference/stockbars): pagination
  is symbol-major with a combined row limit, and may return short pages.
- [Account agreement](https://files.alpaca.markets/disclosures/library/AcctAppMarginAndCustAgmt.pdf):
  applicable account/use terms need operator review; technical access and a $0
  price do not establish permitted retention, training or redistribution.

## Safe execution contract

Importing the adapter does nothing external. `AlpacaAdapter.fetch` defaults to
`live=False`; it accepts an injected synthetic transport and refuses the built-in
HTTP transport in that mode. Synthetic mode never reads environment keys. The
built-in HTTP transport is the only transport accepted for `live=True`.

An explicitly authorized live caller must supply all of the following:

1. Existing secure environment bindings `APCA_API_KEY_ID` and
   `APCA_API_SECRET_KEY`. The adapter does not read dotenv files, obtain keys,
   sign up, or change access grants. Never put keys in YAML, command arguments,
   notebooks or reports. Credentials have no printable representation; errors
   contain fixed status codes rather than exception bodies/headers/URLs/tokens.
2. A nonempty `entitlement_evidence` reference attesting free Basic access and an
   explicit ledger `RetentionPolicy` for the intended use. Rights are independent
   from authentication. `allow_raw=False` omits raw persistence while retaining
   permitted hashes and normalized data; expired/disallowed rights block requests.
3. A private `Ledger` outside the repository and public artifacts. See
   [event storage](event-storage.md). Raw successful page bytes and their hashes
   are retained only under the supplied policy; provider errors are never stored.
4. A `dict[symbol, ListingBinding]` with stable listing IDs, effective interval
   and evidence. Query intervals must lie inside each binding. Split requests at
   ticker reuse/rename boundaries. Provider rename mapping is disabled; the
   adapter does not infer listing identity from ticker strings.

For example, a caller first creates `FreeOnlyPolicy.load('config/providers/free-only.yaml')`,
then `AlpacaAdapter(policy=policy)`, and supplies the explicit bindings and rights
to `fetch(ledger, symbols=bindings, feed='sip', timeframe='1Min', start=..., end=...,
retention=retention, entitlement_evidence=..., live=True)`. This is an interface
example, not approval to run a live smoke. Do not execute it in default CI.

## Counts, bounds and temporal meaning

Pagination continues until `next_page_token` is null, including empty/short
pages and later symbols. Repeated tokens fail explicitly. The default bound is
100 requests per fetch (configurable downwards), 10 seconds per request and
8 MiB per page. There are no automatic retries or sleeps: 429, timeouts and 5xx
return immediately with existing counts. A later authorized invocation may retry.
The shared SQLite rolling-minute budget reserves requests atomically before
transport across adapter instances/processes for the same OS user, regardless of
feed/key. Default state: `~/.local/state/ccr/alpaca-budget/requests.sqlite`.
Use one collector across machines/accounts sharing provider quota; it is not a
distributed limiter. Do not delete or give collectors distinct quota paths to
reset the budget. Clock rollback/corrupt quota state fails closed.

`FetchResult` is a safe collection-run summary, separate from the model Factory
RunManifest. It binds policy/capability/request/retention SHA256, feed, response hashes and record
IDs, and includes the exact capability manifest. Save this summary in private
operator run evidence. It contains no provider text, credentials or pagination
tokens. `requests` counts attempted transport calls; `pages` counts structurally
valid responses. `received = written + reused + rejected`; per-symbol counts
are accepted rows, including reused rows. Failures retain prior counts, and
`drained=False` means a request/rate/pagination/error bound stopped collection.
Null/invalid rows are rejected rather than filled with zero. Malformed pages
have no trustworthy row denominator and report `invalid_response`.

A drained response is `empty`, `partial_data` (rejected rows or a requested symbol
with no accepted bars), or `ok`. `missing_symbols` and `by_symbol` expose observed
coverage; `coverage='unverified'` always remains because no calendar expected-row
denominator or provider coverage guarantee exists. Transport errors are distinct
from empty history. Do not infer closed markets, no trades or full coverage from
missing bars. IEX and SIP remain distinct in records and summaries.

Minute/hour frames (`1–59Min`, `1–23Hour`) and `1Day` are supported. Other provider
frames fail closed. The provider's query end is inclusive by bar start. Normalized
bars use `[start,end)`; daily aggregation ends at the next New York midnight,
including DST, and is conservatively unavailable until then. `session_id` is the
New York aggregation date and `calendar_version='alpaca-aggregation-v1'` is an
aggregation rule, not an exchange holiday/half-day calendar. PIT dataset callers
must supply their independently verified exchange calendar. Incomplete bars and
SIP bars whose completion is still within the delay window are rejected.

JSON numeric prices are parsed as Decimal, never float; OHLCV uses decimal strings.
Only exactly representable timestamps are accepted: nanosecond zero suffixes may
be normalized to microseconds, nonzero lost precision is rejected. Extra provider
fields remain in the private raw page, not fabricated schema fields. Successful
rows use **collector observation proxy** availability, never historical bar-end
availability: today's download cannot enter yesterday's causal snapshot.
Corrections append revisions; identical OHLCV replays reuse the first observation
and original page hash only when the immutable retention policy also matches.
A changed policy reports a conflict; use explicit ledger rights-revocation
operations to remove previously retained data rather than treating replay as a
rights migration. Volume/trade-count/VWAP interpretation does not imply a
tradable quote. Synthetic and live identities are separate.

## Capability and readiness

`capability()` returns the existing `ProviderCapabilityManifest` schema, keeping
authentication, free entitlement, rights and technical access separate:
`needs_user_key` for missing credentials, `rights_blocked` for missing rights,
`docs_only` for an unvalidated configured live path, `access_denied` after 401/403,
and `synthetic_only` for fixtures. HTTP 422 is `request_rejected` and is never
worked around with another feed. No successful fetch promotes to `validated_live`.

A separately authorized, bounded read-only free-account smoke must still verify
liquid and lower-coverage symbols, intervals, actual delays/timestamps, entitlement,
rights, pagination and coverage. Record gaps, unavailable states and the exact
budget privately, without uploading provider payloads. Neither implementation
completion nor fixture success establishes live access or predictive value.

## Offline verification

`python -m pytest tests/market/test_alpaca_adapter.py -q` runs synthetic HTTP tests,
including a mocked opener for the credentialed code path; it makes no provider
call and uses no real key. Fixtures in `tests/fixtures/market/alpaca/` carry seed,
source/rights, license, SHA256 and edge-case metadata. The default offline core
runner also includes these tests with its network-attempt guard. No provider SDK,
ML dependency, new paid runner or optional model is installed/imported.
