# Private event storage and as-of snapshots

Status: **synthetic_only**. This local POSIX implementation uses the #286 schema
contracts. It does not contact providers, download models or execute trades.

## Operator entry points

Configure `Ledger(path, raw_path=..., normalized_path=...)` with private absolute
locations outside every Git working tree. Defaults put both CAS directories under
`path`. Directories use 0700 and files 0600. Each CAS has one owning ledger;
sharing a CAS across ledgers or nesting the two CAS paths is rejected. Do not
place these paths in public artifact uploads, backups or synchronized folders.
The application rejects repository locations, and gitignore adds defense in depth
for conventional private storage names. Existing CI exports only its explicit
runtime reports; it never exports a storage directory. Encryption at rest and
backup erasure remain operator responsibilities.

The following example is executable using only project-authored synthetic data:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import hashlib
from cognitive_runtime.adapters.finance.testing.examples import examples, source_payload
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical
from cognitive_runtime.data.snapshot import build_snapshot, load_snapshot

with TemporaryDirectory(dir='/tmp') as private:
    with Ledger(Path(private) / 'market-private') as ledger:
        record = examples()['NewsEvent']
        raw = canonical(source_payload(record)).encode()
        record = replace(record, provenance=replace(record.provenance,
                         raw_sha256=hashlib.sha256(raw).hexdigest()))
        policy = RetentionPolicy(rights='project-authored-fixture',
            evidence='MIT project-authored synthetic example', allow_raw=True,
            allow_normalized=True, retain_provenance=True)
        ledger.append(record, raw, policy)
        result = build_snapshot(ledger, decision_at='2025-01-06T14:30:00Z',
            price_seconds=3600, text_seconds=7200,
            calendar_id='synthetic', calendar_version='1', calendar_sha256='a'*64,
            preprocessing_sha256='b'*64)
        assert load_snapshot(ledger, result.context.hash) == result
```

Adapters append exact inference observation types with exact original bytes and
matching provenance SHA256. The index stores immutable record/source IDs,
revision links, first-observation time, content hashes and explicit rights
policy. Normalized payloads are separate content-addressed files, allowing their
physical erasure without overwriting logical IDs. A duplicate exact record and
policy is idempotent; any changed content or clock under that ID is rejected.
Labels, terminal outcomes and future training targets are rejected at append.
Do not encode source text in IDs, rights evidence or deletion reasons: these
fields are retained minimal provenance, not a place for prohibited payloads.

## Knowledge time and deterministic replay

`strict_replay` requires both availability and collector observation at/before
the decision. `historical_source_as_of` (the #286 wire spelling) requires evidenced
or synthetic availability but permits later observation. Its caveat explicitly
identifies reconstruction and incomplete revision coverage. Unknown availability
is quarantined. An explicit collector-observation proxy is permitted only in
strict mode, and is marked in the resulting caveats.

Future event times and future corporate-action announcements are excluded even
if a source has inconsistent earlier availability. Complete bars enter only at
or after their end. Resolve knowledge-eligible revisions before judging price
usability: a known bad correction suppresses its older value. Incomplete/missing/
filled bars and crossed quotes stay outside inference inputs but remain bound as
`quality_inputs` in the manifest and `SnapshotBundle.quality_records`. Loading a
snapshot verifies those hashes and their retention rights too.
A known announcement about a future effective action may be stored as context;
the ledger never retrospectively adjusts prices. Actual action-aware labels,
exchange calendars, historical universe construction and session windows remain
#291. This package supports independently specified elapsed price/text windows
and persists the operator-supplied calendar identity/version/hash.

A logical revision series is `(schema_name, source_id, source_record_id)`.
Only eligible revisions compete. Select the maximum tuple
`(revision, available_at, observed_at, record_id, normalized_sha256)` using parsed
UTC clocks, independent of physical append order. Missing predecessors quarantine
a series until resolved. Links may arrive in either order; a present predecessor
must share the series, be exactly one revision earlier, and have no later first
observation. Conflicting forks are retained and resolve by the published tie rule.
`ingested_at` remains an immutable payload clock, not a selection-order input.
Reordering the same immutable inputs leaves the snapshot hash unchanged; changing
an input clock creates a different input, not a replay-equivalent observation.

The snapshot manifest binds exact input IDs/hashes, tie-rule version, identity
relations, story-cluster membership and arrival clocks, calendar, windows,
preprocessing, masks and fill policy. The #286 ContextSnapshot references that
manifest hash. Snapshot metadata is persisted append-only; its own hash is the
`load_snapshot` key. Later corrections cannot change saved or rebuilt earlier
strict snapshots. Rights erasure can make a saved snapshot unavailable; it does
not rewrite the saved identity or pretend missing payloads can be reproduced.

## Identity, text and quality

`resolve_alias` accepts eligible snapshot records and requires exactly one active
namespace/alias to security/listing/issuer/share-class mapping in `[from,to)`.
Conflicts fail closed. A reused ticker is never a security key. Share classes
and listings stay separate. Dedup uses a declared story cluster plus exact entity
identities/ambiguity, preserving every eligible member and its availability and
observation times. The earliest available/observed member is the representative;
record ID/hash break ties. Future reposts cannot backdate or join an old snapshot.
This is deterministic declared-cluster handling, not a fuzzy entity/news model.

`expected_listings` produces absent-price masks. Selected records carry distinct
`empty_text`, `deleted_text`, `missing_price` and `stale_price` reasons. Unusable
price evidence adds `incomplete_price`, `filled_price` or `crossed_quote` reasons
without putting invalid observations into ContextSnapshot selected records. A null
NewsEvent text means empty/no usable text because #286 rejects empty strings.
`signals=(listing, reason, evidence_record_id)` accepts `feed_outage` or
`market_closed` only with selected, listing-related evidence. The adapter owns
the evidence interpretation; unknown status is not inferred as closed or outage.
No reason is replaced with a numeric zero.

Forward fill defaults off. When explicitly enabled, it emits a marked reference
with source/target time and age, never mutates an observation or fabricates a
complete bar. Its bound cannot exceed price freshness. Missing/stale observations,
known outages and closed markets cannot fill. Each source/feed/venue/currency/
mark/adjustment/calendar/interval family fills independently; an IEX trade cannot
silently become a SIP mid. A newer known bad observation blocks older fills in
the same family even across interval lengths; a coarser bar is not assumed to
escape an observed feed gap. Quality evidence first available after the cutoff
cannot block an earlier fill.
Feature builders consume ContextSnapshot and its bound inputs/manifest, never an
unfiltered ledger iterator. Generic `validate_context` also rejects future event
and announcement clocks. The legacy simulated StreamEvent time, arrival handling
and stream/layout serialization are unchanged.

## Rights, recovery and migration

An explicit policy must permit normalized storage and minimal immutable
provenance. Raw retention can be disabled independently; expired or prohibited
policies fail admission. Public accessibility is never interpreted as permission.
Policies forbidding even ID/hash/clock/rights provenance are unsupported and fail
closed before append. Expiry denies reads immediately, and `recover()` physically
purges expired bytes; reopen runs recovery automatically. For a long-lived
collector, invoke recovery regularly according to the retention obligation.

`revoke(record_id, reason=...)` first commits immutable denial intents, then
unlinks raw and normalized files. Shared raw content revokes all its normalized
derivatives. Revoked content cannot be reintroduced, and a blocked logical series
cannot resurrect an older revision. The index and permitted snapshot provenance
remain. This is filesystem unlink, not guaranteed secure erasure of SSD blocks,
external backups, or previously held application memory; operators must satisfy
those obligations separately. No public export API is provided.

A local process lock serializes ledger operations, including recovery. SQLite
FULL-synchronous commits follow fsynced atomic blob writes. After an interrupted
append, unreferenced blobs and temporary files are garbage-collected. After an
interrupted deletion, durable intents deny reads and recovery completes unlink.
Missing/corrupt referenced content fails clearly; it is never silently repaired
with fabricated data. Keep the ledger and owned CAS together for backups.

Storage version 1 initializes an empty database and reopens without changing
record hashes. Existing unversioned or newer databases fail with an explicit
migration error. There is no pre-v1 production storage format to convert.
Future breaking migrations must copy into a new private store and preserve
source IDs/hashes; #286 schema migrations remain explicit before append.

## Verification

Default: `python -m unittest tests.market.test_ledger_snapshots -v` (also collected
by the unchanged pytest core runner). Optional module tier:
`python -m pytest tests/market/extended/test_ledger_recovery.py --run-market=extended -q`
or direct `python -m unittest tests.market.extended.test_ledger_recovery -v`.
The extended suite generates 1,000 observations, reorders/reopens real storage,
measures append/snapshot time, kills child processes between writes and commits,
replays pending erasure, and checks expiry and migration rejection. All payloads
are synthetic, MIT, seed 287, with generated SHA256 provenance. The default golden
snapshot hash is pinned in the correction/replay test; no provider fixture is used.
