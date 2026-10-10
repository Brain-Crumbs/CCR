# Issue #287 delivery ledger

## Authority, sources and base

PR-only implementation for [#287](https://github.com/Brain-Crumbs/CCR/issues/287)
under [#283](https://github.com/Brain-Crumbs/CCR/issues/283). Merge/main verification
and issue tracking belong to the parent delivery task. No merge or issue closure.
#287 source updated `2026-10-09T06:34:12Z`; master updated
`2026-10-09T23:44:25Z`; both comment lists empty. Textual parent/checklist verified;
GitHub's native parent endpoint reports no parent. #286 closed completed after
PR #303, dependency available on fresh main. No open PRs at task start.

Base `1a0684c5272476e626566660f9b9acdcd6da0c5f`, tree
`580fa181dcc3b909fbff641162e9e1e4e48eb830`. Clean saved checkout was one merge
behind; supported GitHub app reads reconstructed and SHA-verified the exact base
tree and signed commit before branching `issue-287-immutable-asof-ledger`.
No denied Git/package transport was retried. No AGENTS.md, `.agents`, contributor
file or PR template exists in the checked-out tree; remote `.agents` absent.
Workflow SKILL.md read; both reference resources failed to load, so this ledger
records the required evidence explicitly. Rulesets empty; branch protection read
returned integration 403. Parent must resolve final protection visibility.

## Plan and acceptance matrix

Use separate raw/normalized CAS plus SQLite immutable metadata. Preserve all #286
wire hashes, clocks and exact-type inference boundaries. Persist snapshot provenance
without sensitive payloads. Exercise real local storage, not an in-memory stand-in.
Rollback is removing this opt-in package; it does not migrate legacy stores or
modify legacy stream/layout serialization. Format v1 rejects unknown/unversioned
stores rather than applying an inferred migration.

| ID | Requirement | Implementation / concrete acceptance evidence |
| --- | --- | --- |
| A1 | Real private content-addressed raw and normalized storage, append-only ledger | `data/raw_store.py`, `ledger.py`; exact bytes/raw hash, immutable ID conflict/idempotency, SQL mutation rejection, private path and ownership tests. |
| A2 | Immutable observations/revisions, correction and order invariance | `select_revisions`, bidirectional lineage checks; tomorrow correction preserves golden strict snapshot, shuffled child-first ingestion and deterministic fork tie tests. |
| A3 | Strict versus explicitly reconstructed availability | Availability+observation cutoff; evidenced-only reconstruction; unknown quarantine and marked conservative proxy; correction and proxy tests. |
| A4 | Completed-bar, future/late/revised information exclusion | Event/announcement/end/availability/observation gates; incomplete/crossed exclusions; boundary-bar, skew, future action and correction tests. Generic inference validator also rejects future events/announcements. |
| A5 | PIT issuer/security/listing/share-class aliases, ticker reuse | `identity.py` exact active interval resolution and conflict rejection; ticker-reuse, renamed alias and distinct share-class test. |
| A6 | Deterministic dedup preserves eligible member arrivals | Cluster plus exact entity identity; representative and all member clocks bound in manifest; repost and share-class cluster separation test. |
| A7 | Distinct missing/stale/empty/outage/closed masks; bounded marked fill | Quality manifest, #286 missing/stale IDs, evidenced status signals and separately keyed fill references; explicit reason assertions, age edges and outage/closed no-fill tests. |
| A8 | Immutable snapshot input/tie/identity/calendar/preprocessing hashes | Persisted ContextSnapshot plus content-addressed manifest; reload equality, golden strict snapshot and large shuffled/reopen equality. |
| A9 | Rights/retention and recovery without changing logical IDs | Explicit admission policy; immutable denial before unlink; block historical fallback; default rights tests plus extended process-exit, expiry, orphan and migration checks. |
| A10 | Labels/future separation and legacy/default compatibility | Exact observation allowlist, no training/labels imports in storage; terminal rejection and subprocess import tests. No legacy stream/layout or CI/test-tier changes. Full hosted default gates required. |
| A11 | Meaningful opt-in append/replay/performance/migration evidence | 1,000 synthetic records, reordered disk replay/reopen, two real process interruptions, expired bytes erased, storage-v1 reopen and unsupported/unversioned rejection. |
| A12 | Operator docs, rights/privacy and living evidence | `docs/market/event-storage.md`, BP-006/BP-007 and append-only history; all fixtures project-authored MIT synthetic seed 287, raw hashes generated/verified at append. |

## Initial verification

- `python -m unittest tests.market.test_ledger_snapshots tests.market.extended.test_ledger_recovery -v`: 17 passed, 18.261s. Large case: 1,000 append 1.455s, snapshot 0.695s; shuffled/reopened hash `40a2cec9cb24163c9cf1ed51528de23232913ea781c24638a2c18b71db824e87`.
- Default strict golden hash `18817f159bf8d39942086eb80c506cce46bab0dd2d06130620178b3a46669132`.
- Initial constructor failure paths emitted resource warnings; cleanup was fixed and rerun. Self-review also prevented CAS/index overlap and separated fill source/feed/venue/currency/mark families.
- Environment provides Python/numpy/jsonschema/PyYAML but lacks pytest/xdist/crafter/namesgenerator. No package installation attempt. Direct unittest covers the new default/extended checks; unchanged hosted CI owns full core, second hash seed and 120-second aggregate gate. Existing five-minute CI caps and lightweight coverage remain unchanged.
- Local and hosted exact-head outcomes, independent review findings and resulting revision identities are appended in the draft PR and final handoff. Any new head requires fresh review/CI.

## Limits and next gates

All delivered capabilities are synthetic_only. No provider/model integration,
paid service, model download, live validation or trading. Calendar computation,
session windows, fuzzy entity/news clustering and model features are later work.
Private storage is POSIX/local filesystem only; deletion is unlink, not certified
SSD/backup/memory erasure. Policies forbidding required minimal provenance are
rejected. Unknown/unsupported migration versions fail closed.

Parent must refresh source issue versions, main/base, check results and protection
visibility before any separately authorized merge; then verify main and tracking.


## Published-head review and fixes

Initial draft [#304](https://github.com/Brain-Crumbs/CCR/pull/304) head
`8231b09aec98929033c5bb5a1e2db26c44868c7a`, tree
`33ab2c4eb8d7b9e3abb4f6aef4042955e3f1eaa7`. Independent review verified remote/local
identity and ran 17 tests in 15.693s. It found R1 (P1): unusable prices filtered
before revision selection resurrected older usable values and allowed fills over
known missing bars; R2 (P2): fill grouping omitted adjustment/calendar/interval.
A2/A7 failed on that head despite green tests. Other A1/A3–A6/A8–A9/A11–A12 met;
review initially marked A10 unverified until hosted evidence was available.

[Initial-head CI](https://github.com/Brain-Crumbs/CCR/actions/runs/38007035640)
passed 1,227 core tests, 256 existing skips, 42 subtests; 600 second-seed Factory
checks, 10 existing skips; aggregate 49.604s. Run metadata binds that head to the
base recorded above. This is historical evidence, not repaired-head validation.
The operator documentation example also executed successfully locally.
The full local core command attempted zero tests because pytest was absent;
combined schema/storage unittest ran 38 tests in 1.700s: 37 passed and the existing
neutral bridge was blocked by missing namesgenerator, which passed on hosted CI.

Fixes select latest knowledge-eligible revisions first, preserve unusable prices
as hashed quality evidence outside inference inputs, and use that evidence to
block obsolete values/fills. Fill bases now include adjustment, calendar version
and interval length. Three new regressions exercise bad bar/quote corrections in
both ingestion orders, quality reload/rights checks, future-quality exclusion,
gap blocking and independent adjusted/calendar/duration fills. The explicit empty
quality-input list changes the new manifest format's golden strict hash to
`81d85010fa607719acceb272a748ac0ca8b61fe6c82c3ff8f409c02a4d749a9d`;
no legacy hash changed. Default storage suite: 16 passed in 0.533s. Fresh published
head review and CI follow; exact identities/results are recorded in PR #304.
