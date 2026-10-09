# Evidence-backed development practices

Implementation and review: [PR #301](https://github.com/Brain-Crumbs/CCR/pull/301).

These practices record specific, reviewed failure modes and their regression
coverage. They do not change repository permissions or replace issue acceptance.

## BP-001: Fail closed on required validation inputs

- Status: active
- Scope: checked-in CI fixtures, schemas, runtime evidence and optional smoke inputs.
- Rule: validate required input inventories and provenance hashes unconditionally.
  A missing pair is a failure, not permission to skip validation. Exercise negative
  cases (missing, malformed, changed and orphan inputs) alongside real serializer
  round trips. JSON Schemas must not resolve remote references during offline CI.
- Exception: a genuinely optional capability may be unavailable, but it must be
  explicitly selected and reported unavailable rather than claimed validated.
- Evidence: [#297](https://github.com/Brain-Crumbs/CCR/issues/297),
  `tests/test_factory_artifact_schemas.py`, `tests/test_factory_smoke_contract.py`,
  `tests/market/test_core_budget.py`. The old workflow silently skipped when both
  artifact directories were absent.
- Added/revised: 2026-10-09.

## BP-002: Bound cost without discarding behavioral coverage

- Status: active
- Scope: default PR tests and opt-in modules.
- Rule: profile before changing test membership; prefer tiny real fixtures with
  unchanged assertions. Preserve a bounded explicit compatibility case when
  reducing production-size fixtures. Count all default execution lanes in the
  measured budget. Exclude optional modules before their imports, including
  explicit file/wildcard collection, and require a deliberate tier option.
- Exception: optional modules may use their own documented dependencies/budgets;
  they must never become mandatory merely because those packages are installed.
- Evidence: [#297](https://github.com/Brain-Crumbs/CCR/issues/297),
  `tests/test_crafter_world.py`, `tests/test_replay_smoke.py`,
  `tests/market/extended/test_legacy_crafter_compatibility.py`,
  `tests/market/test_test_tiers.py`. Repeated default-size terrain generation
  dominated the original 166.49-second default suite.
- Added/revised: 2026-10-09.

## BP-003: Verify actual offline execution and actual artifact identities

- Status: active
- Scope: test runners and Factory CLI orchestration.
- Rule: start with empty model caches; deny and record attempted network access
  in the test process and children. Caught network exceptions must still fail
  the outer gate. Use an isolated network namespace in Linux CI. Capture IDs
  from successful CLI output and verify resulting artifacts; never predict IDs
  from display names, and never substitute a mocked command for end-to-end proof.
- Exception: live tiers are explicitly opt-in, separately rights/key-gated, and
  may not publish restricted payloads. They are not implemented by this repair.
- Evidence: [#297](https://github.com/Brain-Crumbs/CCR/issues/297), related
  [#212](https://github.com/Brain-Crumbs/CCR/issues/212),
  `tests/market/test_offline_core_runner.py` and `.github/scripts/run_factory_smoke.py`.
- Added/revised: 2026-10-09.

## BP-004: Version domain boundaries without rewriting legacy identities

- Status: proposed (independent submitted-head review pending)
- Scope: task/model registry, inference inputs and checkpoint continuation.
- Rule: use an explicit new contract format for new domains; preserve serialized
  legacy defaults merely on load. Check task/model identity before loading state,
  and keep evaluation labels outside inference input containers. Exercise actual
  Factory dispatch, artifacts and persisted failure states with tiny fake backends;
  pair them with opt-in real compatibility runs rather than treating mocks as proof.
- Exception: a legacy adapter may retain its existing lifecycle and serialization
  behind the shared entry point to avoid an unnecessary trainer rewrite.
- Evidence: [#285](https://github.com/Brain-Crumbs/CCR/issues/285),
  `tests/test_model_factory_task_backends.py` and
  `tests/market/extended/test_task_backend_compatibility.py` (added; runtime
  verification blocked in the implementation environment). The audited Factory
  constructed ActionWorldModel for every checkpoint and required pixel/action
  contracts; `brain.cortex` also eagerly imported torch before neutral contracts.
- Added/revised: 2026-10-09. Structural separation does not prove causal features;
  future data/label implementations must supply semantic leakage evidence.

## Change history (append-only)

| Date | ID | Change | Reason and evidence |
| --- | --- | --- | --- |
| 2026-10-09 | BP-001 | Added | Missing fixture/schema pairs produced a false-green CI path; #297. |
| 2026-10-09 | BP-002 | Added | Measured default runtime exceeded 120 seconds; preserve coverage while bounding fixtures, #297. |
| 2026-10-09 | BP-003 | Added | Offline attempts and hardcoded Factory IDs need executable failure evidence; #297 / #212. |
| 2026-10-09 | BP-004 | Proposed | Domain-coupled checkpoint construction and eager cortex import found during #285; integration tests added, execution/review pending. |

## BP-005: Separate future outcomes and bind financial semantics explicitly

- Status: proposed (submitted-head independent review pending)
- Scope: versioned market schemas and synthetic fixtures, #286.
- Rule: validate exact typed observation families at inference boundaries;
  labels, terminal outcomes and training targets live in a separate module.
  Bind target units, horizon origin, currency/feed/mark and anchor-dependent
  distribution support in executable validation. Preserve all knowledge clocks
  in immutable hashes. Reject unknown/stale schemas and migrate copies explicitly.
- Exception: a schema boundary does not prove arbitrary numeric feature causality;
  future feature/ledger implementations still require semantic leakage tests.
- Evidence: `tests/market/test_schema_contracts.py`, explicit terminal/unknown-field,
  currency/clock/quantile negatives, generated fixture hashes and documentation;
  optional seeded boundary/prior-version/process checks in
  `tests/market/extended/test_schema_properties.py` actually executed locally.
- Added: 2026-10-09.

### Additional append-only evidence history

| Date | ID | Change | Reason and evidence |
| --- | --- | --- | --- |
| 2026-10-09 | BP-004 | Completion evidence appended | #285 merged in PR #302; exact-head core 1,189 + 600 tests (59.822s), actual six-stage Crafter smoke and neutral GRU compatibility passed per verified issue completion. Earlier proposed/runtime-blocked entry retained as historical evidence. |
| 2026-10-09 | BP-005 | Proposed | #286 explicit financial schema boundary and decimal/clock/distribution negative cases; local default and opt-in checks passed, submitted-head review pending. |
| 2026-10-09 | BP-005 | Review regression evidence appended | Independent review of PR #303 head 682b59e4 found cancellation after rounded division, tiny-anchor support rounding, and placeholder raw provenance. Exact bounded arithmetic and preserved hashed synthetic source inputs now have direct regressions; fresh-head review pending. |

## BP-006: Make causal replay independent of physical ingestion order

- Status: proposed (independent submitted-head review pending).
- Scope: immutable observation ledgers and snapshot manifests, #287.
- Rule: choose among eligible revisions with an explicit stable tie rule; bind
  knowledge clocks, exact selected hashes, identity/dedup evidence and quality
  policy in the snapshot identity. Append corrections instead of overwriting
  earlier observations. Test shuffled ingestion and future perturbation against
  the same golden snapshot, including a real disk reopen.
- Exception: rights revocation can make historical payloads unavailable; preserve
  allowed provenance and fail reads rather than claiming an identical replay.
- Evidence: `tests/market/test_ledger_snapshots.py` correction, tie, identity and
  dedup tests; `tests/market/extended/test_ledger_recovery.py` large shuffled replay.
- Added: 2026-10-09.

## BP-007: Commit denial before erasure and test actual interruption

- Status: proposed (independent submitted-head review pending).
- Scope: private content-addressed storage with append-only provenance, #287.
- Rule: persist a rights-denial intent before physical payload deletion; block
  reads immediately, resume cleanup after interruption, and forbid older-revision
  resurrection. Durably write content before committing its index reference;
  recover precommit orphans. Keep sensitive content out of immutable metadata.
- Exception: unlink cannot erase external backups or already-held memory; record
  those operator obligations and reject policies forbidding required provenance.
- Evidence: real subprocess exit tests between blob/index commit and denial/unlink,
  expiry and rights-deletion tests in `tests/market/extended/test_ledger_recovery.py`.
- Added: 2026-10-09.

| Date | ID | Change | Reason and evidence |
| --- | --- | --- | --- |
| 2026-10-09 | BP-005 | Completion evidence appended | #286 merged as #303; exact-head 1,214 core + 600 Factory checks, 39.683s aggregate, and decimal/provenance regression review verified in issue completion. |
| 2026-10-09 | BP-006 | Proposed | #287 corrections and shuffled physical arrival must not change past strict snapshots; disk/golden regressions added. |
| 2026-10-09 | BP-007 | Proposed | #287 filesystem/SQLite writes cannot share a transaction; interrupted append and rights erasure need replayable intents and actual process-exit tests. |
