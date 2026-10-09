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
