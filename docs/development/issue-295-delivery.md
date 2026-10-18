# Issue #295 delivery ledger

Scope: #295 under textual parent #283 (no native parent relationship). Sources:
#295 updated 2026-10-09T06:34:57Z; #283 updated 2026-10-17T23:15:34Z.
Base: a994ddcc8bca5d51f8bea73e2c51f61947ba9d43, PR #308 merged.
Dependencies #284/#285/#293/#294 closed and checked in master. No open PR or
295 branch at preflight. No applicable AGENTS.md or local skill files found.
Installed github-development-workflow read; its two reference resources failed
retrieval (including relative-resource retry); main workflow remains available.

## Acceptance and verification plan

| ID | Requirement | Planned evidence |
| --- | --- | --- |
| A1 | Synthetic prepare/fit/evaluate/export; preserve Crafter | Real Factory dispatch with tiny fake fit; bounded real baseline smoke; existing Crafter smoke where dependencies available |
| A2 | Pinned deterministic outputs/hashes | Repeat config/data/seed, changed pins, checkpoint roundtrip and incompatible resume tests |
| A3 | Partial failure/cancellation, unknown rights denied | Prepare/fit/export failure tests; immutable attempts; bounded artifact allowlist |
| A4 | No locked-test search | Validation-only objective, explicit one-use final reporting, duplicate evidence reuse |
| M1 | Reuse state/checkpoint/search/registry/budget/lineage | Existing shared APIs and regression suites |
| M2 | Complete truthful provenance | Versioned manifest, source/environment/config and semantic hashes; no fabricated applicable state |
| M3 | Offline cost and claims | Aggregate default lanes <=120 seconds; fake default orchestration, no optional imports/downloads; synthetic_only |

Implementation plan: extend neutral runner with opt-in preallocation, attempt and
export hooks; register bounded market baseline backend; reuse proposal generation
and registry test-use accounting with validation-only search; document operator
workflow and evidence-backed practice. No JEPA or provider integration. Rollback:
remove lazy registration and new backend; Crafter dispatch/schema stays unchanged.

Merge gate: branch protection read HTTP 403 on 2026-10-18; rulesets empty does not
resolve it. Keep PR open until actual protection/approval requirements are known.

## Initial implementation evidence

- Default fake-model orchestration and neutral compatibility: 27 passed in 6.28s;
  added actual killed-worker/immutable-attempt resume regression: market 12 passed
  in 5.15s. Real baseline existing tests plus neutral tests: 36 passed in 9.72s.
- Actual six-stage Crafter CLI canary with existing local CPU torch: passed in
  12.128s, `/tmp/ccr-295-crafter-smoke/summary.json`. Corpus hash
  `423e353ad92d48982d20761075d8223c3efc9bd20d81642c4547bc78c495e5f7`.
- Real zero-baseline market smoke and explicit sealed report succeeded locally;
  full default/offline lanes and submitted-head review remain pending.
- During implementation, fixed resume validation ordering, truthful NumPy fp64
  declaration, and snapshot-provenance vs source-payload redistribution handling.
  These are author checks, not independent review.

## Independent submitted-head review and repairs

PR: https://github.com/Brain-Crumbs/CCR/pull/309. Initial submitted head
`7c5f11bd32af76719cf1cf24e4a067a1bfccd14b` independently reviewed against complete
issue/master sources, code and BP-012. Reviewer independently passed 12 targeted
tests in 4.85s and requested changes:

- P1: outcome-source rights missing from export gate. Reproduced with ordinary
  corpus construction and matching unknown retention policies. New corpora retain
  a content-bound inventory of context AND outcome rights; missing/unknown
  inventories fail export. Existing loaded corpus identities are not rewritten.
- P2: cached success survived deleted checkpoint. Cache now pins/verifies manifest,
  artifact inventory, byte counts/hashes and completed run state before reuse.
- P2: resumed attempt advertised original run commit/dirty status. Attempt manifest
  now records current source identity separately from original immutable execution.

Regression evidence before final re-review: 59 market/Factory/PIT tests passed in
9.27s, plus explicit resume-source identity regression. Tokenizer provenance now
uses the actual text module TOKENIZER_HASH rather than a descriptive proxy.
Fresh review and all final-head checks are still required.

Initial-head aggregate: local 1,355 core + 612 Factory/hash tests, 62.403s; zero
network/model-cache activity. Hosted run 39139790327 all three active jobs passed,
68.999s aggregate (logs verified); manual CPU-torch job intentionally skipped.
Artifact download endpoint returned 403; readable job logs supplied exact-head
counts/runtime evidence. These initial-head checks do not validate later repairs.
