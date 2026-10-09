# Issue #286 delivery ledger

## Authority and source versions

PR-only implementation for [#286](https://github.com/Brain-Crumbs/CCR/issues/286)
under [#283](https://github.com/Brain-Crumbs/CCR/issues/283); parent owns merge and
tracking. No closing keywords, merge or issue mutation. #286 updated
2026-10-09T06:34:07Z; master updated 2026-10-09T23:23:32Z. Both comment lists empty.
Textual parent/checklist relationship verified; native parent endpoint reports none.
Prerequisite #285 closed, PR #302 merged and verified at current main.

Base commit `fd471af6bd1fc76cd12204d07317457adfc0deff`, tree
`759902134d5209ce0d15e30d6beacebf29b832f5`. The environment initially contained
older clean `675cc2f`; supported GitHub app reads reconstructed and SHA-verified
all changed blobs, the exact tree and signed commit before the feature branch.
No denied Git/package transport was retried. Branch: `issue-286-market-schema-contracts`.
No AGENTS.md, contributor template or `.agents/skills` exists in this checkout;
remote `.agents` also absent. Workflow skill read; its reference resources could
not be read, so this explicit ledger records evidence. Rulesets returned empty;
branch protection returned integration 403 (parent must resolve final gate visibility).

## Scoped acceptance matrix

| ID | Requirement | Implementation and verification |
| --- | --- | --- |
| A1 | All strict versioned record families/common envelope | `schemas/base.py`, `records.py`, `labels.py`; `test_all_families_round_trip_and_deep_unknown_fields` tests every family, recursive unknown fields, stale versions and replay. |
| A2 | Terminal and future-target separation; neutral integration | `labels.py` separate from inference exports; exact-type observation allowlist; `test_context_and_neutral_inference_boundary`, import blocker, real `InferenceInput` and `TaskDataContract` bridge tests. No fake model substitutes for schema acceptance. |
| A3 | Distinct decimal-safe equity and contract targets/units | Explicit Target/Unit enums, local 34-digit Decimal arithmetic; independently specified 100→105 and 0.42→0.47 assertions, split-neutral label test. |
| A4 | Anchor support, monotone quantiles, direction/abstention | Forecast constructor and default invalid-boundary tests; optional 1,000 randomized decimal/support cases. No resolution-probability output. |
| A5 | Explicit clocks, currency/feed/mark; terminal fields rejected | Horizon, LabelSpec, RealizedLabel and context validators; naive/precision/clock/endpoint/currency/stale/terminal negatives. Calendar selection itself remains #291. |
| A6 | Stable replay; additive/breaking migration policy | Canonical JSON/hashes, deep immutable records, subprocess replay, explicit copy-on-write synthetic pilot 0.9 converter; breaking/unknown versions reject. Prior corpus actually exercised. |
| A7 | Tiny seeded generator and zero-signal control | AR(1), noisy instrument loadings, log-price/logit paths, lag/noise text; eight-step signal/null fixtures; optional actual 2,048-step process correlation/null check. |
| A8 | Listed handcrafted edge cases | `edge-cases.json` includes delayed/revised/shuffled/syndicated/skewed news, alias reuse, splits/dividends, delisting/halts, stale/crossed/outage, calendar/DST/holiday/half-day, early closure/terminal traps; explicit expected values tested. No future ledger/calendar engine claimed. |
| A9 | Rights/hash metadata; generated matching docs/examples | Strict FixtureManifest, file and record hashes, seed/license/source/invariants; byte-for-byte regeneration and structural schema validation; reference/examples generated from constructor fields. |
| A10 | Legacy hashes and #297 default gates preserved | No edits to legacy serialization/CI/test-tier policy. New default tests use unittest (pytest discovers them); optional directory remains excluded by existing tier gate. Full hosted core evidence pending initial publication. |

## Commands and evidence history

- `python -m unittest tests.market.test_schema_contracts tests.market.extended.test_schema_properties -v`: initial 27 tests passed, 6.697 seconds (23 default + 4 opt-in); real installed `jsonschema` validation included. Later exact-head runs are reported in PR.
- `python -m cognitive_runtime.adapters.finance.testing.generate --output tests/fixtures/market`: succeeded; initial manifest SHA256 `ef33ab1421c2a8ef39d696015258204d04393d65d993bbff211f51b61de73c0a`.
- `python -m cognitive_runtime.adapters.finance.testing.document`: succeeded; test checks all generated output byte-for-byte.
- Environment has numpy/jsonschema/PyYAML, but lacks pytest/xdist/crafter/namesgenerator/torch. No installation retry. Standard-library unittest is executed directly; unchanged hosted CI must establish aggregate core/legacy gates.
- Local checks discovered and fixed a duplicate generator keyword and stale-version parser error ordering before publication.
- No live validation; every delivered data capability is `synthetic_only`. No UCI import, provider calls, model downloads, trading, model/ledger implementation or paid resource use.

## Review and handoff

Submitted head/tree, hosted CI, independent exact-head findings and resolution are
recorded in the draft PR and final handoff. A new head invalidates prior head-specific
review. Parent must refresh issue versions, main, required checks/protection and
review conversations immediately before any authorized merge.

### Initial publication and independent review

Draft PR [#303](https://github.com/Brain-Crumbs/CCR/pull/303), initial head
`682b59e4b4b289617a406b0fd398674ff554078d`, tree
`371bff63f44206c908714b13151bf36b321a2e70`.
Independent reviewer verified that published head and tree, ran 28 tests in 6.904s
(27 pass, real neutral bridge blocked locally by missing namesgenerator), and
reported three P2 blockers: equity cancellation after rounded division; support
rounding for tiny contract anchors; zero placeholder synthetic raw provenance.

Fixes: exact subtraction before a single 34-significant-digit equity division;
explicit 128-significant-digit/128-fractional-place wire limits with exact bounded
contract arithmetic; preserved canonical synthetic source payloads in sources.json
with real raw SHA256 binding, including revisions and nested LabelSpec. New tests
independently assert a 34-digit tiny equity return, 1e-91 anchor upper support and
rejection outside it, scale limits, and every source digest/payload relationship.
Generated examples/docs and fixture manifest regenerated.

Initial-head hosted CI [run 38005308259](https://github.com/Brain-Crumbs/CCR/actions/runs/38005308259)
passed core (1,213 passed, 256 existing skips, 42 subtests), second-seed Factory and
aggregate core gate (57.818s / 120s). Test-merge SHA
`59edb61e60c146799a1c847d3bbd9ad2073965e4` binds initial head to base fd471af6.
The real neutral bridge passed there with installed base dependencies. This is
historical evidence only; repaired head requires fresh CI and review.
