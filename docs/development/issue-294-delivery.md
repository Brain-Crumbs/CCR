# Issue #294 delivery ledger

Scope: only [#294](https://github.com/Brain-Crumbs/CCR/issues/294), under textual
master [#283](https://github.com/Brain-Crumbs/CCR/issues/283). No #295 work.
Initial/fresh main: `5a876027c9fb0e56687894dc49fe54c93a90ef4b` (PR #307 merged).
Requirements read in full on 2026-10-15: #294 updated 2026-10-09T06:35:27Z;
#283 updated 2026-10-15T18:59:42Z. Both issue comment lists were empty.
#291/#293 are closed and checked in #283. Native parent lookup returns 404;
the checklist/textual relationship is verified independently. Fresh open-PR
search empty; no overlapping remote branch or local worktree. Clean isolated
branch `issue-294-calibration-evaluation` created from fetched main.

No AGENTS.md, contributor or local .agents/skills found. Workflow skill read;
its delivery-record/review-and-practices reference resources fail to resolve
(as in prior delivery), so this ledger follows the complete main skill and
existing repository conventions. CI/test tiers and living practices read.

## Acceptance matrix and planned evidence

| ID | Requirement | Implementation / verification |
| --- | --- | --- |
| A1 | Validation-only calibration; mismatched snapshot/label IDs fail | `bind_population`, `fit_calibrator`, immutable application; exact partition/spec/hash/anchor negatives and hand-computed corrections. |
| A2 | Failed/abstained/censored populations retained | Coverage-first reports, suite failure artifacts, new immutable censored fixture; every held-out ID required. |
| A3 | Oracle rejected; null cannot be promoted as edge | Declared future-return and concrete oracle-label injection rejected; all-arm zero-signal fixture; `edge_promotable=false` and synthetic caveats. |
| A4 | Exact baseline, N, uncertainty and population | Payload/label hashes, paired IDs, source/instrument/horizon, seeded time-block intervals and explicit unavailable evidence. |
| A5 | Forecast quality separate from P&L | No orders/integration/P&L; explicit null P&L in report. |
| M1 | Proper scores and no invented probabilities/distributions | Hand pinball/median MAE/outer interval and explicit-head direction Brier tests; no CRPS from sparse quantiles, no resolution scores. |
| M2 | Equal-budget baseline/control arms, no JEPA dependency | Nine arms, same split/labels/seeds/config limits; actual usable fit rows; partition-local shuffle and causal lag; latent arms await #296. |
| M3 | Regime/liquidity/missingness and disjoint slices | Exhaustive external slice memberships or visible unknown; issuer/family slice counts plus validated SplitPlan group policy. |
| M4 | Locked test, validation-only Factory score, immutable inputs | Exclusive report write, guarded objective, immutable predictions and feature-vector controls; #295 lifecycle integration remains separate. |
| M5 | Bounded offline evidence, provenance, public docs and practices | Default no-download tests, optional tiny multi-seed check, existing full offline/120-second gates; provenance inventory and operator guide; BP-011. |

Design: new evaluation package; one default-preserving trusted model-factory
hook in baseline fitting permits controlled derived-vector ablations without
rewriting snapshots. Revert the new package/tests/docs and that hook to roll back;
no legacy serialization or schema changes. Existing schema EvaluationReport is
unchanged; detailed report uses explicit `market-evaluation-v1` format rather
than pretending to be a different existing envelope schema.

## Gate status

Protection endpoint again returns HTTP 403 `Resource not accessible by
integration`. Repository rulesets return `[]`; this does not establish absence
of branch protection or required approvals. **Merge remains blocked** unless
visibility is restored. No alternate credentials/tool bypass. Keep #294 open
and its #283 checkbox unchecked until verified authorized merge.

Exact submitted head/base, fresh independent review and terminal hosted CI are
recorded on the PR to avoid a self-referential commit. Local commands/results
will be recorded before submission. All evidence synthetic_only; no downloads,
credentials, paid calls, trading, live probes, prospective campaign or full
research claim. Prior Hugging Face denial was not retried or bypassed.

## Initial validation

Existing core/dev environment: Python 3.12.14, no optional neural/provider packages.
Targeted evaluation + baseline regressions: **32 passed in 17.00s**.
Optional bounded two-seed check: **1 passed in 3.79s**. Its first standalone
collection found a test import-path error; corrected to repository package import,
then reran successfully. No training campaign or external access.

```sh
python -m pytest tests/market/test_evaluation.py tests/market/test_distributional_baselines.py -q --durations=5
python -m pytest tests/market/extended/test_evaluation_multiseed.py --run-market=extended -q --durations=5
python .github/scripts/run_core_tests.py --collect-only --report /tmp/ccr294-evidence/core-collection.json
python .github/scripts/run_core_tests.py --report /tmp/ccr294-evidence/core-tests.json
PYTHONHASHSEED=12345 python .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/ccr294-evidence/contracts-seed-12345.json
python .github/scripts/check_core_budget.py /tmp/ccr294-evidence
python .github/scripts/validate_artifact_schemas.py
git diff --check
```

Full core: **1,341 passed, 256 skipped, 42 subtests** (39.77s pytest,
40.039s runner). Factory/hash: **600 passed, 10 skipped**. Aggregate **52.321s /
120s**, zero network attempts/cache files, matching collection and dual-seed
memberships. Five schema fixtures passed. Runtime JSON/logs reside under
`/tmp/ccr294-evidence`; hosted CI will supply durable exact-head evidence.

## Initial submitted-head review and repairs

PR #308 initial head `9ec55dabf507c1f08921e0da42eceb3a25b0c0e4`, base unchanged.
Fresh independent reviewer read full source/issues/matrix/practices and reran
12 evaluation tests (10.17s). Three P1 blockers were reproduced: lexical UTC
ordering let a later fractional-second sample donate to an earlier lag control;
inner label listing/spec could disagree with its wrapper; split Sample snapshot
identity could disagree with actual context. Review marked A1/A4/M2 incomplete.

Repairs parse instants for lag ordering/strict comparison, bind loaded feature
sample/snapshot/time to the split, bind inner label spec/listing/decision/endpoint
to the wrapper, and enforce actual label/availability interval and provenance.
Anchor binding also applies to abstained forecasts. New immutable-artifact and
mixed-precision/equivalent-instant regressions cover these failures. Targeted
repair evaluation: **14 passed in 12.21s** before final full rerun.

Initial-head hosted CI [run 38807899209](https://github.com/Brain-Crumbs/CCR/actions/runs/38807899209)
passed all three active jobs; logs verify test-merge `f5dd37f5661ef3672a1b7d7d283eefa7ad272e3d`
combines exact initial head/base. Core 1,341 and Factory/hash 600 passed, aggregate
66.017s / 120s, zero network/cache activity. Manual CPU-torch job skipped by
explicit workflow condition. These results are superseded by repaired-head
validation/review, recorded in the final PR evidence comment.

Repair validation (same commands, report root `/tmp/ccr294-r2-evidence`): core
**1,343 passed, 256 skipped, 42 subtests**, Factory/hash **600 passed, 10 skipped**;
aggregate **49.883s / 120s**, zero network/cache activity. Five schema fixtures
and diff check passed. Optional two-seed rerun **1 passed in 2.80s**. Issue/master
source timestamps refreshed unchanged before repair submission.
