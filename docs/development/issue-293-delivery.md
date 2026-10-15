# Issue #293 delivery ledger

Scope: one issue, [#293](https://github.com/Brain-Crumbs/CCR/issues/293), under
[master #283](https://github.com/Brain-Crumbs/CCR/issues/283). Merge authorized only
when all gates pass. Source requirements refreshed 2026-10-15; #293 last updated
2026-10-09T06:35:24Z. Initial base/main: `571c3328cc74582e46e2f693789146c1e6ee19ba`.

Selection: prerequisites #285/#291/#292 are closed and merged. Master explicitly
puts #293 in Stage B and source integrations #288–#290 in independent Stage C;
this is DAG order, not skipping an outstanding prerequisite. All open PR search
returned empty; remote branches/worktrees showed no overlapping #293 work.
#283's checklist references #293, but GitHub's native parent endpoint returned
404 (no parent relationship). Preserve that distinction. No AGENTS.md was found
in workspace/repository. Installed workflow SKILL.md read successfully; both
linked reference resources returned `failed to read skill resource` (also noted
by prior deliveries). Existing CI, schema guidance and best-practices read.

## Acceptance / verification map

| ID | Requirement | Evidence |
| --- | --- | --- |
| A1 | Six zero/unconditional/volatility/numeric/text/multimodal baselines share forecast IDs, label specs and splits | Parameterized real ledger/corpus test across all six; fixed shared config inventory. |
| A2 | Correct units/support, monotone quantiles and explicit abstention | Contract 0/1 anchors, equity -1 lower bound, missing/stale/expiry paths; Decimal clipping. |
| A3 | Resume fitted model/preprocessing/effective config | Exact forecast/hash disk round-trip for each kind, changed corpus/split/config rejection, integrity test, TF-IDF restoration. |
| A4 | Honest baseline and synthetic labels | Forecast quality and operator documentation: baseline, generator recovery only, uncalibrated, no direction probability. |
| M1 | Train-only fitting, inference without labels/terminal outcomes | Spy forbids held-out outcomes; neutral InferenceInput rejects TrainingBatch; subprocess asserts no label/torch/transformers import. |
| M2 | Preserve neutral/Crafter interfaces and default cost | PredictiveModel implementation; no legacy changes; existing Factory/hash/default CI suites. #295 owns Factory lifecycle registration. |
| M3 | Bounded offline tests, no downloads/provider spend | 512 train rows, 64 text dimensions, 19 quantiles; offline runner and aggregate budget. |
| M4 | Living practice and operator reproduction | BP-010 and docs/market/baselines.md in this PR. |

Implementation: small closed-form NumPy ridge/reference distributions, separately
imported fitting module, inference-only financial wrapper, text-state restoration.
Rollback: revert these new model/config/docs/tests plus the additive text restore
method; no schema migration or existing hash changes. Calibration/full research
evaluation, Factory campaigns and optional JEPA stay in #294/#295/#296.

## Gates and limitations

Main branch protection read returned HTTP 403 `Resource not accessible by integration`.
Repository rulesets endpoint returned `[]`, which does not establish absence of
branch protection. **Merge blocked until protections are readable**. Do not alter
access or bypass protections. Sub-issue and master must remain open/unchecked
until verified merge. Exact submitted-head review and remote CI tracked in PR.
No live data, paid APIs, hosted LLMs, credentials, model downloads or research
campaigns. Synthetic fixtures are not live source validation. Optional recovery
is a tiny fixed CPU demonstration, not an exhaustive null-signal study.

## Initial content validation

Python 3.12.14, existing core/dev-only `/workspace/ccr-core-venv`.
Targeted: **55 passed in 11.62s**; optional tiny CPU recovery **1 passed in 1.66s**.
Repository-required commands:

```sh
python .github/scripts/run_core_tests.py --collect-only --report /tmp/ccr293-evidence/core-collection.json
python .github/scripts/run_core_tests.py --report /tmp/ccr293-evidence/core-tests.json
PYTHONHASHSEED=12345 python .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/ccr293-evidence/contracts-seed-12345.json
python .github/scripts/check_core_budget.py /tmp/ccr293-evidence
python .github/scripts/validate_artifact_schemas.py
```

Core **1,328 passed, 256 skipped, 42 subtests**; Factory/hash **600 passed,
10 skipped**. Aggregate **47.512s / 120s**, zero network attempts/model-cache files,
matching collection/execution and Factory dual-seed membership. Five golden
fixtures against three schemas passed; `git diff --check` passed. Local runner
uses the repository offline hook; hosted CI additionally uses network namespaces.

## Initial submitted-head review and repair

PR #307 initial head `03b7452bbda3945041a9057dce524e598d5b2ce2`, base unchanged.
Fresh independent reviewer read source, requirements, matrix and practices, and
independently reran 55 tests (10.51s). One P2: quarantine filtering matched only
listing/currency/feed, so an incomplete BID observation suppressed a valid TRADE
anchor. Repair matches target record family, mark and raw adjustment as well;
regressions retain same-basis quarantine while ignoring BID/adjusted/Quote rows.
Optional test renamed to avoid implying a genuine null-signal experiment.
Reviewer found no other blocking defects and accepted #295 Factory scope boundary.

Initial-head hosted CI run 38775133885 passed all three active jobs: core 1,328,
Factory/hash 600, aggregate 51.769s / 120s, zero network/cache activity, five
schema fixtures. CPU-torch job skipped by explicit manual-only workflow condition.
These initial-head results are superseded by repair-head validation/review;
current exact evidence belongs in the PR comment to avoid a self-referential SHA.

Repair targeted checks: **56 passed in 10.00s**, optional recovery **1 passed in
1.36s**. Full offline gates rerun with `/tmp/ccr293-r2-evidence`; exact-head results,
review, remote CI and final blockers are recorded on PR #307 after submission.
#293 and #283 requirements refreshed unchanged before repair submission.
