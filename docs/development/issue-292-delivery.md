# Issue #292 delivery ledger

## Authority, sources and scope

- [#292](https://github.com/Brain-Crumbs/CCR/issues/292), updated
  2026-10-09T06:34:46Z; parent [#283](https://github.com/Brain-Crumbs/CCR/issues/283),
  updated 2026-10-12T17:06:56Z. Full bodies read; zero comments reported.
- Dependencies #286 and #291 refreshed: both closed/completed. Freshly fetched
  main/base `46330f12b71f21952268e5e922b078f2a58679ec` contains #291/#305.
- No open PRs and no #292 branch; remote heads checked. Clean starting `work`
  checkout; new descriptive branch `issue-292-pinned-local-text` from origin/main.
- No AGENTS.md or `.agents/skills` found in checkout/workspace. GitHub workflow
  SKILL.md read. Its linked reference resource returned `failed to read skill
  resource`; main workflow and repository delivery conventions followed.
- Authorized endpoint: reviewed **draft PR only**, no merge, issue closure or
  parent checklist edits. Actual final protection/approval gates belong to parent.
- This is a **partial, blocked delivery**, not completion of #292. No paid/hosted
  APIs, live data, credential changes, model downloads or substantial training.

## Acceptance matrix

| ID | Requirement | Evidence | State |
| --- | --- | --- | --- |
| A1 | Lightweight hashing/TF-IDF, train-only fitting | `TextFeatures.fit`, real #291 corpus/split tests and exact TF-IDF counts | Implemented; synthetic_only |
| A2 | Snapshot-selected inputs and IDs/dtype/dimensions/signature | `selected_news`, `FeatureArtifact`, tamper/knowledge-clock/missing/deletion tests | Implemented |
| A3 | Optional local MiniLM, pinned revision/artifacts/tokenizer/license/settings | Strict lock/staged-file validation and bounded worker; candidate card only | **BLOCKED: immutable pins absent** |
| A4 | Optional local Qwen, reviewed pin/license and schema/prompt version | Local worker and extraction validator; prompt/schema digests, fake extraction tests | **BLOCKED: immutable pins absent** |
| A5 | Revision/model/tokenizer/preprocessor cache invalidation | Cache tests, serialized immutable LRU, frozen encoder/extractor regressions | Implemented |
| A6 | No default torch/transformers or automatic downloads; missing weights clear | Import-denial subprocess, missing/dependency/file-status tests, explicit local-only kwargs | Implemented default boundary; real load unverified |
| A7 | Bounded lengths/batches/memory/compute | Baseline caps, local child resource limits and timeout tests | Implemented; real-model memory/runtime unverified |
| A8 | Evidence/uncertainty, schema rejection, no tools | Strict duplicate/extra-field/enum/span validation; injection remains data | Implemented with fake generator; real quality unverified |
| A9 | Honest pretraining/contamination provenance | Unknown cutoff, runtime/model signatures and warnings, operator guide | Implemented; no leakage-free claim |
| A10 | Frozen backbone shareable; trainable projections explicit | `FrozenTextFeatures` artifact provenance; no target-store import | Implemented boundary, JEPA deferred to #296 |
| A11 | Synthetic provenance, offline budget <=120s, default CI <=5 min | `tests/fixtures/market/text/provenance.json`, existing core runner and unchanged default CI caps | Verification pending |
| A12 | Operator docs, delivery ledger, living practices and exact-head review | `text-features.md`, BP-009, this ledger | Submitted-head review pending |

## Pin access blocker and evidence limits

Official candidate cards were readable through the web tool on 2026-10-12.
The execution-environment request to the official Hugging Face metadata API
failed at the proxy CONNECT step with **HTTP 403 Forbidden**. No alternate route,
credential change, guessed pin or artifact download was attempted. Requested
permitted metadata evidence from parent; no verified immutable SHA256 evidence
was available when this ledger was written. The checked-in lock has null
revisions/license, empty files and `access_denied` status; runtime fails as
`unverified_model_pin`. This is an intentionally visible acceptance blocker.

A permitted official source must establish immutable model/tokenizer revisions,
all loaded artifact hashes/sizes and selected-revision license/card evidence.
Only then can reviewed pins enable local staging/use. Optional real MiniLM/Qwen
smoke remains **unrun**, not a passing fixture or `validated_live` claim.

The existing neural package eagerly imports torch. The proposed neutral text
API instead lives in `adapters/finance/text` without changing legacy exports.
Model loading uses the MiniLM card's Transformers pooling approach. Neither
sentence-transformers nor any other optional framework enters default imports.

## Validation and review ledger

Python 3.12.14, Linux; clean `/tmp/ccr292-venv`, installed core/dev with
`python -m pip install -e '.[dev]'`. No optional ML dependencies installed.
Initial targeted suite: `python -m pytest tests/market/test_text_features.py -q`:
23 passed in 2.64s. Full default collection, core, hash-seed lane and budget
are measured by the existing runners; exact evidence will be recorded below
and in the submitted PR. No test membership/CI limit was weakened.

### Pre-submission measured default validation

`PY=/tmp/ccr292-venv/bin/python`; commands run from repository root:

| Command | Result | Runner runtime |
| --- | --- | --- |
| `$PY .github/scripts/run_core_tests.py --collect-only --report /tmp/ccr292-evidence/core-collection.json` | 1,507 selected node IDs; success | 3.623s |
| `$PY .github/scripts/run_core_tests.py --report /tmp/ccr292-evidence/core-tests.json` | 1,296 passed, 256 skipped, 42 pytest subtests passed | 28.794s |
| `PYTHONHASHSEED=12345 $PY .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/ccr292-evidence/contracts-seed-12345.json` | 600 passed, 10 skipped | 9.111s |
| `$PY .github/scripts/check_core_budget.py /tmp/ccr292-evidence` | **41.528s / 120s** aggregate | included above |
| `$PY .github/scripts/validate_artifact_schemas.py` | 5 fixtures / 3 schemas validated | not separately timed |
| `git diff --check` | passed | not timed |

All three runner reports record zero network attempts and zero model-cache files.
Local verification used the Python offline hook and empty caches, not Linux
`unshare`; unchanged GitHub default jobs add OS network isolation and retain
5-minute caps. Optional real-model smoke was not selected or run. Existing skip
policy is unchanged. A11 default local budget is met; submitted-head CI is a
separate pending check.
