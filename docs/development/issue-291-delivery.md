# Issue #291 delivery ledger

## Authority and sources

- Scope: [#291](https://github.com/Brain-Crumbs/CCR/issues/291), CCR-MKT-08;
  parent [#283](https://github.com/Brain-Crumbs/CCR/issues/283). Draft PR only;
  **no merge or tracking closure authorized in this delegated environment**.
- Requirement snapshots: #291 updated 2026-10-09T06:34:34Z; #283 updated
  2026-10-10T00:06:48Z. Both bodies read; zero comments reported.
- #287 closed completed, updated 2026-10-10T00:06:44Z, implemented by #304;
  merge/base `107302bb0f022d5735e1a05b36029de4d6bf8142` present locally and
  confirmed on GitHub. `git fetch origin main` succeeded on 2026-10-12.
- Clean initial checkout on `work` at that base; new branch
  `issue-291-pit-corpora`. No existing #291 branch/PR found in remote heads and
  issue-title/number PR searches. No AGENTS.md or `.agents/skills` in checkout.
- GitHub development workflow SKILL.md read; its linked reference resources
  returned `failed to read skill resource`. Main workflow followed directly.
- `gh api repos/Brain-Crumbs/CCR/branches/main/protection`, `/rulesets` and
  `/issues/291/parent` returned `Forbidden`. No route/credential/security changes
  or alternate access attempted. Native parent relationship and actual required
  protection checks/approvals are **unverified**; textual parent link confirmed.

## Acceptance matrix

| ID | Requirement | Implementation and executable evidence | State |
| --- | --- | --- | --- |
| A1 | PIT immutable snapshot corpus, universe/aliases/actions and delisting caveats | `corpus.py`, `universe.py`; `test_universe_rename_reuse_and_hash_binding`, private corpus roundtrip | Implemented; synthetic only |
| A2 | Explicit calendar ID/version/hash, DST/holiday/half-day/halt/24h clocks | `calendars.py`; 8 golden calendar cases, coverage validation and 24/7 UTC case | Implemented |
| A3 | Independent context and 1/4 trading-hour, 1/5-session and contract horizons | typed existing Horizon/LabelSpec, policy YAML; golden calendar and context-independence tests | Implemented |
| A4 | Separate decision, anchor observation and endpoint; exact delayed SIP/feed constraints | `test_delayed_sip_exact_and_no_mixed_feed`: 14:45 bar observed at 15:00 censored; nominal remains 16:00; IEX/SIP mismatch has no endpoint | Implemented |
| A5 | Exact price anchors/endpoints across session edges | golden label table, tolerance, after-hours next-session close (71 actual elapsed hours) | Implemented |
| A6 | Raw split-neutral ex-dividend returns; outcome-only actual actions | `test_split_only_adjustment_and_future_feature_invariance`, old-announcement full outcome snapshot and duplicate-action censor | Implemented |
| A7 | Absolute contract change and closure/expiry/missing censoring; no settlement | exact 0.42→0.47 = 0.05; four terminal negatives and early closure | Implemented |
| A8 | Censor populations remain visible | corpus population includes 3 requests: 2 realized + 1 missing endpoint; evaluation retains censor | Implemented |
| A9 | Walk-forward label/latent interval purge, strict context option, actual-end embargo | exact split membership, injected overlap rejection, equality boundary and future/context tests | Implemented |
| A10 | Train-only preprocessing/feature selection, validation-only calibration, train-only future targets | real corpus callbacks see only train or validation; test-target attachment rejected; expanded future interval rejected before writes | Implemented |
| A11 | Issuer/family robustness and source/dedup contamination | four group negatives with injected validation failure; stable alias manifest; all context issuers bound | Implemented |
| A12 | Lightweight offline defaults, optional modules, public operator docs and living practices | full core runner + second seed + schema validation; new `pit-corpora.md`, BP-008; core CI caps unchanged | Locally validated; submitted-head CI/review pending |

## Local verification before submission

Python 3.12.14, Linux x86_64, five visible logical CPUs; clean core/dev-only venv
`/tmp/ccr291-venv` installed with `python -m pip install -e '.[dev]'`. No neural or
provider packages. `$PY` below denotes `/tmp/ccr291-venv/bin/python`.

| Command | Result | Runtime |
| --- | --- | --- |
| `$PY -m pytest tests/market/test_pit_datasets.py -q` | 37 passed | 2.48s |
| `$PY .github/scripts/run_core_tests.py --collect-only --report /tmp/ccr291-evidence/core-collection.json` | 1,478 collected; success | 4.274s runner |
| `$PY .github/scripts/run_core_tests.py --report /tmp/ccr291-evidence/core-tests.json` | 1,267 passed, 256 skipped, 42 pytest subtests passed | 39.641s runner |
| `PYTHONHASHSEED=12345 $PY .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/ccr291-evidence/contracts-seed-12345.json` | 600 passed, 10 skipped | 13.529s runner |
| `$PY .github/scripts/check_core_budget.py /tmp/ccr291-evidence` | aggregate <=120s | 57.444s aggregate |
| `$PY .github/scripts/validate_artifact_schemas.py` | 5 fixtures, 3 schemas validated | 0.121s shell invocation |
| `$PY -m pytest tests/market/extended/test_pit_boundaries.py --run-market=extended -q` | 300 generated samples, 6 policies and shuffled order, 1 passed | 9.84s |
| `git diff --check` | passed | not timed |

Offline runner reports: zero network attempts, zero model-cache files; two
single-thread workers, expected seed and offline hook evidence. Local shell tests
did not use Linux `unshare`; unchanged GitHub CI does. Existing skip policy was
not changed. Collection counts and pytest subtests are reported as emitted, not
represented as extra implemented tests. These runs precede the first submitted
head; subsequent head-specific evidence belongs in the PR review/CI record.

Fixture SHA256 is checked against `tests/fixtures/market/pit/provenance.json`.
All runtime feature/label/corpus/calendar/split hashes are deterministic outputs;
no private data artifacts are committed. The config is a documented policy example,
not an ignored runtime argument. Existing market schema wire formats are unchanged.

## Limitations and remaining gates

- **synthetic_only**. No live/paid data, pretrained encoder, training, forecasting
  performance, complete delisting history or operational exchange calendar claim.
- Snapshot availability proves known inputs, not missing external action/lifecycle
  coverage. Operators must supply full historical outcome snapshots and evidence.
- Group checks use declared stable identities, family mapping and dedup/source IDs;
  no fuzzy near-duplicate inference. All dataset artifacts are private operator data.
- Content-hash readers protect integrity, not malicious arbitrary local Python
  access. Fitting callbacks are capability boundaries. Export retention remains
  an operator obligation; source-ledger revocation cannot delete external copies.
- Independent review, exact submitted-head validation/CI and actual protection
  visibility remain final gates. No GitHub approval or merge is claimed by local
  review or passing tests. Parent tracking remains unchecked until verified merge.
