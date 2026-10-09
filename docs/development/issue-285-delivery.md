# Issue #285 delivery record

Scope: [#285](https://github.com/Brain-Crumbs/CCR/issues/285), textual master
[#283](https://github.com/Brain-Crumbs/CCR/issues/283). Draft PR only; no merge,
issue closure or checklist mutation authorized here.

## Source and starting evidence

- #285 open, updated `2026-10-09T06:34:03Z`; #283 open, updated
  `2026-10-09T17:04:11Z`, both read in full.
- #284 and #297 closed. #297 records verified PR #301 merge. Master still has
  #284 unchecked; parent owns tracking reconciliation. Native parent endpoint
  for #285 returned 404; relationship is textual, not native.
- No open PRs at initial refresh. GitHub app resolved main to
  `675cc2f61bad5a430c549d8a194fa744f6b7a5eb`, tree
  `afa898ff194f59abad08aea1318a7c6651d655f3`; both match clean local checkout.
- Branch: `issue-285-neutral-task-contracts`, created from that verified commit.
  Git fetch was proxy-denied (403); no network/security/credential changes.
- No AGENTS.md or repository .agents guidance found; workspace .agents is empty.
  Workflow SKILL.md read; linked references could not be loaded. Parent supplied
  their ledger/review requirements explicitly; these govern this record.

## Acceptance matrix

| ID | Requirement | Implementation | Evidence / status |
| --- | --- | --- | --- |
| A1 | Versioned neutral task/data/feature/model/training/evaluation boundaries | task_contracts.py, model_contracts.py, separate neutral spec format | Typed boundary and fixture tests added; runtime pending |
| A2 | Existing Factory dispatch and thin legacy compatibility | run_trial dispatch, lazy Crafter adapter delegates existing lifecycle; corpus and allocator accept neutral contracts | Two fake-domain lifecycle tests and six-stage real CLI opt-in test added; pending |
| A3 | Prepare/fit/predict/evaluate/save/load, identities, artifacts, cancellation/failure | task_runner reuses allocator/state/TrialResult; JSON checkpoint dispatch | Failure, cancellation, budget, artifact hash, clone/resume tests added; pending |
| A4 | Unknown backend and cross-domain/incompatible continuation rejection | Registry, metadata-first identity checks, checkpoint identity/data/training validation | Negative tests added; pending |
| A5 | Inference isolation and lazy optional imports | Exact slotted numeric InferenceInput; labels separate; lazy cortex exports | 8 stdlib checks passed in 0.000946 s (scope below); complete base-import subprocess test pending |
| A6 | Existing configs/CLI/checkpoints/stream hashes stable merely on load | Legacy defaults and payloads unchanged; inspection does not write | Existing recipe byte hashes verified; new load preservation/legacy config regressions added; neural regression pending |
| A7 | Offline defaults, bounded opt-in real integration, aggregate runtime gate | Existing #297 workflow/tier/budget scripts untouched; new tests follow tiers | Full core, second-seed and aggregate measurement blocked by dependencies |
| A8 | Operator docs and justified best practices/history | task-backends.md and BP-004 | Documentation and static diff review; independent submitted-head review pending |

## Executed checks and blockers

- `python -m compileall -q brain/cortex cognitive_runtime/training/model_factory tests/test_model_factory_task_backends.py tests/market/extended/test_task_backend_compatibility.py`: passed.
- `git diff --check`: passed.
- Standalone stdlib checks: **8 passed, 0.000946 seconds**. Covered input freezing,
  metadata/extra-field/subclass/nonfinite/alignment rejection, torch-free model
  contract import and unchanged existing recipe hashes. This is limited evidence,
  not execution of the Factory lifecycle suite.
- `python -m pytest tests/test_model_factory_task_backends.py -q`: could not start,
  `No module named pytest`.
- `python -m pip install -e '.[dev]'`: build dependency resolution blocked by
  proxy HTTP 403 for setuptools. Existing interpreter lacks numpy/yaml/pytest/
  namesgenerator/crafter/torch; system Python has numpy/yaml but lacks pytest,
  namesgenerator/crafter/torch. No dependency download workaround attempted.
- Full default suite, second hash seed, aggregate gate and optional neural/CLI
  checks are **unverified**, not passing/skipped evidence. They require a prepared
  authorized environment or hosted CI. Prior #297 timings do not validate this head.
- Only compatibility infrastructure is implemented; no production market backend,
  provider integration, live validation, forecasting or gain claim. `synthetic_only`.

## Revision and review evidence

Remote submitted head/tree, base and independent review findings are recorded in
the draft PR and handoff after publication. Any new head invalidates head-specific
review/CI evidence. Parent owns fresh requirements/protection/CI gates, merge,
main verification and tracking completion.
