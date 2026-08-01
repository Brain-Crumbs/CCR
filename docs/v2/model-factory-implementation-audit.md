# Model Factory implementation audit

**Audit date:** 2026-08-01  
**Epic:** [#212 - Model Factory: Reproducible Checkpoint Lineage and Budgeted Experiments](https://github.com/Brain-Crumbs/CCR/issues/212)  
**Workspace revision:** `f91b2c7c3d33e9256d72b6cd76fdbb383bb4bb21` on
`codex/fix-generic-corpus-build`  
**Compared with:** `origin/main` at
`6273f45e62065129f7bd689f82e2e3a1db9067dc`

## Verdict

The Model Factory is **substantially implemented at the library and artifact
contract level, but the epic is not completely delivered as an end-to-end
operator workflow**.

Phases A-D and the later generic-action/navigation work have concrete modules
and broad tests. Eight of the epic's ten final success criteria are directly
verified; the configuration-breeding criterion is verified through Python but
only partially exposed to users, and the durable confirmation/search workflow
is similarly missing CLI orchestration. CI also advertises two invariant gates
that are currently skipped or unable to start because their fixtures are
absent.

The generic corpus currently builds only with the audited branch's `f91b2c7`
fix. That commit is one revision ahead of `origin/main` and must land before the
generic-corpus result can be claimed for the default branch.

Phase E (surrogate/Bayesian search) is explicitly deferred by the epic and is
not counted as missing implementation.

## Documentation changes required by this audit

| Document | Action | Reason |
| --- | --- | --- |
| `docs/how-to/using-model-factory.md` | Create | Give operators an accurate CLI/library workflow, artifact map, and failure triage path. |
| This audit | Create | Preserve criterion-level evidence, test results, revision scope, and discovered gaps. |
| `docs/v2/proposal-model-factory.md` | Modify | Mark it as an implemented design proposal, link current docs, and stop presenting library-only operations as existing CLI commands. |
| `docs/v2/README.md` | Modify | Make the operator guide and implementation status discoverable. |
| `docs/v2/04-contracts-and-data-flow.md` | Modify | Add the factory checkpoint, corpus, lineage, registry, and state-machine boundary. |
| `docs/v2/phases/phase-2-predictive-cortex.md` | Modify | Connect Cortex training to its current reproducible experiment layer. |
| `README.md` | Modify | Link the existing command table to the full guide and identify library-only advanced operations. |

## Evidence by implementation phase

| Epic phase | Status | Implementation evidence | Verification evidence |
| --- | --- | --- | --- |
| A - contracts and manifests | Verified | `contracts.py`, `spec.py`, `artifacts.py`, `naming.py`, `corpus.py` | Contract/artifact/spec/corpus/naming tests; canonical hashes, immutable manifests, environment provenance, content-hashed corpora, and deterministic names covered. |
| B - resumable checkpoints | Verified | `checkpoint.py`, `budget.py`, `state.py`, continuation logic in `runner.py` | Checkpoint, continuation, exact-resume, deadline, state-transition, locking, and stale-worker tests pass. |
| C - runner and A/B comparison | Verified with one portability defect | `runner.py`, `comparison.py`, `promotion.py`, `registry.py`, `confirmation.py` | Paired comparison, gates, population registry, seed confirmation, sealed-test accounting, CLI, and runner tests pass except `AUD-212-02`. |
| D - budgeted search | Verified as Python APIs; partial CLI | `genome.py`, `search.py`, `breeding.py`, `population.py` | Random/LHS proposals, duplicate prevention, successive halving, crossover/mutation, parent/weight-donor lineage, diversity, and compute-ledger tests pass. No `factory search` or `factory breed` CLI exists. |
| Generic action effects | Verified on audit branch | `action_effects.py`, corpus/nursery integration, generic corpus spec | Action label, event-stratification, scenario-mix, and corpus tests pass. Commit `f91b2c7` fixes sealed-test seed handling and is not on `origin/main`. |
| Goal navigation | Verified | goal representation, navigation oracle/reward, navigation scenarios, navigation metrics, goal-navigation corpus spec | Goal anti-gaming, A*, potential reward, scenario solvability/disjointness/replanning, retention, and navigation test suites pass. |
| Factory CI invariants | Partial | `.github/workflows/ci.yml`, `nightly-factory.yml`, schema validator | Workflows and validator exist, but `AUD-212-03` and `AUD-212-04` leave the advertised end-to-end/schema gates ineffective. |

All linked implementation issues #213-#242 were closed at audit time. Closed
issue state was treated as provenance, not proof; the status above comes from
the checked-out code and executable tests.

## Final success criteria

| # | Epic success criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Fresh baseline with unique immutable run ID | Verified | Artifact allocation and runner tests; manifests are locked before training. |
| 2 | Clone best-validation checkpoint into controlled A/B children | Verified | Continuation, runner, CLI clone, and comparison tests. |
| 3 | Resume an interrupted child exactly | Verified | Deterministic 3+3 versus 6 epoch equivalence tests. |
| 4 | Reject incompatible continuation before GPU work | Verified | Architecture/data/training compatibility tests, including action vocabulary, layout, and backbone kwargs. |
| 5 | Inspect lineage graph and complete effective config | Verified | `ccr factory show` and `lineage`; both also succeeded against a real failed run in this workspace. |
| 6 | Retain a bounded diverse population and breed an explicit-lineage child | Partial | Population and breeding APIs/tests pass; no `ccr factory breed` command exists. |
| 7 | Reproducible cosmetic lineage name | Verified | Deterministic naming, ancestry surname, collision, and identity-separation tests. |
| 8 | Select a leading champion from paired validation evidence | Verified | Paired statistics, promotion gates, registry, and CLI promotion tests. |
| 9 | Confirm across training seeds and once on sealed test | Partial operator workflow | Confirmation and final-test APIs/tests pass, and `factory test` exists; no CLI command creates the required seed-confirmation artifact. |
| 10 | Recover an interrupted worker without promoting partial output | Verified | Atomic state, stale recovery, retry, terminal status, and promotion-eligibility tests. |

## Test record

The focused audit selected all `test_model_factory_*.py` modules, the factory
CLI tests, generic action-effect/corpus tests, goal/navigation tests, and the
core-only motor import regression:

| Shard | Result |
| --- | --- |
| contracts, artifacts, checkpoint, budget, state, spec, naming | 141 passed, 1 skipped |
| corpus, comparison, promotion, registry, confirmation | 105 passed |
| genome, proposals, halving, breeding, population, generic/navigation | 287 passed |
| continuation, resume, runner, CLI | 64 passed, 1 failed |

Total focused result: **597 passed, 1 skipped, 1 failed**. The one failed test
is the integration guard described by `AUD-212-02`. Its delegated legacy suite
ran separately with **69 passed and 3 failed**.

Manual checks also confirmed:

- `ccr factory --help` exposes baseline, clone, compare, promote, show,
  lineage, corpus, and test - but not search, breed, or seed confirmation;
- `ccr factory show` and `ccr factory lineage` correctly inspected
  `Crafter-20260801T081206-5a7215` and reported its failed terminal state;
- resolving the current dirty baseline spec accepts horizons `[1, 2]` with a
  `rollout.t+4...` selection metric, reproducing `AUD-212-01` without training.

## Missing implementation and bugs

### AUD-212-01 - selection metric horizon is not validated before training

**Severity:** high (avoidable GPU cost and late failure)

`spec.validate()` checks only that `evaluation.selection_metric` matches a
metric-path regular expression. It does not ensure that `t+N` is present in
`data.horizons_ticks`. The workspace's run
`Crafter-20260801T081206-5a7215` trained for about 12 minutes and then failed:

```text
ValueError: selection metric 'rollout.t+4.model_over_copy_last_mse'
horizon t+4 was not evaluated (available: [1, 2])
```

Fix by cross-validating prediction metric horizons during spec resolution and
adding a regression test proving the invalid spec fails before artifact
allocation or model construction. The current mismatch is in a pre-existing
dirty edit to `specs/crafter-baseline.yaml`; it was not introduced by this
documentation work.

### AUD-212-02 - GPU-enabled test run mixes CUDA models with CPU inputs

**Severity:** medium (GPU developer/CI portability)

With CUDA available, `training.device: auto` leaves the returned model on
CUDA. Three legacy tests create CPU tensors and call the model/encoder directly,
causing `torch.FloatTensor` versus `torch.cuda.FloatTensor` failures. The
Model Factory runner's compatibility guard therefore fails on a GPU-enabled
machine even though the same lane passes in CPU-only CI.

The most conservative fix is to make device intent explicit in these unit
tests (`device="cpu"`) or construct their tensors on the model device. Keep
separate device-selection tests for the real CUDA path.

### AUD-212-03 - nightly factory smoke workflow has no input fixtures

**Severity:** high (advertised end-to-end CI cannot run)

`.github/workflows/nightly-factory.yml` invokes
`.github/fixtures/micro-corpus.yaml` and `micro-baseline.yaml`, but the
`.github/fixtures` directory does not exist. Since `ccr factory --help`
succeeds, the detection step does not skip the job; the first corpus-build
step fails on a missing file.

Add small deterministic corpus/baseline fixtures, avoid hard-coded assumptions
about generated run IDs, and execute the workflow locally or through
`workflow_dispatch` before treating the nightly gate as active.

### AUD-212-04 - artifact schema CI gate is wired but permanently skipped

**Severity:** medium (schema drift is not mechanically prevented)

The validator script exists, but neither `.github/schemas` nor
`tests/fixtures/factory` exists. `ci.yml` therefore prints a skip message. Add
schemas and golden fixtures for at least trial spec, contracts, lineage,
corpus manifest, registry, state, and checkpoint metadata; then make absence a
failure rather than a skip.

### AUD-212-05 - advanced and durable workflows lack CLI orchestration

**Severity:** medium (epic examples are not executable as written)

The proposal shows `factory search` and `factory breed`, but the parser has no
such subcommands. The sealed-test CLI also requires
`metrics/seed_confirmation.json`, while no CLI command runs
`confirm_across_seeds(...)`. The Python APIs are implemented and tested, so
this is an integration/usability gap rather than missing algorithms.

Add `factory search`, `factory breed`, and `factory confirm-seeds` commands,
including dry-run/spec preview, stable campaign identifiers, budget reporting,
and tests that exercise the exact documented lifecycle.

### AUD-212-06 - recorded loss-weight aliases can be ignored by training

**Severity:** high (the effective trainer can diverge from the hashed contract)

`TrainingContract.loss_weights` accepts an unrestricted mapping, but
`runner._training_config()` forwards only keys that exactly match
`ActionWorldModelConfig` dataclass fields. The shipped baseline uses the short
keys `pixel`, `latent`, and `semantic`; changing one changes
`training_contract_hash`, but those keys are silently discarded and the
trainer continues with its defaults. The CLI help similarly suggests
`loss_weights.closed_loop_pixel`, which is not a trainer field.

Define and validate a versioned loss-weight schema, reject unknown keys, change
the baseline/help text to full names such as `pixel_loss_weight` and
`closed_loop_pixel_loss_weight`, and add a runner test proving each resolved
weight reaches `ActionWorldModelConfig`.

## Exit recommendation

Do not close the epic solely because its child issues are closed. Treat the
core research implementation as complete after `f91b2c7` lands, but keep the
epic's operator/CI exit gate open until `AUD-212-01`, `AUD-212-03`,
`AUD-212-04`, and `AUD-212-05` are addressed. `AUD-212-02` should be fixed as a
GPU portability regression in the same cleanup window, and `AUD-212-06` must
be resolved before experiment hashes are treated as faithful records of the
effective optimization procedure.
