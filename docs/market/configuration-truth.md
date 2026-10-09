# Configuration truth: Factory and ActionWorldModel

Implements [CCR-MKT-01 / #284](https://github.com/Brain-Crumbs/CCR/issues/284),
under [#283](https://github.com/Brain-Crumbs/CCR/issues/283) and the existing
[Model Factory #212](https://github.com/Brain-Crumbs/CCR/issues/212).
This is shared Crafter/Factory infrastructure, not a market-specific trainer.
Validation status: **synthetic_only** (offline fixtures and bounded CPU checks).
No financial data, external model downloads, credentials, orders, or paid APIs.

## Requested → resolved → effective

1. **Requested**: the input spec supplied to `run_trial`, before defaults. A
   caller supplying an already-resolved `ExperimentSpec` supplies that as its
   request; the runner cannot recover omitted original template text.
2. **Resolved**: the immutable spec, with the optimizer version and every
   supported optimizer constructor default materialized. Its existing
   `training_contract_hash` includes the complete optimizer block.
3. **Effective**: `effective_config.json` stores both stages above plus the
   actual optimizer class, every observed parameter-group setting, ordered
   model parameter names/shapes, scheduler (`null`), precision (`fp32`),
   resolved execution device, model backend, applied determinism policy,
   actual checkpoint selection metric, and full trainer configuration.
   Its `effective_config_hash` is canonical SHA-256 over the effective block.
   Tuples and frozen mappings normalize to JSON arrays/objects before hashing.

Factory checkpoints and their payload-bound JSON headers also store observed
optimizer evidence and the full effective-config hash. Checkpoint payloads
retain the full effective configuration. A run's manifest is written before
training. Actual trainer output records the optimizer groups again, and
checkpoint save rejects a constructor/settings mismatch with the new contract.

## Supported optimizer settings

The new optimizer version is `torch-optimizer-v2`. Supported names are exactly
`adam` and `adamw`, mapped directly to `torch.optim.Adam` and `torch.optim.AdamW`.
Both trainer objectives, checkpoint reconstruction, and Factory use the same
constructor function. There is no fallback for an unknown name.

Defaults:

```yaml
training:
  optimizer:
    format: torch-optimizer-v2
    name: adamw
    lr: 0.0003
    betas: [0.9, 0.999]
    eps: 0.00000001
    weight_decay: 0.00001
    amsgrad: false
```

These are Factory defaults for both supported names unless overridden; choose
`weight_decay: 0.0` explicitly for unregularized Adam. Unsupported keys (including
custom parameter groups, `momentum`, `fused`, or `grad_clip`) fail closed.
Learning rate, epsilon and decay must be finite/nonnegative; both betas must be
in `[0, 1)`. `amsgrad` must be a boolean. Runtime implementation-specific group
defaults are observed and hashed, so an environment change can require a new run.

## Migration and legacy reading

Historically, Factory recorded AdamW/weight decay while actually constructing
Adam with zero weight decay. Resolving a template now adds `torch-optimizer-v2`
and runs the real requested optimizer. This deliberately changes the training
contract hash. It is **not** exact continuation of those historical runs.

Direct `ActionWorldModelConfig`/nursery callers preserve historical Adam through
an explicit serialized `optimizer_behavior="legacy-adam-v1"` default: `lr` is
used, decay is zero, betas are `(0.9, 0.999)`, epsilon is `1e-8`, and AMSGrad is off.
To opt in directly, set `optimizer_behavior="torch-optimizer-v2"` and supply the
resolved `optimizer` block; its learning rate must agree with `cfg.lr`.
Historical model artifacts remain inspectable/cloneable without claiming their
recorded optimizer was effective. Historical contract objects remain readable
without rewriting their stored hashes.

## Fail-closed execution

Validation runs before Factory corpus/model allocation. The initial supported
execution surface is intentionally narrow:

- No scheduler: any non-null scheduler fails. There is no hidden scheduler state.
- Only FP32. Other requested precisions fail, and Factory explicitly constructs
  FP32 models rather than relying on the process default dtype.
- Step budgets and early-stopping policies are unsupported; use existing epoch,
  wall-time budget and checkpoint-cadence controls.
- Loss weights have a finite, nonnegative allowlist. Unknown weights cannot
  inject trainer settings such as `lr` or `device`. Duplicate aliases fail.
- Objective-inactive losses fail: windowed closed-loop/direct weights cannot be
  specified for autoregressive training, and autoregressive rollout weights
  cannot be specified for windowed training.
- Factory currently has no semantic head. Explicit `semantic` or
  `semantic_loss_weight` overrides, including the semantic gene in historical
  `generic_action_effects_v2`, fail instead of silently doing nothing. The
  historical genome schemas and their pinned hashes remain readable. New CLI
  search/breed defaults use pinned `generic_action_effects_v3`, which removes
  semantic and marks transition balance windowed-only. Only objective-active
  genes are applied to proposed/bred training specs. Existing v2 proposals
  require a new v3 proposal; do not rewrite old artifacts.
- Autoregressive training uses one episode per step: omitted batch size resolves
  to 1; other explicit batch sizes fail. Transition-balance policies are also
  inactive in that objective and fail if supplied.
- Unknown backbone kwargs and implicit attention-head fallback fail. Layer
  counts must be positive integers. GRU-specific unused kwargs fail.
- The determinism flag is applied to training and the caller's prior setting
  (including warn-only mode) is restored. An omitted determinism seed resolves
  to the training seed; contradictory seeds fail.

The v3 genome content hash is
`8b4659dd6cf5097469530d466dbb610bebbeccdd0d5330f34f998a38809c49c3`.

The effective record exposes legacy trainer defaults that are irrelevant under
one objective as part of the complete dataclass, but those values are not
accepted as explicit Factory search overrides. It also records the runner's
actual validation checkpoint metric. The legacy
`checkpoint_selection_policy={metric: validation_loss, mode: min}` is a supported
alias for `evaluation.selection_metric`; only that alias or the exact effective
metric/direction is accepted, and other policies fail closed.

## Exact resume vs a new run

Exact resume compares training/data/architecture contracts, the observed
optimizer constructor/groups, and the full effective configuration before
loading training state. Loading state must not overwrite a changed requested
learning rate, decay, betas, or class. Stored state groups must agree with the
observed evidence, too. The saved run manifest must match the current effective
configuration and hash; missing proof is a hard failure.

Older Factory artifacts lacking effective evidence remain readable, but exact
resume is rejected with guidance to use `clone` or `fine_tune`. Those modes
allocate a separately recorded run and establish a new optimizer trajectory.
New public checkpoint callers must supply expected `effective_config` when the
checkpoint carries its hash. Generic, non-Factory checkpoint callers that save
only optimizer evidence still get strict optimizer identity checks. Resume
optimizers must own the supplied model's actual parameters and have matching
settings before restoration; an arbitrary throwaway optimizer is no longer safe.

## Verification

Default/offline tests: `python -m pytest tests/market/test_effective_config.py -q`.
These use authored tiny parameter/constructor spies and JSON roundtrips, import
no torch, and do not train or fetch anything. Optional CPU tests:
`python -m pytest tests/market/test_effective_config_neural.py -q`.
They lazily import torch in test bodies and check both Adam and AdamW, both AWM
objectives, exact resumed weights, and rejected mismatches. Their 4×4 constant
RGB fixture is generated inline, seed 7, project-authored under MIT. Its SHA-256
is `2be67a06c2b553ff910234d8c14b3438e8ecbb704fd392857b2ca40687f93724`.

The existing Factory runner, checkpoint and resume suites exercise real bounded
Crafter runs, effective manifests and resumed groups. Default CI stays torch-free;
core and Factory jobs have a five-minute cap. Full repository runtime depends on
host speed; report measured aggregate runtime separately from the tiny tests.
