# Factory task and model boundaries (#285)

Factory's public entry point remains `run_trial`, including the existing CLI
callers. Unchanged `model-factory-spec-v1` recipes select the lazy `crafter`
adapter. The adapter delegates to the existing Factory implementation at its
current path, preserving its trainer, evaluator, optimizer continuation, Clinic
exports and binary checkpoints. It does not reinterpret old contracts or add
backend fields to old serialized specs. This compatibility adapter retains the
legacy lifecycle; new backends use the neutral lifecycle described below.

## Neutral backend authors

Use `format: model-factory-task-spec-v1` and `model.backend` for a registered
backend. This deliberately has a distinct identity from the legacy format.
Declare `organism`, `data.corpus_id`, and backend-specific data/model/training/
evaluation settings. There are no pixel shapes, action vocabularies, nursery
scenarios, reward heads or terminal heads in these contracts. Shared defaults
are CPU, fp32, deterministic seed 0 and a 60-second cooperative execution budget.
The backend must implement `validate` and reject all unsupported configuration
before Factory allocates a run. Evolution/search, specialized promotion,
comparison and Clinic visualization for new domains remain later issue scope.

Register a `BackendRegistration(TaskIdentity(...), factory, capabilities, extra)`
from trusted Python integration code. A factory can be a callable or a lazy
`package.module:attribute` string. Specs may only name registered backends;
they cannot provide executable import paths. Unknown names fail clearly.
Registration does not import the implementation. `extra` names the installation
extra to show if loading that backend fails. The built-in Crafter adapter uses
`cognitive-runtime[neural]`; neutral contracts require no additional packages.
No text model/provider extra is introduced before an implementation needs it.

`TaskBackend` defines these boundaries:

1. `prepare` returns immutable data/model identities, a training batch and a
   separate validation batch. `prepare_task_corpus` checks their identities.
2. `build` constructs a fresh model from the versioned model definition.
3. `fit` receives training inputs/targets, effective supported configuration and
   `TaskControl`; a resume-capable backend must restore its complete optimizer,
   RNG and cursor state, while clone loads weights with fresh training state.
   Call `control.checkpoint()` at resumable boundaries during fit. Factory
   persists state and its header under checkpointing/running transitions. A killed
   worker can continue from that boundary; work after it may need replay.
4. `predict` receives **exactly** `InferenceInput`: feature schema, opaque sample
   IDs and finite numeric feature vectors. It receives no training/evaluation
   container or generic metadata mapping. `require_inference_input` rejects
   subclasses and label-bearing containers; slots forbid adding metadata fields.
5. `evaluate` receives predictions and the separate validation labels.
6. `save/load` serialize/restore JSON-compatible backend state. Factory binds the
   checkpoint bytes to a sidecar digest, task identity and model/data/training
   hashes, validates continuation before reconstruction, and never unpickles a
   neutral checkpoint. Constructors are selected through the registry.

`FeatureBuilder` separates training-only fitting from inference transformation.
`TemporalModel` describes the existing `TemporalBackbone` methods without
importing torch: `initial_state`, `step`, `readout`, `forward_sequence`. Numeric
features can use these backbones directly without fake pixels or actions.

This interface enforces structural separation, not proof of causal data lineage:
a malicious producer could encode a label as a number. Future as-of snapshots,
feature provenance and leakage tests must establish the semantic guarantee.
No financial label/settlement store or forecasting implementation is supplied by
this change; those are separate children of #283.

## Runs, failures and continuation

New backends use the existing artifact allocator, naming, lineage and persisted
state machine. `TrialResult` stays the public result. Neutral runs write
`checkpoints/last.json`, its compatibility sidecar, `metrics/validation.json`,
`metrics/budget_report.json` and `experiment_report.json`. The report includes
artifact references with exact byte hashes and failure/cancellation outcomes.
Existing stream and corpus formats are not rewritten.

Clone/resume rejects different task/domain/backend/version/model definitions.
Clone additionally requires matching data; resume requires matching data and
training identities and a stale, still-active original run. Terminal runs are
not reopened. Resume preserves initial manifests and uses the existing atomic
stale-worker claim. Failed and cancelled outcomes are not completed runs.
The task's cooperative budget is checked between lifecycle stages and via
`TaskControl.check()` inside fitting. This is not a hard process timeout; a
backend that does not yield needs an external process deadline. Resource-bound
experiment launching belongs to #298.

## Validation and limits

All new data in tests are project-authored MIT synthetic values (seed/source,
rights, intended edge case and canonical data hash are declared in the fixture).
No raw market/provider payloads, model fetches or credentials are involved.

Default offline contracts and two fake domains through actual Factory:

```sh
python -m pytest tests/test_model_factory_task_backends.py -q
python .github/scripts/run_core_tests.py --report /tmp/core-tests.json
PYTHONHASHSEED=12345 python .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/contracts-seed-12345.json
python .github/scripts/check_core_budget.py /tmp/core-runtime
```

Collect both execution reports under `/tmp/core-runtime` for the aggregate gate.
The unchanged hosted workflow enforces the 120-second aggregate and 5-minute job
cap, and rejects optional frameworks/network/model downloads in the base lane.

With CPU torch installed, explicit optional compatibility checks:

```sh
python -m pytest tests/market/extended/test_task_backend_compatibility.py --run-market=extended -q
```

This invokes the real six-stage legacy CLI canary (180-second process bound)
and a one-step synthetic GRU fit/predict/save/load/clone through neutral Factory.
The GRU case checks the temporal interface without ActionWorldModel construction;
it is a compatibility check, not a forecasting result. Both are `synthetic_only`.
See the delivery record for checks actually executed and outstanding evidence.
