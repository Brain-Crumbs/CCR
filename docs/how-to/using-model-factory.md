# Using the Model Factory

The Model Factory is the reproducible experiment layer around the Predictive
Cortex. It freezes data, resolves an immutable experiment specification,
records checkpoint lineage, compares candidates on paired validation episodes,
and promotes only candidates that pass the declared gates.

For the design rationale, see
[the original proposal](../v2/proposal-model-factory.md). For verified coverage
and known gaps, see the
[implementation audit](../v2/model-factory-implementation-audit.md).

## Install and inspect the command surface

Training and checkpoint operations need PyTorch:

```bash
pip install -e ".[dev,neural]"
ccr factory --help
```

The implemented CLI commands are:

| Command | Purpose |
| --- | --- |
| `ccr factory corpus build <spec>` | Build and freeze a quality-gated train/validation/test corpus. |
| `ccr factory baseline <spec>` | Resolve and launch any `fresh`, `clone`, `resume`, or `fine_tune` spec. |
| `ccr factory clone <run> --set path=value` | Create a controlled clone or fine-tune child from an existing run. |
| `ccr factory compare <baseline> <candidate>...` | Produce paired validation comparisons. |
| `ccr factory promote <run>` | Apply promotion gates and update a registry slot. |
| `ccr factory test <run>` | Perform the one-time sealed-test action after seed confirmation. |
| `ccr factory show [run]` | Show the resolved spec, hashes, name, and terminal status. |
| `ccr factory lineage [run]` | Show configuration parents and weight-donor ancestry. |

Search proposal generation, successive halving, breeding, and independent
training-seed confirmation are implemented as Python APIs, but do not yet have
CLI subcommands. In particular, `ccr factory test` requires a prior successful
`confirm_across_seeds(...)` Python call. This is tracked as `AUD-212-05` in the
implementation audit.

## 1. Build a frozen corpus

The generic action-effects and goal-navigation declarations are versioned
input templates:

```bash
ccr factory corpus build specs/corpora/generic-action-effects-v1.yaml
ccr factory corpus build specs/corpora/goal-navigation-v1.yaml
```

A successful build writes:

```text
corpora/<organism>/<corpus_id>/
  corpus_spec.json
  corpus_manifest.json
  quality_report.json
  split_overlap_report.json
  train_sessions.json
  validation_sessions.json
  test_sessions.json
```

`corpus_manifest.json` contains content hashes, not only cache identities. A
completed corpus is immutable: change its declaration only by choosing a new
`corpus_id`. Controlled trials fail on a missing or modified session rather
than silently recording replacement evidence.

## 2. Validate the experiment template

Start from `specs/crafter-baseline.yaml`. Before spending GPU time, ensure:

- `data.corpus_id` resolves to the frozen corpus;
- every `rollout.t+N` or `direct.t+N` selection metric has `N` in
  `data.horizons_ticks`;
- `training.device` is `auto`, `cpu`, or a valid PyTorch device such as
  `cuda`;
- the selection metric direction and promotion margins match the objective;
- train, validation, and sealed-test sessions remain disjoint.

The horizon/selection-metric relationship is not yet rejected by spec
validation (`AUD-212-01`), so this manual check is currently important.
Corpus `generator.horizons` and trial `data.horizons_ticks` are separate today:
the former is persisted in the frozen `DataContract`, while the latter builds
the model's horizon heads and controls trial evaluation. Neither is inherited
from the other.

Loss-weight keys must use the complete `ActionWorldModelConfig` field name,
such as `closed_loop_pixel_loss_weight`, `pixel_loss_weight`, or
`latent_loss_weight`. Short aliases such as `pixel` are currently accepted
into the recorded contract but ignored by the trainer (`AUD-212-06`).

Resolve without launching training when reviewing a template:

```python
from cognitive_runtime.training.model_factory import load_spec, resolve

spec = resolve(load_spec("specs/crafter-baseline.yaml"))
print(spec.to_dict())
```

## 3. Launch and inspect a baseline

```bash
ccr factory baseline specs/crafter-baseline.yaml
ccr factory show
ccr factory lineage
```

Every launch receives a unique `run_id`. The resolved spec, contracts,
execution environment, display name, lineage, and initial state are written
before training begins. The important identities are `run_id`, the three
contract hashes, and checkpoint SHA; `display_name` is cosmetic.

## 4. Create a controlled A/B comparison

Change one training-contract field at a time:

```bash
ccr factory clone <baseline-run> \
  --set training.loss_weights.closed_loop_pixel_loss_weight=0.125
ccr factory clone <baseline-run> \
  --set training.loss_weights.closed_loop_pixel_loss_weight=0.500
ccr factory compare <baseline-run> <child-a> <child-b>
```

The comparison uses the same validation sessions and reports paired
per-episode deltas, an interval, win rate, episode count, and the declared
metric. Training loss is diagnostic evidence, never a promotion metric.

## 5. Resume or fine-tune

`mode: resume` reopens the interrupted run and requires identical
architecture, data, and training contracts. It restores model, optimizer,
scheduler, trainer, and RNG state. `clone` and `fine_tune` create new runs,
restore compatible weights, and record a new training trajectory.

Use an explicit parent block containing the parent run, checkpoint filename,
and checkpoint SHA. Compatibility is checked before the parent weights are
loaded.

## 6. Promote, confirm, and sealed-test

Ordinary promotion is validation-only:

```bash
ccr factory promote <candidate-run>
```

A durable champion requires three distinct operations:

1. paired validation comparison and ordinary promotion eligibility;
2. `confirm_across_seeds(...)` over at least two independent training seeds;
3. `ccr factory test <candidate-run>` once against the sealed split, followed
   by `ccr factory promote <candidate-run> --durable`.

Until a CLI command is added, run seed confirmation directly:

```python
from cognitive_runtime.training.model_factory.confirmation import (
    confirm_across_seeds,
)

confirmation = confirm_across_seeds(
    "<candidate-run>",
    seeds=(11, 29, 47),
    organism="Crafter",
)
assert confirmation.confirmed
```

The registry records the leading champion, retained population, decisions,
lineage references, and sealed-test-use budget. It stores checkpoint references
rather than copied weights.

## 7. Search and breeding Python APIs

The bounded search implementation lives in
`cognitive_runtime.training.model_factory.search`:

```python
from cognitive_runtime.training.model_factory import load_spec, resolve
from cognitive_runtime.training.model_factory.genome import get_schema
from cognitive_runtime.training.model_factory.search import (
    propose,
    run_successive_halving,
)

schema = get_schema("generic_action_effects_v1")
base_spec = resolve(load_spec("specs/crafter-baseline.yaml"))
proposals = propose(base_spec, schema, 8, seed=7, method="lhs")
report = run_successive_halving(
    base_spec,
    schema,
    n=8,
    seed=7,
    budgets=(50, 150, 500),
)
```

Genetic configuration breeding lives in
`cognitive_runtime.training.model_factory.breeding`. It records two
configuration parents, one explicit weight donor, the genome diff, mutations,
repairs, generation, seed, and compute ledger. It does not average weights.

## Artifact and failure triage

Each run is stored under `runs/<organism>/<run_id>/`. The state machine uses
`state.json`; `completed`, `budget_exceeded`, `failed`, and `cancelled` are
terminal. Only a completed, fully evaluated run is promotable.

Useful first checks are:

```bash
ccr factory show <run-id>
ccr factory lineage <run-id>
```

Then inspect:

- `state.json` for the terminal reason and transition history;
- `execution.json` for source, dirty-tree, package, device, and determinism
  provenance;
- `contracts.json` for compatibility and evidence identity;
- `metrics/validation.json` and `metrics/comparison.json` for selection
  evidence;
- `checkpoints/last.pt` for resumable partial work and
  `checkpoints/best-validation.pt` for the selected candidate state;
- `experiment_report.json` for the resolved training/evaluation summary.
