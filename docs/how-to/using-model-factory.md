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
| `ccr factory search <spec>` | Evaluate, filter, retain, and breed candidate populations. |
| `ccr factory search <spec> --genome architecture` | Nested architecture (NAS) campaign: evolve architectures, each scored by its own inner training-gene campaign. |
| `ccr factory clone <run> --set path=value` | Create a controlled clone or fine-tune child from an existing run. |
| `ccr factory breed <parent-a> <parent-b>` | Breed two compatible completed runs and launch the explicit-lineage child. |
| `ccr factory compare <baseline> <candidate>...` | Produce paired validation comparisons. |
| `ccr factory promote <run>` | Apply promotion gates and update a registry slot. |
| `ccr factory test <run>` | Perform the one-time sealed-test action after seed confirmation. |
| `ccr factory show [run]` | Show the resolved spec, hashes, name, and terminal status. |
| `ccr factory lineage [run]` | Show configuration parents and weight-donor ancestry. |

Independent training-seed confirmation remains a Python API. In particular,
`ccr factory test` requires a prior successful `confirm_across_seeds(...)`
call. The remaining CLI gap is tracked as `AUD-212-05` in the implementation
audit.

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

Historical v1 declarations contain a `generator.horizons` key, but it never
affected recording or the episode-cache identity. It is accepted as legacy
input and omitted from new corpus contracts. Existing v1 recordings can be
reused by experiments with any horizon set their recorded temporal coverage
can support.

## 2. Validate the experiment template

Start from `specs/crafter-baseline.yaml`. Before spending GPU time, ensure:

- `data.corpus_id` resolves to the frozen corpus;
- the trial `data.horizons_ticks` and built model use the same ordered horizon
  set;
- every frozen train/validation episode has enough recorded frames and tick
  span for the requested horizons and training window;
- every `rollout.t+N` or `direct.t+N` selection metric has `N` in that set;
- `training.device` is `auto`, `cpu`, or a valid PyTorch device such as
  `cuda`;
- the selection metric direction and promotion margins match the objective;
- train, validation, and sealed-test sessions remain disjoint.

Both relationships are enforced before GPU work: spec resolution rejects a
selection metric whose `t+N` is absent from the experiment horizons, and trial
launch checks the loaded recordings' frame counts and recorded tick spans.
`ccr factory show` reports trial/model horizon alignment and whether the corpus
contract includes temporal-coverage metadata.

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

## 7. Search and breeding

Preview the population layout and exact fresh proposals without loading a
corpus or starting PyTorch:

```bash
ccr factory search specs/crafter-baseline.yaml \
  --schema generic_action_effects_v2 \
  --population-size 8 --populations 4 --mutation-rate 0.2 \
  --seed 7 --run-id-prefix crafter-evo-7 --dry-run
```

Remove `--dry-run` to launch the evolutionary campaign. The initial population
uses the selected random/LHS sampler. Each population then:

1. evaluates every new candidate on the fixed validation split;
2. removes incomplete candidates and candidates that fail the rollout quality filters;
3. sorts the remainder by `evaluation.selection_metric` and retains at most the best half;
4. carries those checkpoints forward without retraining and breeds children until the next population reaches `--population-size`.

`--populations` counts the initial population. One survivor is allowed to
self-breed with mutation; zero survivors ends the campaign. The JSON report
records every quality-filter and retention decision, offspring IDs, total
trials/epochs, final survivors, and the optional champion comparison. Candidate
run IDs are stable under the chosen prefix:
`crafter-evo-7-p<population>-c<candidate>`.

To continue from completed compatible runs, add repeatable `--seed-run`
arguments. Seeded runs occupy the first population-zero slots, reuse stored
validation evidence and checkpoints without retraining, and leave the remaining
slots for fresh proposals. Architecture, corpus/data contract, non-genome
training settings, evaluation contract, validation episodes, and genome values
must match. Child generation numbers continue after the highest seeded
generation.

```bash
ccr factory search specs/crafter-baseline.yaml \
  --seed-run <prior-champion-a> --seed-run <prior-champion-b> \
  --population-size 8 --populations 4 \
  --run-id-prefix crafter-evo-continued
```

For reproducibility of an older campaign, explicitly passing
`--budgets 50 150 500` selects the legacy successive-halving engine.

## 7a. Architecture search (NAS)

`--genome architecture` evolves the *architecture* instead of the training
procedure. It is a nested campaign: an outer evolutionary loop over
architectures, where each candidate is scored by running a complete inner
training-gene campaign (the engine from section 7, unchanged) with that
architecture held fixed. Every architecture therefore gets the same inner
search budget before it is judged, so its score is a property of the
architecture rather than of one arbitrary hyperparameter setting.

```bash
ccr factory search specs/crafter-baseline.yaml \
  --genome architecture \
  --outer-schema architecture_search_v1 --schema generic_action_effects_v1 \
  --outer-population-size 4 --outer-populations 2 \
  --population-size 3 --populations 1 \
  --seed 7 --run-id-prefix crafter-nas-7 --dry-run
```

`--population-size`/`--populations`/`--mutation-rate` keep their meaning as
the *inner* campaign's parameters; the outer loop has its own `--outer-*`
flags. Run IDs extend the same convention with the outer coordinates:
`crafter-nas-7-arch<generation>-a<architecture>-p<population>-c<candidate>`.

**Cost multiplies.** The bound is
`outer-population-size × outer-populations × population-size × populations`
full trainings — the 4×2×3×1 above is already 24 — so the command prints
that bound (and its wall-clock equivalent, from the spec's
`training.max_training_seconds`) to stderr on every invocation, dry run or
not, before anything launches. Preview with `--dry-run` first.

The evolvable architecture genes are deliberately few:
`backbone_preset` (one categorical gene over curated whole
`{backbone, backbone_kwargs}` presets, so a transformer never inherits a
convolution's kwargs), `latent_width`, `hidden_dim`, `action_embed_dim` and
`context_length`. Corpus-derived `ArchitectureContract` fields are not free
choices and are not evolvable; `reconstruction_size` is excluded from v1
because changing decoder output resolution across candidates would make raw
pixel MSE comparisons unfair.

A gene the chosen backbone ignores is snapped to its declared default before
a candidate is built — `context_length` is a windowed-backbone ring-buffer
capacity that the GRU ignores, so GRU candidates always record the default.
Without that, four GRU genomes differing only in `context_length` would build
one identical model but spend four full inner budgets and carry four
different `architecture_hash` values, and the outer loop would rank the
training noise between them as architectural signal.

An architecture-changing child **trains from scratch** (`mode: fresh`, no
`parent` block): there is no compatible checkpoint to inherit, and the
factory never averages or splices weights across parents. Its lineage still
records both configuration parents and the fitter one as `weight_donor` —
a provenance label, not a checkpoint reference.

Breed two compatible completed parents after inspecting the child first:

```bash
ccr factory breed <parent-a> <parent-b> \
  --schema generic_action_effects_v2 --seed 19 --dry-run
ccr factory breed <parent-a> <parent-b> \
  --schema generic_action_effects_v2 --seed 19
```

The tier is inferred from both parents' budget reports. Use `--tier fast` only
when the reports are unavailable. The launched child records two configuration
parents, exactly one checkpoint weight donor, the genome diff, mutations,
repairs, generation, and seed. The retained population tracks the associated
compute ledger separately. Breeding never averages weights.

The same operations remain available as Python APIs:

The bounded search implementation lives in
`cognitive_runtime.training.model_factory.search`:

```python
from cognitive_runtime.training.model_factory import load_spec, resolve
from cognitive_runtime.training.model_factory.genome import get_schema
from cognitive_runtime.training.model_factory.search import (
    propose,
    run_evolutionary_search,
)

schema = get_schema("generic_action_effects_v2")
base_spec = resolve(load_spec("specs/crafter-baseline.yaml"))
proposals = propose(base_spec, schema, 8, seed=7, method="lhs")
report = run_evolutionary_search(
    base_spec,
    schema,
    population_size=8,
    population_count=4,
    seed=7,
    mutation_rate=0.2,
    seed_run_ids=("prior-champion-a", "prior-champion-b"),
)
```

The nested architecture campaign lives beside it, in
`cognitive_runtime.training.model_factory.architecture_search`:

```python
from cognitive_runtime.training.model_factory.architecture_genome import (
    get_architecture_schema,
)
from cognitive_runtime.training.model_factory.architecture_search import (
    estimate_cost,
    preview_architecture_search,
    run_architecture_search,
)

outer_schema = get_architecture_schema("architecture_search_v1")
sizes = dict(
    outer_population_size=4,
    outer_population_count=2,
    inner_population_size=3,
    inner_population_count=1,
)
# Always look at this before launching: outer and inner budgets multiply.
print(estimate_cost(base_spec, **sizes).estimated_max_trials)
preview = preview_architecture_search(
    base_spec, outer_schema, schema, seed=7, **sizes,
)
report = run_architecture_search(
    base_spec, outer_schema, schema, seed=7, **sizes,
)
```

The breeding API can likewise be composed directly with the trial runner:

```python
from cognitive_runtime.training.model_factory import (
    breed,
    get_schema,
    load_parent,
    record_breeding_lineage,
    run_trial,
)

schema = get_schema("generic_action_effects_v2")
parent_a = load_parent("runs", "Crafter", "<parent-a>", schema, tier="fast")
parent_b = load_parent("runs", "Crafter", "<parent-b>", schema, tier="fast")
bred = breed(
    parent_a,
    parent_b,
    schema,
    objective="windowed_rollout",
    generation=1,
    seed=19,
)
trial = run_trial(bred.child_spec)
record_breeding_lineage(trial.directory, bred.lineage)
```

The notebook remains the visualization and diagnosis surface; factory specs,
CLI/API campaigns, lineage, execution, and promotion are the source of truth.

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
