# Spatial rollout-collapse handoff

## Symptom

Notebook overfit run `Test-v2-overfit-02` completed after 100 epochs on its
two exact `approach_entity` training recordings, but did not beat copy-last:

| Horizon | model MSE | copy-last MSE |
| --- | ---: | ---: |
| T+1 | 0.0081150 | 0.0081145 |
| T+4 | 0.0131170 | 0.0131160 |

Both direct and rollout paths report `frozen_rollout`: prediction dispersion
`0.00000`, target dispersion `0.01293`. This is a model failure, not a
generalization failure.

Semantic learning is healthy (`semantic_loss: 2.9498 -> 0.016381`), but the
RGB residual decoder has learned effectively zero visual change.

## Reproduce

Run the notebook's overfit cell in
`notebooks/build_and_diagnose_organism.ipynb` (currently configured for
Crafter, transformer, `approach_entity`, horizons `(1, 4)`, 100 epochs), then
inspect the generated trace/clinic summary. It must be evaluated on the exact
train recordings (`overfit_evaluation=True`).

Relevant code:

- `brain/cortex/predictive.py`: `SpatialResidualDecoder`,
  `PredictiveCortex.forward_horizons`
- `cognitive_runtime/training/action_world_model.py`:
  `_spatial_pixel_loss`, `_train_windowed_rollout_objective`,
  `evaluate_action_world_model`
- `cognitive_runtime/training/nursery.py`: Crafter `approach_entity`

## Likely boundary

The residual decoder is allowed to minimize pixel loss by retaining the
observed reference frame. Do not suppress the frozen-rollout warning or relax
the copy-last metric. Diagnose why dynamic-target residuals/masks receive too
little useful gradient despite the semantic head learning the same data.

Check, at minimum:

1. predicted delta magnitude and change-mask values on dynamic training
   windows;
2. pixel residual loss split by target motion/static frames;
3. reference-frame propagation during scheduled sampling and
   `forward_horizons`;
4. whether semantic supervision can be coupled to RGB change without leaking
   targets at evaluation.

## Acceptance

On the exact-training overfit run, both T+1 and T+4 should beat copy-last and
rollout prediction dispersion must be non-frozen. Preserve the explicit
reference-frame API and the static-scene behavior; do not replace the model
with a generic frame generator or weaken the diagnostic.

Add a fast deterministic regression test for a dynamic spatial sequence that
would fail if all rollout frames equal their initial reference.
