# Synthetic Factory smoke inputs

These two project-authored MIT recipes repair the legacy missing-fixture
contract in [#297](https://github.com/Brain-Crumbs/CCR/issues/297), following
[#242](https://github.com/Brain-Crumbs/CCR/issues/242) and the
[Factory epic #212](https://github.com/Brain-Crumbs/CCR/issues/212).
`provenance.json` records source, explicit seeds, rights, SHA-256 and intended
edge case for each input. Hashes cover the exact checked-in recipe bytes.

- `micro-corpus.yaml`: six locally generated Crafter optical-flow episodes,
  at most six ticks each, with distinct train, validation and sealed-test
  seeds. Quality and split-overlap gates remain enabled.
- `micro-baseline.yaml`: one CPU epoch, one attention layer, latent width 8,
  hidden width 16, 4-pixel reconstruction and a 15-second training cap.
  This canary checks integration, not learned model quality or live capability.

After installing CPU PyTorch and `.[dev,neural]`, run from the repository root:

```sh
python .github/scripts/run_factory_smoke.py --output /tmp/ccr-factory-smoke
```

The output path must not exist. The helper fails on absent inputs, unavailable
CLI/dependencies, nonzero exits, expired bounds, incomplete trials, missing
artifacts or non-evaluable comparisons. It captures current CLI `run_id`
outputs, uses `training.loss_weights.closed_loop_pixel_loss_weight` for controlled clones,
and records stage durations and fixture hashes in `smoke-summary.json` even
when a started smoke fails. The total CLI bound is 150 seconds by default
(maximum 180); each command has an additional 60-second bound.

A local CPU verification on 2026-10-09 completed all six CLI stages in
19.723 seconds (PyTorch 2.14.1+cpu, one thread). The exact workflow regression
selection then passed 20 tests in 17.84 seconds. Hosted-runner timings are
reported separately; this is synthetic integration evidence only.

The manual `nightly-factory.yml` workflow also runs explicit, bounded existing
resume-equivalence, budget/checkpoint/promotion and sealed-split regression
nodes. It is an opt-in module job, not a default PR requirement. No credentials,
live API, GPU, external dataset or pretrained model is used. Its artifacts are
`synthetic_only`; generated recordings/checkpoints are not uploaded by CI.
