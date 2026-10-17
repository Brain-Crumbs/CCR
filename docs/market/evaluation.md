# Calibration and baseline evaluation (#294)

This package evaluates immutable #293 forecasts against separate #291 label
artifacts. Readiness is **synthetic_only**, not live validated. There is no
trading, P&L, model download, provider access or claimed edge. No positive gain
is required. #295 owns Factory lifecycle/search wiring; this package exposes a
validation-only objective guard for that integration. #296 must exist before
latent-only/combined arms are registered.

## Reproduce the bounded offline checks

```sh
python -m pytest tests/market/test_evaluation.py -q --durations=5
python -m pytest tests/market/extended/test_evaluation_multiseed.py --run-market=extended -q
```

The first command is default/core, torch-free and network-free. The second is
an optional two-seed, 12-sample synthetic check, not an experiment campaign.
Full prospective holdout evaluation and real-source validation were not run.
The existing aggregate default target is 120 seconds; CI jobs cap at 5 minutes.

## Use existing corpus and model artifacts

```python
from cognitive_runtime.adapters.finance.evaluation.calibration import fit_calibrator
from cognitive_runtime.adapters.finance.evaluation.report import make_report, seal_report, markdown

validation = model.forecast([corpus.feature(k) for k in plan.validation])
calibrator = fit_calibrator(corpus, plan, validation)  # test fitting fails
raw_test = model.forecast([corpus.feature(k) for k in plan.test])
calibrated = calibrator.apply(corpus, plan, raw_test)
zero_test = zero_model.forecast([corpus.feature(k) for k in plan.test])
report = make_report(corpus, plan, calibrated, baseline=zero_test,
    baseline_name='zero-change:' + zero_model.hash, run_id=model.hash,
    population='synthetic', pretraining_cutoff='not_applicable_non_pretrained',
    caveats=('Project-authored synthetic corpus; no historical universe claim.',))
seal_report(report, 'sealed-test.json')  # exclusive creation; existing path fails
print(markdown(report))
```

The report binds the corpus, split, label specification, every label hash,
forecast IDs, prediction payload and exact baseline payload. It rejects missing,
duplicate or mismatched population IDs, foreign anchors and incompatible grids.
Censored, missing/stale and abstained observations stay in the denominator.
Failed runs require a reason and may retain partial scores; a suite fit failure
produces explicit `model_failed` abstentions for every held-out sample.

A corpus has one target/horizon/spec. Reports partition by source/instrument;
never aggregate different target units, horizons or synthetic/play-money/financial
populations. `population` is an operator declaration for real-data provenance:
`synthetic` is checked against the snapshot flag; identifying real play-money
versus financial sources still requires the caller's rights/provider policy.
No function promotes any report as edge evidence (`edge_promotable=false`).

## Metrics and calibration

Mean pinball averages the explicitly named quantile grid. MAE is computed only
at an explicit median. Interval coverage/width use the outer two quantiles,
whose levels remain visible. Direction Brier is available only when the schema
contains an explicit separately calibrated positive-price-change head; zero
change is negative for the `value > 0` event. There is no resolution Brier.
Sparse quantiles are not a full distribution: CRPS is unavailable in v1.

Calibration fits one residual quantile correction per level on **validation
only**, retaining all validation IDs and exact label hashes, with usable rows
forming the fit. Corrections are monotonically rearranged then clipped to each
forecast's exact Decimal support. It is empirical calibration, not a finite-sample
coverage guarantee. Applying it returns new immutable results and rejects another
model/grid, changed corpus/split/spec or a second calibration pass. Validation
scores after validation fitting are descriptive/in-sample for calibration; use
the sealed test report for final performance, never iterative search.

Uncertainty uses seeded nonoverlapping UTC-epoch time blocks, resampled as
clusters. Identical decision timestamps remain together. Choose block length
before evaluation to reflect dependence/horizon; the default one-day block is
only a reference setting. Fewer than two occupied blocks yields explicit
`insufficient_time_blocks`, not a fabricated interval. Bootstrap is capped at
10,000 scored observations and 2,000 replicates per metric. This lightweight
reference implementation is not an unbounded large-population engine.

Paired pinball improvement is baseline loss minus candidate loss on the exact
jointly scored forecast IDs. Candidate and baseline coverage are reported
separately, so abstention cannot silently manufacture population-wide gains.
Regime/liquidity/missingness slices require explicit predeclared, exhaustive,
nonoverlapping memberships; omitted memberships appear as `unknown`. Issuer and
event-family slices retain overlapping multi-entity membership and an unknown
bucket. Only a validated `SplitPlan.group` of `issuer` or `event_family` establishes
that disjoint policy; a slice alone is not an unseen-issuer experiment.

## Equal-budget and negative controls

```python
from cognitive_runtime.adapters.finance.evaluation.ablations import Budget, run_suite
reports = run_suite(corpus, plan, listing_id='contract:demo', population='synthetic',
                    budget=Budget(seeds=(294,), max_train_samples=32, text_dimensions=8))
```

Explicit invocation runs nine arms: zero, unconditional, volatility, numeric,
text, supervised multimodal, shuffled text, lagged text and non-pretrained.
All arms share split/labels/seed list, maximum train rows, dimensions and ridge.
These are closed-form reference fits, so no optimizer-step budget is invented;
actual algorithm and usable fit IDs are reported. Non-pretrained explicitly
aliases hashing multimodal since **all** v1 arms are non-pretrained. Ordinary
fits are deterministic; only shuffled control changes with seed.

Text controls transform derived text-vector blocks for both training and held-out
predictions. Numeric features, target labels and causal snapshots remain unchanged.
Shuffle donors stay within the same partition (deliberately noncausal diagnostic,
not deployable inference). Lag donors use strictly earlier decisions in that
partition; the first has a masked missing-text vector. No test information
enters training. Control hashes bind mode/seed and donor map. These controls
are evaluation-only models; do not load them through the ordinary baseline
checkpoint loader. Their lifecycle/checkpoint integration belongs to #295.

The negative fixture sets all synthetic targets to zero independently of varying
context/text and runs all arms; no gain is a valid result. The oracle fixture
injects a future-return channel/label into the actual inference boundary, which
rejects it; declared held-out fitting is also rejected. Interface checks cannot
prove arbitrary opaque features or pretrained weights free of memorization.
Unknown pretraining cutoff, survivorship and source coverage remain visible.

`validation_objective(report, metric=..., predeclared_metric=...)` rejects sealed
test reports, failed runs, unavailable metrics and undeclared loss choices.
`seal_report` prevents accidental overwrite of a path, not an adversarial operator
with filesystem access. Predeclare the report path and evaluation protocol and
do not repeatedly inspect test results to tune. No test-driven Factory campaign
is implemented by this issue. JSON and Markdown templates live beside this guide.
