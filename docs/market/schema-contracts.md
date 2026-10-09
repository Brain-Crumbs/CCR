# Market schema contracts v1

Issue #286 under #283 adds schema and synthetic-fixture foundations only.
Status: **synthetic_only**. No provider access, model download, trained model,
ledger, calendar engine or predictive-edge result is supplied by these contracts.

## Reproduce and inspect

```sh
python -m unittest tests.market.test_schema_contracts -v
python -m cognitive_runtime.adapters.finance.testing.generate --output /tmp/market-fixtures
python -m cognitive_runtime.adapters.finance.testing.document
# Optional larger synthetic corpus (not in default CI):
python -m cognitive_runtime.adapters.finance.testing.generate --output /tmp/market-large --steps 2048
# Optional reproducible randomized boundary and prior-version corpus checks:
python -m pytest --run-market=extended tests/market/extended/test_schema_properties.py
```

The [generated reference](schema-reference.md), [examples](schema-examples.json)
and JSON Schemas in `schemas/` derive from executable constructors. Structural
JSON Schema validation is necessary but insufficient: use `parse_record` and the
constructors for semantic checks. Both reject unknown fields recursively. No
inference schema contains a resolution-probability prediction.

## Prices, targets and clocks

Wire money, prices, quantile levels, probabilities and metrics are finite plain
**decimal strings**, at most 128 significant digits and 128 fractional places. Floats/exponents/NaN are
rejected. Contract subtraction and support bounds are exact within those limits. Equity
calculation takes an exact difference before division and rounds once to 34
significant digits (ROUND_HALF_EVEN), independently of caller precision. Equity target is
`equity_simple_return`, unit `return_fraction`: endpoint / anchor - 1, with a
positive anchor and lower support -1. Equity labels are split-neutral and exclude
dividends; raw input prices remain unchanged and split adjustments name actions.
Contract target is `contract_absolute_probability_price_change`, unit
`probability_price_fraction`: endpoint - anchor. `0.42 → 0.47` is `0.05`, or 5
percentage points, not 5% relative return. Support is exactly `[-anchor, 1-anchor]`.
Payout scale is separate from normalized quoted probability-price. Settlement
payouts are never substitutes for missing pre-terminal market prices.

Quantile levels strictly increase; values are nondecreasing and within support.
A separately calibrated positive-change head names its own calibrator; direction
cannot be inferred from a handful of quantiles. Abstentions carry a reason and no
quantiles/direction output. Every forecast carries quality, model identity, mark,
feed, currency, anchor and decision clocks. Label validation rejects mixed
currencies/feeds/marks, stale anchors and invalid endpoint tolerances.

Persisted timestamps require UTC `Z`, with at most microsecond precision. Original
timezone and precision are declared separately. Event time may differ from
knowledge time (including skew). Availability is this exact revision's earliest
evidenced availability; unknown remains null. An observation proxy must equal
first observation. Observed and ingested times are separate, immutable serialized
fields; replay includes both in the hash. Revisions must name their predecessor.
No legacy StreamEvent/spec/checkpoint serialization is modified.

Horizon origin is decision time, independent of the anchor time and price/text
context windows. Supported equity clocks are 1/4 exchange trading hours or 1/5
exchange sessions and require a versioned calendar. Contracts use positive UTC
elapsed seconds. Calendar resolution (DST, holidays, half-days, halts) belongs to
#291; this package retains tiny explicit test facts, not a substitute calendar.
A LabelSpec names anchor maximum age, endpoint first-after/last-before tolerance,
action policy and terminal censor policy. Nominal and actual endpoints differ
explicitly. No filling of halted/missing quotes into a false zero return.

## Separation and neutral integration

`finance.schemas` exports only inference-safe records. `finance.labels` separately
owns LabelSpec, RealizedLabel, ContractTerminalMetadata and FutureTarget. Future
training targets require split `train` and stop-gradient; their record references
are not accepted by inference. `validate_context` only accepts exact observation
types and verifies ordered IDs/hashes and availability/observation cutoffs.
Strict replay requires both knowledge clocks ≤ decision. Historical reconstruction
requires evidenced availability plus caveats. Crossed quotes are rejected;
stale/missing observations require explicit masks. Snapshots cannot contain inline
arbitrary metadata, terminal fields or future-target objects.

`to_inference_input` converts validated context plus an already prepared finite
numeric vector to the existing neutral `InferenceInput`, with the snapshot ID as
sample identity and explicit feature schema. It imports no training framework.
It does not certify caller-supplied feature causality or implement a model backend;
train-only feature fitting and actual models belong to later issues. Labels stay
in the neutral TrainingBatch/EvaluationBatch boundary, outside predict.

## Serialization and migrations

Canonical serialization is UTF-8 JSON, sorted object keys, compact separators,
no NaN, preserving array order and exact decimal/timestamp spellings. SHA256 binds
all fields. Semantically equal but differently spelled decimals intentionally
have different record hashes: canonicalization promises replay stability, not
normalization of source facts. Constructors deep-convert nested records/arrays
into frozen dataclasses/tuples; callers cannot mutate an input dict to change a
record. Duplicate JSON keys fail. Schemas reject stale versions on ordinary load.

The explicitly supported **synthetic pilot 0.9.0** corpus omits
`availability_confidence`. It is a project-authored migration test format, not a
claim of previously released market records. `migrate_legacy_0_9` copies that
payload, adds conservative confidence `0`, validates 1.0.0 and creates a new hash;
source bytes remain untouched. Additive required fields need explicit migration;
minor/major versions are never silently accepted. Breaking target/unit/clock
changes require a new reviewed converter and version. Unsupported versions fail
closed. Keep original source bytes/hash and new record hash in a migration ledger
when a persistence layer is introduced. Legacy CCR hashes are outside this policy.

## Fixture rights and intended controls

All committed records are project-authored MIT synthetic data. Manifest carries
seed, generator version, source/rights, per-file SHA256, record hashes, edge cases
and expected invariants. `sources.json` preserves canonical synthetic constructor
inputs (excluding provenance); every raw_sha256 is verified against those exact
source bytes, including revised stories. The default corpus is eight steps per signal/null arm.
The generator uses a seeded AR(1) latent factor, noisy instrument loadings, equity
log-price paths, bounded logistic contract prices, and lagged noisy synthetic text.
The null arm zeros the latent contribution to prices while retaining the text
process. Oracle latent state is explicitly training/testing-only and kept outside
inference records. Signal recovery would show only generator recovery, never edge.

Handcrafted cases preserve delayed/revised stories, shuffled arrival, syndication,
clock skew, alias reuse, splits/dividends, delisting/halts, stale/crossed/outage
quotes, holidays/half-days/DST, early closure and terminal traps. They form inputs
and assertions for this package's contracts; as-of selection, calendar resolution
and label construction are deferred to #287/#291. No UCI or external raw payload
is imported. Missing pytest/core packages in an execution environment must be
reported; CI acceptance checks must still execute and cannot silently skip.
