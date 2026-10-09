# Offline core and opt-in test modules

This is the test-infrastructure repair for [#297](https://github.com/Brain-Crumbs/CCR/issues/297),
linked to [#283](https://github.com/Brain-Crumbs/CCR/issues/283) and the existing
[Factory epic #212](https://github.com/Brain-Crumbs/CCR/issues/212). It implements no
market feed, forecasting claim or #298 experiment pipeline. Evidence is
**synthetic_only**; live-source capabilities remain unvalidated.

## Clean default installation and offline execution

```sh
python -m venv .venv-core
. .venv-core/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python .github/scripts/validate_artifact_schemas.py
python .github/scripts/run_core_tests.py --collect-only --report /tmp/core-collection.json
python .github/scripts/run_core_tests.py --report /tmp/core-tests.json
PYTHONHASHSEED=12345 python .github/scripts/run_core_tests.py tests/test_model_factory_*.py tests/test_factory_artifact_schemas.py tests/market/test_effective_config.py --report /tmp/contracts-seed-12345.json
python .github/scripts/check_core_budget.py /tmp
```

Install-time package retrieval is separate from offline test execution. The clean
core environment has pytest/jsonschema/pytest-xdist plus existing runtime dependencies; it
must contain no torch, transformers, sentence-transformers or provider SDKs.
No model/data download, GPU, API key or substantive training is allowed.

The runner starts each invocation with empty temporary Hugging Face/Torch/data
caches, offline flags and two CPU test workers with single-thread numerical libraries.
Use `--workers 0` for sequential diagnostic runs; test selection is identical.
`--dist=loadfile` keeps module fixtures together, while independent modules run
on two ordinary CPU cores. Selected node IDs and executed-test counts are
checked across collection and both seeds; collection-only cannot masquerade as
a passing test run. An inherited Python
audit hook denies DNS/IP socket operations in collection, tests and Python
children; attempts fail the outer invocation even when a test catches the error.
Linux CI additionally runs each test command in a fresh network namespace via
`sudo -E unshare --net -- "$(command -v python)" ...`, so non-Python children
cannot contact external services either. For local Linux environments supporting
unprivileged namespaces, prefix the command with
`unshare --user --map-root-user --net --`. No host firewall is changed.

Every default job has a five-minute cap. Runtime evidence from collection, the
full default suite (hash seed 0) and the second Factory pass (hash seed 12345) is required, summed, and
must stay at or below 120 seconds. This is execution time, excluding dependency
installation and Actions startup. A missing/failed report is a failure. A budget
regression requires profiling/optimization or justified execution splitting,
not new skips, removed assertions or moving lightweight tests out of default.
The scripts print per-test durations, counts and machine/Python metadata.
The old workflow ran every Factory test three times (inside core and in both
seed passes). Core now explicitly uses seed 0, so two executions preserve all
tests and both hash seeds without the redundant third pass. The second pass
also includes the new golden serializer/hash and effective-configuration modules.

## Test membership and opt-in imports

Existing lightweight tests remain enabled. The pre-existing legacy/deferred
selection (`--run-deferred`) is unchanged. The existing bounded neural optimizer
regressions now require `--run-market=extended`; they no longer start training
merely because torch happens to be installed.

Registered module tiers are:

- `market_extended`: bounded offline integration/property/compatibility checks
- `market_text`: optional local text-encoder checks
- `market_jepa`: optional bounded neural/JEPA checks
- `market_live`: explicitly authorized, rights/key-gated provider checks

Declare a module marker statically (`pytestmark = pytest.mark.market_text`) or
place the module under `tests/market/text/` (similarly `extended`, `jepa`, `live`).
Tier detection happens **before importing the module**, including explicit file
paths and shell-expanded wildcards. Marker aliases and literal `getattr` names
are supported; dynamically constructed marker names are not a supported module
boundary and must use a tier directory. Mixed modules containing an optional
marker are treated conservatively as optional as a whole; put lightweight tests
in a separate unmarked module. `-m market_text` alone does not opt in.
Repeat `--run-market=TIER` if multiple tiers are needed.

Only the extended compatibility tests exist today. Text, JEPA and live marker
names reserve future module boundaries; they are not claims that those packages
or extras are implemented. Owning issues must add their tests incrementally,
name/install only their needed extras, and keep eager optional imports behind
these boundaries. Selecting an empty module is not successful validation.

## Existing bounded module commands

Keep optional dependencies out of the core virtual environment:

```sh
python -m venv .venv-neural
. .venv-neural/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,neural]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest tests/market/test_effective_config_neural.py --run-market=extended -q --durations=10
# No torch needed for this real default-size Crafter compatibility check:
python -m pytest tests/market/extended/test_legacy_crafter_compatibility.py \
  --run-market=extended -q --durations=5
python .github/scripts/run_factory_smoke.py --output /tmp/factory-smoke-new
```

The Factory output directory must be new. Its micro corpus uses six synthetic
short episodes, strict quality/overlap gates, fixed source seeds and frozen
content hashes. The runner performs corpus build/reuse, one-epoch baseline,
two one-field clones and paired comparison using actual CLI-returned IDs.
It verifies artifacts and a non-accessed sealed test split. Individual commands
are capped at 60 seconds, the complete smoke at 150 seconds (180 maximum).
Failures write a partial summary and are never converted to a skip.

`.github/workflows/nightly-factory.yml` retains its historical filename but is
now manual-only. It separately runs exact resume-equivalence, budget/loadability,
missing-frozen-corpus and sealed-split regression nodes, capped at 90 seconds.
Only synthetic JSON/log evidence is uploaded, retained seven days; no model
checkpoint or restricted provider payload is published.

The broad, slow legacy neural sweep is also manual-only through CI's
`legacy_neural` input; it is not a required PR check. A manual run is a deliberate
compute choice, not permission to fetch pretrained weights or call a provider.

## Fixture contract and provenance

- `.github/fixtures/`: executable micro corpus/baseline configurations, provenance
  hashes and rights/source notes.
- `.github/schemas/`: strict offline legacy artifact schemas.
- `tests/fixtures/factory/`: current-writer goldens, independent generation config
  and hash/rights/seed/edge-case inventory.

The schema validator always runs. Both missing directories, missing schema or
fixture, mismatched hash, malformed data/schema and orphan fixtures must fail.
Round-trip regressions use production writers/readers rather than merely
validating self-authored JSON against an equally permissive schema. All committed
inputs are project-authored synthetic MIT fixtures, not provider response data.

## Measurements

See the PR's exact-head evidence and Actions runtime artifacts for final counts
and durations. The pre-repair default suite on Python 3.12 / Intel Xeon Platinum
8370C took 166.49 seconds (1063 passed, 263 skipped). Profiling found repeated
64x64 terrain generation dominated Crafter seam/replay tests. Using real 16x16
worlds with unchanged 64x64 pixel rendering and assertions reduced those 21
tests to 9.66 seconds. A separate opt-in three-tick production-default 64x64
record/replay check passed in 8.76 seconds. No existing lightweight test was
removed or newly skipped. CPU timings vary; the hosted aggregate gate remains
authoritative for each submitted head.

Final implementation evidence is tracked in [PR #301](https://github.com/Brain-Crumbs/CCR/pull/301).
The two-worker local core completed 1173 passing tests and 256 pre-existing skips
in 37.32 seconds. Both worker environments verified the inherited offline guard
and the correct hash seed; no network attempt or model-cache file was observed.
