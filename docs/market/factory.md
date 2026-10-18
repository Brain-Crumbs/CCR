# Market Factory lifecycle (#295)

Readiness is **synthetic_only**. The registered `market-baseline` backend runs
existing NumPy baselines through `model_factory.runner.run_trial`. It does not
fetch providers/models, use credentials, invoke hosted models, execute trades or
establish forecasting quality. JEPA remains optional future work (#296).

## Reproduce

From the repository with base dependencies installed:

```sh
python -m scripts.factory_market_smoke --output /tmp/market-factory-example
# Explicitly consume the one-use final-test protocol on a separate fresh run:
python -m scripts.factory_market_smoke --output /tmp/market-factory-final --final-test
```

The checked-in `config/market/factory-smoke.yaml` specifies seed, bounded baseline
and seconds. The MIT project-authored generator (`testing/factory.py`, version 1)
creates twelve synthetic binary-contract samples, including 0/1 boundary anchors,
separate outcome snapshots and a purged walk-forward split. It writes exact
content hashes through the existing ledger/corpus stores. `resolved-spec.json`
pins corpus and split before `run_trial`. Keep `private/` local; export only the
explicit artifact inventory. This is not a real prediction-market dataset.

For an existing synthetic corpus use `factory.market_spec(corpus, plan, ...)`;
then `run_trial(spec, root=..., corpus_root=corpus.root.parent)`. The helper's
listing default is `contract:demo`; other listings can be set in a resolved task
spec. Only explicit registered baseline kinds/configurations are supported.
Optional text encoders, ablation control models and neural JEPA checkpoints are
rejected rather than pretending their states are interchangeable.

## Evidence and continuation

Factory retains the existing run identity, spec, contracts, lineage, execution,
state and heartbeat at the run root. Each attempt records its actual current
source/commit separately from the preserved original-run execution identity. Every worker invocation has a fresh immutable
attempt ID under `attempts/<id>/`; checkpoints, metrics, budget, result and
`run_manifest.json` live there. Running manifests are partial and finalized in
place; terminal attempt evidence is never overwritten by resume. An abrupt kill
leaves its initial partial manifest. Cooperative cancellation/budget failure writes
a terminal partial manifest and available checkpoint. Preparation failures now
have an allocated run and partial evidence too. Invalid specs/parent identities
are rejected before execution and do not claim a run happened.

Resume requires a stale active worker, the same run ID and matching
model/data/training contracts, using `parent.checkpoint` relative to the run root
(e.g. `attempts/<id>/checkpoints/last.json`). It creates a new attempt, preserving
previous checkpoints and reports. Terminal runs cannot reopen: use a new clone
with explicit lineage for a retry after a terminal failure. An unfitted checkpoint
restarts the bounded fit; a completed closed-form fit reloads without refitting.
Changed corpus/split/availability/label horizons or baseline/text architecture
fail compatibility checks. No optimizer, scheduler or EMA exists in these
closed-form models; the manifest records that fact, along with fit cursor and
non-applicable RNG state. Seeds still bind generator/report identities.

`market-factory-run-v1` is an explicit Factory task manifest namespace, leaving
Crafter run schemas unchanged. It records source content and commit, resolved and
effective settings, installed environment versions, hardware, seed, semantic
hashes, outcome, resource measurements and artifact references. Missing pre-fit
identities are null, never invented. NumPy computation is honestly declared
CPU/fp64. Supported repeatability is exact in the same environment; cross-library
or cross-hardware bitwise equivalence is not promised. IDs/runtime/RSS differ
between attempts, while `reproducibility_sha256` binds stable inputs/results.
Budgets are cooperative wall-clock limits; RSS is process high-water measurement,
not per-attempt allocation or hard memory enforcement. #298 owns hard isolation.

Export registers only checkpoint/sidecar and validation report references after
checking the corpus's content-bound inventory of both context and outcome source
records' synthetic MIT redistribution rights, safe regular
paths, SHA256, size and the 1 MiB backend cap. Unknown rights or older corpora without the inventory fail export;
no recursive raw-store export occurs. Arbitrary exception text is not persisted.
Export inventory is not an upload. A trusted writer who changes files or rewrites
rights can invalidate these guarantees; this is not an adversarial sandbox.

## Search and locked final reporting

`model_factory.search.propose_market` reuses seeded Factory Latin-hypercube
sampling and bounded Gene definitions (ridge and, for text models, dimensions).
It leaves data, splits, seeds, rights and evaluation policy fixed. `run_market_search`
runs at most sixteen supplied fresh candidates using the same runner and registry
lock. Exact spec/source/environment duplicates reuse evidence, including failures;
failed/cancelled/over-budget candidates have no selection score. Only validation
pinball is selectable. These are bounded closed-form candidates, not a neural
successive-halving campaign.

`factory.final_test` is an explicit separate reporting action. It records the
completed candidate in the existing registry, charges its test-use ledger and
reserves the exact held-out features/labels/spec protocol once across that run root **before reading
final outcomes**. A reporting failure consumes the reservation. Reports are
exclusively created and never returned as search objectives. Do not move runs to
a new root or delete reservations to repeatedly tune against the test. Guards
protect ordinary API use, not an operator with arbitrary filesystem access.

Default tests use fake fits with real Factory, corpus and evaluation boundaries.
The explicit smoke above exercises the existing real zero baseline. Run the
existing Crafter canary with an already installed CPU neural extra:

```sh
python .github/scripts/run_factory_smoke.py --output /tmp/crafter-compatibility
```

All default lanes still count toward the aggregate 120-second gate. No live
validation or scientific/production-readiness claim follows from these checks.
