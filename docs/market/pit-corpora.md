# Point-in-time research corpora

Issue [#291](https://github.com/Brain-Crumbs/CCR/issues/291), under
[#283](https://github.com/Brain-Crumbs/CCR/issues/283). Status: **synthetic_only**.
No provider, model, credential, trading, paid service or live data is used by this
package. Complete real-world corporate-action, lifecycle, delisting and calendar
history is not supplied. Missing source coverage is not evidence of no event.

## Reproduction and interfaces

Install the existing core/dev extras, then run:

```sh
python -m pytest tests/market/test_pit_datasets.py -q
python -m pytest tests/market/extended/test_pit_boundaries.py --run-market=extended -q
```

The first test module contains the executable end-to-end example
`test_corpus_stores_population_and_train_validation_gates`: append project-authored
records to a private `Ledger`, freeze decision and outcome snapshots, pass their
hashes in `Request` objects to `build_corpus`, create a `walk_forward` plan, and
consume the permitted feature/label views. The constructor takes a typed
`LabelSpec`; `config/market/horizons.yaml` documents the supported policy vocabulary
and example tolerances, rather than silently overriding a supplied spec.
One corpus binds one spec/horizon; build separate immutable corpora for each horizon.

Use `freeze_outcomes` for the separate full-history outcome snapshot. Its window
starts at the oldest known event so an old announcement with a later effective
split date is retained. Manually supplied outcome snapshots must cover the full
relevant action/lifecycle history; the label routine rejects windows that do not
even cover the decision/anchor interval, but cannot infer absent external history.
The outcome cutoff must be late enough to observe eligible endpoints and actions.
The exact frozen outcome snapshot hash is recorded: later corrections never rewrite
an existing corpus. Unknown external history remains a coverage limitation.

Store ledger and corpus roots outside the repository. Each corpus has separate
private `features/`, `labels/`, `targets/` and `manifests/` content-addressed stores.
`FeatureStore` imports no label/target/training modules and exposes only a feature
reader. Features contain only the loaded decision snapshot's observations,
quality records and frozen universe; label artifacts bind both snapshot hashes and
the spec. Reads check content hashes. Dataset roots inherit #287's private-path
and atomic CAS rules. Copied derived artifacts require the same operator retention
and deletion controls as their source: ledger revocation does not erase exports.

A corpus manifest hash identifies the population, spec, calendar, feature/label
hashes and explicit market-to-event-family mapping. Split plans hash the exact
sample metadata, record every purged ID/reason and can be serialized with
`canonical_json()`. No arbitrary file is a training target: attachment checks the
exact `FutureTarget`, immutable ledger identities, strictly future observations,
permitted train membership, and expanded intervals before writing a new manifest.

## Time and price semantics

The calendar is an explicit, versioned, hashed UTC schedule plus original IANA
zone, coverage bounds, declared closed dates, sessions and halt intervals. Every
covered local date must be accounted for. No network calendar lookup or guessed
weekday schedule is performed. A calendar is an operator-supplied schedule,
not a claim that unexpected historical closures were known prospectively; freeze
separate versions when knowledge changes. Decision and outcome snapshots must
bind the same calendar hash as the requested equity label.

- Equity horizons: 1/4 trading hours count open segments, excluding halts and
  closed periods. 1/5 sessions mean the nth scheduled close strictly after the
  decision. At a close, a one-session horizon ends at the next session's close.
- Closed decisions either roll to the next trading segment or fail explicitly,
  according to the calendar policy. A price at a session close is eligible;
  a price inside a halt (including halt start) is not. A nominal endpoint at a
  halt boundary may therefore censor when no eligible price meets tolerance.
- Contract horizons count explicit UTC seconds continuously across DST/weekends.
- Anchor time, collector observation time, decision time, nominal endpoint and
  actual endpoint remain separate. The last eligible anchor must be fresh by
  observation-price time; a SIP bar observed now but ending 15 minutes ago fails
  a 60-second maximum age. This version does not offer a last-known-price task.
- The first observation at/after the nominal endpoint must be usable and within
  tolerance. Known bad first observations are censored, never skipped to improve
  outcomes. Anchor and endpoint must match source, listing, feed, venue, mark,
  currency, raw adjustment, calendar version and interval width (or contract
  market/token/side/question version/payout basis). No IEX/SIP mixing.
- Raw equity endpoint price is multiplied by realized within-interval split
  ratios before division by the anchor. Dividends are excluded. Total return is
  unsupported and rejected by `LabelSpec`; same-time ambiguous split actions
  censor. Halt/delisting/successor evidence within the price interval censors.
- Contract values are absolute price changes. Closure, halt, expiry and missing
  prices censor; settlement records cannot enter the observation ledger and
  are never used as endpoints.

`LabelOutcome` retains missing/stale anchors as well as endpoint/terminal censors.
Successful outcomes wrap the existing `RealizedLabel` schema and disclose actual
anchor-to-end elapsed seconds. `population()` counts **all** requested samples by
realized/censor reason; `evaluation()` returns censored outcomes as well. Consumers
must report coverage before metrics and must not treat a censor as zero return.

## Leakage controls and fitting scope

`walk_forward` orders train → validation/calibration → sealed test. Test has
precedence. Earlier partitions lose observations whose actual label/future-target
intervals intersect held-out observations, including equality at boundaries.
Conservative corpus label ends cover nominal endpoint tolerance and observed
label availability. Embargo starts at each sample's actual end, with an optional
extra elapsed margin; it is not a guessed fixed horizon. Optional strict context
purging extends overlap checks back to both samples' context starts.

Source-record and declared dedup-cluster repetitions cannot cross retained
partitions. Source checks mean `(provider, source_record_id)` repetition, not a ban
on using one provider across time. Issuer robustness uses stable issuer IDs from
all instruments and news entities included in the feature snapshot, so aliases
cannot escape the boundary by renaming a ticker. Event-family robustness requires
an explicit mapping for all context contracts; missing identities fail closed.
These checks do not infer fuzzy duplicate stories or undisclosed issuer relations.

`fit_preprocessing(plan, fit)` gives a callback only train features, including for
vocabularies and feature selection. `calibrate(plan, fit)` gives only validation
features/outcomes. `training(plan)` exposes attached future targets only in train;
held-out `evaluation()` has no future-target branch. These are explicit API
capabilities, not protection from a caller with unrestricted Python/filesystem
access. Model fitting, calibrated models, scoring and Factory campaign integration
belong to subsequent issues and are not claimed here.

## Fixtures and limitations

`tests/fixtures/market/pit/calendars.json` is manually authored, MIT, seed 291,
with SHA256 in `provenance.json`. Tests also construct synthetic records using
#286's shared envelope/source sealing, binding exact generated raw bytes to hashes.
The tables deliberately exercise US-like DST, holiday/half-day schedules, halts,
after-hours, delayed feeds, early contract closure and split/dividend separation;
they are not redistribution of any exchange data. Extended tests generate 300
intervals and check temporal/group/context variants and shuffled order.

No live source, production exchange calendar, survivorship-free universe,
forecast accuracy, investment value, or pretrained model is validated. The
lightweight core test runner and unchanged five-minute core CI job caps remain
the default gates; extended tests are opt-in.
