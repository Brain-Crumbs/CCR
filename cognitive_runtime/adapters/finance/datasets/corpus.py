"""Immutable private corpus stores; feature readers do not import outcome types.

Fit/calibration views are explicit capabilities, not protection against a caller
with arbitrary filesystem/Python access. No estimator or model is trained here.
"""
from collections import Counter
from dataclasses import dataclass, replace
from datetime import timedelta
import json

from ..schemas.base import Strict, timestamp, sha256
from ..schemas.records import Instrument, PredictionContractState, NewsEvent, PriceBar, Quote
from cognitive_runtime.data.ledger import canonical, digest
from cognitive_runtime.data.raw_store import RawStore, private_path
from cognitive_runtime.data.snapshot import load_snapshot
from .calendars import utc
from .universe import freeze_universe, contamination_keys
from .splits import Sample, SplitPlan, validate_split


@dataclass(frozen=True, kw_only=True)
class Request(Strict):
    sample_id: str
    snapshot_sha256: str
    outcome_snapshot_sha256: str
    listing_id: str
    universe_id: str
    event_family: str | None = None

    def validate(self):
        sha256(self.snapshot_sha256)
        sha256(self.outcome_snapshot_sha256)


class FeatureStore:
    """Inference-safe reader. There is deliberately no label/target method."""
    def __init__(self, path):
        self._store = RawStore(path)

    def read(self, content_hash):
        value = json.loads(self._store.get(content_hash))
        if set(value) != {'version', 'sample_id', 'context', 'universe', 'records', 'quality_records'} or value['version'] != 1:
            raise ValueError('not a feature artifact')
        return value


class Corpus:
    def __init__(self, root, manifest_sha256):
        self.root = private_path(root)
        self.manifest_sha256 = manifest_sha256
        self.manifest = json.loads(RawStore(self.root / 'manifests').get(manifest_sha256))
        self.samples = tuple(Sample(**s) for s in self.manifest['samples'])
        self.features = FeatureStore(self.root / 'features')

    def _outcome(self, sample_id):
        from .labels import LabelOutcome
        return LabelOutcome.from_json(RawStore(self.root / 'labels').get(self.manifest['labels'][sample_id]))

    def population(self):
        """Coverage-first accounting includes censored observations unchanged."""
        reasons = Counter(self._outcome(s.sample_id).censor_reason or 'realized' for s in self.samples)
        return {'total': len(self.samples), 'reasons': dict(sorted(reasons.items()))}

    def feature(self, sample_id):
        return self.features.read(self.manifest['features'][sample_id])

    def fit_preprocessing(self, plan, fit):
        """Feature selection/vocabulary/scalers receive train features only."""
        validate_split(self.samples, plan)
        return fit(tuple(self.feature(key) for key in plan.train))

    def calibrate(self, plan, fit):
        """Validation only, including explicit censor outcomes for accounting."""
        validate_split(self.samples, plan)
        return fit(tuple((self.feature(key), self._outcome(key)) for key in plan.validation))

    def evaluation(self, plan, partition='test'):
        validate_split(self.samples, plan)
        if partition not in ('validation', 'test'):
            raise ValueError('evaluation requires held-out partition')
        return tuple((self.feature(key), self._outcome(key)) for key in getattr(plan, partition))

    def training(self, plan):
        validate_split(self.samples, plan)
        from ..labels import FutureTarget
        result = []
        for key in plan.train:
            target_hash = self.manifest['targets'].get(key)
            target = None if target_hash is None else FutureTarget.from_json(RawStore(self.root / 'targets').get(target_hash))
            result.append((self.feature(key), self._outcome(key), target))
        return tuple(result)


def build_corpus(ledger, root, requests, *, spec, calendar=None, coverage='synthetic_only',
                 limitations=('Synthetic universe; no complete historical delisting coverage.',),
                 event_family_evidence=None):
    """One horizon/spec per immutable corpus; build multiple corpora for multiple
    horizons. Every request survives censoring. Event-family assignments must be
    explicitly evidenced; market/token IDs are not proxies for event families.
    """
    from .labels import derive_label
    root = private_path(root)
    requests = tuple(requests)
    if not requests or len({r.sample_id for r in requests}) != len(requests):
        raise ValueError('nonempty unique sample IDs required')
    stores = {kind: RawStore(root / kind) for kind in ('features', 'labels', 'targets', 'manifests')}
    features, labels, samples = {}, {}, []
    rights_inventory = {}
    for request in sorted(requests, key=lambda r: r.sample_id):
        decision = load_snapshot(ledger, request.snapshot_sha256)
        outcomes = load_snapshot(ledger, request.outcome_snapshot_sha256)
        # Bind redistribution provenance of both feature AND label sources.
        # Only hashes/rights metadata enter this inventory, never future prices
        # or text; inference still receives only the separate feature store.
        for snapshot in (decision, outcomes):
            for record in (*snapshot.records, *snapshot.quality_records):
                rights_inventory[record.hash] = {key: getattr(record.provenance, key)
                    for key in ('rights', 'license', 'synthetic')}
        universe = freeze_universe(decision, request.universe_id, coverage=coverage, limitations=limitations)
        if coverage == 'synthetic_only' and not decision.context.provenance.synthetic:
            raise ValueError('real data cannot claim synthetic coverage')
        result = derive_label(decision, outcomes, listing_id=request.listing_id, spec=spec, calendar=calendar)
        if request.listing_id not in {row[2] for row in universe.identities}:
            result = replace(result, label=None, anchor_to_endpoint_seconds=None, censor_reason='outside_pit_universe')
        feature = {'version': 1, 'sample_id': request.sample_id, 'context': decision.context.to_dict(),
                   'universe': universe.to_dict(), 'records': [r.to_dict() for r in decision.records],
                   'quality_records': [r.to_dict() for r in decision.quality_records]}
        features[request.sample_id] = stores['features'].put(canonical(feature).encode())
        labels[request.sample_id] = stores['labels'].put(result.canonical_json().encode())
        dedup, source, news_issuers = contamination_keys(decision)
        issuer_ids = {r.issuer_id for r in decision.records if type(r) is Instrument}
        families = ()
        contracts = [r for r in decision.records if type(r) is PredictionContractState]
        requested_contracts = [r for r in contracts if r.listing_id == request.listing_id]
        if request.event_family is not None:
            if not requested_contracts or not event_family_evidence or any(
                    event_family_evidence.get(r.market_id) != request.event_family for r in requested_contracts):
                raise ValueError('event family requires explicit market mapping')
            if any(r.market_id not in event_family_evidence for r in contracts):
                raise ValueError('every context contract needs an event family')
            families = tuple(sorted({event_family_evidence[r.market_id] for r in contracts}))
        # Censored windows extend through endpoint tolerance. Realized windows
        # include label knowledge time, conservatively guarding delayed targets.
        end = timestamp(result.nominal_endpoint_at) + timedelta(seconds=spec.endpoint_tolerance_seconds)
        if result.label:
            end = max(end, timestamp(result.label.actual_endpoint_at), timestamp(result.label.target_available_at))
        else:
            # Absence/terminal censoring is established only at the frozen
            # outcome cutoff, never backdated to the nominal endpoint.
            end = max(end, timestamp(outcomes.context.decision_at))
        start = timestamp(decision.context.decision_at) - timedelta(seconds=max(
            decision.context.price_context.count, decision.context.text_context.count))
        samples.append(Sample(sample_id=request.sample_id, snapshot_sha256=decision.context.hash,
            decision_at=decision.context.decision_at, context_start_at=utc(start), label_end_at=utc(end),
            issuer_ids=tuple(sorted(issuer_ids | set(news_issuers))), event_families=families, dedup_keys=dedup, source_keys=source))
    manifest = {'version': 1, 'spec': spec.to_dict(), 'calendar': None if calendar is None else calendar.to_dict(),
                'event_family_evidence': event_family_evidence or {},
                'samples': [s.to_dict() for s in samples], 'features': features, 'labels': labels, 'targets': {},
                'source_rights': dict(sorted(rights_inventory.items()))}
    key = stores['manifests'].put(canonical(manifest).encode())
    return Corpus(root, key)


def attach_training_targets(corpus, ledger, plan, targets):
    """Version a corpus with train-only targets, checking each future record and
    the expanded actual interval against every held-out partition before write.
    Return a new corpus and re-bound split; never mutate the old manifest.
    """
    from ..labels import FutureTarget
    from cognitive_runtime.data.snapshot import eligible
    from .splits import _sample_hash
    validate_split(corpus.samples, plan)
    targets = dict(targets)
    if not set(targets) <= set(plan.train):
        raise ValueError('future targets restricted to permitted training partitions')
    by_id = {s.sample_id: s for s in corpus.samples}
    for key, target in targets.items():
        if type(target) is not FutureTarget:
            raise TypeError('exact FutureTarget required')
        sample = by_id[key]
        snapshot = load_snapshot(ledger, sample.snapshot_sha256)
        if target.context_snapshot_id != snapshot.context.record_id or target.window_start_at != sample.decision_at:
            raise ValueError('target/context window mismatch')
        future_records = []
        for identity in target.future_record_ids:
            record = ledger.get(identity)
            if not eligible(record, target.window_end_at, 'strict_replay') or timestamp(record.event_at) <= timestamp(sample.decision_at):
                raise ValueError('future record outside permitted training window')
            future_records.append(record)
        evidence = _future_groups(ledger, target.window_end_at, future_records,
                                  corpus.manifest['event_family_evidence'])
        by_id[key] = replace(sample, future_end_at=target.window_end_at,
            issuer_ids=tuple(sorted(set(sample.issuer_ids) | evidence['issuers'])),
            event_families=tuple(sorted(set(sample.event_families) | evidence['families'])),
            dedup_keys=tuple(sorted(set(sample.dedup_keys) | evidence['dedup'])),
            source_keys=tuple(sorted(set(sample.source_keys) | evidence['sources'])))
    samples = tuple(by_id[s.sample_id] for s in corpus.samples)
    new_plan = replace(plan, samples_sha256=_sample_hash(samples))
    validate_split(samples, new_plan)
    manifest = {**corpus.manifest, 'samples': [s.to_dict() for s in samples], 'targets': dict(corpus.manifest['targets'])}
    store = RawStore(corpus.root / 'targets')
    for key, target in targets.items():
        manifest['targets'][key] = store.put(target.canonical_json().encode())
    key = RawStore(corpus.root / 'manifests').put(canonical(manifest).encode())
    return Corpus(corpus.root, key), new_plan


def freeze_outcomes(ledger, *, decision_at, calendar_id, calendar_version, calendar_sha256,
                    preprocessing_sha256, mode='strict_replay'):
    """Freeze full known observation history for label derivation, including old
    announcements with later effective dates. Snapshot is separate from context.
    This is bounded by the operator's input ledger, not a history-completeness
    claim. A later append creates a new snapshot, never repairs an old one.
    """
    from cognitive_runtime.data.snapshot import build_snapshot
    with ledger.locked():
        records = ledger._records()
    at = timestamp(decision_at)
    oldest = min([timestamp(r.event_at) for r in records] + [at])
    seconds = max(1, int((at - oldest).total_seconds()) + 1)
    return build_snapshot(ledger, decision_at=decision_at, price_seconds=seconds, text_seconds=seconds,
        calendar_id=calendar_id, calendar_version=calendar_version, calendar_sha256=calendar_sha256,
        preprocessing_sha256=preprocessing_sha256, mode=mode)


def _future_groups(ledger, end_at, records, family_mapping):
    """Future price/text views participate in the same robustness boundaries.

    Resolve point-in-time identities from eligible revisions at the future
    cutoff, then check their business-valid intervals at each target event.
    Missing/ambiguous identity is not permission to omit group metadata.
    """
    from cognitive_runtime.data.snapshot import select_revisions
    from cognitive_runtime.data.identity import active
    with ledger.locked():
        lineage = {r['record_id']: r['supersedes'] for r in ledger.db.execute('SELECT record_id,supersedes FROM events')}
        identities = tuple(r for r in select_revisions(ledger._records(), end_at, 'strict_replay', lineage)
                           if type(r) is Instrument)
    result = {'issuers': set(), 'families': set(), 'dedup': set(), 'sources': set()}
    for record in records:
        if type(record) is NewsEvent:
            if not record.entities or any(e.ambiguous for e in record.entities):
                raise ValueError('future news requires unambiguous issuer evidence')
            result['issuers'].update(e.issuer_id for e in record.entities)
            result['dedup'].add(record.dedup_cluster_id)
            result['sources'].add(record.source_id + ':' + record.source_record_id)
        elif type(record) in (PriceBar, Quote, PredictionContractState):
            at = record.end_at if type(record) is PriceBar else record.event_at
            matches = [i for i in identities if i.listing_id == record.listing_id and active(i, at)]
            if len(matches) != 1:
                raise ValueError('future price requires unambiguous PIT issuer identity')
            result['issuers'].add(matches[0].issuer_id)
            if type(record) is PredictionContractState:
                if record.market_id not in family_mapping:
                    raise ValueError('future contract requires event-family evidence')
                result['families'].add(family_mapping[record.market_id])
        else:
            raise ValueError('unsupported future-target observation family')
    return result
