"""Causal snapshot selection and immutable, content-addressed manifests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json

from cognitive_runtime.adapters.finance.schemas.base import AvailabilityBasis, Provenance, timestamp, sha256
from cognitive_runtime.adapters.finance.schemas.records import (
    ContextSnapshot, ContextWindow, Instrument, InstrumentAlias, PriceBar, Quote,
    CorporateAction, NewsEvent,
)
from .identity import active, identity_manifest, dedup_manifest
from .ledger import canonical, digest

TIE_RULE = 'revision,available_at,observed_at,record_id,normalized_sha256:max:v1'
MODES = ('strict_replay', 'historical_source_as_of')


def eligible(record, cutoff, mode):
    if mode not in MODES:
        raise ValueError('unknown as-of mode')
    at = timestamp(cutoff)
    if record.available_at is None or timestamp(record.available_at) > at or timestamp(record.event_at) > at:
        return False
    if mode == 'strict_replay':
        if timestamp(record.observed_at) > at:
            return False
    elif record.availability_basis not in (AvailabilityBasis.SYNTHETIC, AvailabilityBasis.EVIDENCED):
        return False
    if type(record) is PriceBar:
        return record.complete and not record.filled and timestamp(record.end_at) <= at
    if type(record) is Quote and record.crossed:
        return False
    if type(record) is CorporateAction and timestamp(record.announced_at) > at:
        return False
    return True


def select_revisions(records, cutoff, mode, lineage):
    candidates = {}
    def complete(record):
        # Missing predecessors remain quarantined until resolved. Index metadata
        # survives permitted rights deletion, so a prior payload is not required.
        current = record.record_id
        seen = set()
        while current is not None:
            if current in seen or current not in lineage:
                return False
            seen.add(current)
            current = lineage[current]
        return True
    for record in records:
        if eligible(record, cutoff, mode) and complete(record):
            key = (record.schema_name, record.source_id, record.source_record_id)
            order = (record.revision, timestamp(record.available_at), timestamp(record.observed_at), record.record_id, record.hash)
            if key not in candidates or order > candidates[key][0]:
                candidates[key] = (order, record)
    return tuple(value[1] for _, value in sorted(candidates.items()))


@dataclass(frozen=True)
class SnapshotBundle:
    context: ContextSnapshot
    records: tuple
    manifest_json: str

    @property
    def manifest(self):
        return json.loads(self.manifest_json)


def build_snapshot(ledger, *, decision_at, price_seconds, text_seconds,
                   calendar_id, calendar_version, calendar_sha256, preprocessing_sha256,
                   feature_schema='market-observations-v1', mode='strict_replay',
                   expected_listings=(), price_max_age_seconds=60, forward_fill_seconds=0,
                   signals=()):
    """Build elapsed-window context, with externally evidenced status observations.

    signals are (listing_id, reason, evidence_record_id) tuples. Evidence must be
    selected and eligible. Status reasons describe feed_outage/market_closed;
    the caller owns their interpretation and versioned calendar. Session-window
    calculation and model feature computation are later packages.
    """
    if mode not in MODES:
        raise ValueError('unknown as-of mode')
    cutoff = timestamp(decision_at)
    for count in (price_seconds, text_seconds):
        if type(count) is not int or count <= 0:
            raise ValueError('positive elapsed context required')
    for count in (price_max_age_seconds, forward_fill_seconds):
        if type(count) is not int or count < 0:
            raise ValueError('nonnegative age limit required')
    if forward_fill_seconds > price_max_age_seconds:
        raise ValueError('forward fill cannot exceed freshness bound')
    sha256(calendar_sha256)
    sha256(preprocessing_sha256)
    if not calendar_id or not calendar_version:
        raise ValueError('calendar identity/version required')
    if any(type(x) is not str or not x for x in expected_listings):
        raise ValueError('expected listing IDs required')
    with ledger.locked():
        records = ledger._records()
        lineage = {r['record_id']: r['supersedes'] for r in ledger.db.execute('SELECT record_id,supersedes FROM events')}
        records = select_revisions(records, decision_at, mode, lineage)
        selected = []
        for record in records:
            if type(record) in (Instrument, InstrumentAlias):
                include = active(record, decision_at)
            else:
                seconds = text_seconds if type(record) is NewsEvent else price_seconds
                clock = record.end_at if type(record) is PriceBar else record.event_at
                include = timestamp(clock) >= cutoff - timedelta(seconds=seconds)
            if include:
                selected.append(record)
        selected = tuple(sorted(selected, key=lambda r: r.record_id))
        by_id = {r.record_id: r for r in selected}
        masks, stale, missing, fills = [], set(), set(), []
        prices = {}
        for record in selected:
            if type(record) in (PriceBar, Quote):
                clock = record.end_at if type(record) is PriceBar else record.event_at
                age = (cutoff - timestamp(clock)).total_seconds()
                prices.setdefault(record.listing_id, []).append((timestamp(clock), record.record_id, record, age))
                if record.stale or age > price_max_age_seconds:
                    stale.add(record.record_id)
                    masks.append({'record_id': record.record_id, 'reason': 'stale_price'})
                if record.missing:
                    missing.add(record.record_id)
                    masks.append({'record_id': record.record_id, 'reason': 'missing_price'})
            if type(record) is NewsEvent:
                if record.deleted or record.text is None:
                    missing.add(record.record_id)
                    masks.append({'record_id': record.record_id, 'reason': 'deleted_text' if record.deleted else 'empty_text'})
        for listing in sorted(set(expected_listings)):
            if listing not in prices:
                masks.append({'listing_id': listing, 'reason': 'missing_price'})
        for listing, reason, evidence in sorted(set(signals)):
            if reason not in ('feed_outage', 'market_closed') or evidence not in by_id:
                raise ValueError('status mask requires selected evidence and supported reason')
            record = by_id[evidence]
            related = getattr(record, 'listing_id', None) == listing
            if type(record) is NewsEvent:
                identities = [r for r in selected if type(r) is Instrument and r.listing_id == listing]
                related = any(i.security_id in e.instrument_ids for i in identities for e in record.entities)
            if not related:
                raise ValueError('status evidence does not identify listing')
            masks.append({'listing_id': listing, 'reason': reason, 'evidence_record_id': evidence})
        blocked = {m['listing_id'] for m in masks if m.get('reason') in ('feed_outage', 'market_closed')}
        fill_groups = {}
        for listing, candidates in prices.items():
            for item in candidates:
                record = item[2]
                basis = (listing, record.schema_name, record.source_id, record.venue, record.feed,
                         record.currency, record.mark.value if type(record) is PriceBar else 'bid_ask')
                fill_groups.setdefault(basis, []).append(item)
        for basis, candidates in sorted(fill_groups.items()):
            listing = basis[0]
            _, _, record, age = max(candidates, key=lambda item: (item[0], item[1]))
            if (forward_fill_seconds and 0 < age <= forward_fill_seconds and listing not in blocked
                    and record.record_id not in stale | missing):
                # A reference/view, never a fabricated zero or mutated complete bar.
                fills.append({'listing_id': listing, 'basis': list(basis), 'source_record_id': record.record_id,
                              'source_at': record.end_at if type(record) is PriceBar else record.event_at,
                              'target_at': decision_at, 'age_seconds': age, 'filled': True})
        caveats = []
        if mode == 'historical_source_as_of':
            caveats.append('Reconstructed from evidenced source availability; later observation allowed; historical revision coverage is not guaranteed.')
        if any(r.availability_basis == AvailabilityBasis.OBSERVED for r in selected):
            caveats.append('Conservative collector-observation proxy; not evidenced original publication availability.')
        manifest = {
            'version': 1, 'decision_at': decision_at, 'mode': mode, 'tie_rule': TIE_RULE,
            'inputs': [[r.record_id, r.hash] for r in selected],
            'identities': identity_manifest(selected, decision_at), 'dedup': dedup_manifest(selected),
            'calendar': [calendar_id, calendar_version, calendar_sha256],
            'preprocessing_sha256': preprocessing_sha256, 'feature_schema': feature_schema,
            'price_seconds': price_seconds, 'text_seconds': text_seconds,
            'expected_listings': sorted(set(expected_listings)),
            'price_max_age_seconds': price_max_age_seconds, 'forward_fill_seconds': forward_fill_seconds,
            'masks': sorted(masks, key=canonical), 'forward_fills': fills, 'caveats': caveats,
        }
        manifest_hash = digest(manifest)
        context = ContextSnapshot(
            schema_name='ContextSnapshot', schema_version='1.0.0', record_id='snapshot:' + manifest_hash,
            source_id='local-ledger-v1', source_record_id=manifest_hash, revision=1, supersedes_record_id=None,
            event_at=decision_at, available_at=decision_at, observed_at=decision_at, ingested_at=decision_at,
            availability_basis='collector_observation_proxy', availability_confidence='1',
            original_timezone='UTC', timestamp_precision='microsecond',
            provenance=Provenance(raw_sha256=manifest_hash, adapter_version='ledger-v1',
                rights='minimal-provenance-only', license='operator-policy',
                synthetic=all(r.provenance.synthetic for r in selected), untrusted_source_text=False),
            decision_at=decision_at, as_of_mode=mode, selected_record_ids=tuple(r.record_id for r in selected),
            selected_record_hashes=tuple(r.hash for r in selected),
            price_context=ContextWindow(clock='utc_elapsed_seconds', count=price_seconds, calendar_id=None, calendar_version=None),
            text_context=ContextWindow(clock='utc_elapsed_seconds', count=text_seconds, calendar_id=None, calendar_version=None),
            missing_record_ids=tuple(sorted(missing)), stale_record_ids=tuple(sorted(stale)),
            feature_schema=feature_schema, preprocessing_sha256=preprocessing_sha256,
            manifest_sha256=manifest_hash, caveats=tuple(caveats))
        payload = canonical({'context': context.to_dict(), 'manifest': manifest})
        ledger.db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?)', (context.hash, payload))
        return SnapshotBundle(context, selected, canonical(manifest))


def load_snapshot(ledger, snapshot_hash):
    sha256(snapshot_hash)
    with ledger.locked():
        row = ledger.db.execute('SELECT payload FROM snapshots WHERE hash=?', (snapshot_hash,)).fetchone()
        if row is None:
            raise KeyError(snapshot_hash)
        payload = json.loads(row[0])
        context = ContextSnapshot(**payload['context'])
        if context.hash != snapshot_hash or digest(payload['manifest']) != context.manifest_sha256:
            raise ValueError('snapshot manifest integrity failure')
        records = []
        for identity, expected in zip(context.selected_record_ids, context.selected_record_hashes):
            row = ledger.db.execute('SELECT * FROM events WHERE record_id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('snapshot references missing ledger entry')
            record = ledger._read(row)
            if record.hash != expected:
                raise ValueError('snapshot input integrity failure')
            records.append(record)
        return SnapshotBundle(context, tuple(records), canonical(payload['manifest']))
