"""Freeze universe evidence from immutable decision snapshots, never today's list."""
from dataclasses import dataclass

from ..schemas.base import Strict, sha256
from ..schemas.records import Instrument, InstrumentAlias, CorporateAction, NewsEvent
from cognitive_runtime.data.identity import active
from cognitive_runtime.data.ledger import digest


@dataclass(frozen=True, kw_only=True)
class UniverseManifest(Strict):
    snapshot_sha256: str
    decision_at: str
    universe_id: str
    identities: tuple[tuple[str, ...], ...]
    aliases: tuple[tuple[str, ...], ...]
    actions: tuple[tuple[str, ...], ...]
    coverage: str
    limitations: tuple[str, ...]

    def validate(self):
        sha256(self.snapshot_sha256)
        if self.coverage not in ('synthetic_only', 'partial_historical') or not self.limitations:
            raise ValueError('explicit historical/delisting coverage limitations required')


def verify_bundle(bundle):
    """Detect changed records/manifests; ledger reload remains the trust boundary."""
    c, m = bundle.context, bundle.manifest
    if (digest(m) != c.manifest_sha256 or
            tuple(r.record_id for r in bundle.records) != c.selected_record_ids or
            tuple(r.hash for r in bundle.records) != c.selected_record_hashes or
            m['quality_inputs'] != [[r.record_id, r.hash] for r in bundle.quality_records]):
        raise ValueError('snapshot bundle integrity failure')


def freeze_universe(bundle, universe_id, *, coverage, limitations):
    verify_bundle(bundle)
    records, at = bundle.records, bundle.context.decision_at
    instruments = [r for r in records if type(r) is Instrument and r.universe_id == universe_id and active(r, at)]
    listings = {r.listing_id: r for r in instruments}
    if len(listings) != len(instruments):
        raise ValueError('ambiguous listing identity')
    aliases = []
    for r in records:
        if type(r) is InstrumentAlias and r.listing_id in listings and active(r, at):
            if r.security_id != listings[r.listing_id].security_id:
                raise ValueError('alias/security conflict')
            aliases.append((r.namespace, r.alias, r.security_id, r.listing_id, r.record_id, r.hash))
    securities = {r.security_id for r in instruments}
    return UniverseManifest(snapshot_sha256=bundle.context.hash, decision_at=at, universe_id=universe_id,
        identities=tuple(sorted((r.issuer_id, r.security_id, r.listing_id, r.share_class, r.venue, r.record_id, r.hash) for r in instruments)),
        aliases=tuple(sorted(aliases)), actions=tuple(sorted((r.record_id, r.hash) for r in records
            if type(r) is CorporateAction and r.security_id in securities)), coverage=coverage, limitations=tuple(limitations))


def contamination_keys(bundle):
    """Source keys identify repeated source records (not mere common providers)."""
    verify_bundle(bundle)
    news = [r for r in bundle.records if type(r) is NewsEvent]
    return (tuple(sorted({r.dedup_cluster_id for r in news})),
            tuple(sorted({r.source_id + ':' + r.source_record_id for r in news})),
            tuple(sorted({e.issuer_id for r in news for e in r.entities})))
