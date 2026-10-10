"""Point-in-time identities and explicit story clusters, never ticker joins."""
from cognitive_runtime.adapters.finance.schemas.base import timestamp
from cognitive_runtime.adapters.finance.schemas.records import Instrument, InstrumentAlias, NewsEvent


def active(record, at):
    at = timestamp(at)
    return timestamp(record.effective_from) <= at and (record.effective_to is None or at < timestamp(record.effective_to))


def resolve_alias(records, *, alias, namespace, at):
    """Input must be the eligible snapshot records, not the unfiltered ledger."""
    instruments = {r.listing_id: r for r in records if type(r) is Instrument and active(r, at)}
    matches = set()
    for record in records:
        if type(record) is InstrumentAlias and record.alias == alias and record.namespace == namespace and active(record, at):
            instrument = instruments.get(record.listing_id)
            if instrument and instrument.security_id == record.security_id:
                matches.add((instrument.issuer_id, instrument.security_id, instrument.listing_id, instrument.share_class))
    if len(matches) != 1:
        raise ValueError('missing or ambiguous point-in-time alias')
    return next(iter(matches))


def identity_manifest(records, at):
    values = {}
    for record in records:
        if type(record) is Instrument and active(record, at):
            identity = (record.issuer_id, record.security_id, record.listing_id, record.share_class, record.venue)
            if record.listing_id in values and values[record.listing_id] != identity:
                raise ValueError('conflicting point-in-time listing identities')
            values[record.listing_id] = identity
    return [list(values[key]) for key in sorted(values)]


def dedup_manifest(records):
    """Declared cluster + exact entity identities. No fuzzy matching/backdating.

    Preserve every member/version clock. The representative is the earliest
    eligible available/observed version, with record ID/hash as final ties.
    """
    groups = {}
    for record in records:
        if type(record) is NewsEvent:
            entities = tuple(sorted((e.issuer_id, tuple(sorted(e.instrument_ids)), e.ambiguous) for e in record.entities))
            groups.setdefault((record.dedup_cluster_id, entities), []).append(record)
    result = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: (timestamp(r.available_at), timestamp(r.observed_at), r.record_id, r.hash))
        result.append({'cluster': key[0], 'entities': key[1], 'representative': members[0].record_id,
                       'members': [{'record_id': r.record_id, 'available_at': r.available_at,
                                    'observed_at': r.observed_at, 'hash': r.hash} for r in members]})
    return result
