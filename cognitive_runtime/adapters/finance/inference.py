"""Validate selected context and bind it to neutral inference inputs."""
from brain.cortex.model_contracts import InferenceInput
from .schemas import (ContextSnapshot, Instrument, InstrumentAlias, PriceBar, Quote,
                      CorporateAction, NewsEvent, PredictionContractState)
from .schemas.base import timestamp, AvailabilityBasis

OBSERVATIONS = (Instrument, InstrumentAlias, PriceBar, Quote, CorporateAction, NewsEvent, PredictionContractState)


def validate_context(snapshot, records):
    if type(snapshot) is not ContextSnapshot:
        raise TypeError('expected exact ContextSnapshot')
    records = tuple(records)
    if any(type(record) not in OBSERVATIONS for record in records):
        raise TypeError('terminal, labels and future targets are forbidden in context')
    if tuple(r.record_id for r in records) != snapshot.selected_record_ids or tuple(r.hash for r in records) != snapshot.selected_record_hashes:
        raise ValueError('selected record identities/hashes differ')
    cutoff = timestamp(snapshot.decision_at)
    for record in records:
        if record.available_at is None or timestamp(record.available_at) > cutoff:
            raise ValueError('unknown or future availability')
        if timestamp(record.event_at) > cutoff:
            raise ValueError('future event in context')
        if isinstance(record, CorporateAction) and timestamp(record.announced_at) > cutoff:
            raise ValueError('future announcement in context')
        if snapshot.as_of_mode == 'strict_replay':
            if timestamp(record.observed_at) > cutoff:
                raise ValueError('future observation in strict replay')
        elif record.availability_basis not in (AvailabilityBasis.EVIDENCED, AvailabilityBasis.SYNTHETIC):
            raise ValueError('historical reconstruction requires evidenced availability')
        if isinstance(record, PriceBar) and (not record.complete or record.filled):
            raise ValueError('incomplete/filled prices cannot enter context')
        if isinstance(record, Quote) and record.crossed:
            raise ValueError('crossed quotes cannot enter context')
        if getattr(record, 'stale', False) and record.record_id not in snapshot.stale_record_ids:
            raise ValueError('stale observation missing from context mask')
        if getattr(record, 'missing', False) and record.record_id not in snapshot.missing_record_ids:
            raise ValueError('missing observation absent from context mask')
    return records


def to_inference_input(snapshot, records, features):
    """Bind an already prepared feature vector to validated context identity.

    This does not fit preprocessing or certify arbitrary external features as
    causal. Feature builders in later issues own that semantic guarantee.
    """
    validate_context(snapshot, records)
    return InferenceInput(snapshot.feature_schema, (snapshot.record_id,), (tuple(features),))
