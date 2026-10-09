"""Public inference record catalog; deliberately excludes label/training types."""
from .base import VERSION, Record, Provenance
from .semantics import Target, Unit, Clock, Mark, Horizon
from .records import (
    Instrument, InstrumentAlias, PriceBar, Quote, CorporateAction, NewsEvent,
    PredictionContractState, ContextSnapshot, ForecastDistribution,
    ProviderCapabilityManifest, RunManifest, EvaluationReport,
)

RECORD_TYPES = (Instrument, InstrumentAlias, PriceBar, Quote, CorporateAction,
                NewsEvent, PredictionContractState, ContextSnapshot,
                ForecastDistribution, ProviderCapabilityManifest, RunManifest, EvaluationReport)
CATALOG = {kind.__name__: kind for kind in RECORD_TYPES}


def parse_record(payload):
    if type(payload) is not dict or payload.get('schema_name') not in CATALOG:
        raise ValueError('unknown or label/training-only record')
    if payload.get('schema_version') != VERSION:
        raise ValueError('stale/unknown schema version; explicit migration required')
    return CATALOG[payload['schema_name']](**payload)
