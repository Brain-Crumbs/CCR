"""Strict immutable wire types. No provider, model, or training imports."""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import re
import types
from typing import Union, get_args, get_origin, get_type_hints

VERSION = '1.0.0'


def decimal(value: str) -> Decimal:
    """Wire decimals are finite plain strings; never accept binary floats."""
    if type(value) is not str or not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?', value):
        raise ValueError('decimal must be a plain decimal string')
    result = Decimal(value)
    if not result.is_finite() or len(result.as_tuple().digits) > 34:
        raise ValueError('decimal exceeds finite 34-digit contract')
    return result


def timestamp(value: str) -> datetime:
    if type(value) is not str or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z', value):
        raise ValueError('timestamp requires explicit UTC Z, at most microseconds')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def sha256(value: str) -> None:
    if not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('expected lowercase SHA256')


def _coerce(kind, value):
    origin, args = get_origin(kind), get_args(kind)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return _coerce(choice, value)
            except (TypeError, ValueError):
                pass
        raise ValueError('value does not match union')
    if kind is type(None):
        if value is not None:
            raise TypeError('expected null')
        return None
    if origin is tuple:
        if type(value) not in (tuple, list):
            raise TypeError('expected array')
        return tuple(_coerce(args[0], item) for item in value)
    if isinstance(kind, type) and issubclass(kind, Strict):
        if type(value) is kind:
            return value
        if type(value) is dict:
            return kind(**value)
        raise TypeError('expected exact record type')
    if isinstance(kind, type) and issubclass(kind, Enum):
        if type(value) is kind:
            return value
        if type(value) is not str:
            raise TypeError('expected enum string')
        return kind(value)
    if type(value) is not kind:
        raise TypeError(f'expected {kind}, received {type(value)}')
    if kind is str and not value:
        raise ValueError('empty strings are forbidden; use null where supported')
    return value


def _wire(value):
    if isinstance(value, Strict):
        return {f.name: _wire(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_wire(x) for x in value]
    return value


@dataclass(frozen=True, kw_only=True)
class Strict:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            object.__setattr__(self, field.name, _coerce(hints[field.name], getattr(self, field.name)))
        self.validate()

    def validate(self):
        pass

    def to_dict(self):
        return _wire(self)

    def canonical_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

    @property
    def hash(self):
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    @classmethod
    def from_json(cls, data):
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError('duplicate JSON key')
                out[key] = value
            return out
        return cls(**json.loads(data, object_pairs_hook=unique))


@dataclass(frozen=True, kw_only=True)
class Provenance(Strict):
    raw_sha256: str
    adapter_version: str
    rights: str
    license: str
    synthetic: bool
    untrusted_source_text: bool

    def validate(self):
        sha256(self.raw_sha256)


class AvailabilityBasis(str, Enum):
    SYNTHETIC = 'synthetic_ground_truth'
    EVIDENCED = 'evidenced_historical'
    OBSERVED = 'collector_observation_proxy'
    UNKNOWN = 'unknown'


@dataclass(frozen=True, kw_only=True)
class Record(Strict):
    schema_name: str
    schema_version: str
    record_id: str
    source_id: str
    source_record_id: str
    revision: int
    supersedes_record_id: str | None
    event_at: str
    available_at: str | None
    observed_at: str
    ingested_at: str
    availability_basis: AvailabilityBasis
    availability_confidence: str
    original_timezone: str
    timestamp_precision: str
    provenance: Provenance

    def validate(self):
        if self.schema_name != type(self).__name__ or self.schema_version != VERSION:
            raise ValueError('schema name/version mismatch; explicit migration required')
        if self.revision < 1 or (self.revision == 1) != (self.supersedes_record_id is None):
            raise ValueError('revision requires explicit predecessor after version one')
        if self.supersedes_record_id == self.record_id:
            raise ValueError('record cannot supersede itself')
        timestamp(self.event_at)
        observed, ingested = timestamp(self.observed_at), timestamp(self.ingested_at)
        if ingested < observed:
            raise ValueError('ingestion precedes observation')
        if self.available_at is not None:
            timestamp(self.available_at)
        if (self.available_at is None) != (self.availability_basis == AvailabilityBasis.UNKNOWN):
            raise ValueError('unknown availability must remain null')
        if self.availability_basis == AvailabilityBasis.OBSERVED and self.available_at != self.observed_at:
            raise ValueError('observation proxy must equal first observation')
        if not 0 <= decimal(self.availability_confidence) <= 1:
            raise ValueError('confidence outside [0,1]')
        if self.timestamp_precision not in ('second', 'millisecond', 'microsecond'):
            raise ValueError('unsupported timestamp precision')
        precision = {'second': 0, 'millisecond': 3, 'microsecond': 6}[self.timestamp_precision]
        for value in (self.event_at, self.available_at, self.observed_at, self.ingested_at):
            if value and '.' in value and len(value.split('.')[1][:-1]) > precision:
                raise ValueError('clock exceeds declared precision')
