"""Explicit copy-on-write migration. Parsing never silently changes old hashes."""
from copy import deepcopy
from . import parse_record


def migrate_legacy_0_9(payload):
    """Supported pilot 0.9.0 omitted availability_confidence.

    Additive 1.0 requires that field. Confidence is conservatively zero, never
    an invented claim. All old bytes/hashes remain the caller's immutable input.
    Breaking/unknown versions require a separately reviewed converter.
    """
    if type(payload) is not dict or payload.get('schema_version') != '0.9.0':
        raise ValueError('unsupported prior schema version; breaking migration unavailable')
    if 'availability_confidence' in payload:
        raise ValueError('invalid 0.9.0 payload: unexpected confidence field')
    result = deepcopy(payload)
    result.update(schema_version='1.0.0', availability_confidence='0')
    record = parse_record(result)
    return record
