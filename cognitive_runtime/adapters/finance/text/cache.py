"""Bounded process-local cache. Retention/revocation of exported features is external."""
from collections import OrderedDict

from cognitive_runtime.data.ledger import digest


def cache_key(record, *, model_sha256, tokenizer_sha256, preprocessing_sha256):
    return digest({'version': 1, 'record_id': record.record_id, 'revision': record.revision,
                   'record_sha256': record.hash, 'model': model_sha256,
                   'tokenizer': tokenizer_sha256, 'preprocessing': preprocessing_sha256})


class FeatureCache:
    """LRU of immutable values; bounded item bytes as well as entry count."""
    def __init__(self, max_entries=256, max_value_bytes=131072):
        if type(max_entries) is not int or not 1 <= max_entries <= 1024:
            raise ValueError('cache entries must be in [1,1024]')
        if type(max_value_bytes) is not int or not 1 <= max_value_bytes <= 131072:
            raise ValueError('cache item bytes must be in [1,131072]')
        self.max_entries, self.max_value_bytes = max_entries, max_value_bytes
        self._values = OrderedDict()

    def get(self, key):
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key, value):
        # Canonical JSON also rejects non-finite values and avoids mutable aliases.
        from cognitive_runtime.data.ledger import canonical
        data = canonical(value)
        if len(data.encode()) > self.max_value_bytes:
            raise ValueError('cache value exceeds byte bound')
        self._values[key] = data
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def clear(self):
        self._values.clear()
