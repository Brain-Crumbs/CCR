"""Train-fitted, dependency-light hashing/TF-IDF and frozen encoder features.

The fit entry point accepts a Corpus + validated SplitPlan, never an arbitrary
held-out document list. These are API integrity boundaries, not a Python sandbox.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re

from ..schemas.records import ContextSnapshot, NewsEvent
from ..datasets.splits import validate_split
from cognitive_runtime.data.ledger import digest
from cognitive_runtime.data.snapshot import eligible
from .cache import FeatureCache, cache_key

MAX_RECORDS = 128
MAX_CHARS = 4096
TOKENIZER = {'version': 'ascii-words-v1', 'pattern': '[a-z0-9]+', 'lowercase': True}
TOKENIZER_HASH = digest(TOKENIZER)


def selected_news(feature):
    """Read only #291 context records; check exact IDs/content and knowledge clocks.

    Quality and outcome stores are never sources of model text. Missing news is
    represented explicitly by a zero-row artifact, not a made-up neutral story.
    """
    if set(feature) != {'version', 'sample_id', 'context', 'universe', 'records', 'quality_records'} or feature['version'] != 1:
        raise ValueError('expected corpus feature artifact')
    context = ContextSnapshot(**feature['context'])
    selected = dict(zip(context.selected_record_ids, context.selected_record_hashes))
    if len(feature['records']) != len(selected) or {r['record_id'] for r in feature['records']} != set(selected):
        raise ValueError('snapshot record membership mismatch')
    news = []
    for raw in feature['records']:
        if digest(raw) != selected[raw['record_id']]:
            raise ValueError('snapshot record content mismatch')
        if raw['schema_name'] == 'NewsEvent':
            record = NewsEvent(**raw)
            if not eligible(record, context.decision_at, context.as_of_mode):
                raise ValueError('text not known at context cutoff')
            if not record.deleted:
                news.append(record)
    if len(news) > MAX_RECORDS:
        raise ValueError('text snapshot exceeds 128 records; narrow context upstream')
    return context, tuple(sorted(news, key=lambda r: r.record_id))


def source_text(record):
    # Offset convention used by extraction: Unicode code points in title + LF + body.
    title = record.title[:MAX_CHARS]
    remaining = MAX_CHARS - len(title)
    return title + ('\n' + (record.text or '')[:remaining-1] if remaining else '')


def tokens(record):
    return re.findall(TOKENIZER['pattern'], source_text(record).lower())


@dataclass(frozen=True)
class FeatureArtifact:
    sample_id: str
    snapshot_sha256: str
    record_ids: tuple
    record_sha256: tuple
    truncated: tuple
    values: tuple
    dimensions: int
    signature: str
    provenance_json: str
    dtype: str = 'float32'

    def to_dict(self):
        return {'version': 1, 'sample_id': self.sample_id, 'snapshot_sha256': self.snapshot_sha256,
                'record_ids': list(self.record_ids), 'record_sha256': list(self.record_sha256),
                'truncated': list(self.truncated), 'values': [list(row) for row in self.values],
                'dimensions': self.dimensions, 'dtype': self.dtype, 'feature_signature': self.signature,
                'provenance': json.loads(self.provenance_json), 'missing_text': not self.record_ids}

    @property
    def hash(self):
        return digest(self.to_dict())


def finite_vector(value, dimensions):
    # numpy is core; float32 conversion makes the advertised storage dtype truthful.
    import numpy as np
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dimensions,) or not np.isfinite(array).all():
        raise ValueError('encoder returned wrong dimensions or nonfinite values')
    return tuple(float(x) for x in array)


class TextFeatures:
    """Frozen hash counts or train-only TF-IDF vocabulary; no implicit refitting."""
    def __init__(self, mode='hashing', dimensions=256, *, cache=None):
        if mode not in ('hashing', 'tfidf') or type(dimensions) is not int or not 1 <= dimensions <= 4096:
            raise ValueError('expected hashing/tfidf and dimensions in [1,4096]')
        self._mode, self._dimensions = mode, dimensions
        self.cache = cache if cache is not None else FeatureCache()
        self._state = None

    @property
    def mode(self):
        return self._mode

    @property
    def dimensions(self):
        return self._dimensions

    def fit(self, corpus, plan):
        if self._state is not None:
            raise ValueError('already fitted; create a new feature transformer')
        validate_split(corpus.samples, plan)
        if not plan.train or len(plan.train) > 4096:
            raise ValueError('fit requires 1..4096 training samples')
        # Bound total train documents and vocabulary growth before materialization.
        df, seen, documents = Counter(), set(), 0
        for key in plan.train:
            _, records = selected_news(corpus.feature(key))
            for record in records:
                if record.hash in seen:
                    continue
                seen.add(record.hash)
                documents += 1
                if documents > 4096:
                    raise ValueError('fit exceeds 4096 unique train text records')
                if self.mode == 'tfidf':
                    df.update(set(tokens(record)))
                    if len(df) > 65536:
                        raise ValueError('train vocabulary exceeds 65536 terms')
        vocab = sorted(df, key=lambda word: (-df[word], word))[:self.dimensions]
        self._state = {'version': 1, 'mode': self.mode, 'dimensions': self.dimensions,
                       'vocabulary': vocab, 'idf': [math.log((1 + documents)/(1 + df[w])) + 1 for w in vocab],
                       'train_records': sorted(seen), 'train_sample_ids': list(plan.train),
                       'corpus_sha256': corpus.manifest_sha256, 'split_sha256': plan.hash,
                       'tokenizer_sha256': TOKENIZER_HASH, 'max_chars': MAX_CHARS,
                       'normalization': 'l2', 'dtype': 'float32', 'hash': 'sha256-first8-big-unsigned-v1'}
        return self

    @property
    def state(self):
        if self._state is None:
            raise ValueError('fit on a validated training partition before transform')
        return json.loads(json.dumps(self._state))

    @property
    def signature(self):
        return digest(self.state)

    def _vector(self, record):
        counts = Counter(tokens(record))
        vector = [0.] * self.dimensions
        if self.mode == 'hashing':
            for word, count in counts.items():
                index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'big') % self.dimensions
                vector[index] += count
        else:
            for index, (word, idf) in enumerate(zip(self._state['vocabulary'], self._state['idf'])):
                vector[index] = counts[word] * idf
        norm = math.sqrt(sum(v*v for v in vector))
        return finite_vector([v/norm if norm else 0. for v in vector], self.dimensions)

    def transform(self, feature):
        signature = self.signature
        context, records = selected_news(feature)
        values = []
        for record in records:
            key = cache_key(record, model_sha256=signature, tokenizer_sha256=TOKENIZER_HASH,
                            preprocessing_sha256=signature)
            cached = self.cache.get(key)
            vector = self._vector(record) if cached is None else finite_vector(json.loads(cached), self.dimensions)
            if cached is None:
                self.cache.put(key, vector)
            values.append(vector)
        return artifact(feature, context, records, values, self.dimensions, signature,
                        {'kind': self.mode, 'pretrained': False, 'training': self.state,
                         'trainable_projections': 'not included; owned by downstream supervised/JEPA model'})


def artifact(feature, context, records, values, dimensions, signature, provenance):
    from cognitive_runtime.data.ledger import canonical
    return FeatureArtifact(feature['sample_id'], context.hash, tuple(r.record_id for r in records),
        tuple(r.hash for r in records), tuple(len(r.title) + 1 + len(r.text or '') > MAX_CHARS for r in records),
        tuple(values), dimensions, signature, canonical(provenance))


class FrozenTextFeatures:
    """Share immutable backbone outputs; context/target projections remain external.

    Encoder contract: dimensions, signature, tokenizer_sha256, provenance, and
    encode(list[str]). Fake encoders exercise this path without optional imports.
    """
    def __init__(self, encoder, *, cache=None):
        if type(encoder.dimensions) is not int or not 1 <= encoder.dimensions <= 4096:
            raise ValueError('invalid encoder dimensions')
        self.encoder = encoder
        self.cache = cache if cache is not None else FeatureCache()

    def transform(self, feature):
        context, records = selected_news(feature)
        preprocessing = digest({'version': 1, 'max_chars': MAX_CHARS, 'source': 'title-LF-body'})
        signature = digest({'encoder': self.encoder.signature, 'tokenizer': self.encoder.tokenizer_sha256,
                            'preprocessor': preprocessing})
        values = []
        # Each batch is <=8; processing whole-snapshot cache misses stays bounded.
        for offset in range(0, len(records), 8):
            batch = records[offset:offset+8]
            keys = [cache_key(r, model_sha256=self.encoder.signature,
                    tokenizer_sha256=self.encoder.tokenizer_sha256, preprocessing_sha256=preprocessing) for r in batch]
            rows = [self.cache.get(k) for k in keys]
            missing = [i for i, row in enumerate(rows) if row is None]
            if missing:
                encoded = self.encoder.encode([source_text(batch[i]) for i in missing])
                if len(encoded) != len(missing):
                    raise ValueError('encoder returned wrong batch length')
                for i, value in zip(missing, encoded):
                    row = finite_vector(value, self.encoder.dimensions)
                    self.cache.put(keys[i], row)
                    rows[i] = json.dumps(row)
            values.extend(finite_vector(json.loads(row), self.encoder.dimensions) for row in rows)
        return artifact(feature, context, records, values, self.encoder.dimensions, signature,
                        {**self.encoder.provenance, 'frozen_backbone': True,
                         'trainable_projections': 'not included; owned by downstream supervised/JEPA model'})
