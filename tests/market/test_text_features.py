"""MIT project-authored synthetic text, seed 292; manifest binds static traps."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cognitive_runtime.adapters.finance.schemas.records import NewsEvent
from cognitive_runtime.adapters.finance.testing.examples import examples, envelope, seal_source, source_payload
from cognitive_runtime.adapters.finance.datasets.corpus import Corpus, Request, build_corpus
from cognitive_runtime.adapters.finance.datasets.splits import walk_forward
from cognitive_runtime.adapters.finance.schemas.semantics import Horizon, Clock, Target, Unit, Mark
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical, digest
from cognitive_runtime.data.snapshot import build_snapshot
from cognitive_runtime.adapters.finance.text.features import (
    TextFeatures, FrozenTextFeatures, selected_news, source_text, MAX_CHARS,
)
from cognitive_runtime.adapters.finance.text.cache import FeatureCache, cache_key
from cognitive_runtime.adapters.finance.text.extractor import StructuredExtractor, validate_output, SCHEMA_HASH, PROMPT_HASH
from cognitive_runtime.adapters.finance.text.local import (
    LocalModel, OptionalCapabilityError, verify_staged, read_pin, REQUIRED, SETTINGS, MODELS,
)

FIXTURES = Path(__file__).parents[1] / 'fixtures/market/text'
TRAPS = json.loads((FIXTURES / 'traps.json').read_text())
POLICY = RetentionPolicy(rights='project-authored-fixture', evidence='MIT synthetic seed 292',
                        allow_raw=True, allow_normalized=True, retain_provenance=True)


def news(text, identity='news', day=6, **changes):
    at = f'2025-01-{day:02}T14:30:00Z'
    return seal_source(replace(examples()['NewsEvent'], **{**envelope(NewsEvent, identity, at),
        'original_published_at': at, 'first_seen_at': at, 'title': 'Synthetic', 'text': text,
        'dedup_cluster_id': identity, **changes}))


def feature(records, sample_id='sample'):
    # Explicit snapshot contract, not standalone text passed directly to a model.
    at = max([r.observed_at for r in records] + ['2025-01-06T14:30:00Z'])
    context = replace(examples()['ContextSnapshot'], decision_at=at,
        selected_record_ids=tuple(r.record_id for r in records), selected_record_hashes=tuple(r.hash for r in records))
    return {'version': 1, 'sample_id': sample_id, 'context': context.to_dict(), 'universe': {},
            'records': [r.to_dict() for r in records], 'quality_records': []}


@pytest.fixture
def corpus(tmp_path):
    # Real #291 builder, disk reopen and split validation. Censored rows survive;
    # fitting has no reason to read their separately stored labels.
    with Ledger(tmp_path / 'ledger') as ledger:
        requests = []
        for day, text in zip((6, 7, 8), ('apple apple banana', 'validationonly', 'testonly')):
            record = news(text, f'news-{day}', day)
            ledger.append(record, canonical(source_payload(record)).encode(), POLICY)
            snapshot = build_snapshot(ledger, decision_at=record.observed_at, price_seconds=60,
                text_seconds=60, calendar_id='synthetic', calendar_version='1', calendar_sha256='a'*64,
                preprocessing_sha256='b'*64)
            outcome = build_snapshot(ledger, decision_at=record.observed_at.replace('14:30', '14:31'),
                price_seconds=86400, text_seconds=86400, calendar_id='synthetic', calendar_version='1',
                calendar_sha256='a'*64, preprocessing_sha256='b'*64)
            requests.append(Request(sample_id=str(day), snapshot_sha256=snapshot.context.hash,
                outcome_snapshot_sha256=outcome.context.hash, listing_id='contract:demo', universe_id='synthetic-universe-v1'))
        spec = replace(examples()['LabelSpec'], target=Target.CONTRACT_CHANGE, unit=Unit.PRICE_CHANGE,
            mark=Mark.MID, action_policy='none', horizon=Horizon(clock=Clock.ELAPSED, count=60, calendar_id=None, calendar_version=None))
        built = build_corpus(ledger, tmp_path / 'corpus', requests, spec=spec)
    result = Corpus(built.root, built.manifest_sha256)
    plan = walk_forward(result.samples, validation_at='2025-01-07T00:00:00Z', test_at='2025-01-08T00:00:00Z')
    return result, plan


def test_fixture_provenance():
    manifest = json.loads((FIXTURES / 'provenance.json').read_text())
    assert manifest['synthetic'] is True and manifest['license'] == 'MIT' and manifest['seed'] == 292
    assert hashlib.sha256((FIXTURES / 'traps.json').read_bytes()).hexdigest() == manifest['sha256']


@pytest.mark.parametrize('mode', ['hashing', 'tfidf'])
def test_train_only_fit_and_artifacts(corpus, mode):
    corpus, plan = corpus
    touched = []
    read = corpus.feature
    corpus.feature = lambda key: (touched.append(key), read(key))[1]
    model = TextFeatures(mode, 16).fit(corpus, plan)
    assert touched == ['6']
    assert not {'validationonly', 'testonly'} & set(model.state['vocabulary'])
    before = model.signature
    artifact = model.transform(corpus.feature('8'))
    assert model.signature == before and artifact.dimensions == 16
    assert artifact.dtype == 'float32' and artifact.record_ids == ('news-8',)
    assert artifact.hash == model.transform(corpus.feature('8')).hash
    assert model.state['train_sample_ids'] == ['6']
    # Reopened corpus labels cannot be consulted by preprocessing.
    assert not any('label' in name for name in artifact.to_dict())
    state = model.state
    state['vocabulary'].append('pollution')
    assert model.signature == before
    with pytest.raises(ValueError, match='already fitted'):
        model.fit(corpus, plan)
    with pytest.raises(ValueError):
        TextFeatures(mode).fit(corpus, replace(plan, train=plan.test, test=plan.train))


def test_tfidf_exact_train_counts(corpus):
    corpus, plan = corpus
    model = TextFeatures('tfidf', 4).fit(corpus, plan)
    assert model.state['vocabulary'] == ['apple', 'banana', 'synthetic']
    assert model.state['idf'] == [1., 1., 1.]
    assert model.transform(corpus.feature('6')).values[0] == pytest.approx([2/6**.5, 1/6**.5, 1/6**.5, 0])
    # Unseen held-out words get no vocabulary slot, only known title survives.
    assert model.transform(corpus.feature('8')).values[0] == (0., 0., 1., 0.)


def test_missing_deleted_truncated_and_invalid(corpus):
    corpus, plan = corpus
    model = TextFeatures().fit(corpus, plan)
    assert model.transform(feature([])).to_dict()['missing_text']
    assert not model.transform(feature([news(None, deleted=True)])).values
    long = news('x' * (MAX_CHARS*2))
    assert len(source_text(long)) == MAX_CHARS
    assert model.transform(feature([long])).truncated == (True,)
    invalid = feature([news('valid')])
    invalid['records'][0]['text'] = 'tampered'
    with pytest.raises(ValueError, match='content mismatch'):
        model.transform(invalid)
    invalid = feature([news('valid')])
    invalid['context']['decision_at'] = '2025-01-05T14:30:00Z'
    with pytest.raises(ValueError, match='not known'):
        model.transform(invalid)
    with pytest.raises(ValueError, match='128'):
        model.transform(feature([news('a', str(i)) for i in range(129)]))
    with pytest.raises(ValueError, match='fit'):
        TextFeatures().transform(feature([]))


class FakeEncoder:
    dimensions = 3
    signature = 'a'*64
    tokenizer_sha256 = 'b'*64
    provenance = {'kind': 'fake', 'pretrained': False, 'capability': 'synthetic_only'}
    def __init__(self):
        self.calls = []
    def encode(self, texts):
        self.calls.append(texts)
        return [[1., 0., float(len(text))] for text in texts]


def test_frozen_cache_revision_and_identity():
    encoder = FakeEncoder()
    cache = FeatureCache()
    model = FrozenTextFeatures(encoder, cache=cache)
    original = news('apple')
    baseline = model.transform(feature([original]))
    assert model.transform(feature([original])).hash == baseline.hash and len(encoder.calls) == 1
    for revised in (replace(original, text='banana'), replace(original, revision=2, supersedes_record_id="previous"), replace(original, record_id='different')):
        model.transform(feature([revised]))
    assert len(encoder.calls) == 4
    encoder.tokenizer_sha256 = 'c'*64
    model.transform(feature([original]))
    encoder.signature = 'd'*64
    model.transform(feature([original]))
    assert len(encoder.calls) == 6
    assert baseline.to_dict()['provenance']['frozen_backbone']
    keys = [cache_key(original, model_sha256='a'*64, tokenizer_sha256='b'*64, preprocessing_sha256=h*64) for h in ('c','d')]
    assert keys[0] != keys[1]


def test_encoder_bounds_and_cache_eviction():
    encoder = FakeEncoder()
    model = FrozenTextFeatures(encoder, cache=FeatureCache(1))
    model.transform(feature([news('a', str(i)) for i in range(17)]))
    assert [len(batch) for batch in encoder.calls] == [8,8,1]
    encoder.encode = lambda texts: [[float('nan')]*3 for t in texts]
    with pytest.raises(ValueError, match='nonfinite'):
        model.transform(feature([news('bad')]))
    encoder.encode = lambda texts: []
    with pytest.raises(ValueError, match='batch length'):
        model.transform(feature([news('bad')]))
    cache = FeatureCache(1, 20)
    cache.put('a', [1])
    cache.put('b', [2])
    assert cache.get('a') is None
    with pytest.raises(ValueError, match='byte bound'):
        cache.put('c', 'x'*30)
    cache.clear()
    assert cache.get('b') is None


class FakeGenerator:
    signature = 'c'*64
    tokenizer_sha256 = 'd'*64
    provenance = {'kind': 'fake', 'capability': 'synthetic_only'}
    def __init__(self, output):
        self.output, self.calls = output, []
    def generate(self, text):
        self.calls.append(text)
        return self.output


def claim(text):
    return {'claims': [{'kind':'production','direction':'unknown','uncertainty':'high',
        'start':0,'end':len(text),'evidence':text}]}


def test_extraction_evidence_uncertainty_cache_and_provenance():
    record = news('α fictional production')
    text = source_text(record)
    generator = FakeGenerator(json.dumps(claim(text)))
    extractor = StructuredExtractor(generator)
    result = extractor.transform(feature([record]))
    assert result.hash == extractor.transform(feature([record])).hash and len(generator.calls) == 1
    value = result.to_dict()
    assert value['records'][0]['claims'][0]['evidence'] == text
    assert value['schema_sha256'] == SCHEMA_HASH and value['prompt_sha256'] == PROMPT_HASH
    assert value['claim_status'] == 'model_derived_not_verified_facts'
    assert value['records'][0]['claims'][0]['uncertainty'] == 'high'
    generator.tokenizer_sha256 = 'f'*64
    extractor.transform(feature([record]))
    assert len(generator.calls) == 2


@pytest.mark.parametrize('raw', TRAPS['invalid_outputs'])
def test_schema_rejects_malicious_malformed_output(raw):
    with pytest.raises(ValueError):
        validate_output(raw, 'Synthetic\nfictional production')


def test_evidence_mismatch_and_injection_is_data():
    text = TRAPS['injection']
    generator = FakeGenerator('{"claims": []}')
    result = StructuredExtractor(generator).transform(feature([news(text)]))
    assert result.to_dict()['records'][0]['claims'] == []
    assert generator.calls == ['Synthetic\n'+text]  # no action channel exists
    value = claim('wrong evidence')
    with pytest.raises(ValueError, match='exact source'):
        validate_output(json.dumps(value), source_text(news(text)))
    value = claim('x')
    value['claims'][0]['start'] = True
    with pytest.raises(ValueError, match='span'):
        validate_output(json.dumps(value), text)


def fake_pin(tmp_path, model='minilm'):
    # Clearly synthetic identity; never checked into the production lock.
    root = tmp_path / model
    root.mkdir()
    files = {}
    for name in REQUIRED[model] | {'README.md','LICENSE'}:
        content = ('synthetic fixture ' + name).encode()
        (root / name).write_bytes(content)
        files[name] = {'size':len(content),'sha256':hashlib.sha256(content).hexdigest(),
                      'evidence_url':'https://huggingface.co/'+MODELS[model]+'/synthetic-test-only'}
    pin = {'status':'verified','repository':MODELS[model], 'revision':'a'*40,'tokenizer_revision':'a'*40,
           'license':'Apache-2.0','files':files,'settings':SETTINGS[model],'training_cutoff':'unknown'}
    lock = tmp_path/'lock.json'
    lock.write_text(json.dumps({'version':1,'models':{model:pin}}))
    return pin, root, lock


def test_pin_and_local_inventory_fail_closed(tmp_path):
    pin, root, lock = fake_pin(tmp_path)
    assert read_pin(lock, 'minilm') == pin
    assert verify_staged(pin, root) == root
    with pytest.raises(OptionalCapabilityError, match='missing_local_artifacts'):
        verify_staged(pin, tmp_path/'absent')
    (root/'config.json').write_bytes(b'x'*pin['files']['config.json']['size'])
    with pytest.raises(OptionalCapabilityError, match='SHA256'):
        verify_staged(pin, root)
    (root/'extra.py').write_text('raise RuntimeError()')
    with pytest.raises(OptionalCapabilityError, match='inventory'):
        verify_staged(pin, root)
    pin['revision']='main'
    lock.write_text(json.dumps({'version':1,'models':{'minilm':pin}}))
    with pytest.raises(ValueError, match='immutable'):
        read_pin(lock, 'minilm')


def test_shipped_unresolved_pin_is_explicit_blocker(tmp_path):
    lock = Path(__file__).parents[2]/'config/market/models.lock.json'
    for name in MODELS:
        with pytest.raises(OptionalCapabilityError, match='unverified_model_pin'):
            LocalModel(name, lock_path=lock, staged=tmp_path)


def test_optional_import_boundary():
    code = '''
import sys
class Deny:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','transformers','sentence_transformers'):
   raise AssertionError('optional import: '+fullname)
sys.meta_path.insert(0,Deny())
from cognitive_runtime.adapters.finance.text import features, cache, local, extractor
assert not {'torch','transformers'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_missing_optional_dependency_status(tmp_path):
    _, root, lock = fake_pin(tmp_path)
    # Base suite must run without torch/transformers; fake file hashes are valid,
    # but their contents must never be treated as real model validation.
    with pytest.raises(OptionalCapabilityError, match='missing_optional_dependencies'):
        LocalModel('minilm', lock_path=lock, staged=root)


def test_local_request_bounds_offline_flags_and_timeout(tmp_path, monkeypatch):
    import cognitive_runtime.adapters.finance.text.local as local
    _, root, lock = fake_pin(tmp_path)
    monkeypatch.setattr(local.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(local.importlib.metadata, 'version', lambda name: 'synthetic-test-only')
    model = LocalModel('minilm', lock_path=lock, staged=root)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({'network_attempts':0,'metrics':{},'result':[[1.,0.,0.]]}), '')
    monkeypatch.setattr(local.subprocess, 'run', run)
    assert model.encode(['Synthetic']) == [[1.,0.,0.]]
    command, kwargs = calls[0]
    assert command[-1] == 'cognitive_runtime.adapters.finance.text.worker'
    assert kwargs['timeout'] == 60 and kwargs['env']['HF_HUB_OFFLINE'] == '1'
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and kwargs['env']['OMP_NUM_THREADS'] == '2'
    assert json.loads(kwargs['input'])['staged'] == str(root)
    for texts in ([], ['a']*9, ['a'*(MAX_CHARS+1)]):
        with pytest.raises(ValueError, match='bound'):
            model.encode(texts)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 60)
    monkeypatch.setattr(local.subprocess, 'run', timeout)
    with pytest.raises(OptionalCapabilityError, match='resource_limit'):
        model.encode(['x'])


def test_cache_value_immutable():
    value = [1.]
    cache = FeatureCache()
    cache.put('key', value)
    value.append(2.)
    assert json.loads(cache.get('key')) == [1.]
