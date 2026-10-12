"""Explicit opt-in, pinned, local-only model execution in a bounded CPU child.

No download helper is exposed. Operators stage a reviewed allowlist themselves.
Model lock verification precedes optional imports, including inside the worker.
"""
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from cognitive_runtime.data.ledger import digest, canonical
from .features import MAX_CHARS

CONTAMINATION = ('Training cutoff unknown. Historical memorization/contamination is possible; '
                 'compare non-pretrained baselines and prospective/post-release holdouts. '
                 'Artifact identity and name masking do not establish leakage freedom.')
MODELS = {'minilm': 'sentence-transformers/all-MiniLM-L6-v2', 'qwen': 'Qwen/Qwen2.5-0.5B-Instruct'}
REQUIRED = {'minilm': {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                       'special_tokens_map.json', 'vocab.txt'},
            'qwen': {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                     'vocab.json', 'merges.txt', 'generation_config.json'}}
SETTINGS = {'minilm': {'dimensions': 384, 'max_tokens': 256, 'pooling': 'attention-mask-mean',
                        'normalization': 'l2', 'dtype': 'float32', 'frozen': True},
            'qwen': {'max_tokens': 1536, 'max_new_tokens': 384, 'do_sample': False,
                     'dtype': 'float32', 'frozen': True}}


class OptionalCapabilityError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f'{status}: {message}')


def read_pin(lock_path, model):
    if model not in MODELS:
        raise ValueError('unsupported local model')
    lock = json.loads(Path(lock_path).read_text())
    if lock.get('version') != 1:
        raise ValueError('unsupported model lock version')
    pin = lock['models'][model]
    if pin.get('status') != 'verified':
        raise OptionalCapabilityError('unverified_model_pin', 'immutable artifact metadata must be reviewed before local use')
    if pin.get('repository') != MODELS[model] or pin.get('license') != 'Apache-2.0':
        raise ValueError('model/license not approved')
    if not re.fullmatch('[0-9a-f]{40}', pin.get('revision', '')) or pin.get('tokenizer_revision') != pin['revision']:
        raise ValueError('immutable model and tokenizer revision required')
    if pin.get('settings') != SETTINGS[model] or pin.get('training_cutoff') != 'unknown':
        raise ValueError('unsupported settings/cutoff; requires a reviewed implementation update')
    files = pin['files']
    if set(files) != REQUIRED[model] | {'README.md', 'LICENSE'}:
        raise ValueError('exact reviewed artifact inventory required')
    total = 0
    for name, entry in files.items():
        if type(entry) is not dict or set(entry) != {'sha256', 'size', 'evidence_url'}:
            raise ValueError('artifact must include hash, size and official evidence URL')
        if not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('artifact SHA256 required')
        if type(entry['size']) is not int or not 0 < entry['size'] <= 2*1024**3:
            raise ValueError('artifact size outside bound')
        if not entry['evidence_url'].startswith('https://huggingface.co/' + MODELS[model] + '/'):
            raise ValueError('official artifact evidence required')
        total += entry['size']
    if total > 2*1024**3:
        raise ValueError('staged model exceeds 2 GiB artifact bound')
    return pin


def verify_staged(pin, staged):
    root = Path(staged).resolve()
    if not root.is_dir():
        raise OptionalCapabilityError('missing_local_artifacts', 'stage reviewed pinned files explicitly; downloads are disabled')
    # Reject sidecar configs/code and alternate weights, not just corrupt weights.
    found = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if found != set(pin['files']):
        raise OptionalCapabilityError('missing_local_artifacts', 'staged directory must contain exactly the pinned inventory')
    for name, metadata in pin['files'].items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != metadata['size']:
            raise OptionalCapabilityError('artifact_mismatch', f'invalid pinned artifact: {name}')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != metadata['sha256']:
            raise OptionalCapabilityError('artifact_mismatch', f'SHA256 mismatch: {name}')
    return root


class LocalModel:
    """One bounded subprocess per request; no GPU, hosted calls, tools or training.

    4 GiB virtual-address ceiling, 60s wall, 50s CPU, 2 threads. Unsupported
    enforcement fails closed. This intentionally trades model reload cost for
    simple isolation. A successful real-model run is not assumed from fake tests.
    """
    def __init__(self, model, *, lock_path, staged):
        self.model = model
        self.pin = read_pin(lock_path, model)
        self.staged = verify_staged(self.pin, staged)
        absent = [name for name in ('torch', 'transformers', 'safetensors') if importlib.util.find_spec(name) is None]
        if absent:
            raise OptionalCapabilityError('missing_optional_dependencies', 'install the market-text extra explicitly')
        runtime = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'safetensors')}
        self.dimensions = 384 if model == 'minilm' else 0
        self.tokenizer_sha256 = digest({k: v for k, v in self.pin['files'].items()
            if k in {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'vocab.json', 'merges.txt'}})
        self.signature = digest({'implementation': 'local-cpu-v1', 'pin': self.pin, 'tokenizer': self.tokenizer_sha256, 'runtime': runtime})
        self.provenance = {'kind': model, 'pretrained': True, 'model_sha256': self.signature,
                           'tokenizer_sha256': self.tokenizer_sha256, 'pin': self.pin,
                           'training_cutoff': 'unknown', 'contamination_warning': CONTAMINATION,
                           'runtime_versions': runtime, 'capability': 'local_only_not_live_validated'}
        self.last_metrics = None

    def _run(self, texts):
        limit = 8 if self.model == 'minilm' else 1
        if type(texts) is not list or not 1 <= len(texts) <= limit or any(type(t) is not str or len(t) > MAX_CHARS for t in texts):
            raise ValueError('local model text/batch bound exceeded')
        request = {'model': self.model, 'pin': self.pin, 'staged': str(self.staged), 'texts': texts}
        with tempfile.TemporaryDirectory(prefix='ccr-text-') as tmp:
            env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_DATASETS_OFFLINE': '1', 'HF_HOME': tmp, 'TORCH_HOME': tmp,
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2',
                   'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
            try:
                result = subprocess.run([sys.executable, '-m', 'cognitive_runtime.adapters.finance.text.worker'],
                    input=canonical(request), text=True, capture_output=True, env=env, timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise OptionalCapabilityError('resource_limit', 'local model exceeded 60s wall limit') from exc
        if result.returncode:
            # Never echo untrusted source text or a library traceback into artifacts.
            raise OptionalCapabilityError('local_model_failed', 'optional child failed (dependencies, artifacts or resource bound); no fallback')
        value = json.loads(result.stdout)
        if value['network_attempts'] != 0:
            raise OptionalCapabilityError('network_denied', 'local child attempted network access')
        self.last_metrics = value['metrics']
        return value['result']

    def encode(self, texts):
        if self.model != 'minilm':
            raise ValueError('only MiniLM encodes features')
        return self._run(texts)

    def generate(self, text):
        if self.model != 'qwen':
            raise ValueError('only Qwen extracts')
        return self._run([text])
