"""Tiny seeded synthetic foundation; no downloads, provider calls or model fitting.

Run: python -m cognitive_runtime.adapters.finance.testing.generate --output DIR
Use --steps 2048 for an explicitly larger optional benchmark, never market evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from cognitive_runtime.adapters.finance.schemas.base import Strict, decimal, sha256, timestamp
from cognitive_runtime.adapters.finance.schemas import PriceBar, NewsEvent, PredictionContractState
from .examples import examples, envelope, T, seal_source, source_payload

GENERATOR_VERSION = 'synthetic-market-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class FixtureEntry(Strict):
    path: str
    sha256: str
    record_hashes: tuple[str, ...]
    intended_edge_cases: tuple[str, ...]
    expected_invariants: tuple[str, ...]

    def validate(self):
        sha256(self.sha256)
        for value in self.record_hashes:
            sha256(value)
        if Path(self.path).is_absolute() or '..' in Path(self.path).parts or not self.expected_invariants:
            raise ValueError('unsafe path or missing fixture invariants')


@dataclass(frozen=True, kw_only=True)
class FixtureManifest(Strict):
    schema_name: str
    schema_version: str
    synthetic: bool
    generator_version: str
    seed: int
    source: str
    license: str
    files: tuple[FixtureEntry, ...]

    def validate(self):
        if self.schema_name != 'FixtureManifest' or self.schema_version != '1.0.0' or not self.synthetic:
            raise ValueError('versioned synthetic fixture manifest required')
        if self.source != 'project-authored' or self.license != 'MIT':
            raise ValueError('uncleared fixture rights')
        if len({x.path for x in self.files}) != len(self.files):
            raise ValueError('duplicate fixture path')


def at(step):
    return (timestamp(T) + timedelta(minutes=step)).isoformat().replace('+00:00', 'Z')


def generate(seed=286, steps=8, lag=1, text_noise=0.1, zero_signal=False):
    if type(steps) is not int or not 2 <= steps <= 100000 or type(lag) is not int or lag < 0:
        raise ValueError('invalid bounded steps/lag')
    if not math.isfinite(text_noise) or text_noise < 0:
        raise ValueError('invalid text noise')
    rng = random.Random(seed)
    templates = examples()
    latent, equity_log, contract_logit = 0.0, math.log(100), 0.0
    equity_loading, contract_loading = rng.gauss(0.02, 0.002), rng.gauss(0.15, 0.02)
    records, oracle = [], []
    for step in range(steps):
        latent = 0.8 * latent + rng.gauss(0, 0.5)
        signal = 0 if zero_signal else latent
        equity_log += equity_loading * signal + rng.gauss(0, 0.003)
        contract_logit = 0.8 * contract_logit + contract_loading * signal + rng.gauss(0, 0.02)
        equity = f'{math.exp(equity_log):.8f}'
        probability = f'{1 / (1 + math.exp(-contract_logit)):.8f}'
        time = at(step)
        bar = replace(templates['PriceBar'], **envelope(PriceBar, f'bar:{step}', time), start_at=at(step - 1), end_at=time,
                      open=equity, high=equity, low=equity, close=equity)
        contract = replace(templates['PredictionContractState'], **envelope(PredictionContractState, f'contract:{step}', time),
                           probability_price=probability)
        reveal = latent + rng.gauss(0, text_noise)
        news = replace(templates['NewsEvent'], **{**envelope(NewsEvent, f'news:{step}', time), 'available_at': at(step + lag),
                       'observed_at': at(step + lag), 'ingested_at': at(step + lag)}, first_seen_at=at(step + lag),
                       original_published_at=time, dedup_cluster_id=f'story:{step}',
                       title=f'Synthetic latent report {step}', text=f'Fictional latent reading {reveal:.6f}.')
        records.extend([seal_source(r).to_dict() for r in (bar, contract, news)])
        oracle.append({'step': step, 'latent': latent, 'equity_log': equity_log, 'contract_logit': contract_logit,
                       'text_reading': reveal})
    return {'synthetic': True, 'seed': seed, 'generator_version': GENERATOR_VERSION, 'lag': lag, 'text_noise': text_noise,
            'zero_signal': zero_signal, 'loadings': {'equity': equity_loading, 'contract': contract_loading},
            'records': records, 'training_only_oracle': oracle,
            'expectation': 'zero latent contribution to prices' if zero_signal else 'lagged noisy text reveals AR(1) factor; no asserted model gain'}


def edge_cases():
    e = examples()
    news, quote, alias, action, contract = (e[k] for k in ('NewsEvent', 'Quote', 'InstrumentAlias', 'CorporateAction', 'PredictionContractState'))
    delayed = replace(news, record_id='news:delayed', available_at=at(10), observed_at=at(11), ingested_at=at(11), first_seen_at=at(11))
    revised = replace(news, record_id='news:revision', revision=2, supersedes_record_id=news.record_id,
                      available_at=at(20), observed_at=at(20), ingested_at=at(20), first_seen_at=at(20), text='Revised fictional production.')
    syndicated = replace(news, record_id='news:syndicated', source_id='synthetic-repost', available_at=at(5),
                         observed_at=at(5), ingested_at=at(5), first_seen_at=at(5))
    skew = replace(news, record_id='news:skew', event_at=at(5))
    old_alias = replace(alias, record_id='alias:old', effective_to=at(5))
    new_alias = replace(alias, record_id='alias:reused', security_id='security:other', listing_id='listing:other', effective_from=at(5))
    dividend = replace(action, record_id='action:dividend', kind='dividend', split_ratio=None, cash='1', currency='USD')
    delisted = replace(action, record_id='action:delisting', kind='delisting', split_ratio=None)
    halted = replace(action, record_id='action:halt', kind='halt', split_ratio=None)
    stale = replace(quote, record_id='quote:stale', stale=True)
    crossed = replace(quote, record_id='quote:crossed', bid='102', crossed=True)
    missing = replace(quote, record_id='quote:outage', bid=None, ask=None, missing=True)
    early = replace(contract, record_id='contract:early-close', lifecycle='closed')
    scenarios = {
        'delayed_revised_news': ([news, delayed, revised], 'Only the original is eligible at the original decision cutoff.'),
        'shuffled_arrival': ([revised, delayed, news], 'Input order does not change recorded availability or identities.'),
        'syndication': ([news, syndicated], 'Same dedup cluster; repost availability remains later.'),
        'clock_skew': ([skew], 'Future event clock is preserved separately from observation, never substituted for knowledge time.'),
        'alias_reuse': ([old_alias, new_alias], 'Same ticker maps to distinct securities in disjoint business intervals.'),
        'split_dividend': ([action, dividend], 'Raw price preserved; split ratio and dividend cash have separate semantics.'),
        'delisting_halt': ([delisted, halted], 'No manufactured endpoint or carry-forward zero return.'),
        'stale_crossed_outage': ([stale, crossed, missing], 'Explicit stale/crossed/missing masks; crossed quotes rejected at inference boundary.'),
        'early_closure_terminal_trap': ([early, e['ContractTerminalMetadata']], 'Known closure may enter context; actual terminal metadata cannot.'),
    }
    result = {name: {'records': [seal_source(x).to_dict() for x in values], 'expected_invariant': invariant} for name, (values, invariant) in scenarios.items()}
    # Calendar facts are test inputs, not an implemented calendar engine (#291).
    result['calendar_boundaries'] = {
        'records': [], 'expected_invariant': 'Calendar version and clock stay explicit; elapsed seconds cannot masquerade as exchange sessions.',
        'sessions': [
            {'case': 'holiday', 'date': '2025-12-25', 'open_at': None, 'close_at': None},
            {'case': 'half_day', 'date': '2025-11-28', 'open_at': '2025-11-28T14:30:00Z', 'close_at': '2025-11-28T18:00:00Z'},
            {'case': 'dst_before', 'date': '2025-03-07', 'open_at': '2025-03-07T14:30:00Z', 'close_at': '2025-03-07T21:00:00Z'},
            {'case': 'dst_after', 'date': '2025-03-10', 'open_at': '2025-03-10T13:30:00Z', 'close_at': '2025-03-10T20:00:00Z'},
        ], 'calendar_id': 'synthetic-exchange', 'calendar_version': '1', 'original_timezone': 'America/New_York',
    }
    return result


def write_fixtures(output: Path, seed=286, steps=8):
    output.mkdir(parents=True, exist_ok=True)
    payloads = {'signal.json': generate(seed, steps), 'null.json': generate(seed, steps, zero_signal=True),
                'edge-cases.json': edge_cases(), 'examples.json': {k: v.to_dict() for k, v in examples().items()}}
    prior = examples()['Quote'].to_dict()
    prior['schema_version'] = '0.9.0'
    del prior['availability_confidence']
    prior['provenance']['raw_sha256'] = digest(source_payload(prior))
    payloads['prior-0.9.json'] = prior
    all_records = [*payloads['signal.json']['records'], *payloads['null.json']['records'],
                   *payloads['examples.json'].values(), prior,
                   *(record for case in payloads['edge-cases.json'].values() for record in case['records'])]
    payloads['sources.json'] = {record['provenance']['raw_sha256']: source_payload(record) for record in all_records}
    files = []
    for name, value in payloads.items():
        raw = canonical(value) + '\n'
        (output / name).write_text(raw)
        records = value.get('records', [])
        if name == 'examples.json':
            records = list(value.values())
        elif name == 'edge-cases.json':
            records = [r for case in value.values() for r in case['records']]
        elif name == 'prior-0.9.json':
            records = [value]
        cases = tuple(value) if name == 'edge-cases.json' else (name[:-5],)
        files.append(FixtureEntry(path=name, sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                  record_hashes=tuple(digest(r) for r in records), intended_edge_cases=cases,
                                  expected_invariants=('Deterministic project-authored synthetic bytes; strict typed parsing except explicitly prior-version corpus.',)))
    manifest = FixtureManifest(schema_name='FixtureManifest', schema_version='1.0.0', synthetic=True,
                               generator_version=GENERATOR_VERSION, seed=seed, source='project-authored', license='MIT', files=tuple(files))
    (output / 'manifest.json').write_text(manifest.canonical_json() + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=286)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    print(write_fixtures(args.output, args.seed, args.steps).hash)
