"""Project-authored MIT synthetic Factory corpus, generator version 1.

Seed changes the bounded price signal, never a claim of market edge.
"""
from dataclasses import replace
from pathlib import Path


from cognitive_runtime.adapters.finance.testing.examples import examples, envelope, seal_source, source_payload
from cognitive_runtime.adapters.finance.schemas.records import NewsEvent, PredictionContractState
from cognitive_runtime.adapters.finance.schemas.semantics import Horizon, Clock
from cognitive_runtime.adapters.finance.datasets.corpus import Corpus, Request, build_corpus
from cognitive_runtime.adapters.finance.datasets.splits import walk_forward
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical
from cognitive_runtime.data.snapshot import build_snapshot



def prepare_synthetic(root, seed=295):
    root = Path(root)
    base = examples()
    policy = RetentionPolicy(rights='project-authored-fixture', evidence=f'MIT generator factory-v1 seed {seed}', allow_raw=True,
                             allow_normalized=True, retain_provenance=True)
    with Ledger(root / 'ledger') as ledger:
        def append(r):
            r = seal_source(r)
            ledger.append(r, canonical(source_payload(r)).encode(), policy)
            return r
        append(replace(base['Instrument'], listing_id='contract:demo', kind='binary_contract'))
        requests = []
        for day in range(6, 18):
            at = f'2025-01-{day:02}T14:30:00Z'
            # Include exact 0/1 anchors in sealed test. No settlement used.
            price = 0.0 if day == 15 else 1.0 if day == 16 else .4
            signal = (.02 + (seed % 7) * .002) * (1 if day % 2 else -1)
            for clock, value in [('14:29', .4-signal), ('14:30', price), ('14:31', min(1., max(0., price+signal)))]:
                stamp = at.replace('14:30', clock)
                append(replace(base['PredictionContractState'], **envelope(PredictionContractState, f'p-{day}-{clock}', stamp),
                    probability_price=str(value), scheduled_expiry_at='2025-02-01T00:00:00Z'))
            append(replace(base['NewsEvent'], **envelope(NewsEvent, f'n-{day}', at),
                original_published_at=at, first_seen_at=at, dedup_cluster_id=f'story-{day}',
                title='rise' if signal > 0 else 'fall', text='synthetic signal'))
            def snapshot(stamp):
                return build_snapshot(ledger, decision_at=stamp, price_seconds=180, text_seconds=180,
                    calendar_id='synthetic', calendar_version='1', calendar_sha256='a'*64, preprocessing_sha256='b'*64)
            decision, outcome = snapshot(at), snapshot(at.replace('14:30', '14:31'))
            requests.append(Request(sample_id=str(day), snapshot_sha256=decision.context.hash,
                outcome_snapshot_sha256=outcome.context.hash, listing_id='contract:demo', universe_id='synthetic-universe-v1'))
        spec = replace(base['LabelSpec'], horizon=Horizon(clock=Clock.ELAPSED, count=60, calendar_id=None, calendar_version=None))
        c = build_corpus(ledger, root / 'corpus', requests, spec=spec)
    c = Corpus(c.root, c.manifest_sha256)
    plan = walk_forward(c.samples, validation_at='2025-01-12T00:00:00Z', test_at='2025-01-15T00:00:00Z')
    return c, plan
