"""Project-authored synthetic MIT fixtures, seed 291; no source market data."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from cognitive_runtime.adapters.finance.datasets.calendars import Calendar, elapsed_endpoint, utc
from cognitive_runtime.adapters.finance.datasets.labels import derive_label
from cognitive_runtime.adapters.finance.datasets.universe import freeze_universe
from cognitive_runtime.adapters.finance.datasets.corpus import (Request, build_corpus, attach_training_targets)
from cognitive_runtime.adapters.finance.datasets.splits import Sample, walk_forward, validate_split
from cognitive_runtime.adapters.finance.labels import FutureTarget
from cognitive_runtime.adapters.finance.schemas.base import timestamp
from cognitive_runtime.adapters.finance.schemas.semantics import Horizon, Clock, Target, Unit, Mark
from cognitive_runtime.adapters.finance.testing.examples import examples, envelope, seal_source
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical
from cognitive_runtime.data.snapshot import build_snapshot

FIXTURE = Path(__file__).parents[1] / 'fixtures/market/pit/calendars.json'
CALENDARS = tuple(Calendar(**c) for c in json.loads(FIXTURE.read_text())['calendars'])
T = '2025-03-07T15:00:00Z'
END = '2025-03-07T16:00:00Z'
POLICY = RetentionPolicy(rights='project-authored-fixture', evidence='MIT synthetic seed 291',
    allow_raw=True, allow_normalized=True, retain_provenance=True)


@pytest.fixture
def ledger():
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp, Ledger(Path(tmp) / 'ledger') as ledger:
        yield ledger


def add(ledger, record):
    from cognitive_runtime.adapters.finance.testing.examples import source_payload
    record = seal_source(record)
    ledger.append(record, canonical(source_payload(record)).encode(), POLICY)
    return record


def bar(at, price='100', **changes):
    value = examples()['PriceBar']
    return replace(value, **{**envelope(type(value), 'bar:' + at, at), 'start_at': utc(timestamp(at) - timedelta(minutes=1)),
        'end_at': at, 'open': price, 'close': price, 'low': price, 'high': price,
        'calendar_version': '1', 'session_id': at[:10], **changes})


def identity(at=T, **changes):
    value = examples()['Instrument']
    return replace(value, **{**envelope(type(value), 'instrument:' + at, at), 'effective_from': at, **changes})


def contract(at, price='0.42', **changes):
    value = examples()['PredictionContractState']
    return replace(value, **{**envelope(type(value), 'contract:' + at, at), 'probability_price': price,
                            'scheduled_expiry_at': '2025-03-20T00:00:00Z', **changes})


def snap(ledger, at=T, calendar=CALENDARS[0], **changes):
    return build_snapshot(ledger, **{**dict(decision_at=at, price_seconds=90*86400, text_seconds=86400,
        calendar_id=calendar.calendar_id, calendar_version=calendar.version, calendar_sha256=calendar.hash,
        preprocessing_sha256='b'*64, price_max_age_seconds=60), **changes})


def spec(calendar=CALENDARS[0], clock=Clock.TRADING_HOURS, count=1, **changes):
    value = examples()['LabelSpec']
    equity = clock != Clock.ELAPSED
    return replace(value, **{**dict(target=Target.EQUITY_RETURN if equity else Target.CONTRACT_CHANGE,
        unit=Unit.RETURN if equity else Unit.PRICE_CHANGE, mark=Mark.TRADE if equity else Mark.MID,
        action_policy='split_neutral_ex_dividend' if equity else 'none',
        horizon=Horizon(clock=clock, count=count, calendar_id=calendar.calendar_id if equity else None,
                        calendar_version=calendar.version if equity else None)), **changes})


def label(ledger, decision=T, outcome=END, calendar=CALENDARS[0], **kwargs):
    return derive_label(snap(ledger, decision, calendar), snap(ledger, outcome, calendar),
        listing_id='listing:demo', spec=spec(calendar), calendar=calendar, **kwargs)


@pytest.mark.parametrize('index,decision,clock,count,expected', [
    (0, '2025-03-07T20:30:00Z', Clock.TRADING_HOURS, 1, '2025-03-10T14:00:00Z'),
    (0, '2025-03-07T22:00:00Z', Clock.TRADING_HOURS, 1, '2025-03-10T14:30:00Z'),
    (0, '2025-03-07T20:00:00Z', Clock.TRADING_HOURS, 4, '2025-03-10T16:30:00Z'),
    (0, '2025-03-07T21:00:00Z', Clock.SESSIONS, 1, '2025-03-10T20:00:00Z'),
    (0, '2025-03-07T15:00:00Z', Clock.SESSIONS, 5, '2025-03-13T20:00:00Z'),
    (1, '2025-11-28T17:30:00Z', Clock.TRADING_HOURS, 1, '2025-12-01T15:00:00Z'),
    (1, '2025-11-27T20:00:00Z', Clock.SESSIONS, 1, '2025-11-28T18:00:00Z'),
    (1, '2025-12-01T14:30:00Z', Clock.TRADING_HOURS, 1, '2025-12-01T16:00:00Z'),
])
def test_golden_calendar_endpoints(index, decision, clock, count, expected):
    cal = CALENDARS[index]
    assert cal.endpoint(decision, spec(cal, clock, count).horizon) == expected


@pytest.mark.parametrize('decision,end', [
    ('2025-03-07T20:30:00Z', '2025-03-10T14:00:00Z'),
    ('2025-03-10T13:30:00Z', '2025-03-10T14:30:00Z'),
    ('2025-11-28T17:30:00Z', '2025-12-01T15:00:00Z'),
])
def test_golden_exact_anchor_endpoint_values(ledger, decision, end):
    cal = CALENDARS[1] if decision.startswith('2025-11') else CALENDARS[0]
    add(ledger, identity(decision))
    a = add(ledger, bar(decision, calendar_version=cal.version))
    e = add(ledger, bar(end, '110', calendar_version=cal.version))
    result = label(ledger, decision, end, cal)
    # Completion at halt start is deliberately unusable (not a fabricated fill).
    if end == '2025-12-01T15:00:00Z':
        assert result.censor_reason == 'endpoint_outside_session'
    else:
        assert result.label.value == '0.1'
        assert (result.label.anchor_record_id, result.label.endpoint_record_id) == (a.record_id, e.record_id)
        assert result.label.nominal_endpoint_at == end


def test_calendar_validation_and_24_7():
    cal = CALENDARS[0]
    with pytest.raises(ValueError, match='unaccounted'):
        replace(cal, sessions=cal.sessions[:-1])
    with pytest.raises(ValueError, match='outside trading'):
        replace(cal, edge_policy='reject_closed').endpoint('2025-03-08T15:00:00Z', spec().horizon)
    with pytest.raises(ValueError, match='coverage'):
        cal.endpoint('2025-03-14T19:00:00Z', spec(count=4).horizon)
    assert elapsed_endpoint('2025-03-09T06:30:00Z', spec(clock=Clock.ELAPSED, count=3600).horizon) == '2025-03-09T07:30:00Z'


def test_delayed_sip_exact_and_no_mixed_feed(ledger):
    add(ledger, identity())
    add(ledger, bar('2025-03-07T14:45:00Z', available_at=T, observed_at=T, ingested_at=T, feed='SIP'))
    add(ledger, bar(END, '110', feed='SIP'))
    d, o = snap(ledger), snap(ledger, END)
    result = derive_label(d, o, listing_id='listing:demo', spec=spec(feed='SIP'), calendar=CALENDARS[0])
    assert result.censor_reason == 'stale_anchor'
    assert result.nominal_endpoint_at == END  # never 15:45 or 16:15
    assert result.anchor_observed_at == T
    assert result.label is None
    add(ledger, bar(T, feed='IEX'))
    result = derive_label(snap(ledger), o, listing_id='listing:demo', spec=spec(feed='IEX'), calendar=CALENDARS[0])
    assert result.censor_reason == 'missing_endpoint'
    assert result.nominal_endpoint_at == END


def test_anchor_endpoint_tolerance_and_context_independence(ledger):
    add(ledger, identity())
    a = add(ledger, bar('2025-03-07T14:59:30Z'))
    add(ledger, bar('2025-03-07T16:00:30Z', '110'))
    result = label(ledger, outcome='2025-03-07T16:00:30Z')
    assert result.label.value == '0.1'
    assert result.anchor_to_endpoint_seconds == 3660
    assert result.label.anchor_at == a.end_at
    assert result.label.nominal_endpoint_at == END
    d = snap(ledger, price_seconds=3600, text_seconds=7200)
    result2 = derive_label(d, snap(ledger, '2025-03-07T16:00:30Z'), listing_id='listing:demo', spec=spec(), calendar=CALENDARS[0])
    assert result2.label.value == result.label.value
    assert d.context.price_context.count != d.context.text_context.count
    with pytest.raises(ValueError, match='cover label interval'):
        derive_label(d, snap(ledger, '2025-03-07T16:00:30Z', price_seconds=30),
                     listing_id='listing:demo', spec=spec(), calendar=CALENDARS[0])


@pytest.mark.parametrize('changes,reason', [({'stale': True}, 'invalid_endpoint'),
    ({'complete': False, 'missing': True}, 'invalid_endpoint'), ({'feed': 'other'}, 'missing_endpoint'),
    ({'adjustment': 'split_adjusted'}, 'missing_endpoint'), ({'calendar_version': 'other'}, 'missing_endpoint')])
def test_unusable_endpoints_are_censored(ledger, changes, reason):
    add(ledger, identity())
    add(ledger, bar(T))
    add(ledger, bar(END, '110', **changes))
    assert label(ledger).censor_reason == reason


def test_split_only_adjustment_and_future_feature_invariance(ledger):
    add(ledger, identity())
    add(ledger, bar(T))
    before = snap(ledger)
    add(ledger, bar(END, '55'))
    a = examples()['CorporateAction']
    split = add(ledger, replace(a, **envelope(type(a), 'split', END), announced_at=END, effective_at=END))
    add(ledger, replace(a, **envelope(type(a), 'dividend', END), announced_at=END, effective_at=END,
                        kind='dividend', split_ratio=None, cash='5', currency='USD'))
    result = label(ledger)
    assert result.label.value == '0.1'
    assert result.label.action_record_ids == (split.record_id,)
    assert result.label.split_multiplier == '2'
    assert snap(ledger) == before


@pytest.mark.parametrize('terminal', ['closed', 'expiry', 'missing', 'halted'])
def test_contract_censoring_no_settlement(ledger, terminal):
    add(ledger, contract(T, scheduled_expiry_at=END if terminal == 'expiry' else '2025-03-20T00:00:00Z'))
    if terminal != 'missing':
        add(ledger, contract(END, '0.47', lifecycle=terminal if terminal in ('closed', 'halted') else 'open'))
    result = derive_label(snap(ledger), snap(ledger, END), listing_id='contract:demo', spec=spec(clock=Clock.ELAPSED, count=3600))
    assert result.label is None and result.censor_reason is not None
    assert result.nominal_endpoint_at == END


def test_contract_absolute_change_and_early_closure(ledger):
    add(ledger, contract(T))
    add(ledger, contract(END, '0.47'))
    d = snap(ledger)
    result = derive_label(d, snap(ledger, END), listing_id='contract:demo', spec=spec(clock=Clock.ELAPSED, count=3600))
    assert result.label.value == '0.05'
    add(ledger, contract('2025-03-07T15:30:00Z', None, lifecycle='closed'))
    result = derive_label(d, snap(ledger, END), listing_id='contract:demo', spec=spec(clock=Clock.ELAPSED, count=3600))
    assert result.censor_reason == 'closed_expired_or_halted'


def sample(key, hour, end, **changes):
    return Sample(**{**dict(sample_id=key, snapshot_sha256='a'*64, decision_at=f'2025-03-07T{hour:02}:00:00Z',
        context_start_at=f'2025-03-07T{hour-1:02}:00:00Z', label_end_at=f'2025-03-07T{end:02}:30:00Z',
        issuer_ids=(key,), event_families=(key,)), **changes})


def plan(samples, **changes):
    return walk_forward(samples, validation_at='2025-03-07T12:00:00Z', test_at='2025-03-07T16:00:00Z', **changes)


def test_exact_split_memberships_and_injected_leakage():
    samples = (sample('a', 8, 9), sample('b', 11, 12), sample('c', 12, 13), sample('d', 15, 17), sample('e', 16, 17))
    p = plan(samples)
    assert (p.train, p.validation, p.test) == (('a',), ('c',), ('e',))
    assert {x[0] for x in p.purged} == {'b', 'd'}
    with pytest.raises(ValueError, match='leakage|embargo'):
        validate_split(samples, replace(p, train=('a','b'), purged=tuple(x for x in p.purged if x[0] != 'b')))


def test_latent_context_and_embargo_boundaries():
    samples = (sample('a', 10, 10), sample('b', 12, 13), sample('c', 16, 17))
    assert plan(samples).train == ('a',)
    assert not plan(samples, embargo_seconds=5400).train  # exact boundary
    assert not plan((replace(samples[0], future_end_at='2025-03-07T12:00:00Z'), *samples[1:])).train
    contextual = (samples[0], replace(samples[1], context_start_at='2025-03-07T10:00:00Z'), samples[2])
    assert plan(contextual).train == ('a',)
    assert not plan(contextual, strict_context=True).train


@pytest.mark.parametrize('field,group', [('issuer_ids','issuer'),('event_families','event_family'),('dedup_keys','temporal'),('source_keys','temporal')])
def test_group_and_story_contamination(field, group):
    samples = (sample('renamed',8,9, **{field: ('same',)}), sample('heldout',12,13, **{field: ('same',)}), sample('test',16,17))
    p = plan(samples, group=group)
    assert p.train == ()
    assert field in p.purged[0][1]
    with pytest.raises(ValueError, match='contamination'):
        validate_split(samples, replace(p, train=('renamed',), purged=()))


def test_universe_rename_reuse_and_hash_binding(ledger):
    i = add(ledger, identity())
    a = examples()['InstrumentAlias']
    add(ledger, replace(a, **envelope(type(a), 'alias-old', T), effective_from=T))
    before = snap(ledger)
    m = freeze_universe(before, i.universe_id, coverage='synthetic_only', limitations=('No complete delisting history.',))
    add(ledger, replace(a, **envelope(type(a), 'alias-new', END), alias='RENAMED', effective_from=END))
    future = freeze_universe(snap(ledger, END), i.universe_id, coverage='synthetic_only', limitations=m.limitations)
    assert {x[1] for x in future.aliases} == {'DEMO','RENAMED'}
    assert {x[0] for x in m.identities} == {x[0] for x in future.identities} == {i.issuer_id}
    assert snap(ledger) == before
    with pytest.raises(ValueError, match='integrity'):
        freeze_universe(replace(before, records=()), i.universe_id, coverage='synthetic_only', limitations=m.limitations)


def test_corpus_stores_population_and_train_validation_gates(ledger):
    requests = []
    for day in (10,11,12):
        at, end = f'2025-03-{day}T14:00:00Z', f'2025-03-{day}T15:00:00Z'
        add(ledger, identity(at, record_id=f'i{day}', source_record_id=f'i{day}', listing_id=f'l{day}',
                             security_id=f's{day}', issuer_id=f'issuer{day}'))
        add(ledger, bar(at, listing_id=f'l{day}'))
        if day != 12:
            add(ledger, bar(end, '110', listing_id=f'l{day}'))
        d, o = snap(ledger, at, price_seconds=3600), snap(ledger, end)
        requests.append(Request(sample_id=str(day), snapshot_sha256=d.context.hash, outcome_snapshot_sha256=o.context.hash,
                                listing_id=f'l{day}', universe_id='synthetic-universe-v1'))
    with tempfile.TemporaryDirectory(dir='/tmp') as root:
        corpus = build_corpus(ledger, root, requests, spec=spec(), calendar=CALENDARS[0])
        assert corpus.population() == {'total':3,'reasons':{'missing_endpoint':1,'realized':2}}
        p = walk_forward(corpus.samples, validation_at='2025-03-11T00:00:00Z', test_at='2025-03-12T00:00:00Z')
        assert (p.train,p.validation,p.test) == (('10',),('11',),('12',))
        seen = corpus.fit_preprocessing(p, lambda rows: [r['sample_id'] for r in rows])
        assert seen == ['10']
        assert corpus.calibrate(p, lambda rows: [r[0]['sample_id'] for r in rows]) == ['11']
        assert corpus.evaluation(p)[0][1].censor_reason == 'missing_endpoint'
        assert 'labels' not in corpus.feature('10') and 'targets' not in corpus.feature('10')
        target_record = add(ledger, bar('2025-03-10T15:30:00Z', '120', listing_id='l10'))
        target = FutureTarget(**envelope(FutureTarget, 'future', '2025-03-10T15:30:00Z'), split='train',
            context_snapshot_id=corpus.feature('10')['context']['record_id'], window_start_at='2025-03-10T14:00:00Z',
            window_end_at='2025-03-10T15:30:00Z', future_record_ids=(target_record.record_id,), stop_gradient=True)
        new, new_plan = attach_training_targets(corpus, ledger, p, {'10':target})
        assert new.training(new_plan)[0][2] == target
        assert new.manifest_sha256 != corpus.manifest_sha256
        assert corpus.manifest['targets'] == {}
        with pytest.raises(ValueError, match='training partitions'):
            attach_training_targets(corpus, ledger, p, {'12':target})
        with pytest.raises(ValueError, match='leakage|embargo'):
            attach_training_targets(corpus, ledger, p, {'10':replace(target, window_end_at='2025-03-11T15:00:00Z')})
        # Label corruption cannot be silently used by calibration/evaluation.
        path = corpus.root / 'labels'
        from cognitive_runtime.data.raw_store import RawStore
        RawStore(path).path(corpus.manifest['labels']['12']).write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='corrupt'):
            corpus.evaluation(p)


def test_fixture_hash_and_import_boundary():
    expected = json.loads((FIXTURE.parent / 'provenance.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == expected['sha256']
    subprocess.run([sys.executable, '-c', '''
import sys
from cognitive_runtime.adapters.finance.datasets.corpus import FeatureStore
assert 'cognitive_runtime.adapters.finance.labels' not in sys.modules
assert not any(x.startswith(('torch', 'transformers', 'cognitive_runtime.training')) for x in sys.modules)
'''], check=True)


def test_old_split_announcement_full_outcome_snapshot(ledger):
    from cognitive_runtime.adapters.finance.datasets.corpus import freeze_outcomes
    old = '2025-01-01T00:00:00Z'
    add(ledger, identity())
    add(ledger, bar(T))
    add(ledger, bar(END, '55'))
    a = examples()['CorporateAction']
    add(ledger, replace(a, **envelope(type(a), 'old-split', old), announced_at=old, effective_at=END))
    cal = CALENDARS[0]
    outcomes = freeze_outcomes(ledger, decision_at=END, calendar_id=cal.calendar_id,
        calendar_version=cal.version, calendar_sha256=cal.hash, preprocessing_sha256='b'*64)
    result = derive_label(snap(ledger), outcomes, listing_id='listing:demo', spec=spec(), calendar=cal)
    assert result.label.value == '0.1'
    add(ledger, replace(a, **envelope(type(a), 'duplicate-split', END), announced_at=END, effective_at=END))
    assert label(ledger).censor_reason == 'ambiguous_split_actions'


def test_endpoint_exceeds_tolerance_and_bad_revision_no_resurrection(ledger):
    add(ledger, identity())
    add(ledger, bar(T))
    original = add(ledger, bar('2025-03-07T16:02:00Z', '110'))
    assert label(ledger, outcome=original.end_at).censor_reason == 'endpoint_outside_tolerance'
    first = add(ledger, bar(END, '110'))
    add(ledger, replace(first, record_id='bad-revision', revision=2, supersedes_record_id=first.record_id,
                        complete=False, missing=True))
    assert label(ledger, outcome=original.end_at).censor_reason == 'invalid_endpoint'


def test_exact_session_horizons_and_after_hours(ledger):
    decision = '2025-03-07T21:00:30Z'
    add(ledger, identity('2025-03-07T21:00:00Z'))
    a = add(ledger, bar('2025-03-07T21:00:00Z'))
    end = '2025-03-10T20:00:00Z'
    e = add(ledger, bar(end, '110'))
    d, o = snap(ledger, decision), snap(ledger, end)
    result = derive_label(d, o, listing_id='listing:demo', spec=spec(clock=Clock.SESSIONS), calendar=CALENDARS[0])
    assert result.label.value == '0.1'
    assert result.label.anchor_record_id == a.record_id and result.label.endpoint_record_id == e.record_id
    assert result.label.nominal_endpoint_at == end
    assert result.anchor_to_endpoint_seconds == 71*3600


@pytest.mark.parametrize('late_kind', ['endpoint', 'split'])
def test_late_observation_extends_target_availability_and_purge(ledger, late_kind):
    observed = '2025-03-10T15:00:00Z'
    add(ledger, identity())
    add(ledger, bar(T))
    add(ledger, bar(END, '55' if late_kind == 'split' else '110',
                    observed_at=observed if late_kind == 'endpoint' else END,
                    ingested_at=observed if late_kind == 'endpoint' else END))
    if late_kind == 'split':
        a = examples()['CorporateAction']
        add(ledger, replace(a, **{**envelope(type(a), 'late-split', END), 'observed_at':observed,
            'ingested_at':observed}, announced_at=END, effective_at=END))
    d, o = snap(ledger), snap(ledger, observed)
    result = derive_label(d,o,listing_id='listing:demo',spec=spec(),calendar=CALENDARS[0])
    assert result.label.value == '0.1'
    assert result.label.target_available_at == observed
    request = Request(sample_id='late',snapshot_sha256=d.context.hash,outcome_snapshot_sha256=o.context.hash,
                      listing_id='listing:demo',universe_id='synthetic-universe-v1')
    with tempfile.TemporaryDirectory(dir='/tmp') as root:
        corpus = build_corpus(ledger,root,[request],spec=spec(),calendar=CALENDARS[0])
        assert corpus.samples[0].label_end_at == observed
        held = replace(corpus.samples[0], sample_id='held',decision_at='2025-03-10T14:00:00Z',
                       context_start_at='2025-03-10T13:00:00Z',label_end_at='2025-03-10T16:00:00Z')
        p = walk_forward((*corpus.samples,held),validation_at='2025-03-10T00:00:00Z',test_at='2025-03-11T00:00:00Z')
        assert p.train == () and p.validation == ('held',)


def test_outside_pit_universe_retains_explicit_censor(ledger):
    add(ledger,identity())
    add(ledger,bar(T))
    add(ledger,bar(END,'110'))
    d,o=snap(ledger),snap(ledger,END)
    request=Request(sample_id='outside',snapshot_sha256=d.context.hash,outcome_snapshot_sha256=o.context.hash,
                    listing_id='listing:demo',universe_id='not-the-universe')
    with tempfile.TemporaryDirectory(dir='/tmp') as root:
        corpus=build_corpus(ledger,root,[request],spec=spec(),calendar=CALENDARS[0])
        assert corpus.population() == {'total':1,'reasons':{'outside_pit_universe':1}}
        assert corpus.feature('outside')['universe']['identities'] == []
        assert corpus._outcome('outside').label is None


@pytest.mark.parametrize('group', ['temporal','issuer'])
def test_future_news_contamination_revalidated_before_write(ledger,group):
    from cognitive_runtime.data.raw_store import RawStore
    future_at='2025-03-10T15:30:00Z'
    news=examples()['NewsEvent']
    future=add(ledger,replace(news,**envelope(type(news),'future-news',future_at),
        first_seen_at=future_at,original_published_at=future_at,
        entities=(replace(news.entities[0],issuer_id='held-issuer'),)))
    requests=[]
    for day in (10,11):
        at,end=f'2025-03-{day}T14:00:00Z',f'2025-03-{day}T15:00:00Z'
        issuer='train-issuer' if day==10 else 'held-issuer'
        # Expire the first identity before validation so group isolation is real.
        add(ledger,identity(at,record_id=f'i{day}',source_record_id=f'i{day}',listing_id=f'l{day}',
            security_id=f's{day}',issuer_id=issuer,effective_to='2025-03-11T00:00:00Z' if day==10 else None))
        add(ledger,bar(at,listing_id=f'l{day}'))
        add(ledger,bar(end,'110',listing_id=f'l{day}'))
        d,o=snap(ledger,at,price_seconds=3600),snap(ledger,end)
        requests.append(Request(sample_id=str(day),snapshot_sha256=d.context.hash,outcome_snapshot_sha256=o.context.hash,
                                listing_id=f'l{day}',universe_id='synthetic-universe-v1'))
    with tempfile.TemporaryDirectory(dir='/tmp') as root:
        corpus=build_corpus(ledger,root,requests,spec=spec(),calendar=CALENDARS[0])
        p=walk_forward(corpus.samples,validation_at='2025-03-11T00:00:00Z',test_at='2025-03-12T00:00:00Z',group=group)
        assert p.train==('10',) and p.validation==('11',)
        target=FutureTarget(**envelope(FutureTarget,'target',future_at),split='train',
            context_snapshot_id=corpus.feature('10')['context']['record_id'],window_start_at='2025-03-10T14:00:00Z',
            window_end_at=future_at,future_record_ids=(future.record_id,),stop_gradient=True)
        before=list((Path(root)/'targets').rglob('*'))
        with pytest.raises(ValueError,match='contamination'):
            attach_training_targets(corpus,ledger,p,{'10':target})
        assert list((Path(root)/'targets').rglob('*'))==before
        assert corpus.manifest['targets']=={}


def test_future_price_identity_and_contract_family_fail_closed(ledger):
    from cognitive_runtime.adapters.finance.datasets.corpus import _future_groups
    at='2025-03-10T15:30:00Z'
    future=add(ledger,bar(at))
    with pytest.raises(ValueError,match='PIT issuer identity'):
        _future_groups(ledger,at,[future],{})
    add(ledger,identity('2025-03-10T14:00:00Z'))
    assert _future_groups(ledger,at,[future],{})['issuers']=={'issuer:demo'}
    c=add(ledger,contract(at))
    add(ledger,identity('2025-03-10T14:00:00Z',record_id='contract-i',source_record_id='contract-i',
        listing_id='contract:demo',security_id='contract-security',issuer_id='contract-issuer',kind='binary_contract'))
    with pytest.raises(ValueError,match='event-family'):
        _future_groups(ledger,at,[c],{})
    groups=_future_groups(ledger,at,[c],{'market:demo':'family-held'})
    assert groups['families']=={'family-held'}
    assert groups['issuers']=={'contract-issuer'}
    samples=(sample('train',8,9,event_families=tuple(groups['families'])),
             sample('held',12,13,event_families=('family-held',)),sample('test',16,17))
    assert plan(samples,group='event_family').train==()
