"""Tiny project-authored synthetic storage/leakage fixtures; MIT, seed 287.

Every raw payload is generated and bound to its schema provenance at runtime.
No provider data or external service is used.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cognitive_runtime.adapters.finance.testing.examples import examples, source_payload
from cognitive_runtime.adapters.finance.schemas.base import AvailabilityBasis
from cognitive_runtime.data.ledger import Ledger, RetentionPolicy, canonical
from cognitive_runtime.data.snapshot import build_snapshot, load_snapshot
from cognitive_runtime.data.identity import resolve_alias

T = '2025-01-06T14:30:00Z'
LATE = '2025-01-07T14:30:00Z'
POLICY = RetentionPolicy(rights='project-authored-fixture', evidence='MIT synthetic seed 287',
                         allow_raw=True, allow_normalized=True, retain_provenance=True)


def sealed(record):
    raw = canonical(source_payload(record)).encode()
    return replace(record, provenance=replace(record.provenance, raw_sha256=hashlib.sha256(raw).hexdigest())), raw


def append(ledger, record, policy=POLICY):
    record, raw = sealed(record)
    ledger.append(record, raw, policy)
    return record


def snapshot(ledger, **kwargs):
    args = dict(decision_at=T, price_seconds=3600, text_seconds=7200,
                calendar_id='synthetic', calendar_version='1', calendar_sha256='a' * 64,
                preprocessing_sha256='b' * 64)
    args.update(kwargs)
    return build_snapshot(ledger, **args)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Ledger(Path(self.tmp.name) / 'ledger')
        self.addCleanup(self.ledger.close)
        self.ex = examples()

    def test_immutable_storage_and_raw_binding(self):
        record = append(self.ledger, self.ex['NewsEvent'])
        self.assertEqual(self.ledger.raw_payload(record.record_id), sealed(record)[1])
        self.assertEqual(self.ledger.get(record.record_id), record)
        self.assertEqual(append(self.ledger, record), record)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            append(self.ledger, replace(record, text='changed'))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.ledger.append(record, b'wrong', POLICY)
        for table in ('events', 'snapshots', 'deletions'):
            with self.assertRaises(sqlite3.IntegrityError):
                # Triggers execute per row; ensure there is one.
                if table == 'snapshots':
                    snapshot(self.ledger)
                if table == 'deletions':
                    self.ledger.db.execute("INSERT INTO deletions VALUES ('raw',?,'test')", ('f' * 64,))
                self.ledger.db.execute(f'DELETE FROM {table}')

    def test_correction_and_ingestion_order_invariance_and_golden(self):
        original = self.ex['NewsEvent']
        revised = replace(original, record_id='news:revision:2', revision=2, supersedes_record_id=original.record_id,
                          text='Corrected tomorrow', observed_at=LATE, first_seen_at=LATE, ingested_at=LATE)
        original = append(self.ledger, original)
        before = snapshot(self.ledger)
        append(self.ledger, revised)
        self.assertEqual(snapshot(self.ledger), before)
        self.assertEqual(load_snapshot(self.ledger, before.context.hash), before)
        reconstructed = snapshot(self.ledger, mode='historical_source_as_of')
        self.assertEqual(reconstructed.records[0].record_id, revised.record_id)
        self.assertIn('Reconstructed', reconstructed.context.caveats[0])
        with Ledger(Path(self.tmp.name) / 'shuffled') as other:
            append(other, revised)
            self.assertEqual(snapshot(other).records, ())  # missing predecessor quarantined
            append(other, original)
            self.assertEqual(snapshot(other), before)
            self.assertEqual(snapshot(other, mode='historical_source_as_of'), reconstructed)
        self.assertEqual(before.context.hash, '81d85010fa607719acceb272a748ac0ca8b61fe6c82c3ff8f409c02a4d749a9d')

    def test_deterministic_revision_ties(self):
        original = append(self.ledger, self.ex['NewsEvent'])
        for identity in ('z-revision', 'a-revision'):
            append(self.ledger, replace(original, revision=2, record_id=identity,
                                       supersedes_record_id=original.record_id, text=identity))
        self.assertEqual(snapshot(self.ledger).records[0].record_id, 'z-revision')
        self.assertIn('record_id', snapshot(self.ledger).manifest['tie_rule'])

    def test_completed_bar_future_action_unknown_and_clock_skew(self):
        bar = append(self.ledger, self.ex['PriceBar'])
        self.assertEqual(snapshot(self.ledger, decision_at='2025-01-06T14:29:59Z').records, ())
        self.assertEqual(snapshot(self.ledger).records, (bar,))
        append(self.ledger, replace(self.ex['NewsEvent'], available_at=None, availability_basis=AvailabilityBasis.UNKNOWN))
        append(self.ledger, replace(self.ex['Quote'], event_at=LATE))
        append(self.ledger, replace(self.ex['CorporateAction'], announced_at=LATE))
        self.assertEqual(snapshot(self.ledger).records, (bar,))
        append(self.ledger, replace(self.ex['Quote'], record_id='incomplete', source_record_id='incomplete', crossed=True, bid='102'))
        self.assertEqual(snapshot(self.ledger).records, (bar,))

    def test_observation_proxy_explicit_and_not_reconstructed(self):
        append(self.ledger, replace(self.ex['NewsEvent'], availability_basis=AvailabilityBasis.OBSERVED))
        self.assertIn('proxy', snapshot(self.ledger).context.caveats[0])
        self.assertEqual(snapshot(self.ledger, mode='historical_source_as_of').records, ())

    def test_ticker_reuse_classes_and_renames(self):
        i = append(self.ledger, self.ex['Instrument'])
        a = append(self.ledger, self.ex['InstrumentAlias'])
        self.assertEqual(resolve_alias(snapshot(self.ledger).records, alias='DEMO', namespace='ticker:SYNTH', at=T)[1], i.security_id)
        newer = replace(i, record_id='instrument:new', source_record_id='instrument:new', security_id='security:new',
                        listing_id='listing:new', share_class='preferred', effective_from=LATE, event_at=LATE,
                        available_at=LATE, observed_at=LATE, ingested_at=LATE)
        append(self.ledger, newer)
        append(self.ledger, replace(a, record_id='alias:new', source_record_id='alias:new', security_id='security:new',
                                   listing_id='listing:new', effective_from=LATE, event_at=LATE,
                                   available_at=LATE, observed_at=LATE, ingested_at=LATE))
        self.assertEqual(len(snapshot(self.ledger).manifest['identities']), 1)
        # Expiration known only tomorrow cannot retroactively rewrite yesterday.
        append(self.ledger, replace(a, record_id='alias:expired', revision=2, supersedes_record_id=a.record_id,
                                   effective_to=LATE, observed_at=LATE, ingested_at=LATE))
        future = snapshot(self.ledger, decision_at=LATE)
        self.assertEqual(resolve_alias(future.records, alias='DEMO', namespace='ticker:SYNTH', at=LATE)[1], 'security:new')
        self.assertEqual(len(future.manifest['identities']), 2)
        append(self.ledger, replace(a, record_id='alias:renamed', source_record_id='alias:renamed', alias='NEWNAME'))
        self.assertEqual(resolve_alias(snapshot(self.ledger).records, alias='NEWNAME', namespace='ticker:SYNTH', at=T)[2], i.listing_id)

    def test_dedup_retains_member_arrivals_and_does_not_backdate(self):
        first = append(self.ledger, self.ex['NewsEvent'])
        second = replace(first, record_id='repost', source_record_id='repost', available_at=LATE,
                         observed_at=LATE, first_seen_at=LATE, ingested_at=LATE)
        append(self.ledger, second)
        self.assertEqual(len(snapshot(self.ledger).manifest['dedup'][0]['members']), 1)
        result = snapshot(self.ledger, decision_at=LATE, text_seconds=172800)
        self.assertEqual(len(result.manifest['dedup'][0]['members']), 2)
        self.assertEqual(result.manifest['dedup'][0]['representative'], first.record_id)
        self.assertEqual(result.manifest['dedup'][0]['members'][1]['observed_at'], LATE)
        separate = replace(first, record_id='class-b-story', source_record_id='class-b-story',
                           entities=(replace(first.entities[0], instrument_ids=('security:class-b',)),))
        append(self.ledger, separate)
        self.assertEqual(len(snapshot(self.ledger).manifest['dedup']), 2)

    def test_missing_stale_outage_closed_and_bounded_fill(self):
        append(self.ledger, self.ex['Instrument'])
        bar = append(self.ledger, self.ex['PriceBar'])
        news = append(self.ledger, replace(self.ex['NewsEvent'], text=None))
        later = '2025-01-06T14:30:30Z'
        filled = snapshot(self.ledger, decision_at=later, forward_fill_seconds=30)
        self.assertTrue(filled.manifest['forward_fills'][0]['filled'])
        self.assertFalse(bar.filled)
        self.assertIn(news.record_id, filled.context.missing_record_ids)
        for reason in ('feed_outage', 'market_closed'):
            result = snapshot(self.ledger, decision_at=later, forward_fill_seconds=30,
                              signals=(('listing:demo', reason, news.record_id),))
            self.assertEqual(result.manifest['forward_fills'], [])
            self.assertIn(reason, [m['reason'] for m in result.manifest['masks']])
        stale = snapshot(self.ledger, decision_at='2025-01-06T14:32:00Z', expected_listings=('absent',))
        self.assertIn(bar.record_id, stale.context.stale_record_ids)
        self.assertEqual({m['reason'] for m in stale.manifest['masks']}, {'empty_text', 'missing_price', 'stale_price'})
        self.assertEqual(stale.manifest['forward_fills'], [])
        self.assertFalse(snapshot(self.ledger, decision_at=later, forward_fill_seconds=29).manifest['forward_fills'])
        with self.assertRaises(ValueError):
            snapshot(self.ledger, forward_fill_seconds=61)
        with self.assertRaises(ValueError):
            snapshot(self.ledger, signals=(('listing:demo', 'feed_outage', 'future-evidence'),))

    def test_bad_price_revisions_suppress_old_values_and_preserve_quality(self):
        for kind, changes, reason in (
                ('PriceBar', dict(complete=False, missing=True), 'missing_price'),
                ('PriceBar', dict(complete=False), 'incomplete_price'),
                ('Quote', dict(crossed=True, bid='102'), 'crossed_quote')):
            original = self.ex[kind]
            revised = replace(original, record_id='bad-' + reason, revision=2,
                              supersedes_record_id=original.record_id, **changes)
            snapshots = []
            for index, records in enumerate(((original, revised), (revised, original))):
                with Ledger(Path(self.tmp.name) / (reason + str(index))) as ledger:
                    for record in records:
                        append(ledger, record)
                    result = snapshot(ledger, decision_at='2025-01-06T14:30:30Z', forward_fill_seconds=30)
                    self.assertEqual(result.records, ())
                    self.assertEqual([r.record_id for r in result.quality_records], [revised.record_id])
                    self.assertEqual(result.manifest['masks'][0]['reason'], reason)
                    self.assertEqual(result.manifest['forward_fills'], [])
                    self.assertEqual(load_snapshot(ledger, result.context.hash), result)
                    snapshots.append(result)
            self.assertEqual(*snapshots)

    def test_missing_bar_blocks_fill_but_future_quality_cannot_change_past(self):
        original = append(self.ledger, self.ex['PriceBar'])
        before = snapshot(self.ledger, forward_fill_seconds=60)
        missing = replace(original, record_id='gap', source_record_id='gap',
                          end_at='2025-01-06T14:30:30Z', complete=False, missing=True,
                          available_at='2025-01-06T14:30:30Z', observed_at='2025-01-06T14:30:30Z',
                          ingested_at='2025-01-06T14:30:30Z')
        append(self.ledger, missing)
        self.assertEqual(snapshot(self.ledger, forward_fill_seconds=60), before)
        result = snapshot(self.ledger, decision_at='2025-01-06T14:30:30Z', forward_fill_seconds=60)
        self.assertEqual(result.manifest['forward_fills'], [])
        self.assertEqual(result.manifest['masks'][0]['reason'], 'missing_price')
        self.assertEqual(len(result.manifest['quality_inputs']), 1)
        self.assertEqual(load_snapshot(self.ledger, result.context.hash), result)
        self.ledger.revoke(missing.record_id, reason='synthetic rights deletion')
        with self.assertRaises(PermissionError):
            load_snapshot(self.ledger, result.context.hash)

    def test_forward_fill_separates_adjustment_calendar_and_interval(self):
        original = self.ex['PriceBar']
        variants = [original,
                    replace(original, record_id='adjusted', source_record_id='adjusted', adjustment='split_adjusted'),
                    replace(original, record_id='duration', source_record_id='duration', start_at='2025-01-06T14:28:00Z'),
                    replace(original, record_id='calendar', source_record_id='calendar', calendar_version='other-v1')]
        results = []
        for index, records in enumerate((variants, list(reversed(variants)))):
            with Ledger(Path(self.tmp.name) / ('basis' + str(index))) as ledger:
                for record in records:
                    append(ledger, record)
                result = snapshot(ledger, decision_at='2025-01-06T14:30:30Z', forward_fill_seconds=30)
                fills = result.manifest['forward_fills']
                self.assertEqual(len(fills), 4)
                self.assertEqual(len({tuple(f['basis']) for f in fills}), 4)
                self.assertEqual({f['source_record_id'] for f in fills}, {r.record_id for r in variants})
                results.append(result)
        self.assertEqual(*results)

    def test_rights_revocation_and_no_resurrection(self):
        original = append(self.ledger, self.ex['NewsEvent'])
        revised = append(self.ledger, replace(original, revision=2, supersedes_record_id=original.record_id, record_id='revision-2'))
        saved = snapshot(self.ledger)
        self.ledger.revoke(revised.record_id, reason='synthetic permission revoked')
        self.assertFalse(self.ledger.raw.path(revised.provenance.raw_sha256).exists())
        self.assertFalse(self.ledger.normalized.path(revised.hash).exists())
        self.assertEqual(snapshot(self.ledger).records, ())
        with self.assertRaises(PermissionError):
            load_snapshot(self.ledger, saved.context.hash)
        with self.assertRaises(PermissionError):
            append(self.ledger, revised)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 2)

    def test_retention_admission_and_normalized_only(self):
        for policy in (replace(POLICY, allow_normalized=False), replace(POLICY, expires_at=T)):
            with self.assertRaises(PermissionError):
                append(self.ledger, self.ex['NewsEvent'], policy)
        with self.assertRaises(ValueError):
            replace(POLICY, retain_provenance=False)
        record = append(self.ledger, self.ex['NewsEvent'], replace(POLICY, allow_raw=False))
        with self.assertRaises(PermissionError):
            self.ledger.raw_payload(record.record_id)
        self.assertFalse(self.ledger.raw.path(record.provenance.raw_sha256).exists())
        self.assertEqual(self.ledger.get(record.record_id), record)

    def test_reject_labels_invalid_lineage_and_storage_paths(self):
        for kind in ('FutureTarget', 'ContractTerminalMetadata', 'RealizedLabel'):
            with self.assertRaises(TypeError):
                append(self.ledger, self.ex[kind])
        news = append(self.ledger, self.ex['NewsEvent'])
        with self.assertRaises(ValueError):
            append(self.ledger, replace(news, record_id='bad', source_record_id='other', revision=2, supersedes_record_id=news.record_id))
        with self.assertRaises(ValueError):
            Ledger(Path(__file__).resolve().parents[2] / 'private-data')
        with self.assertRaises(ValueError):
            Ledger(Path(self.tmp.name) / 'other', raw_path=self.ledger.raw.root)

    def test_recovery_and_corruption_fail_closed(self):
        record = append(self.ledger, self.ex['NewsEvent'])
        orphan = self.ledger.raw.put(b'orphan')
        (self.ledger.normalized.root / '.pending-test').write_bytes(b'incomplete')
        self.ledger.recover()
        self.assertFalse(self.ledger.raw.path(orphan).exists())
        self.assertFalse((self.ledger.normalized.root / '.pending-test').exists())
        self.ledger.normalized.path(record.hash).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.ledger.recover()

    def test_import_boundary(self):
        code = '''
import sys
import cognitive_runtime.data.ledger, cognitive_runtime.data.snapshot
assert not any(x.startswith(('torch', 'transformers', 'cognitive_runtime.training')) for x in sys.modules)
assert 'cognitive_runtime.adapters.finance.labels' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
