"""Explicit offline extended tier: synthetic append/replay and real crash tests."""
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cognitive_runtime.data.ledger import Ledger
from tests.market.test_ledger_snapshots import append, snapshot, examples, POLICY


class LedgerExtendedTests(unittest.TestCase):
    def test_large_append_replay_performance_and_reopen(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            records = [replace(examples()['Quote'], record_id=f'quote:{i:05d}', source_record_id=f'quote:{i:05d}') for i in range(1000)]
            started = time.monotonic()
            with Ledger(Path(tmp) / 'ordered') as ledger:
                for record in records:
                    append(ledger, record)
                expected = snapshot(ledger)
            appended = time.monotonic() - started
            random.Random(287).shuffle(records)
            with Ledger(Path(tmp) / 'shuffled') as ledger:
                for record in records:
                    append(ledger, record)
                start = time.monotonic()
                actual = snapshot(ledger)
                replayed = time.monotonic() - start
                self.assertEqual(actual, expected)
            with Ledger(Path(tmp) / 'ordered') as ledger:
                self.assertEqual(snapshot(ledger), expected)
                self.assertEqual(ledger.db.execute('PRAGMA user_version').fetchone()[0], 1)
            # Broad module-local guard, independent from unchanged default caps.
            self.assertLess(appended, 60)
            self.assertLess(replayed, 15)
            print(f'ledger extended: 1000 append={appended:.3f}s snapshot={replayed:.3f}s hash={expected.context.hash}')

    def test_process_crash_after_blob_before_commit_and_during_delete(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = str(Path(tmp) / 'ledger')
            setup = '''
import os,sys
from cognitive_runtime.data.ledger import Ledger
from tests.market.test_ledger_snapshots import append,examples
ledger=Ledger(sys.argv[1])
'''
            crash_append = setup + '''
original=ledger.normalized.put
def stop(payload):
    original(payload)
    os._exit(71)
ledger.normalized.put=stop
append(ledger,examples()['NewsEvent'])
'''
            result = subprocess.run([sys.executable, '-c', crash_append, root])
            self.assertEqual(result.returncode, 71)
            with Ledger(root) as ledger:
                self.assertEqual(ledger.records(), ())
                self.assertEqual([p.name for p in ledger.raw.root.iterdir()], ['.ledger-owner'])
                record = append(ledger, examples()['NewsEvent'])
            crash_delete = setup + '''
def stop():
    os._exit(72)
ledger._cleanup=stop
ledger.revoke('NewsEvent:demo:r1',reason='synthetic revocation')
'''
            result = subprocess.run([sys.executable, '-c', crash_delete, root])
            self.assertEqual(result.returncode, 72)
            with Ledger(root) as ledger:
                self.assertFalse(ledger.raw.path(record.provenance.raw_sha256).exists())
                self.assertFalse(ledger.normalized.path(record.hash).exists())
                with self.assertRaises(PermissionError):
                    ledger.get(record.record_id)
                self.assertEqual(ledger.db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)

    def test_retention_expiry_physically_deletes_payload(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with Ledger(Path(tmp) / 'ledger') as ledger:
                now = datetime.now(timezone.utc)
                expiry = (now + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
                record = append(ledger, examples()['NewsEvent'], replace(POLICY, expires_at=expiry))
                with patch('cognitive_runtime.data.ledger.datetime') as clock:
                    clock.now.return_value = now + timedelta(days=2)
                    with self.assertRaises(PermissionError):
                        ledger.get(record.record_id)
                    ledger.recover()
                self.assertFalse(ledger.raw.path(record.provenance.raw_sha256).exists())
                self.assertFalse(ledger.normalized.path(record.hash).exists())

    def test_migration_rejects_future_and_unversioned_databases(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            for version in (0, 2):
                root = Path(tmp) / str(version)
                root.mkdir()
                db = sqlite3.connect(root / 'index.sqlite')
                db.execute('CREATE TABLE legacy(payload TEXT)')
                db.execute(f'PRAGMA user_version={version}')
                db.close()
                with self.assertRaisesRegex(ValueError, 'version|migration'):
                    Ledger(root)


if __name__ == '__main__':
    unittest.main()
