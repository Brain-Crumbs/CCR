"""Append-only observation metadata; rights-aware CAS and crash recovery.

SQLite transactions commit references only after fsynced blobs. A process lock
serializes append, snapshot reads and recovery across processes on local POSIX
filesystems. No network filesystems or external database services are supported.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import sqlite3

from cognitive_runtime.adapters.finance.schemas.base import Strict, timestamp
from cognitive_runtime.adapters.finance.schemas.records import (
    Instrument, InstrumentAlias, PriceBar, Quote, CorporateAction, NewsEvent,
    PredictionContractState,
)
from .raw_store import RawStore, private_path

OBSERVATIONS = (Instrument, InstrumentAlias, PriceBar, Quote, CorporateAction, NewsEvent, PredictionContractState)
TYPES = {t.__name__: t for t in OBSERVATIONS}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RetentionPolicy(Strict):
    """Explicit operator evidence, not an inference from public accessibility.

    Only policies permitting immutable minimal ID/hash/clock/rights provenance
    are supported. A policy forbidding even that metadata must be rejected.
    """
    rights: str
    evidence: str
    allow_raw: bool
    allow_normalized: bool
    retain_provenance: bool
    expires_at: str | None = None

    def validate(self):
        if not self.retain_provenance:
            raise ValueError('append-only provenance retention is not permitted')
        if self.expires_at is not None:
            timestamp(self.expires_at)


class Ledger:
    def __init__(self, path, *, raw_path=None, normalized_path=None):
        self.root = private_path(path)
        self.raw = RawStore(raw_path or self.root / 'raw')
        self.normalized = RawStore(normalized_path or self.root / 'normalized')
        if (self.raw.root == self.normalized.root or self.raw.root in self.normalized.root.parents
                or self.normalized.root in self.raw.root.parents):
            raise ValueError('raw and normalized stores must be distinct')
        for store in (self.raw, self.normalized):
            if store.root == self.root or store.root in self.root.parents:
                raise ValueError('CAS cannot contain the ledger index')
            owner = store.root / '.ledger-owner'
            identity = hashlib.sha256(str(self.root).encode()).hexdigest()
            if owner.exists():
                if owner.read_text() != identity:
                    raise ValueError('CAS belongs to another ledger')
            else:
                if any(store.root.iterdir()):
                    raise ValueError('CAS directory must be empty or owned by this ledger')
                with owner.open('x') as stream:
                    stream.write(identity)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(owner, 0o600)
        self._lock = open(self.root / '.lock', 'a+b')
        os.chmod(self.root / '.lock', 0o600)
        self.db = sqlite3.connect(self.root / 'index.sqlite', isolation_level=None)
        os.chmod(self.root / 'index.sqlite', 0o600)
        self.db.row_factory = sqlite3.Row
        try:
            with self.locked():
                self.db.execute('PRAGMA synchronous=FULL')
                version = self.db.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, 1):
                    raise ValueError('unsupported ledger version; explicit migration required')
                if version == 0:
                    # Refuse to reinterpret any existing unversioned schema.
                    if self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                        raise ValueError('unversioned ledger requires explicit migration')
                    self.db.executescript('''BEGIN IMMEDIATE;
                        CREATE TABLE events (
                          record_id TEXT PRIMARY KEY, family TEXT NOT NULL, source_id TEXT NOT NULL,
                          source_record_id TEXT NOT NULL, revision INTEGER NOT NULL,
                          supersedes TEXT, observed_at TEXT NOT NULL, raw_hash TEXT NOT NULL,
                          normalized_hash TEXT NOT NULL, policy TEXT NOT NULL);
                        CREATE INDEX lineage ON events(family,source_id,source_record_id);
                        CREATE TABLE deletions (kind TEXT NOT NULL, hash TEXT NOT NULL,
                          reason TEXT NOT NULL, PRIMARY KEY(kind,hash));
                        CREATE TABLE snapshots (hash TEXT PRIMARY KEY, payload TEXT NOT NULL);
                        PRAGMA user_version=1;
                        COMMIT;''')
                for table in ('events', 'deletions', 'snapshots'):
                    for operation in ('UPDATE', 'DELETE'):
                        self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only table'); END")
            self.recover()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def locked(self):
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock, fcntl.LOCK_UN)

    def close(self):
        self.db.close()
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _denied(self, kind, value):
        return self.db.execute('SELECT 1 FROM deletions WHERE kind=? AND hash=?', (kind, value)).fetchone() is not None

    def append(self, record, raw_payload, policy):
        if type(record) not in OBSERVATIONS or type(policy) is not RetentionPolicy:
            raise TypeError('exact inference observation and retention policy required')
        if type(raw_payload) is not bytes:
            raise TypeError('raw payload must be bytes')
        if hashlib.sha256(raw_payload).hexdigest() != record.provenance.raw_sha256:
            raise ValueError('raw provenance mismatch')
        if policy.rights != record.provenance.rights:
            raise ValueError('rights evidence mismatch')
        if policy.expires_at and timestamp(policy.expires_at) <= datetime.now(timezone.utc):
            raise PermissionError('retention expired')
        if not policy.allow_normalized:
            raise PermissionError('normalized retention not allowed')
        encoded = record.canonical_json().encode()
        with self.locked():
            old = self.db.execute('SELECT * FROM events WHERE record_id=?', (record.record_id,)).fetchone()
            if old:
                if old['normalized_hash'] != record.hash or old['policy'] != policy.canonical_json():
                    raise ValueError('immutable record ID conflict')
                if self._denied('normalized', record.hash):
                    raise PermissionError('record was revoked')
                return record.record_id
            if self._denied('raw', record.provenance.raw_sha256) or self._denied('normalized', record.hash):
                raise PermissionError('revoked content cannot be reintroduced')
            # Validate both directions to permit deterministic shuffled ingestion.
            family = (record.schema_name, record.source_id, record.source_record_id)
            neighbors = self.db.execute('SELECT * FROM events WHERE record_id=? OR supersedes=?',
                                        (record.supersedes_record_id, record.record_id)).fetchall()
            for row in neighbors:
                if (row['family'], row['source_id'], row['source_record_id']) != family:
                    raise ValueError('supersedes crosses logical identity')
                if row['record_id'] == record.supersedes_record_id:
                    valid = row['revision'] + 1 == record.revision and timestamp(row['observed_at']) <= timestamp(record.observed_at)
                else:
                    valid = record.revision + 1 == row['revision'] and timestamp(record.observed_at) <= timestamp(row['observed_at'])
                if not valid:
                    raise ValueError('invalid revision/observation lineage')
            # Commit reference after CAS durability; recovery removes precommit orphans.
            if policy.allow_raw:
                self.raw.put(raw_payload)
            self.normalized.put(encoded)
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (record.record_id, *family, record.revision, record.supersedes_record_id,
                             record.observed_at, record.provenance.raw_sha256, record.hash, policy.canonical_json()))
        return record.record_id

    def _read(self, row):
        policy = RetentionPolicy.from_json(row['policy'])
        if (self._denied('normalized', row['normalized_hash']) or
                (policy.expires_at and timestamp(policy.expires_at) <= datetime.now(timezone.utc))):
            raise PermissionError('record retention expired or revoked')
        record = TYPES[row['family']].from_json(self.normalized.get(row['normalized_hash']))
        if record.record_id != row['record_id'] or record.hash != row['normalized_hash']:
            raise ValueError('index/content mismatch')
        return record

    def get(self, record_id):
        with self.locked():
            row = self.db.execute('SELECT * FROM events WHERE record_id=?', (record_id,)).fetchone()
            if row is None:
                raise KeyError(record_id)
            return self._read(row)

    def raw_payload(self, record_id):
        with self.locked():
            row = self.db.execute('SELECT * FROM events WHERE record_id=?', (record_id,)).fetchone()
            if row is None:
                raise KeyError(record_id)
            self._read(row)
            if not RetentionPolicy.from_json(row['policy']).allow_raw or self._denied('raw', row['raw_hash']):
                raise PermissionError('raw retention unavailable')
            return self.raw.get(row['raw_hash'])

    def _records(self):
        rows = self.db.execute('SELECT * FROM events ORDER BY record_id').fetchall()
        # Never resurrect an older version after a rights tombstone. A logical
        # stream with unavailable revisions is unavailable as a whole.
        blocked = set()
        for row in rows:
            policy = RetentionPolicy.from_json(row['policy'])
            if (self._denied('normalized', row['normalized_hash']) or
                    (policy.expires_at and timestamp(policy.expires_at) <= datetime.now(timezone.utc))):
                blocked.add((row['family'], row['source_id'], row['source_record_id']))
        return tuple(self._read(row) for row in rows
                     if (row['family'], row['source_id'], row['source_record_id']) not in blocked)

    def records(self):
        with self.locked():
            return self._records()

    def revoke(self, record_id, *, reason):
        """Irreversible hash-wide revocation, including shared raw/normalized bytes.

        The caller must have permission to retain minimal provenance. Persist the
        denial before deleting bytes; recovery finishes any interrupted cleanup.
        """
        if not reason or type(reason) is not str:
            raise ValueError('nonempty rights reason required')
        with self.locked():
            row = self.db.execute('SELECT * FROM events WHERE record_id=?', (record_id,)).fetchone()
            if row is None:
                raise KeyError(record_id)
            self.db.execute('BEGIN IMMEDIATE')
            try:
                # Sharing a prohibited source payload also invalidates derived records.
                rows = self.db.execute('SELECT raw_hash, normalized_hash FROM events WHERE raw_hash=?', (row['raw_hash'],)).fetchall()
                for item in rows:
                    for kind in ('raw', 'normalized'):
                        self.db.execute('INSERT OR IGNORE INTO deletions VALUES (?,?,?)', (kind, item[kind + '_hash'], reason))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise
            self._cleanup()

    def _cleanup(self):
        for row in self.db.execute('SELECT kind,hash FROM deletions'):
            getattr(self, row['kind']).delete(row['hash'])

    def recover(self):
        """Finish deletions, remove temporary/unreferenced blobs; detect data loss."""
        with self.locked():
            # Expired records fail reads even before this physical purge executes.
            expired = [r for r in self.db.execute('SELECT * FROM events')
                       if (p := RetentionPolicy.from_json(r['policy'])).expires_at and
                       timestamp(p.expires_at) <= datetime.now(timezone.utc)]
            with self.db:
                for original in expired:
                    shared = self.db.execute('SELECT * FROM events WHERE raw_hash=?', (original['raw_hash'],)).fetchall()
                    for row in shared:
                      for kind in ('raw', 'normalized'):
                        self.db.execute('INSERT OR IGNORE INTO deletions VALUES (?,?,?)',
                                        (kind, row[kind + '_hash'], 'retention_expired'))
            self._cleanup()
            for kind in ('raw', 'normalized'):
                store = getattr(self, kind)
                referenced = {r[0] for r in self.db.execute(f'SELECT {kind}_hash FROM events')}
                for path in store.root.iterdir():
                    if path.name != '.ledger-owner' and path.is_file() and (path.name.startswith('.pending-') or path.name not in referenced):
                        path.unlink()
            # Referenced non-deleted normalized blobs must remain durable and valid.
            for row in self.db.execute('SELECT * FROM events'):
                if not self._denied('normalized', row['normalized_hash']):
                    self._read(row)
                    if RetentionPolicy.from_json(row['policy']).allow_raw and not self._denied('raw', row['raw_hash']):
                        self.raw.get(row['raw_hash'])
