"""Durable content-addressed bytes in explicitly private, non-repository paths."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from cognitive_runtime.adapters.finance.schemas.base import sha256


def private_path(path):
    path = Path(path).expanduser().resolve()
    if any((p / '.git').is_file() or (p / '.git' / 'HEAD').exists() for p in (path, *path.parents)):
        raise ValueError('storage must be outside a repository')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RawStore:
    """Low-level CAS. Use Ledger for policy-checked reads and coordinated writes."""

    def __init__(self, path):
        self.root = private_path(path)

    def path(self, digest):
        sha256(digest)
        return self.root / digest

    def put(self, payload: bytes):
        if type(payload) is not bytes:
            raise TypeError('raw payload must be bytes')
        digest = hashlib.sha256(payload).hexdigest()
        target = self.path(digest)
        if target.exists():
            self.get(digest)  # detect corruption, never silently replace it
            return digest
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            sync_directory(self.root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest

    def get(self, digest):
        payload = self.path(digest).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('content-addressed blob is corrupt')
        return payload

    def delete(self, digest):
        self.path(digest).unlink(missing_ok=True)
        sync_directory(self.root)
