"""Small transport and evidence contracts shared by read-only providers."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = field(repr=False)


class Transport(Protocol):
    def get(self, url: str, *, params: dict, headers: dict, timeout: float) -> Response: ...


@dataclass
class FetchResult:
    """Safe summary: never includes bodies, headers, tokens or exception text.

    received = written + reused + rejected. Counts cover received bar entries,
    not unobserved market intervals. drained does not imply complete coverage.
    """
    feed: str
    config_sha256: str
    capability_sha256: str
    request_sha256: str | None = None
    retention_sha256: str | None = None
    capability_manifest: dict = field(default_factory=dict)
    status: str = 'pending'
    readiness: str = 'docs_only'
    requests: int = 0
    pages: int = 0
    received: int = 0
    written: int = 0
    reused: int = 0
    rejected: int = 0
    drained: bool = False
    by_symbol: dict = field(default_factory=dict)
    missing_symbols: list = field(default_factory=list)
    coverage: str = 'unverified'
    reasons: dict = field(default_factory=dict)
    record_ids: list = field(default_factory=list)
    response_sha256: list = field(default_factory=list)
    http_status: int | None = None

    def reject(self, reason):
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
