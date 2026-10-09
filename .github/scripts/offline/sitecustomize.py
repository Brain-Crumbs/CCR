"""Inherited Python network guard for the core test runner (including children).

The append-only attempt log makes swallowed denial exceptions fail the runner.
CI also uses a network namespace, covering non-Python subprocesses.
"""
import os
import sys

_LOG = os.environ.get("CCR_OFFLINE_ATTEMPTS")


def _audit(event, args):
    blocked = event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        address = args[-1]
        blocked = isinstance(address, tuple)  # AF_INET/6, leave local IPC intact
    if blocked:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(event + "\n")  # never log hosts, tokens or raw payloads
        raise RuntimeError(f"offline core forbids network operation: {event}")


if _LOG:
    sys.addaudithook(_audit)
