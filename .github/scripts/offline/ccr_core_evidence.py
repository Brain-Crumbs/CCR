"""Controller-owned membership/execution evidence for sequential and xdist runs."""
import json
import os
from pathlib import Path
import sys

import pytest

_NODEIDS = []
_STARTED = set()
_WORKERS = {}


def pytest_collection_finish(session):
    _NODEIDS[:] = [item.nodeid for item in session.items]


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    if _NODEIDS and _NODEIDS != ids:
        raise pytest.UsageError("core workers disagree on selected test membership")
    _NODEIDS[:] = ids


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    _WORKERS[node.gateway.id] = node.workeroutput.get("ccr_environment", {})


def pytest_runtest_logstart(nodeid, location):
    _STARTED.add(nodeid)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        guard = sys.modules.get("sitecustomize")
        session.config.workeroutput["ccr_environment"] = {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "offline_guard": bool(guard and getattr(guard, "_LOG", None) == os.environ.get("CCR_OFFLINE_ATTEMPTS")),
        }
        return  # only the controller writes final merged evidence
    Path(os.environ["CCR_CORE_NODEIDS"]).write_text(
        json.dumps({"nodeids": _NODEIDS, "executed_tests": len(_STARTED), "worker_environments": _WORKERS}) + "\n",
        encoding="utf-8",
    )
