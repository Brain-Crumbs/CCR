"""Regression coverage for the persisted trial state machine, atomic locked
writes, and stale-worker recovery (epic #212 Phase B step 6, §16, issue #222).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from cognitive_runtime.training.model_factory.state import (
    ACTIVE_STATES,
    ALL_STATES,
    STATE_BUDGET_EXCEEDED,
    STATE_CANCELLED,
    STATE_CHECKPOINTING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_QUEUED,
    STATE_RUNNING,
    TERMINAL_STATES,
    DeviceReservationError,
    IllegalTransitionError,
    IncompleteRunError,
    StateError,
    create_state,
    heartbeat_path,
    is_stale,
    load_state,
    recover_stale_worker,
    release_devices,
    require_completed_for_promotion,
    reserve_devices,
    retry_failed_run,
    state_path,
    transition,
    write_heartbeat,
)

# --------------------------------------------------------------------- AC1: legal/illegal transitions


def test_every_legal_transition_is_permitted(tmp_path):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    assert load_state(path).state == STATE_QUEUED

    transition(path, STATE_RUNNING)
    transition(path, STATE_CHECKPOINTING)
    transition(path, STATE_RUNNING)
    transition(path, STATE_CHECKPOINTING)
    final = transition(path, STATE_COMPLETED)

    assert final.state == STATE_COMPLETED
    # queued->running->checkpointing->running->checkpointing->completed
    assert [entry["to"] for entry in final.history] == [
        STATE_QUEUED, STATE_RUNNING, STATE_CHECKPOINTING, STATE_RUNNING,
        STATE_CHECKPOINTING, STATE_COMPLETED,
    ]


@pytest.mark.parametrize("terminal_branch", [STATE_BUDGET_EXCEEDED, STATE_FAILED, STATE_CANCELLED])
def test_running_and_checkpointing_may_branch_to_every_non_promotable_terminal(tmp_path, terminal_branch):
    path = state_path(tmp_path / "from-running")
    create_state(path, "run-a")
    transition(path, STATE_RUNNING)
    final = transition(path, terminal_branch, reason="synthetic")
    assert final.state == terminal_branch
    assert final.is_terminal

    path2 = state_path(tmp_path / "from-checkpointing")
    create_state(path2, "run-b")
    transition(path2, STATE_RUNNING)
    transition(path2, STATE_CHECKPOINTING)
    final2 = transition(path2, terminal_branch, reason="synthetic")
    assert final2.state == terminal_branch


def test_queued_trial_may_be_cancelled_before_a_worker_ever_launches(tmp_path):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    final = transition(path, STATE_CANCELLED, reason="operator cancelled before launch")
    assert final.state == STATE_CANCELLED


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (STATE_COMPLETED, STATE_RUNNING),  # the epic's own named example
        (STATE_COMPLETED, STATE_QUEUED),
        (STATE_FAILED, STATE_COMPLETED),
        (STATE_FAILED, STATE_RUNNING),
        (STATE_BUDGET_EXCEEDED, STATE_RUNNING),
        (STATE_CANCELLED, STATE_RUNNING),
        (STATE_QUEUED, STATE_COMPLETED),
        (STATE_QUEUED, STATE_CHECKPOINTING),
        (STATE_RUNNING, STATE_QUEUED),
    ],
)
def test_every_illegal_transition_raises(tmp_path, from_state, to_state):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    # Drive the record to `from_state` via a legal path before probing the
    # illegal edge under test.
    route = {
        STATE_QUEUED: (),
        STATE_RUNNING: (STATE_RUNNING,),
        STATE_CHECKPOINTING: (STATE_RUNNING, STATE_CHECKPOINTING),
        STATE_COMPLETED: (STATE_RUNNING, STATE_COMPLETED),
        STATE_BUDGET_EXCEEDED: (STATE_RUNNING, STATE_BUDGET_EXCEEDED),
        STATE_FAILED: (STATE_RUNNING, STATE_FAILED),
        STATE_CANCELLED: (STATE_CANCELLED,),
    }[from_state]
    for step in route:
        transition(path, step)
    assert load_state(path).state == from_state

    with pytest.raises(IllegalTransitionError) as excinfo:
        transition(path, to_state)
    assert excinfo.value.current_state == from_state
    assert excinfo.value.requested_state == to_state
    # The rejected attempt must not have perturbed the persisted record.
    assert load_state(path).state == from_state


def test_unknown_state_name_is_rejected(tmp_path):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    with pytest.raises(StateError, match="unknown trial state"):
        transition(path, "not_a_real_state")


def test_every_state_name_is_covered_by_the_transition_table():
    from cognitive_runtime.training.model_factory.state import _LEGAL_TRANSITIONS

    assert set(_LEGAL_TRANSITIONS) == set(ALL_STATES)
    assert TERMINAL_STATES == {STATE_COMPLETED, STATE_BUDGET_EXCEEDED, STATE_FAILED, STATE_CANCELLED}
    for terminal in TERMINAL_STATES:
        assert _LEGAL_TRANSITIONS[terminal] == frozenset()


def test_create_state_refuses_to_overwrite_an_existing_record(tmp_path):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    with pytest.raises(FileExistsError):
        create_state(path, "run-a")


# --------------------------------------------------------------------- AC2: stale-worker watchdog


def test_watchdog_detects_stale_worker_after_declared_timeout_and_never_marks_completed(tmp_path):
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)
    create_state(state_file, "run-a")
    transition(state_file, STATE_RUNNING)

    clock = {"now": 1000.0}

    def fake_clock() -> float:
        return clock["now"]

    write_heartbeat(heartbeat_file, run_id="run-a", clock=fake_clock)

    # Still within the grace period: not stale, no-op.
    clock["now"] = 1050.0
    assert not is_stale(load_state(state_file), heartbeat_file, timeout_seconds=300.0, clock=fake_clock)
    assert recover_stale_worker(state_file, heartbeat_file, timeout_seconds=300.0, clock=fake_clock) is None
    assert load_state(state_file).state == STATE_RUNNING

    # The killed worker never heartbeats again; time passes the declared
    # timeout -- the watchdog must classify it stale and fail it, not
    # complete it.
    clock["now"] = 1301.0
    assert is_stale(load_state(state_file), heartbeat_file, timeout_seconds=300.0, clock=fake_clock)
    recovered = recover_stale_worker(state_file, heartbeat_file, timeout_seconds=300.0, clock=fake_clock)
    assert recovered is not None
    assert recovered.state == STATE_FAILED
    assert recovered.reason == "stale_worker_watchdog_timeout"

    # And a stale-recovered run can never later be marked completed.
    with pytest.raises(IllegalTransitionError):
        transition(state_file, STATE_COMPLETED)
    assert load_state(state_file).state == STATE_FAILED


def test_watchdog_ignores_a_run_with_no_live_worker_expected(tmp_path):
    """A queued (not yet launched) or already-terminal run is never "stale"
    -- there is no live worker to have gone quiet."""
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)  # deliberately never written
    create_state(state_file, "run-a")

    assert not is_stale(load_state(state_file), heartbeat_file, timeout_seconds=1.0)
    assert recover_stale_worker(state_file, heartbeat_file, timeout_seconds=1.0) is None
    assert load_state(state_file).state == STATE_QUEUED


def test_a_freshly_launched_worker_gets_its_full_grace_period_before_its_first_heartbeat(tmp_path):
    """Regression: a watchdog poll landing between `queued -> running` and
    the worker's first `write_heartbeat` call must not treat the missing
    file as an infinite gap -- a newly launched, responsive worker with a
    long declared timeout must not be failed immediately."""
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)  # not written yet
    create_state(state_file, "run-a")
    transition(state_file, STATE_RUNNING)

    assert not is_stale(load_state(state_file), heartbeat_file, timeout_seconds=300.0)
    assert recover_stale_worker(state_file, heartbeat_file, timeout_seconds=300.0) is None
    assert load_state(state_file).state == STATE_RUNNING


def test_a_never_heartbeated_run_becomes_stale_once_its_grace_period_elapses(tmp_path):
    """A worker that dies before its first heartbeat is exactly as stale as
    one that stopped heartbeating -- once its grace period actually elapses,
    never mistaken for fresh forever."""
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)  # never written
    create_state(state_file, "run-a")
    transition(state_file, STATE_RUNNING)

    time.sleep(0.1)

    assert is_stale(load_state(state_file), heartbeat_file, timeout_seconds=0.01)
    recovered = recover_stale_worker(state_file, heartbeat_file, timeout_seconds=0.01)
    assert recovered.state == STATE_FAILED


def test_a_heartbeat_belonging_to_a_different_run_id_is_not_trusted(tmp_path):
    """A leftover or misdirected heartbeat stamped with a *different*
    run_id must not keep a genuinely dead run classified as healthy just
    because its own timestamp looks fresh."""
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)
    create_state(state_file, "run-a")

    import datetime as _dt

    import cognitive_runtime.training.model_factory.state as state_module

    old_timestamp = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=1)).isoformat()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(state_module, "_now_iso", lambda: old_timestamp)
        transition(state_file, STATE_RUNNING)  # "entered running" an hour ago

    # A brand-new heartbeat -- but stamped with a different run's id.
    write_heartbeat(heartbeat_file, run_id="some-other-run")

    state = load_state(state_file)
    assert is_stale(state, heartbeat_file, timeout_seconds=5.0)
    recovered = recover_stale_worker(state_file, heartbeat_file, timeout_seconds=5.0)
    assert recovered is not None
    assert recovered.state == STATE_FAILED


def test_recovery_uses_the_timeout_declared_at_creation_when_not_overridden(tmp_path):
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)
    create_state(state_file, "run-a", heartbeat_timeout_seconds=0.01)
    transition(state_file, STATE_RUNNING)
    write_heartbeat(heartbeat_file, run_id="run-a")

    time.sleep(0.1)

    recovered = recover_stale_worker(state_file, heartbeat_file)  # no override supplied
    assert recovered is not None
    assert recovered.state == STATE_FAILED


def test_is_stale_requires_a_timeout_from_somewhere(tmp_path):
    state_file = state_path(tmp_path)
    create_state(state_file, "run-a")  # no heartbeat_timeout_seconds declared
    transition(state_file, STATE_RUNNING)

    with pytest.raises(StateError, match="no heartbeat_timeout_seconds"):
        is_stale(load_state(state_file), heartbeat_path(tmp_path))


def test_is_stale_rejects_an_override_that_conflicts_with_the_declared_timeout(tmp_path):
    state_file = state_path(tmp_path)
    create_state(state_file, "run-a", heartbeat_timeout_seconds=60.0)
    transition(state_file, STATE_RUNNING)

    with pytest.raises(StateError, match="conflicts with"):
        is_stale(load_state(state_file), heartbeat_path(tmp_path), timeout_seconds=30.0)

    # A non-conflicting (matching) override is accepted.
    assert is_stale(
        load_state(state_file), heartbeat_path(tmp_path), timeout_seconds=60.0,
    ) in (True, False)


def test_a_backward_clock_adjustment_is_treated_as_a_fresh_heartbeat_not_an_error(tmp_path):
    """A wall-clock rollback after a heartbeat was written must not raise
    inside the watchdog -- that would stop a polling loop from recovering
    any later, genuinely stale worker."""
    state_file = state_path(tmp_path)
    heartbeat_file = heartbeat_path(tmp_path)
    create_state(state_file, "run-a")
    transition(state_file, STATE_RUNNING)

    clock = {"now": 1_000_000.0}

    def fake_clock() -> float:
        return clock["now"]

    write_heartbeat(heartbeat_file, run_id="run-a", clock=fake_clock)
    clock["now"] = 999_999.0  # the clock moved backward after the heartbeat

    # Must not raise ValueError from a negative age reaching watchdog_expired.
    assert not is_stale(
        load_state(state_file), heartbeat_file, timeout_seconds=300.0, clock=fake_clock,
    )
    assert recover_stale_worker(
        state_file, heartbeat_file, timeout_seconds=300.0, clock=fake_clock,
    ) is None


_SIGKILL_WORKER_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {path!r})
    from cognitive_runtime.training.model_factory.state import (
        create_state, transition, write_heartbeat, STATE_RUNNING,
    )

    state_file = {state_file!r}
    heartbeat_file = {heartbeat_file!r}
    create_state(state_file, "killed-run")
    transition(state_file, STATE_RUNNING)
    write_heartbeat(heartbeat_file, run_id="killed-run")
    # Signal readiness, then hang so the parent can SIGKILL a genuinely
    # live process instead of one that already exited on its own.
    with open({ready_file!r}, "w") as handle:
        handle.write("ready")
    while True:
        pass
    """
)


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL semantics are POSIX-specific")
def test_a_sigkilled_worker_is_classified_stale_by_the_watchdog_and_never_completed(tmp_path):
    """AC: a SIGKILLed worker leaves a state record the watchdog classifies
    as stale, and recovery never marks it completed -- exercised against a
    real killed OS process, not a mock."""
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    state_file = tmp_path / "state.json"
    heartbeat_file = tmp_path / "heartbeat.json"
    ready_file = tmp_path / "ready"

    script = _SIGKILL_WORKER_SCRIPT.format(
        path=repo_root, state_file=str(state_file), heartbeat_file=str(heartbeat_file),
        ready_file=str(ready_file),
    )
    worker = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 30.0
        while not ready_file.exists():
            assert time.monotonic() < deadline, "worker never signalled readiness"
            time.sleep(0.05)
        assert load_state(state_file).state == STATE_RUNNING

        worker.kill()  # SIGKILL: no cleanup, no final heartbeat, no graceful exit
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)

    recovered = recover_stale_worker(state_file, heartbeat_file, timeout_seconds=0.001)
    assert recovered is not None
    assert recovered.state == STATE_FAILED
    assert recovered.state != STATE_COMPLETED

    with pytest.raises(IllegalTransitionError):
        transition(state_file, STATE_COMPLETED)


# --------------------------------------------------------------------- AC3: concurrent writers


_RACE_WORKER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {path!r})
    from cognitive_runtime.training.model_factory.state import transition, STATE_RUNNING

    state_file = {state_file!r}
    ready_file = {ready_file!r}
    go_file = {go_file!r}
    out_file = {out_file!r}

    with open(ready_file, "w") as handle:
        handle.write("ready")
    deadline = time.monotonic() + 30.0
    while not __import__("os").path.exists(go_file):
        if time.monotonic() > deadline:
            raise SystemExit("timed out waiting for go file")
        pass

    result = {{}}
    try:
        transition(state_file, STATE_RUNNING)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{{type(exc).__name__}}: {{exc}}"

    import json
    with open(out_file, "w") as handle:
        json.dump(result, handle)
    """
)


def test_concurrent_transition_from_two_real_subprocesses_is_serialized(tmp_path):
    """AC: concurrent writes from two real processes produce a valid
    document, with one write serialised behind the other -- exactly one of
    two racing ``queued -> running`` attempts wins."""
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    state_file = tmp_path / "state.json"
    create_state(state_file, "run-a")

    procs = []
    outcomes = []
    for label in ("a", "b"):
        ready_file = tmp_path / f"ready-{label}"
        go_file = tmp_path / "go"
        out_file = tmp_path / f"out-{label}.json"
        script = _RACE_WORKER_SCRIPT.format(
            path=repo_root, state_file=str(state_file), ready_file=str(ready_file),
            go_file=str(go_file), out_file=str(out_file),
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        procs.append(proc)
        outcomes.append((ready_file, out_file))

    # Wait for both real OS processes to reach their busy-wait, then release
    # them together so the race genuinely contends on the lock.
    deadline = time.monotonic() + 30.0
    for ready_file, _out_file in outcomes:
        while not ready_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    (tmp_path / "go").write_text("go", encoding="utf-8")

    for proc in procs:
        assert proc.wait(timeout=30) == 0

    results = []
    for _ready_file, out_file in outcomes:
        assert out_file.exists()
        results.append(json.loads(out_file.read_text(encoding="utf-8")))

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert "IllegalTransitionError" in failures[0]["error"]

    # The document is valid JSON with exactly one queued->running edge --
    # the lock prevented a lost update, not merely a torn write.
    final = load_state(state_file)
    assert final.state == STATE_RUNNING
    transitions_to_running = [e for e in final.history if e["to"] == STATE_RUNNING]
    assert len(transitions_to_running) == 1


def test_concurrent_create_state_is_also_serialized(tmp_path):
    """The same guarantee applies to the very first write: two launchers
    racing to initialize the same run's state must not both succeed."""
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    state_file = tmp_path / "state.json"

    script = textwrap.dedent(
        """
        import sys, time, os, json
        sys.path.insert(0, {path!r})
        from cognitive_runtime.training.model_factory.state import create_state

        ready_file = {ready_file!r}
        go_file = {go_file!r}
        out_file = {out_file!r}
        with open(ready_file, "w") as handle:
            handle.write("ready")
        deadline = time.monotonic() + 30.0
        while not os.path.exists(go_file):
            if time.monotonic() > deadline:
                raise SystemExit("timeout")
        result = {{}}
        try:
            create_state({state_file!r}, "run-a")
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["error"] = type(exc).__name__
        with open(out_file, "w") as handle:
            json.dump(result, handle)
        """
    )
    procs, outcomes = [], []
    for label in ("a", "b"):
        ready_file = tmp_path / f"ready-{label}"
        out_file = tmp_path / f"out-{label}.json"
        rendered = script.format(
            path=repo_root, state_file=str(state_file), ready_file=str(ready_file),
            go_file=str(tmp_path / "go"), out_file=str(out_file),
        )
        proc = subprocess.Popen([sys.executable, "-c", rendered])
        procs.append(proc)
        outcomes.append((ready_file, out_file))

    deadline = time.monotonic() + 30.0
    for ready_file, _ in outcomes:
        while not ready_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    (tmp_path / "go").write_text("go", encoding="utf-8")

    for proc in procs:
        assert proc.wait(timeout=30) == 0

    results = [json.loads(out_file.read_text(encoding="utf-8")) for _, out_file in outcomes]
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert failures[0]["error"] == "FileExistsError"
    assert load_state(state_file).state == STATE_QUEUED


# --------------------------------------------------------------------- AC4: promotion gate


def test_completed_run_is_promotable(tmp_path):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    transition(path, STATE_RUNNING)
    final = transition(path, STATE_COMPLETED)
    require_completed_for_promotion(final)  # must not raise


@pytest.mark.parametrize(
    "steps",
    [
        (),  # still queued
        (STATE_RUNNING,),  # interrupted mid-flight
        (STATE_RUNNING, STATE_CHECKPOINTING),
        (STATE_RUNNING, STATE_BUDGET_EXCEEDED),
        (STATE_RUNNING, STATE_FAILED),
        (STATE_CANCELLED,),
    ],
)
def test_every_non_completed_run_is_rejected_by_the_promotion_path(tmp_path, steps):
    path = state_path(tmp_path)
    create_state(path, "run-a")
    state = load_state(path)
    for step in steps:
        state = transition(path, step)

    with pytest.raises(IncompleteRunError):
        require_completed_for_promotion(state)


# --------------------------------------------------------------------- AC5: retry lineage


def test_retrying_a_failed_run_creates_a_new_run_id_naming_the_failed_run(tmp_path):
    failed_path = state_path(tmp_path / "run-a")
    create_state(failed_path, "run-a")
    transition(failed_path, STATE_RUNNING)
    transition(failed_path, STATE_FAILED, reason="crashed")

    retry_path = state_path(tmp_path / "run-a-retry-1")
    retried = retry_failed_run(failed_path, retry_path, "run-a-retry-1")

    assert retried.run_id == "run-a-retry-1"
    assert retried.state == STATE_QUEUED
    assert retried.retried_from == "run-a"
    assert load_state(retry_path).retried_from == "run-a"


@pytest.mark.parametrize("steps", [(), (STATE_RUNNING,), (STATE_RUNNING, STATE_COMPLETED)])
def test_retry_is_refused_unless_the_source_run_actually_failed(tmp_path, steps):
    source_path = state_path(tmp_path / "run-a")
    create_state(source_path, "run-a")
    for step in steps:
        transition(source_path, step)

    with pytest.raises(StateError, match="not 'failed'|not \"failed\""):
        retry_failed_run(source_path, state_path(tmp_path / "run-a-retry"), "run-a-retry")


def test_retry_rejects_reusing_the_failed_runs_own_id(tmp_path):
    failed_path = state_path(tmp_path / "run-a")
    create_state(failed_path, "run-a")
    transition(failed_path, STATE_RUNNING)
    transition(failed_path, STATE_FAILED)

    with pytest.raises(StateError, match="new run_id"):
        retry_failed_run(failed_path, state_path(tmp_path / "run-a-again"), "run-a")


# --------------------------------------------------------------------- device reservation (scope)


def test_device_reservation_is_exclusive_and_idempotent_for_the_same_run(tmp_path):
    reserve_devices(tmp_path, "run-a", ["cuda:0"])
    # Re-reserving its own device (e.g. a crash-and-relaunch) is not a conflict.
    reserve_devices(tmp_path, "run-a", ["cuda:0"])

    with pytest.raises(DeviceReservationError):
        reserve_devices(tmp_path, "run-b", ["cuda:0"])

    release_devices(tmp_path, "run-a")
    reserve_devices(tmp_path, "run-b", ["cuda:0"])  # now free


def test_release_devices_is_a_no_op_for_a_run_holding_nothing(tmp_path):
    release_devices(tmp_path, "run-with-no-reservation")  # must not raise


# --------------------------------------------------------------------- AC6: torch-free import


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.state  # noqa: F401
print("OK")
"""


def test_state_module_imports_cleanly_without_torch():
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=repo_root)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# --------------------------------------------------------------------- misc

def test_active_states_are_exactly_running_and_checkpointing():
    assert ACTIVE_STATES == {STATE_RUNNING, STATE_CHECKPOINTING}
