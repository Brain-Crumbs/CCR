"""Persisted trial state machine, atomic locked writes, and stale-worker
recovery (epic #212 Phase B step 6, §16, issue #222).

```text
queued -> running -> checkpointing -> completed
                          |-> budget_exceeded | failed | cancelled
```

Every state change is a locked read-validate-write critical section: a
single-process file lock (POSIX ``flock``/Windows ``msvcrt.locking``) guards
the transition, and the record itself is written with the same
temp-write/rename primitive as every other Model Factory manifest
(:func:`cognitive_runtime.training.model_factory.artifacts.atomic_write_json`).
This closes the epic risk table's "a local worker dies or two writers race":
two concurrent writers can never interleave into a corrupt document, and
never both believe they won the same transition.

A bounded watchdog (:func:`recover_stale_worker`) detects a worker whose
heartbeat has gone quiet past its declared timeout and fails the run --
never completes it. :func:`require_completed_for_promotion` is the promotion
path's gate: an incomplete or interrupted run can never be promoted.
:func:`retry_failed_run` is the only sanctioned way to retry a failed run --
as a brand-new run whose lineage names the failed one, never by reusing or
reviving the failed run's own state record. Resuming an interrupted run
instead reuses its own run directory under MF-B2's strict resume contract
(:mod:`cognitive_runtime.training.model_factory.checkpoint`); this module
does not duplicate that decision, only the fact that a stale/failed record
is never silently reopened as ``completed``.

Pure Python + filesystem -- no torch -- so this module is usable in the
core-only install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.budget import watchdog_expired

STATE_FORMAT = "model-factory-trial-state-v1"
HEARTBEAT_FORMAT = "model-factory-trial-heartbeat-v1"
DEVICE_LEDGER_FORMAT = "model-factory-device-ledger-v1"
CANCEL_REQUEST_FORMAT = "model-factory-cancel-request-v1"

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_CHECKPOINTING = "checkpointing"
STATE_COMPLETED = "completed"
STATE_BUDGET_EXCEEDED = "budget_exceeded"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

ALL_STATES: Tuple[str, ...] = (
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_CHECKPOINTING,
    STATE_COMPLETED,
    STATE_BUDGET_EXCEEDED,
    STATE_FAILED,
    STATE_CANCELLED,
)

#: A run occupies one of these states only while a worker is expected to be
#: alive and heartbeating; every other state is either not yet launched or
#: already terminal.
ACTIVE_STATES = frozenset({STATE_RUNNING, STATE_CHECKPOINTING})

#: No transition ever leaves a terminal state -- this is what makes
#: ``completed -> running`` (and every other post-terminal edge) illegal.
TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_BUDGET_EXCEEDED, STATE_FAILED, STATE_CANCELLED})

# epic #212 §16's diagram, plus the checkpointing<->running cycle a periodic
# checkpoint boundary implies (MF-B3's checkpoint_due), and letting a queued
# trial be cancelled before a worker ever launches.
_LEGAL_TRANSITIONS: Dict[str, frozenset] = {
    STATE_QUEUED: frozenset({STATE_RUNNING, STATE_CANCELLED}),
    STATE_RUNNING: frozenset({
        STATE_CHECKPOINTING, STATE_COMPLETED, STATE_BUDGET_EXCEEDED, STATE_FAILED, STATE_CANCELLED,
    }),
    STATE_CHECKPOINTING: frozenset({
        STATE_RUNNING, STATE_COMPLETED, STATE_BUDGET_EXCEEDED, STATE_FAILED, STATE_CANCELLED,
    }),
    STATE_COMPLETED: frozenset(),
    STATE_BUDGET_EXCEEDED: frozenset(),
    STATE_FAILED: frozenset(),
    STATE_CANCELLED: frozenset(),
}


class StateError(ValueError):
    """A malformed, unknown, or otherwise invalid trial state record."""


class IllegalTransitionError(StateError):
    """Raised when a requested transition is not a legal state-machine edge."""

    def __init__(self, run_id: str, current_state: str, requested_state: str) -> None:
        self.run_id = run_id
        self.current_state = current_state
        self.requested_state = requested_state
        super().__init__(
            f"illegal trial state transition for {run_id!r}: "
            f"{current_state!r} -> {requested_state!r}"
        )


class IncompleteRunError(StateError):
    """Raised when the promotion path is offered a non-``completed`` run."""


class DeviceReservationError(StateError):
    """Raised when a declared device is already reserved by another run."""


@dataclasses.dataclass(frozen=True)
class TrialState:
    """A trial's persisted state-machine record.

    ``history`` is append-only: every accepted transition adds one entry
    rather than replacing the record's provenance, so a stale/failed run's
    full lifecycle stays inspectable after the fact.
    """

    format: str
    run_id: str
    state: str
    created_at: str
    updated_at: str
    reason: Optional[str] = None
    devices: Tuple[str, ...] = ()
    retried_from: Optional[str] = None
    heartbeat_timeout_seconds: Optional[float] = None
    history: Tuple[Dict[str, Any], ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "run_id": self.run_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reason": self.reason,
            "devices": list(self.devices),
            "retried_from": self.retried_from,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "history": [dict(entry) for entry in self.history],
        }


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise StateError(f"expected a JSON object: {path}")
    return value


def _lock_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.lock"


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    """Serialize the critical section guarded by ``lock_path`` across
    processes (a runner, its watchdog, a retry launcher, ...).

    A single-process ``flock``/``msvcrt.locking`` is deliberately sufficient
    for the first factory release (epic §16: "local and single-process");
    the lock is held only for one read-validate-write, never across GPU or
    training work.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.02)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def state_path(run_directory: Union[str, Path]) -> Path:
    """The conventional state-record path within an MF-A3 run directory."""
    return Path(run_directory) / "state.json"


def heartbeat_path(run_directory: Union[str, Path]) -> Path:
    """The conventional heartbeat path within an MF-A3 run directory."""
    return Path(run_directory) / "heartbeat.json"


def load_state(path: Union[str, Path]) -> TrialState:
    """Load and validate a persisted trial state record."""
    target = Path(path)
    payload = _read_json(target)
    if payload.get("format") != STATE_FORMAT:
        raise StateError(f"unsupported trial state format {payload.get('format')!r} for {target}")
    if payload.get("state") not in ALL_STATES:
        raise StateError(f"unknown trial state {payload.get('state')!r} for {target}")
    return TrialState(
        format=payload["format"],
        run_id=payload["run_id"],
        state=payload["state"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        reason=payload.get("reason"),
        devices=tuple(payload.get("devices") or ()),
        retried_from=payload.get("retried_from"),
        heartbeat_timeout_seconds=payload.get("heartbeat_timeout_seconds"),
        history=tuple(dict(entry) for entry in payload.get("history") or ()),
    )


def create_state(
    path: Union[str, Path],
    run_id: str,
    *,
    devices: Sequence[str] = (),
    retried_from: Optional[str] = None,
    heartbeat_timeout_seconds: Optional[float] = None,
) -> TrialState:
    """Write a run's initial ``queued`` state record.

    Refuses to overwrite an existing record -- exactly one process may
    create a run's state, mirroring the run directory's own identity guard
    (MF-A3's ``experiment_directory``). Locked for the same reason every
    other write here is: two launchers racing to create the same path must
    not both believe they initialized it.
    """
    target = Path(path)
    with _locked(_lock_path_for(target)):
        if target.exists():
            raise FileExistsError(f"trial state already exists: {target}")
        timestamp = _now_iso()
        state = TrialState(
            format=STATE_FORMAT,
            run_id=run_id,
            state=STATE_QUEUED,
            created_at=timestamp,
            updated_at=timestamp,
            devices=tuple(devices),
            retried_from=retried_from,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            history=({"from": None, "to": STATE_QUEUED, "at": timestamp, "reason": "queued"},),
        )
        atomic_write_json(target, state.to_dict())
        return state


def _apply_transition(current: TrialState, to_state: str, *, reason: Optional[str]) -> TrialState:
    """The pure state-machine edge: validate, then build the new record.

    Never performs I/O itself, so callers that already hold the lock (the
    watchdog's :func:`recover_stale_worker`) can reuse it without acquiring
    the lock a second time -- ``flock`` is per open-file-description, so a
    second acquisition from the same process would block on itself exactly
    as it would against a different process.
    """
    if to_state not in ALL_STATES:
        raise StateError(f"unknown trial state {to_state!r}")
    allowed = _LEGAL_TRANSITIONS[current.state]
    if to_state not in allowed:
        raise IllegalTransitionError(current.run_id, current.state, to_state)
    timestamp = _now_iso()
    entry = {"from": current.state, "to": to_state, "at": timestamp, "reason": reason}
    return dataclasses.replace(
        current,
        state=to_state,
        updated_at=timestamp,
        reason=reason if reason is not None else current.reason,
        history=current.history + (entry,),
    )


def transition(path: Union[str, Path], to_state: str, *, reason: Optional[str] = None) -> TrialState:
    """Atomically validate and apply one state-machine edge.

    The read-validate-write critical section is serialized by a
    single-process file lock so two concurrent writers cannot both observe
    the same pre-transition state and both believe they won it -- exactly
    the "two writers race" risk epic #212's risk table names. Raises
    :class:`IllegalTransitionError` for any edge not in the declared graph,
    including every transition out of a terminal state.
    """
    target = Path(path)
    with _locked(_lock_path_for(target)):
        current = load_state(target)
        updated = _apply_transition(current, to_state, reason=reason)
        atomic_write_json(target, updated.to_dict())
        return updated


def write_heartbeat(
    path: Union[str, Path], *, run_id: str, clock: Callable[[], float] = time.time,
) -> Path:
    """Persist a liveness heartbeat atomically.

    The timestamp is carried in the payload rather than relied on from the
    file's own mtime, which is not trustworthy across every filesystem and
    clock-skew scenario a real worker can run under.
    """
    return atomic_write_json(
        Path(path), {"format": HEARTBEAT_FORMAT, "run_id": run_id, "heartbeat_seconds": clock()},
    )


def cancel_request_path(run_directory: Union[str, Path]) -> Path:
    """The conventional cancellation-flag path within an MF-A3 run directory."""
    return Path(run_directory) / "cancel_requested.json"


def request_cancellation(run_directory: Union[str, Path], *, reason: Optional[str] = None) -> Path:
    """Ask a running trial to stop at its next checkpoint boundary.

    A plain flag file, not a state-machine transition: the run stays
    ``running``/``checkpointing`` until its own training loop notices this
    request and transitions itself to ``cancelled`` -- the same "honored at
    the next checkpoint boundary" discipline :mod:`.budget` already uses for
    a graceful budget-exceeded stop, just triggered by this flag instead of
    a time/epoch projection. Idempotent: writing this twice (or after the
    run has already finished) is harmless, since nothing ever deletes it and
    a terminal run never reads it again.
    """
    return atomic_write_json(
        cancel_request_path(run_directory),
        {"format": CANCEL_REQUEST_FORMAT, "requested_at": _now_iso(), "reason": reason},
    )


def cancellation_requested(run_directory: Union[str, Path]) -> bool:
    """Whether :func:`request_cancellation` has been called for this run."""
    return cancel_request_path(run_directory).exists()


def _iso_to_epoch(value: str) -> float:
    return dt.datetime.fromisoformat(value).timestamp()


def _read_heartbeat(path: Path) -> Optional[Dict[str, Any]]:
    """Return the validated heartbeat payload, or ``None`` if never written."""
    if not path.exists():
        return None
    payload = _read_json(path)
    if payload.get("format") != HEARTBEAT_FORMAT:
        raise StateError(f"invalid trial heartbeat file: {path}")
    return payload


def heartbeat_age_seconds(
    path: Union[str, Path], *, clock: Callable[[], float] = time.time,
) -> float:
    """Seconds since the last heartbeat, or ``math.inf`` if none was ever
    written -- a worker that dies before its first heartbeat is exactly as
    stale as one that stopped heartbeating, never mistaken for fresh.

    Clamped at zero: a wall-clock rollback must never surface as a negative
    age, which :func:`~cognitive_runtime.training.model_factory.budget.watchdog_expired`
    rejects outright.
    """
    payload = _read_heartbeat(Path(path))
    if payload is None:
        return math.inf
    return max(clock() - float(payload["heartbeat_seconds"]), 0.0)


def _heartbeat_age_for_state(
    state: TrialState, heartbeat_file: Union[str, Path], *, clock: Callable[[], float],
) -> float:
    """Heartbeat age for ``state``, falling back to time-since-active when
    there is no heartbeat this run can be judged by.

    Two distinct cases share this fallback: the worker's first heartbeat
    hasn't landed yet (its declared grace period must still apply, not an
    instant ``math.inf`` false-positive), and ``heartbeat_file`` holds a
    leftover or misdirected record stamped with a *different* ``run_id`` --
    which must never vouch for this run's liveness just because its own
    timestamp looks fresh.
    """
    payload = _read_heartbeat(Path(heartbeat_file))
    if payload is not None and payload.get("run_id") == state.run_id:
        return max(clock() - float(payload["heartbeat_seconds"]), 0.0)
    return max(clock() - _iso_to_epoch(state.updated_at), 0.0)


def _resolve_timeout(state: TrialState, timeout_seconds: Optional[float]) -> float:
    """Prefer the timeout declared at :func:`create_state` time; an explicit
    caller-supplied value is only accepted when it agrees with that
    declaration, so a restarted watchdog can never silently enforce a
    different contract than the one persisted on the run."""
    declared = state.heartbeat_timeout_seconds
    if timeout_seconds is None:
        if declared is None:
            raise StateError(
                f"no heartbeat_timeout_seconds declared for run {state.run_id!r} "
                "and none supplied"
            )
        return declared
    if declared is not None and declared != timeout_seconds:
        raise StateError(
            f"timeout_seconds={timeout_seconds!r} conflicts with the "
            f"heartbeat_timeout_seconds={declared!r} declared at creation for "
            f"run {state.run_id!r}"
        )
    return timeout_seconds


def is_stale(
    state: TrialState,
    heartbeat_file: Union[str, Path],
    *,
    timeout_seconds: Optional[float] = None,
    clock: Callable[[], float] = time.time,
) -> bool:
    """True when ``state`` is active and its heartbeat gap exceeds its
    watchdog timeout. A non-active (not-yet-launched or already terminal)
    run is never stale -- there is no live worker to have gone quiet.

    ``timeout_seconds`` defaults to the value declared at
    :func:`create_state` time; see :func:`_resolve_timeout` for the
    override rule. See :func:`_heartbeat_age_for_state` for how a missing
    or mismatched-``run_id`` heartbeat is judged.
    """
    if state.state not in ACTIVE_STATES:
        return False
    resolved_timeout = _resolve_timeout(state, timeout_seconds)
    age = _heartbeat_age_for_state(state, heartbeat_file, clock=clock)
    return watchdog_expired(age, resolved_timeout)


def recover_stale_worker(
    path: Union[str, Path],
    heartbeat_file: Union[str, Path],
    *,
    timeout_seconds: Optional[float] = None,
    clock: Callable[[], float] = time.time,
) -> Optional[TrialState]:
    """Detect and, if stale, atomically fail a run whose worker stopped
    heartbeating -- the watchdog's one write path.

    Never marks a stale run ``completed``: the only legal destination here
    is ``failed``, recorded with a distinct machine-readable reason so a
    later operator can tell a watchdog kill apart from any other failure.
    Returns ``None`` (a no-op) when the heartbeat is still current, so a
    polling loop can call this unconditionally every tick. ``timeout_seconds``
    follows :func:`is_stale`'s declared-timeout rule.
    """
    target = Path(path)
    with _locked(_lock_path_for(target)):
        current = load_state(target)
        if not is_stale(current, heartbeat_file, timeout_seconds=timeout_seconds, clock=clock):
            return None
        updated = _apply_transition(current, STATE_FAILED, reason="stale_worker_watchdog_timeout")
        atomic_write_json(target, updated.to_dict())
        return updated


def reconcile_stale_runs(
    root: Union[str, Path], organism: str, *, clock: Callable[[], float] = time.time,
) -> Tuple[TrialState, ...]:
    """Sweep every run under ``root/organism`` and fail any whose worker has
    gone stale (:func:`recover_stale_worker`), returning the ones actually
    recovered.

    Nothing in this package calls :func:`recover_stale_worker` on its own --
    a caller (a control plane's job-status poll, a periodic maintenance
    task) must invoke this itself. Cheap and idempotent to call repeatedly:
    a run with no ``state.json`` yet (not allocated) or a fresh heartbeat is
    left untouched, exactly :func:`recover_stale_worker`'s own no-op rule,
    applied across every run directory found rather than one at a time.
    """
    organism_root = Path(root) / organism
    if not organism_root.is_dir():
        return ()
    recovered = []
    for run_directory in sorted(organism_root.iterdir()):
        candidate_state_path = state_path(run_directory)
        if not candidate_state_path.is_file():
            continue
        result = recover_stale_worker(candidate_state_path, heartbeat_path(run_directory), clock=clock)
        if result is not None:
            recovered.append(result)
    return tuple(recovered)


def reconcile_after_kill(
    path: Union[str, Path], *, to_state: str = STATE_FAILED, reason: str = "killed_by_operator",
) -> Optional[TrialState]:
    """Atomically transition a run whose worker process the caller has just
    killed into a terminal state.

    Unlike :func:`recover_stale_worker`, this never checks the heartbeat --
    the caller already knows there is no live worker (it just sent the kill
    signal), so there is nothing to wait out. Use ``to_state=STATE_FAILED``
    for an unplanned/hung-process kill, or ``to_state=STATE_CANCELLED`` when
    the kill enforces a prior :func:`request_cancellation` the worker did
    not reach a checkpoint boundary in time to honor itself. A no-op
    (returns ``None``) when the run is not in an active state, so a control
    plane can call this unconditionally after a kill without first
    re-reading state.json itself.
    """
    if to_state not in (STATE_FAILED, STATE_CANCELLED):
        raise StateError(
            f"reconcile_after_kill only targets {STATE_FAILED!r} or {STATE_CANCELLED!r}; got {to_state!r}"
        )
    target = Path(path)
    with _locked(_lock_path_for(target)):
        current = load_state(target)
        if current.state not in ACTIVE_STATES:
            return None
        updated = _apply_transition(current, to_state, reason=reason)
        atomic_write_json(target, updated.to_dict())
        return updated


def require_completed_for_promotion(state: TrialState) -> None:
    """The promotion path's gate: an incomplete artifact can never enter
    promotion.

    Callers on the promotion path must check the persisted trial state --
    not merely a ``training_stats["completion_status"]`` string, which a
    corrupted or partially-written report could misreport -- before
    considering a run's checkpoint eligible.
    """
    if state.state != STATE_COMPLETED:
        raise IncompleteRunError(
            f"run {state.run_id!r} is not eligible for promotion: trial state "
            f"is {state.state!r}, not {STATE_COMPLETED!r}"
        )


def retry_failed_run(
    failed_state_path: Union[str, Path],
    new_state_path: Union[str, Path],
    new_run_id: str,
    *,
    devices: Sequence[str] = (),
    heartbeat_timeout_seconds: Optional[float] = None,
) -> TrialState:
    """Start a fresh retry of a failed run.

    A failed worker is retried only as a *new* run naming the failed one in
    its lineage (epic #212 §16) -- never by reusing the failed run_id or
    reviving its state record in place. An interrupted-but-not-yet-failed
    run instead resumes in its own run directory under MF-B2's strict
    resume contract; this function is specifically the failed-run path.
    """
    failed = load_state(Path(failed_state_path))
    if failed.state != STATE_FAILED:
        raise StateError(
            f"cannot retry run {failed.run_id!r}: trial state is {failed.state!r}, "
            f"not {STATE_FAILED!r}"
        )
    if new_run_id == failed.run_id:
        raise StateError("a retry must use a new run_id, not the failed run's own id")
    return create_state(
        new_state_path,
        new_run_id,
        devices=devices,
        retried_from=failed.run_id,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )


def _read_ledger(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    if payload.get("format") != DEVICE_LEDGER_FORMAT:
        raise StateError(f"invalid device ledger file: {path}")
    return dict(payload.get("devices", {}))


def reserve_devices(root: Union[str, Path], run_id: str, devices: Sequence[str]) -> Dict[str, str]:
    """Atomically reserve every declared device before a runner launches.

    All-or-nothing: if any requested device is already held by a different
    run_id, no reservation is made. Idempotent for a run's own repeated
    reservation of the same devices, so a crash-and-relaunch of the same
    run_id is not treated as a conflict with itself.
    """
    if not devices:
        return {}
    ledger_path = Path(root) / "device_ledger.json"
    with _locked(_lock_path_for(ledger_path)):
        ledger = _read_ledger(ledger_path)
        conflicts = {
            device: holder for device, holder in ledger.items()
            if device in devices and holder != run_id
        }
        if conflicts:
            raise DeviceReservationError(
                f"cannot reserve devices for {run_id!r}: already reserved by {conflicts!r}"
            )
        for device in devices:
            ledger[device] = run_id
        atomic_write_json(ledger_path, {"format": DEVICE_LEDGER_FORMAT, "devices": ledger})
        return dict(ledger)


def release_devices(root: Union[str, Path], run_id: str) -> None:
    """Release every device currently reserved by ``run_id``.

    A no-op (not an error) when the run holds no reservation, so a terminal
    transition's cleanup can call this unconditionally.
    """
    ledger_path = Path(root) / "device_ledger.json"
    with _locked(_lock_path_for(ledger_path)):
        ledger = _read_ledger(ledger_path)
        remaining = {device: holder for device, holder in ledger.items() if holder != run_id}
        if remaining != ledger:
            atomic_write_json(ledger_path, {"format": DEVICE_LEDGER_FORMAT, "devices": remaining})


__all__ = [
    "STATE_FORMAT",
    "HEARTBEAT_FORMAT",
    "DEVICE_LEDGER_FORMAT",
    "CANCEL_REQUEST_FORMAT",
    "STATE_QUEUED",
    "STATE_RUNNING",
    "STATE_CHECKPOINTING",
    "STATE_COMPLETED",
    "STATE_BUDGET_EXCEEDED",
    "STATE_FAILED",
    "STATE_CANCELLED",
    "ALL_STATES",
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "StateError",
    "IllegalTransitionError",
    "IncompleteRunError",
    "DeviceReservationError",
    "TrialState",
    "state_path",
    "heartbeat_path",
    "cancel_request_path",
    "load_state",
    "create_state",
    "transition",
    "write_heartbeat",
    "request_cancellation",
    "cancellation_requested",
    "heartbeat_age_seconds",
    "is_stale",
    "recover_stale_worker",
    "reconcile_stale_runs",
    "reconcile_after_kill",
    "require_completed_for_promotion",
    "retry_failed_run",
    "reserve_devices",
    "release_devices",
]
