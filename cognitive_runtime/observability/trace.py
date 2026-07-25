"""Run tracing: a crash-safe, machine-readable record of one CCR run.

A *run* is one invocation of a pipeline -- ``ccr nursery joint``, ``ccr
nursery run walk_forward``, a notebook cell that calls
``run_nursery_joint`` -- and a *trace* is the append-only JSONL log of what
that run did, written line-by-line as it happens:

    <trace_dir>/<run_id>/
        manifest.json    run identity: config, git commit, package versions,
                         device, status, duration, metric summaries
        trace.jsonl      one event per line, flushed on write

The design constraint is that V2 crafter training runs are long (record
scenarios -> gate -> train N epochs -> evaluate -> probe) and the existing
``--report`` JSON only exists if the run reaches the end.  A trace is
written *during* the run, so a crash at epoch 25/30 still leaves the config,
the timings, the per-epoch loss curve up to the crash, and the traceback on
disk.

Three primitives, all no-ops when no run is active, so library code can call
them unconditionally without importing or threading a run object:

    from cognitive_runtime.observability import span, trace_event, trace_metrics

    with span("cortex.train", epochs=30):          # timed, nestable
        trace_event("dataset", frames=1024)        # a point in time
        trace_metrics(step=epoch, total_loss=0.31) # a scalar series

Every span and event is also emitted on the ``ccr.trace`` logger, so the
human-readable console narrative and the machine-readable trace come from
the same instrumentation -- see ``observability.logs``.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

TRACE_SCHEMA = "ccr-trace-v1"

#: Root directory for traces when neither ``--trace-dir`` nor
#: ``CCR_TRACE_DIR`` says otherwise.
DEFAULT_TRACE_DIR = "runs/traces"

#: Packages whose versions are worth pinning to a training result.
_TRACKED_PACKAGES = ("torch", "numpy", "crafter", "cognitive-runtime")

log = logging.getLogger("ccr.trace")

_ACTIVE_TRACE: contextvars.ContextVar[Optional["RunTrace"]] = contextvars.ContextVar(
    "ccr_active_trace", default=None
)
_SPAN_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "ccr_span_stack", default=()
)

#: Fallback for the notebook, which is the project's primary entrypoint:
#: ipykernel runs each cell in its own asyncio Task, so a context variable
#: set while executing one cell is not visible from the next.  A started run
#: is also recorded here so a trace opened in the setup cell still covers the
#: training cell.  The context variable wins where both are set, which keeps
#: nesting and threading correct.
_GLOBAL_TRACE: Optional["RunTrace"] = None


# --------------------------------------------------------------------------- environment capture


def _git_info() -> Dict[str, Any]:
    """Best-effort commit/branch/dirty state.  Never raises, never blocks
    long: a training result that can't be traced back to a commit is a
    result you can't reproduce."""

    def _run(*argv: str) -> Optional[str]:
        try:
            out = subprocess.run(
                argv, capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run("git", "rev-parse", "HEAD")
    if commit is None:
        return {}
    status = _run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _package_versions() -> Dict[str, str]:
    """Version of every tracked package, read from metadata so that asking
    doesn't *import* torch into a CLI run that never needed it."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return {}
    versions: Dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - absent package is normal
            continue
    return versions


def _device_info() -> Dict[str, Any]:
    """Torch device facts, but only if torch is *already* imported -- at run
    start it usually isn't (nothing to report), at run end it is (the
    manifest is rewritten then, so the facts land in the same file)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    info: Dict[str, Any] = {"torch_version": getattr(torch, "__version__", None)}
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = int(torch.cuda.device_count())
        info["default_dtype"] = str(torch.get_default_dtype())
        info["threads"] = int(torch.get_num_threads())
    except Exception:  # noqa: BLE001 - diagnostics must not break a run
        pass
    return info


def _environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "packages": _package_versions(),
        "git": _git_info(),
    }


# --------------------------------------------------------------------------- JSON coercion


def _jsonable(value: Any) -> Any:
    """Coerce anything an instrumentation call-site might hand us into JSON.

    Config dataclasses, tensors, numpy scalars and Path objects all show up
    in this codebase's call sites; a trace that raises on one of them is
    worse than a trace that records its repr.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar / 0-d torch tensor
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:  # noqa: BLE001
            pass
    return repr(value)


# --------------------------------------------------------------------------- metric summaries


@dataclasses.dataclass
class _MetricSummary:
    count: int = 0
    first: Optional[float] = None
    last: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    last_step: Optional[int] = None

    def observe(self, value: float, step: Optional[int]) -> None:
        self.count += 1
        if self.first is None:
            self.first = value
        self.last = value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)
        if step is not None:
            self.last_step = step


# --------------------------------------------------------------------------- the trace


class RunTrace:
    """One run's trace.  Usable as a context manager; safe to construct with
    ``enabled=False``, in which case it logs but writes nothing."""

    def __init__(
        self,
        name: str,
        *,
        trace_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.run_id = run_id or new_run_id(name)
        self.enabled = enabled
        self.root_dir = trace_dir if trace_dir is not None else default_trace_dir()
        self.dir = os.path.join(self.root_dir, self.run_id) if enabled else ""
        self.config: Dict[str, Any] = dict(_jsonable(config or {}))
        self.status = "pending"
        self.error: Optional[Dict[str, Any]] = None
        self.started_at: Optional[float] = None
        self.duration_s: Optional[float] = None
        self.metrics_summary: Dict[str, _MetricSummary] = {}
        self._counters: Dict[str, float] = {}
        self._handle = None
        self._lock = threading.Lock()
        self._span_seq = 0
        self._token = None
        self._stack_token = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def trace_path(self) -> str:
        return os.path.join(self.dir, "trace.jsonl") if self.enabled else ""

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.dir, "manifest.json") if self.enabled else ""

    def start(self) -> "RunTrace":
        global _GLOBAL_TRACE
        if self.started_at is not None:
            return self
        self.started_at = time.time()
        self.status = "running"
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
                # Truncating, not appending: a repeated explicit --run-id
                # overwrites the manifest anyway, so appending would leave one
                # manifest describing two concatenated runs -- `trace show`
                # would merge both runs' spans, metrics and warnings into a
                # single bogus picture.  One run id, one run.
                if os.path.exists(self.trace_path) and os.path.getsize(self.trace_path):
                    log.warning("run id %s already has a trace; overwriting %s",
                                self.run_id, self.trace_path)
                self._handle = open(self.trace_path, "w", encoding="utf-8")
                self._write_manifest()
                self._link_latest()
            except OSError as exc:
                # Tracing is diagnostics: an unwritable --trace-dir (or an
                # unwritable cwd) must not stop the training run the user
                # actually asked for.  Matches the fail-open behaviour of
                # every later write.
                self.enabled = False
                self._handle = None
                self.dir = ""
                log.warning("run tracing disabled: cannot write to %s (%s)",
                            os.path.join(self.root_dir, self.run_id), exc)
        self._token = _ACTIVE_TRACE.set(self)
        self._stack_token = _SPAN_STACK.set(())
        _GLOBAL_TRACE = self
        # Logged only once the run is the active one, so the JSON log format
        # can stamp it with the run id it is announcing.
        if self.enabled:
            log.info("run %s  trace=%s", self.run_id, self.trace_path)
        else:
            log.debug("run %s  (tracing disabled)", self.run_id)
        self._emit("run.start", self.name, data={"config": self.config})
        return self

    def finish(self, status: str = "ok", error: Optional[BaseException] = None) -> None:
        global _GLOBAL_TRACE
        if self.started_at is None or self.status in ("ok", "error", "failed", "interrupted"):
            return
        self.duration_s = round(time.time() - self.started_at, 3)
        self.status = status
        if error is not None:
            self.error = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self._emit(
            "run.end", self.name,
            data={"status": status, "duration_s": self.duration_s, "error": self.error},
        )
        if self.enabled:
            self._write_manifest()
            handle, self._handle = self._handle, None
            if handle is not None:
                handle.close()
            log.info(
                "run %s finished  status=%s  duration=%.1fs  trace=%s",
                self.run_id, status, self.duration_s, self.trace_path,
            )
        # A token can only be reset in the context that created it.  In a
        # notebook the run is started in one cell and finished in another --
        # ipykernel runs each in its own Task -- so the reset legitimately
        # fails there; clearing the module-level fallback below is what
        # actually ends the run.
        for variable, token in ((_ACTIVE_TRACE, self._token), (_SPAN_STACK, self._stack_token)):
            if token is not None:
                try:
                    variable.reset(token)
                except ValueError:
                    pass
        self._token = None
        self._stack_token = None
        if _GLOBAL_TRACE is self:
            _GLOBAL_TRACE = None

    def __enter__(self) -> "RunTrace":
        return self.start()

    def managed(self) -> "RunTrace":
        """Explicit notebook-friendly spelling of the trace context manager.

        Keeping this as a method makes the intended lifecycle obvious in a
        notebook cell: exceptions close the manifest before they are reraised.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.finish("ok")
        elif isinstance(exc, KeyboardInterrupt):
            self.finish("interrupted", exc)
        elif isinstance(exc, SystemExit) and exc.code in (None, 0):
            self.finish("ok")
        else:
            # Manifest consumers distinguish an expected interruption from a
            # real exception with the stable ``error`` terminal status.
            self.finish("error", exc)
        return False

    # -- emission ----------------------------------------------------------

    def _emit(self, kind: str, name: str, **fields: Any) -> None:
        record: Dict[str, Any] = {
            "t": round(time.time() - (self.started_at or time.time()), 4),
            "wall": round(time.time(), 3),
            "kind": kind,
            "name": name,
        }
        stack = _SPAN_STACK.get()
        if stack:
            record["span"] = stack[-1]
            if len(stack) > 1:
                record["parent"] = stack[-2]
        for key, value in fields.items():
            if value is not None and value != {}:
                record[key] = value
        if not self.enabled or self._handle is None:
            return
        line = json.dumps(record, default=repr)
        with self._lock:
            try:
                self._handle.write(line + "\n")
                self._handle.flush()  # crash safety beats throughput here
            except (OSError, ValueError):  # closed/full disk: never kill a run
                self.enabled = False

    def event(self, name: str, **fields: Any) -> None:
        """Record a point in time -- something happened, with context."""
        data = {k: _jsonable(v) for k, v in fields.items()}
        self._emit("event", name, data=data)
        log.debug("%s %s", name, _format_fields(data))

    def metrics(self, step: Optional[int] = None, **values: Any) -> None:
        """Record a batch of named scalars at one step (one trace line)."""
        numeric: Dict[str, float] = {}
        for key, value in values.items():
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                continue
        if not numeric:
            return
        for key, value in numeric.items():
            self.metrics_summary.setdefault(key, _MetricSummary()).observe(value, step)
        self._emit("metrics", "metrics", step=step, data=numeric)
        log.debug("metrics step=%s %s", step, _format_fields(numeric))

    def metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Record a single named scalar."""
        self.metrics(step=step, **{name: value})

    def counter(self, name: str, amount: float = 1.0) -> None:
        """Accumulate a count that lands in the manifest (episodes recorded,
        gate failures, NaN batches skipped...)."""
        self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator["_Span"]:
        """Time a phase of the run.  Nests; records duration and outcome even
        when the body raises."""
        with self._lock:
            self._span_seq += 1
            span_id = f"s{self._span_seq}"
        stack = _SPAN_STACK.get()
        token = _SPAN_STACK.set(stack + (span_id,))
        data = {k: _jsonable(v) for k, v in fields.items()}
        started = time.perf_counter()
        self._emit("span.start", name, data=data)
        log.debug("-> %s %s", name, _format_fields(data))
        handle = _Span(self, name)
        try:
            yield handle
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            duration = time.perf_counter() - started
            self._emit(
                "span.end", name,
                dur_s=round(duration, 4), ok=False,
                data={**data, **handle.result,
                      "error": {"type": type(exc).__name__, "message": str(exc)}},
            )
            log.warning("<- %s failed after %.2fs: %s: %s",
                        name, duration, type(exc).__name__, exc)
            raise
        else:
            duration = time.perf_counter() - started
            self._emit(
                "span.end", name,
                dur_s=round(duration, 4), ok=True,
                data={**data, **handle.result},
            )
            log.info("<- %s  %.2fs%s", name, duration,
                     (" " + _format_fields(handle.result)) if handle.result else "")
        finally:
            _SPAN_STACK.reset(token)

    # -- manifest ----------------------------------------------------------

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        manifest = {
            "schema": TRACE_SCHEMA,
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "started_at_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at))
                if self.started_at else None
            ),
            "duration_s": self.duration_s,
            "error": self.error,
            "config": self.config,
            "environment": _environment(),
            "device": _device_info(),
            "counters": dict(self._counters),
            "metrics": {
                key: dataclasses.asdict(summary)
                for key, summary in sorted(self.metrics_summary.items())
            },
        }
        tmp = self.manifest_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, default=repr)
            os.replace(tmp, self.manifest_path)
        except OSError:
            pass

    def _link_latest(self) -> None:
        """``runs/traces/latest`` always points at the newest run."""
        link = os.path.join(self.root_dir, "latest")
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.basename(self.dir), link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            try:  # Windows / no-symlink filesystems
                with open(os.path.join(self.root_dir, "latest.txt"), "w", encoding="utf-8") as fh:
                    fh.write(self.run_id + "\n")
            except OSError:
                pass


class _Span:
    """Handle yielded by :meth:`RunTrace.span`, for attaching results that
    are only known once the body has run (frames written, loss reached)."""

    def __init__(self, trace: "RunTrace", name: str) -> None:
        self._trace = trace
        self.name = name
        self.result: Dict[str, Any] = {}

    def set(self, **fields: Any) -> None:
        self.result.update({k: _jsonable(v) for k, v in fields.items()})


class _NullSpan(_Span):
    def __init__(self) -> None:  # noqa: D107 - see _Span
        self.name = ""
        self.result = {}

    def set(self, **fields: Any) -> None:
        return None


# --------------------------------------------------------------------------- module-level API


def new_run_id(name: str) -> str:
    """Sortable, unique, filesystem-safe: ``nursery-joint-20260724T1830-1a2b3c``."""
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-") or "run"
    return f"{slug}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def default_trace_dir() -> str:
    return os.environ.get("CCR_TRACE_DIR") or DEFAULT_TRACE_DIR


def tracing_disabled_by_env() -> bool:
    return os.environ.get("CCR_TRACE", "").strip().lower() in ("0", "off", "false", "no")


def current_trace() -> Optional[RunTrace]:
    """The run trace this code is executing inside, or ``None``.

    Only a *running* trace counts: a run finished from another context (the
    notebook case) can leave its context variable set in the context that
    started it, and emitting into a closed trace would be worse than not
    tracing at all.
    """
    for trace in (_ACTIVE_TRACE.get(), _GLOBAL_TRACE):
        if trace is not None and trace.status == "running":
            return trace
    return None


@contextmanager
def start_run(
    name: str,
    *,
    trace_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    enabled: bool = True,
) -> Iterator[RunTrace]:
    """Open a run trace for the duration of the block.

        with start_run("nursery.joint", config={"epochs": 30}):
            model, report = run_nursery_joint(...)
    """
    trace = RunTrace(
        name, trace_dir=trace_dir, run_id=run_id, config=config,
        enabled=enabled and not tracing_disabled_by_env(),
    )
    with trace:
        yield trace


@contextmanager
def span(name: str, **fields: Any) -> Iterator[_Span]:
    """Time a phase of whatever run is active; a no-op outside a run."""
    trace = current_trace()
    if trace is None:
        started = time.perf_counter()
        log.debug("-> %s %s", name, _format_fields(fields))
        try:
            yield _NullSpan()
        finally:
            log.debug("<- %s  %.2fs", name, time.perf_counter() - started)
        return
    with trace.span(name, **fields) as handle:
        yield handle


def trace_event(name: str, **fields: Any) -> None:
    """Record a point in time on the active run (no-op outside a run)."""
    trace = current_trace()
    if trace is None:
        log.debug("%s %s", name, _format_fields(fields))
        return
    trace.event(name, **fields)


def trace_metrics(step: Optional[int] = None, **values: Any) -> None:
    """Record named scalars at one step on the active run."""
    trace = current_trace()
    if trace is None:
        return
    trace.metrics(step=step, **values)


def trace_metric(name: str, value: Any, step: Optional[int] = None) -> None:
    """Record one named scalar on the active run."""
    trace_metrics(step=step, **{name: value})


def trace_counter(name: str, amount: float = 1.0) -> None:
    """Increment a run-level counter that lands in the manifest."""
    trace = current_trace()
    if trace is not None:
        trace.counter(name, amount)


def _format_fields(fields: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)
