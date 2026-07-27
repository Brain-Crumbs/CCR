"""Read back what a run did: ``ccr trace list`` / ``ccr trace show``.

A trace is only worth writing if it's cheap to read.  These helpers load a
run directory (or a bare ``trace.jsonl``) and render it as a phase tree with
durations, a metric table, and the run's identity -- the post-mortem you
want when a training run took 40 minutes and produced a number you don't
trust.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from cognitive_runtime.observability.trace import default_trace_dir


def resolve_run_dir(target: Optional[str] = None, trace_dir: Optional[str] = None) -> str:
    """Accept a run directory, a ``trace.jsonl`` path, a bare run id, or
    nothing (meaning "the latest run")."""
    root = trace_dir or default_trace_dir()
    if target:
        if os.path.isdir(target):
            return target
        if os.path.isfile(target):
            return os.path.dirname(os.path.abspath(target))
        candidate = os.path.join(root, target)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"no trace found for {target!r} (looked in {root})")
    latest = os.path.join(root, "latest")
    if os.path.isdir(latest):
        return latest
    pointer = os.path.join(root, "latest.txt")
    if os.path.isfile(pointer):
        with open(pointer, encoding="utf-8") as handle:
            return os.path.join(root, handle.read().strip())
    runs = list_runs(root)
    if not runs:
        raise FileNotFoundError(f"no traces under {root}")
    return runs[-1]["_dir"]


def load_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def iter_events(run_dir: str) -> Iterator[Dict[str, Any]]:
    """Stream a trace's events, tolerating a truncated final line -- a trace
    from a killed run is exactly the one you most need to read."""
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_trace(target: Optional[str] = None, trace_dir: Optional[str] = None
               ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run_dir = resolve_run_dir(target, trace_dir)
    return load_manifest(run_dir), list(iter_events(run_dir))


def list_runs(trace_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every run under ``trace_dir``, oldest first, each manifest annotated
    with its directory."""
    root = trace_dir or default_trace_dir()
    if not os.path.isdir(root):
        return []
    runs: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        if entry == "latest" or not os.path.isdir(run_dir) or os.path.islink(run_dir):
            continue
        manifest = load_manifest(run_dir)
        if not manifest:
            continue
        manifest["_dir"] = run_dir
        runs.append(manifest)
    runs.sort(key=lambda m: m.get("started_at") or 0.0)
    return runs


# --------------------------------------------------------------------------- rendering


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m"


def span_tree(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spans in start order, each with its nesting ``depth``.

    A span with no matching ``span.end`` is reported as still open rather
    than dropped -- in a killed run, the open span is precisely the phase
    that was running when the process died.
    """
    parents: Dict[str, Optional[str]] = {}
    order: List[str] = []
    started: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event.get("kind") == "span.start" and event.get("span"):
            parents[event["span"]] = event.get("parent")
            started[event["span"]] = event
            order.append(event["span"])
    ended = {
        event["span"]: event
        for event in events
        if event.get("kind") == "span.end" and event.get("span")
    }
    rows: List[Dict[str, Any]] = []
    for span_id in order:
        depth = 0
        cursor = parents.get(span_id)
        while cursor is not None and depth < 32:
            depth += 1
            cursor = parents.get(cursor)
        begin = started.get(span_id, {})
        end = ended.get(span_id)
        rows.append({
            "span": span_id,
            "depth": depth,
            "name": (end or begin).get("name", "?"),
            "dur_s": (end or {}).get("dur_s"),
            "ok": None if end is None else end.get("ok", True),
            "data": (end or begin).get("data", {}),
            "open": end is None,
        })
    return rows


def metric_summaries(events: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Rebuild per-metric ``count/first/last/min/max`` from the event stream.

    The manifest carries the same summary, but only from the run's *last*
    write -- a run killed mid-training never gets one.  The events do, so a
    crashed run's loss curve is still summarizable.
    """
    summaries: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "metrics":
            continue
        step = event.get("step")
        for key, value in (event.get("data") or {}).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            entry = summaries.setdefault(
                key,
                {"count": 0, "first": number, "last": number,
                 "min": number, "max": number, "last_step": None},
            )
            entry["count"] += 1
            entry["last"] = number
            entry["min"] = min(entry["min"], number)
            entry["max"] = max(entry["max"], number)
            if step is not None:
                entry["last_step"] = step
    return summaries


def _format_span_data(data: Dict[str, Any], limit: int = 4) -> str:
    parts = []
    for key, value in list(data.items())[:limit]:
        if key == "error":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        elif isinstance(value, (dict, list)):
            parts.append(f"{key}=<{type(value).__name__}>")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_trace_summary(
    manifest: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    *,
    tail: int = 0,
    show_metrics: bool = True,
) -> str:
    lines: List[str] = []
    status = manifest.get("status", "?")
    if status == "running":
        # No run.end line: either it is still going or the process was killed
        # (OOM, SIGKILL, a lost session) without unwinding.
        status = "running / killed (no run.end)"
    lines.append(f"run       {manifest.get('run_id', '?')}  ({manifest.get('name', '?')})")
    lines.append(f"status    {status}   duration {_duration(manifest.get('duration_s'))}"
                 f"   started {manifest.get('started_at_iso', '?')}")
    env = manifest.get("environment", {})
    git = env.get("git", {})
    if git:
        dirty = "  [dirty worktree]" if git.get("dirty") else ""
        lines.append(f"commit    {str(git.get('commit', '?'))[:12]} "
                     f"({git.get('branch', '?')}){dirty}")
    packages = env.get("packages", {})
    if packages:
        lines.append("packages  " + "  ".join(f"{k}={v}" for k, v in sorted(packages.items())))
    device = manifest.get("device", {})
    if device:
        lines.append("device    " + "  ".join(f"{k}={v}" for k, v in sorted(device.items())))
    if manifest.get("error"):
        error = manifest["error"]
        lines.append(f"error     {error.get('type')}: {error.get('message')}")

    config = manifest.get("config") or {}
    if config:
        lines.append("")
        lines.append("config")
        for key, value in sorted(config.items()):
            rendered = json.dumps(value, default=repr) if isinstance(value, (dict, list)) else value
            text = str(rendered)
            lines.append(f"  {key:<24} {text[:96]}")

    rows = span_tree(events)
    if rows:
        lines.append("")
        lines.append("phases")
        for row in rows:
            marker = "  " if row["ok"] else ("!!" if row["ok"] is False else "..")
            indent = "  " * row["depth"]
            detail = _format_span_data(row.get("data") or {})
            lines.append(
                f"{marker} {_duration(row['dur_s']):>9}  {indent}{row['name']}"
                + (f"   {detail}" if detail else "")
            )

    counters = manifest.get("counters") or {}
    if counters:
        lines.append("")
        lines.append("counters")
        for key, value in sorted(counters.items()):
            lines.append(f"  {key:<32} {value:g}")

    # Prefer the event stream over the manifest's copy: they agree for a
    # finished run, and only the events exist for a killed one.
    metrics = metric_summaries(events) or (manifest.get("metrics") or {})
    if show_metrics and metrics:
        lines.append("")
        lines.append(f"{'metric':<32} {'n':>5} {'first':>12} {'last':>12} "
                     f"{'min':>12} {'max':>12}")
        for key, summary in sorted(metrics.items()):
            lines.append(
                f"{key:<32} {summary.get('count', 0):>5} "
                f"{_num(summary.get('first')):>12} {_num(summary.get('last')):>12} "
                f"{_num(summary.get('min')):>12} {_num(summary.get('max')):>12}"
            )

    warnings = [e for e in events if e.get("kind") == "event" and e.get("name") == "log"]
    if warnings:
        lines.append("")
        lines.append(f"warnings/errors ({len(warnings)})")
        for event in warnings[-10:]:
            data = event.get("data", {})
            lines.append(f"  [{data.get('level')}] {data.get('logger')}: {data.get('message')}")

    if tail:
        lines.append("")
        lines.append(f"last {tail} events")
        for event in list(events)[-tail:]:
            lines.append(f"  +{event.get('t', 0):>8.2f}s {event.get('kind'):<10} "
                         f"{event.get('name')}  {json.dumps(event.get('data', {}), default=repr)[:100]}")
    return "\n".join(lines)


def _num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def format_run_list(runs: Sequence[Dict[str, Any]]) -> str:
    if not runs:
        return "no traces found"
    lines = [f"{'run_id':<44} {'status':<12} {'duration':>10}  started"]
    for manifest in runs:
        lines.append(
            f"{manifest.get('run_id', '?'):<44} {manifest.get('status', '?'):<12} "
            f"{_duration(manifest.get('duration_s')):>10}  "
            f"{manifest.get('started_at_iso', '?')}"
        )
    return "\n".join(lines)
