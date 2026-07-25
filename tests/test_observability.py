"""Tracing + logging (``cognitive_runtime.observability``).

The properties that matter for a V2 crafter training run: a trace exists on
disk *while* the run is happening (not only after it succeeds), it records
what produced the numbers (config, git commit, package versions), the
instrumentation primitives are safe no-ops outside a run, and nothing in
here can take a training run down with it.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from cognitive_runtime.observability import (
    RunTrace,
    configure_logging,
    current_trace,
    format_run_list,
    format_trace_summary,
    list_runs,
    load_trace,
    span,
    start_run,
    trace_counter,
    trace_event,
    trace_metric,
    trace_metrics,
)
from cognitive_runtime.observability.read import span_tree


def _events(run_dir: str):
    with open(os.path.join(run_dir, "trace.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest(run_dir: str):
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_run_writes_manifest_and_events(tmp_path):
    with start_run("nursery.joint", trace_dir=str(tmp_path), config={"epochs": 3}) as trace:
        run_dir = trace.dir
        trace_event("dataset", transitions=128)
        trace_metrics(step=1, total_loss=0.5)

    manifest = _manifest(run_dir)
    assert manifest["status"] == "ok"
    assert manifest["name"] == "nursery.joint"
    assert manifest["config"] == {"epochs": 3}
    assert manifest["duration_s"] is not None
    assert manifest["environment"]["python"]
    # metric summaries are what `ccr trace show` renders
    assert manifest["metrics"]["total_loss"]["last"] == 0.5

    kinds = [event["kind"] for event in _events(run_dir)]
    assert kinds[0] == "run.start"
    assert kinds[-1] == "run.end"
    assert "event" in kinds and "metrics" in kinds


def test_events_are_flushed_during_the_run(tmp_path):
    """A trace's whole point is being readable before the run ends."""
    with start_run("mid-run", trace_dir=str(tmp_path)) as trace:
        trace_event("recording", seed=0)
        events = _events(trace.dir)
        assert [event["name"] for event in events] == ["mid-run", "recording"]
        assert os.path.exists(trace.manifest_path)
        assert _manifest(trace.dir)["status"] == "running"


def test_spans_nest_and_carry_durations(tmp_path):
    with start_run("spans", trace_dir=str(tmp_path)) as trace:
        with span("outer", scenario="walk_forward"):
            with span("inner", seed=3) as inner:
                inner.set(frames=42)

    rows = span_tree(_events(trace.dir))
    assert [(row["name"], row["depth"]) for row in rows] == [("outer", 0), ("inner", 1)]
    assert all(row["ok"] and row["dur_s"] is not None for row in rows)
    assert rows[1]["data"]["frames"] == 42
    assert rows[1]["data"]["seed"] == 3


def test_failing_span_and_run_record_the_error_and_reraise(tmp_path):
    with pytest.raises(ValueError, match="quality gate"):
        with start_run("failing", trace_dir=str(tmp_path)) as trace:
            run_dir = trace.dir
            with span("nursery.quality_gate"):
                raise ValueError("quality gate failed")

    manifest = _manifest(run_dir)
    assert manifest["status"] == "error"
    assert manifest["error"]["type"] == "ValueError"
    ended = [e for e in _events(run_dir) if e["kind"] == "span.end"]
    assert ended[0]["ok"] is False
    assert ended[0]["data"]["error"]["message"] == "quality gate failed"


def test_keyboard_interrupt_is_recorded_as_interrupted(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with start_run("interrupted", trace_dir=str(tmp_path)) as trace:
            run_dir = trace.dir
            raise KeyboardInterrupt
    assert _manifest(run_dir)["status"] == "interrupted"


def test_metric_series_summarized_across_steps(tmp_path):
    with start_run("curve", trace_dir=str(tmp_path)) as trace:
        for step, loss in enumerate([0.9, 0.4, 0.6], start=1):
            trace_metric("train/total_loss", loss, step=step)
        trace_counter("episodes_recorded", 4)
        trace_counter("episodes_recorded")

    manifest = _manifest(trace.dir)
    summary = manifest["metrics"]["train/total_loss"]
    assert (summary["count"], summary["first"], summary["last"]) == (3, 0.9, 0.6)
    assert (summary["min"], summary["max"], summary["last_step"]) == (0.4, 0.9, 3)
    assert manifest["counters"]["episodes_recorded"] == 5


def test_non_numeric_metrics_are_dropped_not_raised(tmp_path):
    with start_run("mixed", trace_dir=str(tmp_path)) as trace:
        trace_metrics(step=1, good=1.5, bad="not a number", also_bad=None)
    assert set(_manifest(trace.dir)["metrics"]) == {"good"}


def test_instrumentation_is_a_no_op_outside_a_run(tmp_path):
    """Library code calls these unconditionally; with no run active they must
    do nothing at all rather than raise or create files."""
    assert current_trace() is None
    trace_event("orphan", value=1)
    trace_metric("orphan", 1.0)
    trace_counter("orphan")
    with span("orphan") as handle:
        handle.set(anything=True)
    assert os.listdir(tmp_path) == []


def test_disabled_trace_writes_nothing(tmp_path):
    with start_run("off", trace_dir=str(tmp_path), enabled=False) as trace:
        trace_event("still callable", n=1)
        with span("still callable"):
            trace_metric("loss", 0.1)
        assert trace.enabled is False
    assert os.listdir(tmp_path) == []


def test_env_var_disables_tracing(tmp_path, monkeypatch):
    monkeypatch.setenv("CCR_TRACE", "0")
    with start_run("env-off", trace_dir=str(tmp_path)) as trace:
        assert trace.enabled is False
    assert os.listdir(tmp_path) == []


def test_dataclass_and_tensor_like_fields_are_serialized(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        epochs: int = 3
        horizons: tuple = (1, 10)

    class FakeTensor:
        def tolist(self):
            return [1.0, 2.0]

    with start_run("coerce", trace_dir=str(tmp_path), config={"cfg": Cfg()}) as trace:
        trace_event("odd", tensor=FakeTensor(), path=os.sep, nested={"a": Cfg()})

    manifest = _manifest(trace.dir)
    assert manifest["config"]["cfg"] == {"epochs": 3, "horizons": [1, 10]}
    event = [e for e in _events(trace.dir) if e["name"] == "odd"][0]
    assert event["data"]["tensor"] == [1.0, 2.0]
    assert event["data"]["nested"]["a"]["epochs"] == 3


def test_unserializable_field_does_not_break_the_run(tmp_path):
    class Exploding:
        def __repr__(self):
            return "<exploding>"

    with start_run("hostile", trace_dir=str(tmp_path)) as trace:
        trace_event("weird", value=Exploding())
    assert [e for e in _events(trace.dir) if e["name"] == "weird"]


def test_latest_pointer_resolves_to_the_newest_run(tmp_path):
    with start_run("first", trace_dir=str(tmp_path)):
        pass
    with start_run("second", trace_dir=str(tmp_path)) as second:
        pass
    manifest, events = load_trace(trace_dir=str(tmp_path))
    assert manifest["run_id"] == second.run_id
    assert events

    runs = list_runs(str(tmp_path))
    assert [run["name"] for run in runs] == ["first", "second"]
    assert second.run_id in format_run_list(runs)


def test_load_trace_accepts_run_id_and_paths(tmp_path):
    with start_run("addressable", trace_dir=str(tmp_path)) as trace:
        trace_event("hello")
    by_id, _ = load_trace(trace.run_id, trace_dir=str(tmp_path))
    by_dir, _ = load_trace(trace.dir)
    by_file, _ = load_trace(trace.trace_path)
    assert by_id["run_id"] == by_dir["run_id"] == by_file["run_id"] == trace.run_id

    with pytest.raises(FileNotFoundError):
        load_trace("no-such-run", trace_dir=str(tmp_path))


def test_truncated_trace_still_reads(tmp_path):
    """A killed run leaves a half-written last line; that trace is exactly
    the one worth reading."""
    with start_run("killed", trace_dir=str(tmp_path)) as trace:
        trace_event("epoch", n=1)
        run_dir = trace.dir
    with open(os.path.join(run_dir, "trace.jsonl"), "a", encoding="utf-8") as handle:
        handle.write('{"kind": "metrics", "na')
    _manifest_, events = load_trace(run_dir)
    assert [e["name"] for e in events if e["kind"] == "event"] == ["epoch"]


def test_killed_run_keeps_its_open_phase_and_loss_curve(tmp_path):
    """SIGKILL/OOM: no run.end, no final manifest write. The trace must still
    show which phase was running and how far the loss got -- the whole reason
    events are flushed instead of buffered."""
    trace = RunTrace("killed", trace_dir=str(tmp_path)).start()
    phase = trace.span("nursery.train", epochs=30)
    phase.__enter__()
    for epoch in (1, 2, 3):
        trace_metrics(step=epoch, **{"train/total_loss": 1.0 / epoch})
    # ...and here the process dies: no span.end, no run.end, no manifest
    # rewrite -- exactly what `os._exit`/SIGKILL leaves behind.

    manifest, events = load_trace(trace.dir)
    assert manifest["status"] == "running"
    assert manifest["metrics"] == {}

    rows = span_tree(events)
    assert rows[0]["open"] is True and rows[0]["dur_s"] is None
    assert rows[0]["name"] == "nursery.train"  # named from span.start
    assert rows[0]["data"]["epochs"] == 30

    text = format_trace_summary(manifest, events)
    assert "killed (no run.end)" in text
    assert "nursery.train" in text
    # the curve is recovered from the flushed events, not the manifest
    assert "train/total_loss" in text
    assert "0.33333" in text


def test_summary_renders_identity_phases_and_metrics(tmp_path):
    with start_run("nursery.joint", trace_dir=str(tmp_path),
                   config={"epochs": 2, "backbone": "gru"}) as trace:
        with span("nursery.record", scenarios=2):
            with span("nursery.record.episode", seed=0):
                pass
        trace_metrics(step=1, **{"train/total_loss": 0.25})

    text = format_trace_summary(*load_trace(trace.dir))
    assert "nursery.joint" in text
    assert "nursery.record.episode" in text
    assert "train/total_loss" in text
    assert "backbone" in text
    assert "status    ok" in text


# --------------------------------------------------------------------------- logging


def test_configure_logging_makes_training_logs_visible(capsys):
    """The bug this module exists to fix: nothing ever configured logging, so
    every ``log.info`` in the training modules went nowhere."""
    logger = logging.getLogger("ccr.training.nursery")
    try:
        configure_logging("info", stream=None)
        logger.info("recording nursery-walk_forward-train-0")
        captured = capsys.readouterr()
        assert "recording nursery-walk_forward-train-0" in captured.err + captured.out
        assert "nursery" in captured.err + captured.out
    finally:
        configure_logging("warning")


def test_configure_logging_is_idempotent():
    logger = logging.getLogger("ccr")
    try:
        for _ in range(3):
            configure_logging("debug")
        # one console handler + one trace forwarder, however many times called
        # (pytest's own capture handlers sit alongside them)
        installed = [h for h in logger.handlers if getattr(h, "_ccr_handler", False)]
        assert len(installed) == 2
    finally:
        configure_logging("warning")


def test_log_level_filters_and_log_file_keeps_full_detail(tmp_path, capsys):
    log_file = tmp_path / "nested" / "run.log"
    logger = logging.getLogger("ccr.training.cortex")
    try:
        configure_logging("warning", log_file=str(log_file))
        logger.debug("epoch 1/30 total=0.4")
        logger.warning("frozen rollout detected")
        console = capsys.readouterr()
        console_text = console.err + console.out
        assert "epoch 1/30" not in console_text
        assert "frozen rollout detected" in console_text
        # the file records at DEBUG regardless of the console level
        contents = log_file.read_text(encoding="utf-8")
        assert "epoch 1/30" in contents and "frozen rollout detected" in contents
    finally:
        configure_logging("warning")


def test_json_log_format_is_one_object_per_line(capsys):
    logger = logging.getLogger("ccr.training.nursery")
    try:
        configure_logging("info", log_format="json")
        logger.info("quality gate passed")
        captured = capsys.readouterr()
        payload = json.loads((captured.err + captured.out).strip().splitlines()[-1])
        assert payload["message"] == "quality gate passed"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "ccr.training.nursery"
    finally:
        configure_logging("warning")


def test_warnings_are_forwarded_into_the_active_trace(tmp_path):
    logger = logging.getLogger("ccr.training.cortex")
    try:
        configure_logging("error")  # console silent; the trace still gets it
        with start_run("warned", trace_dir=str(tmp_path)) as trace:
            logger.info("this is not important enough to trace")
            logger.warning("frozen rollout detected")
        forwarded = [
            event for event in _events(trace.dir)
            if event["kind"] == "event" and event["name"] == "log"
        ]
        assert [event["data"]["message"] for event in forwarded] == ["frozen rollout detected"]
        assert "frozen rollout detected" in format_trace_summary(*load_trace(trace.dir))
    finally:
        configure_logging("warning")


# --------------------------------------------------------------------------- CLI wiring


def test_observability_flags_work_in_either_position():
    from cognitive_runtime.cli import build_parser

    parser = build_parser()
    before = parser.parse_args(["--log-level", "debug", "nursery", "joint"])
    after = parser.parse_args(["nursery", "joint", "--log-level", "debug"])
    assert before.log_level == after.log_level == "debug"
    # an unset subparser copy must not clobber a value given before the
    # subcommand (the argparse default-override trap)
    mixed = parser.parse_args(["--trace-dir", "/tmp/x", "nursery", "joint"])
    assert mixed.trace_dir == "/tmp/x"
    assert not hasattr(parser.parse_args(["dashboard"]), "log_level")


def test_run_name_follows_the_subcommand_path():
    from cognitive_runtime.cli import _run_name, build_parser

    parser = build_parser()
    assert _run_name(parser.parse_args(["nursery", "joint"])) == "nursery.joint"
    assert _run_name(
        parser.parse_args(["nursery", "run", "walk_forward"])
    ) == "nursery.run.walk_forward"


def test_read_only_commands_are_not_traced():
    """Otherwise a few `ccr dashboard` calls bury the training runs."""
    from cognitive_runtime.cli import _run_name, _should_trace, build_parser

    parser = build_parser()

    def traced(argv):
        args = parser.parse_args(argv)
        return _should_trace(args, _run_name(args))

    assert traced(["nursery", "joint"])
    assert traced(["nursery", "run", "walk_forward"])
    assert not traced(["nursery", "list"])
    assert not traced(["dashboard"])
    assert not traced(["trace", "show"])
    assert not traced(["nursery", "joint", "--no-trace"])


def test_run_can_be_finished_from_another_context(tmp_path):
    """The notebook starts a run in one cell and finishes it in another;
    ipykernel runs each cell in its own Task, so the context token cannot be
    reset there and the instrumentation must fall back cleanly."""
    import contextvars

    trace = RunTrace("notebook.Pixel", trace_dir=str(tmp_path)).start()

    def later_cell():
        assert current_trace() is trace  # visible across contexts
        trace_event("training", epochs=3)
        trace.finish("ok")

    contextvars.copy_context().run(later_cell)
    assert current_trace() is None
    assert _manifest(trace.dir)["status"] == "ok"
    assert [e["name"] for e in _events(trace.dir) if e["kind"] == "event"] == ["training"]


def test_trace_survives_a_closed_output_file(tmp_path):
    """Disk problems must degrade tracing, never kill the training run."""
    trace = RunTrace("fragile", trace_dir=str(tmp_path)).start()
    trace._handle.close()  # simulate a full disk / closed descriptor
    trace.event("after", value=1)
    trace.finish("ok")
    assert trace.enabled is False
