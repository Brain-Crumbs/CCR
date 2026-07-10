"""Phase F live childhood run protocol (issue #33): every live
(``--backend remote``) run starts from a checkpoint or explicit ``--fresh``,
always records (frames included), a crashed bridge ends its episode
recoverably instead of killing the process, a real SIGINT leaves a valid
checkpoint that a rerun resumes from, and the ``review`` command compares a
run against baseline sessions on the same curriculum.  Driven entirely
through the Python fake bridge (``bridge/fake/sim_bridge.py``) -- no
Minecraft, no Node.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import main  # noqa: E402
from cognitive_runtime.neural.checkpoint import read_checkpoint_metadata  # noqa: E402
from cognitive_runtime.runtime.replay import iter_cognitive_ticks  # noqa: E402
from cognitive_runtime.tools.review import review_run  # noqa: E402

FAKE_BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "fake" / "sim_bridge.py"


def _use_fake_remote(monkeypatch, crash_after_steps=None):
    monkeypatch.setenv("CCR_MINECRAFT_BRIDGE_CMD", f"{sys.executable} {FAKE_BRIDGE}")
    if crash_after_steps is not None:
        monkeypatch.setenv("CCR_FAKE_BRIDGE_CRASH_AFTER_STEPS", str(crash_after_steps))


# ------------------------------------------------- checkpoint-or-fresh gate


def test_live_run_without_checkpoint_or_fresh_exits_with_actionable_message(
    tmp_path, monkeypatch
):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"
    with pytest.raises(SystemExit, match="--fresh"):
        main([
            "run", "--backend", "remote", "--policy", "actor-critic",
            "--episodes", "1", "--episode-ticks", "5", "--world-size", "16",
            "--actor-critic-model", str(checkpoint),
            "--record-dir", str(tmp_path), "--session-id", "live-no-checkpoint",
        ])
    assert not checkpoint.exists()


def test_live_run_with_fresh_flag_starts_a_new_checkpoint(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"
    main([
        "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
        "--episodes", "1", "--episode-ticks", "10", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "live-fresh",
    ])
    assert checkpoint.exists()


def test_live_run_with_existing_checkpoint_does_not_need_fresh(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"
    main([
        "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
        "--episodes", "1", "--episode-ticks", "10", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "live-seed",
    ])
    first_step = read_checkpoint_metadata(str(checkpoint))["online_optimizer"]["step"]

    # No --fresh this time: an existing checkpoint satisfies the gate.
    main([
        "run", "--backend", "remote", "--policy", "actor-critic",
        "--episodes", "1", "--episode-ticks", "10", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "live-resume",
    ])
    second_step = read_checkpoint_metadata(str(checkpoint))["online_optimizer"]["step"]
    assert second_step > first_step


# --------------------------------------------------- always record, with frames


def test_live_run_without_no_record_flag_rejects_no_record(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"
    with pytest.raises(SystemExit, match="recorded"):
        main([
            "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
            "--episodes", "1", "--episode-ticks", "5", "--world-size", "16",
            "--actor-critic-model", str(checkpoint), "--no-record",
        ])


def test_live_run_auto_records_frames_without_the_flag(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"
    main([
        "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
        "--episodes", "1", "--episode-ticks", "10", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "live-frames",
    ])
    session_dir = str(tmp_path / "live-frames")
    saw_frame = False
    for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        for record in sensory:
            if record.get("stream_id") == "vision.frame.pixels":
                assert not record.get("elided"), "frames must not be elided on a live run"
                saw_frame = True
    assert saw_frame


# --------------------------------------------------------- value estimate stream


def test_actor_critic_publishes_value_estimate_stream(tmp_path):
    checkpoint = tmp_path / "actor-critic.pt"
    main([
        "run", "--policy", "actor-critic",
        "--episodes", "1", "--episode-ticks", "15", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "value-estimate",
    ])
    session_dir = str(tmp_path / "value-estimate")
    saw_value_estimate = False
    for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        for record in sensory:
            if record.get("stream_id") == "model.value_estimate":
                assert isinstance(record["payload"]["value_estimate"], float)
                saw_value_estimate = True
    assert saw_value_estimate


# --------------------------------------------------- bridge crash: recoverable


def test_bridge_crash_mid_episode_ends_the_episode_not_the_process(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch, crash_after_steps=8)
    checkpoint = tmp_path / "actor-critic.pt"

    # Each episode's fake-bridge subprocess crashes after 8 steps -- both
    # episodes here end via the recoverable path, and the run still returns
    # normally instead of raising.
    main([
        "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
        "--episodes", "2", "--episode-ticks", "30", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--actor-critic-save-every", "100000",
        "--record-dir", str(tmp_path), "--session-id", "live-bridge-crash",
    ])

    session_dir = tmp_path / "live-bridge-crash"
    for episode_id in ("episode_00000", "episode_00001"):
        with open(session_dir / f"{episode_id}.summary.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        assert summary["termination_reason"] == "bridge_error"
        assert summary["success"] is False
        assert summary["duration_ticks"] < 30  # cut short by the crash

    # The run's own shutdown checkpoint (loop.py's `finally`) is the final
    # metadata write, but each episode's own bridge-crash checkpoint already
    # ran and is visible in that episode's summary (asserted above); the
    # checkpoint itself must still be valid and have trained.
    metadata = read_checkpoint_metadata(str(checkpoint))
    assert metadata["online_optimizer"]["step"] > 0
    assert checkpoint.exists()


# ------------------------------------------------------------- real SIGINT


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery differs on Windows")
def test_real_sigint_during_live_run_leaves_a_valid_checkpoint_that_resumes(
    tmp_path, monkeypatch
):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "actor-critic.pt"

    def _send_sigint_soon():
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(0.3, _send_sigint_soon)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            main([
                "run", "--backend", "remote", "--policy", "actor-critic", "--fresh",
                "--episodes", "1", "--episode-ticks", "100000", "--world-size", "16",
                "--actor-critic-model", str(checkpoint),
                "--record-dir", str(tmp_path), "--session-id", "live-sigint",
            ])
    finally:
        timer.cancel()

    assert checkpoint.exists()
    metadata = read_checkpoint_metadata(str(checkpoint))
    assert metadata["training_stats"]["last_checkpoint_reason"] == "keyboard_interrupt"
    interrupted_step = metadata["online_optimizer"]["step"]
    assert interrupted_step > 0

    # Rerun resumes both the tick counter and the trained weights.
    main([
        "run", "--backend", "remote", "--policy", "actor-critic",
        "--episodes", "1", "--episode-ticks", "10", "--world-size", "16",
        "--actor-critic-model", str(checkpoint),
        "--record-dir", str(tmp_path), "--session-id", "live-sigint-resume",
    ])
    resumed_step = read_checkpoint_metadata(str(checkpoint))["online_optimizer"]["step"]
    assert resumed_step > interrupted_step


# ------------------------------------------------------------ review command


def test_review_reports_run_summary_and_baseline_comparison(tmp_path):
    record_dir = tmp_path / "sessions"
    common = [
        "--curriculum", "flat-safe", "--episodes", "1", "--episode-ticks", "20",
        "--world-size", "16", "--record-dir", str(record_dir),
    ]
    main(["run", "--policy", "random", "--session-id", "baseline-random", *common])
    main(["run", "--policy", "scripted", "--session-id", "candidate-run", *common])

    report = review_run(str(record_dir / "candidate-run"), record_dir=str(record_dir))
    assert "candidate-run" in report
    assert "baseline:random" in report
    assert "episode_00000" in report
