"""Byte-exact record->replay->compare smoke test (issue #90), extended from
Minecraft to Crafter now that Crafter is deterministic and snapshot-able
(``docs/v2/phases/phase-1-nursery-world.md``). A cheap plumbing check
(publish-order/recorder regressions), not a learning gate.
"""

from __future__ import annotations

import json
import os

import pytest

crafter = pytest.importorskip("crafter")

from cognitive_runtime.policies import RandomPolicy  # noqa: E402
from cognitive_runtime.programs.crafter.actions import ACTION_SPACE as CRAFTER_ACTION_SPACE  # noqa: E402
from cognitive_runtime.programs.crafter.adapter import CrafterWorld  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.tools.replay_runner import replay_session  # noqa: E402

FAST_CONFIG = {"episode_ticks": 60}


def _record_crafter_session(tmp_path, policy, session_id, seed=5):
    config = RuntimeConfig(
        episodes=1, seed=seed, max_ticks_per_episode=60,
        record_dir=str(tmp_path), session_id=session_id, program_config=FAST_CONFIG,
    )
    CognitiveRuntime(
        program=CrafterWorld(config=FAST_CONFIG), policy=policy, config=config
    ).run()
    return os.path.join(str(tmp_path), session_id)


def test_crafter_episode_replays_byte_identically(tmp_path):
    session_dir = _record_crafter_session(
        tmp_path, RandomPolicy(CRAFTER_ACTION_SPACE, seed=3), "crafter-replay"
    )
    results = replay_session(session_dir)
    assert len(results) == 1
    assert results[0].matched, results[0]
    assert results[0].ticks_replayed == 60


def test_crafter_replay_detects_sensory_tampering(tmp_path):
    session_dir = _record_crafter_session(
        tmp_path, RandomPolicy(CRAFTER_ACTION_SPACE, seed=4), "crafter-tamper"
    )
    path = os.path.join(session_dir, "episode_00000.streams.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        record = json.loads(line)
        if record.get("dir") == "sensory" and record.get("stream_id") == "body.health":
            record["payload"] = float(record["payload"]) + 5.0
            lines[i] = json.dumps(record)
            break
    else:
        pytest.fail("no body.health sensory record found to tamper with")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    results = replay_session(session_dir)
    assert not results[0].matched
    assert results[0].first_divergence_stream == "body.health"


def test_crafter_replay_detects_motor_tampering():
    """A flipped **motor** payload makes the world step differently, so the
    regenerated sensory hashes diverge downstream (the other tamper mode
    ``runtime.replay`` documents catching)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_path:
        session_dir = _record_crafter_session(
            tmp_path, RandomPolicy(CRAFTER_ACTION_SPACE, seed=6), "crafter-motor-tamper"
        )
        path = os.path.join(session_dir, "episode_00000.streams.jsonl")
        lines = open(path, encoding="utf-8").read().splitlines()
        flipped = False
        for i, line in enumerate(lines):
            record = json.loads(line)
            if record.get("dir") == "motor" and record.get("stream_id") == "motor.command":
                payload = record.get("payload") or {}
                original = payload.get("action")
                replacement = next(
                    a.name for a in CRAFTER_ACTION_SPACE if a.name != original
                )
                payload["action"] = replacement
                record["payload"] = payload
                lines[i] = json.dumps(record)
                flipped = True
                break
        assert flipped, "no motor.command record found to tamper with"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        results = replay_session(session_dir)
        assert not results[0].matched
