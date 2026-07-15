"""Organism identity (issue #88): `RuntimeConfig.name` threaded through
config resolution, recorded session metadata, and checkpoint bundles."""

from __future__ import annotations

import json
import os

import pytest

from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime

FAST_CONFIG = {"episode_ticks": 50, "world_size": 32, "day_length": 150, "start_time": 100}


# --------------------------------------------------------------------------- config resolution


def test_explicit_name_round_trips():
    config = RuntimeConfig(name="Pixel")
    assert config.resolve_name() == "Pixel"
    assert config.resolve_name() == "Pixel"  # stable across repeated calls


def test_unset_name_resolves_to_a_stable_generated_slug():
    config = RuntimeConfig()
    first = config.resolve_name()
    assert first  # never None/empty
    assert config.resolve_name() == first  # cached, not re-rolled per call

    other = RuntimeConfig()
    assert other.resolve_name() != "" and other.resolve_name() is not None


def test_resolved_session_id_prefixes_generated_name_when_session_id_unset():
    config = RuntimeConfig(name="Pixel")
    session_id = config.resolved_session_id("scripted")
    assert session_id.startswith("Pixel-")


def test_resolved_session_id_respects_explicit_session_id():
    config = RuntimeConfig(name="Pixel", session_id="explicit-id")
    assert config.resolved_session_id("scripted") == "explicit-id"


# --------------------------------------------------------------------------- recorded sessions


def _record(tmp_path, *, name=None, session_id=None):
    config = RuntimeConfig(
        episodes=1, seed=3, max_ticks_per_episode=50,
        record_dir=str(tmp_path), session_id=session_id, program_config=FAST_CONFIG,
        name=name,
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=1),
        config=config,
    )
    runtime.run()
    return runtime.recorder.session_dir


def test_recorded_session_dir_and_metadata_carry_explicit_name(tmp_path):
    session_dir = _record(tmp_path, name="Pixel")
    assert os.path.basename(session_dir).startswith("Pixel-")
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["name"] == "Pixel"


def test_recorded_session_resolves_a_generated_name_when_unset(tmp_path):
    session_dir = _record(tmp_path, session_id="unnamed-run")
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["name"]  # never null


def test_episode_summary_carries_the_organism_name(tmp_path):
    session_dir = _record(tmp_path, name="Pixel", session_id="pixel-run")
    with open(os.path.join(session_dir, "episode_00000.summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["name"] == "Pixel"


# --------------------------------------------------------------------------- checkpoints

torch = pytest.importorskip("torch")

from cognitive_runtime.neural import NeuralAgentCheckpoint  # noqa: E402

LAYOUT_HASH = "organism-name-test-layout"
ACTION_KEYS = ["NULL", "JUMP"]


def test_checkpoint_save_load_preserves_name(tmp_path):
    path = tmp_path / "agent.pt"
    NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS, name="Pixel",
    ).save(reason="test")

    loaded = NeuralAgentCheckpoint(str(path), layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS)
    metadata = loaded.load()

    assert loaded.name == "Pixel"
    assert metadata["name"] == "Pixel"


def test_legacy_nameless_checkpoint_still_loads(tmp_path):
    path = tmp_path / "legacy.pt"
    NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS,
    ).save(reason="test")

    # Simulate a checkpoint written before issue #88 -- no `name` key at all,
    # not even `null` -- by stripping it out of the saved payload.
    payload = torch.load(str(path), weights_only=False)
    del payload["metadata"]["name"]
    torch.save(payload, str(path))

    loaded = NeuralAgentCheckpoint(str(path), layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS)
    metadata = loaded.load()  # must not raise

    assert loaded.name is None
    assert "name" not in metadata
