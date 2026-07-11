"""Deterministic attention controller (issue #59): scoring, budget, dwell/
hysteresis, the `attention="off"` ablation's byte-identical fusion output,
and a simulated run recording an `AttentionState` every tick."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from cognitive_runtime.core.attention import (
    AttentionBudget,
    AttentionCoefficients,
    AttentionConfig,
    AttentionController,
)
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.fusion import TemporalFusion
from cognitive_runtime.core.streams.registry import DEFAULT_STREAM_REGISTRY
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.programs.minecraft.streams import build_survival_stream_specs
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import ATTENTION_WEIGHTS_STREAM, CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard

FAST_CONFIG = {"episode_ticks": 20, "world_size": 32, "max_mobs": 1}

_CATALOG = [
    StreamSpec("body.health", "body", range=(0, 20)),
    StreamSpec("body.hunger", "body", range=(0, 20)),
    StreamSpec("reward.scalar", "reward", range=(-2.0, 2.0)),
    StreamSpec("internal.risk", "event"),
]


def _fill(buffer: TemporalBuffer, t: float, seq: int, health=20.0, reward=0.01, risk=0.05):
    buffer.append(StreamEvent("body.health", "body", t, seq, health))
    buffer.append(StreamEvent("body.hunger", "body", t, seq, 15.0))
    buffer.append(StreamEvent("reward.scalar", "reward", t, seq, {"value": reward}))
    buffer.append(StreamEvent("internal.risk", "event", t, seq, {"value": risk}))


# ------------------------------------------------------------- contract


def test_attention_module_does_not_import_minecraft():
    code = (
        "import sys; import cognitive_runtime.core.attention; "
        "bad = [m for m in sys.modules if m.startswith('cognitive_runtime.programs')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_controller_operates_on_registry_metadata_only_generic_catalog():
    """No Minecraft stream ids anywhere in the fixture catalog; the controller
    only ever consults `DEFAULT_STREAM_REGISTRY`'s generic declarations."""
    ctrl = AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted")
    assert set(ctrl.stream_ids) == {"body.health", "body.hunger", "reward.scalar", "internal.risk"}


# ------------------------------------------------------------- off mode


def test_off_mode_gives_every_stream_uniform_weight_one():
    ctrl = AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="off")
    buffer = TemporalBuffer()
    _fill(buffer, 0.05, 0)
    state = ctrl.compute(0, buffer)
    assert state.mode == "off"
    assert set(state.weights) == set(ctrl.stream_ids)
    assert all(w == 1.0 for w in state.weights.values())
    assert state.reasons == {}


def test_off_mode_fusion_output_is_byte_identical_to_no_attention():
    catalog = [
        StreamSpec("body.health", "body", range=(0, 20)),
        StreamSpec("event.action_rejected", "event"),  # aux_debug, unattended
    ]
    fusion = TemporalFusion(catalog)
    buffer = TemporalBuffer()
    for i in range(5):
        buffer.append(StreamEvent("body.health", "body", 0.05 * i, i, 20.0 - i))

    baseline = fusion.fuse(None, buffer)
    ctrl = AttentionController(catalog, DEFAULT_STREAM_REGISTRY, mode="off")
    state = ctrl.compute(0, buffer)
    gated = fusion.fuse(None, buffer, attention_weights=state.weights)
    assert gated.vector == baseline.vector
    assert gated.layout_hash == baseline.layout_hash


# ------------------------------------------------------------- budget


def test_budget_never_exceeded():
    budget = AttentionBudget(max_total_weight=2.0, max_streams=2)
    ctrl = AttentionController(
        _CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted",
        config=AttentionConfig(budget=budget),
    )
    buffer = TemporalBuffer()
    for i in range(10):
        _fill(buffer, 0.05 * i, i, health=20.0 - i, reward=0.02 * i, risk=0.1)
    state = ctrl.compute(10, buffer)
    assert len(state.selected_streams) <= 2
    assert state.budget_used == pytest.approx(2.0)
    assert sum(state.weights.values()) <= 2.0 + 1e-9


def test_budget_forces_a_choice_among_more_streams_than_max_streams():
    ctrl = AttentionController(
        _CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted",
        config=AttentionConfig(budget=AttentionBudget(max_total_weight=4.0, max_streams=1)),
    )
    buffer = TemporalBuffer()
    _fill(buffer, 0.05, 0)
    state = ctrl.compute(0, buffer)
    assert len(state.selected_streams) == 1
    unselected = set(ctrl.stream_ids) - set(state.selected_streams)
    assert all(state.weights[sid] == 0.0 for sid in unselected)


# ------------------------------------------------------------- salience / focus


def test_salience_spike_captures_focus_within_one_tick():
    ctrl = AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted")
    buffer = TemporalBuffer()
    t = 0.0
    for i in range(20):
        t += 0.05
        _fill(buffer, t, i, health=20.0, reward=0.0, risk=0.05)
    ctrl.compute(20, buffer)  # settle into a baseline (no strong focus reason)

    # A damage spike: body.health drops sharply, reward turns negative.
    t += 0.05
    _fill(buffer, t, 21, health=5.0, reward=-1.0, risk=0.8)
    state = ctrl.compute(21, buffer)
    assert state.focus_stream == "body.health"
    assert state.weights["body.health"] > state.weights["body.hunger"]


def test_dwell_hysteresis_prevents_thrash_on_alternating_equal_spikes():
    config = AttentionConfig(dwell_ticks=5, displacement_margin=0.5)
    ctrl = AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted", config=config)
    buffer = TemporalBuffer()
    t = 0.0
    for i in range(20):
        t += 0.05
        _fill(buffer, t, i)
    state = ctrl.compute(20, buffer)
    first_focus = state.focus_stream

    # Alternate small, equal-magnitude bumps on the two body streams for a
    # few ticks -- neither should out-spike the captured focus by the
    # displacement margin, and dwell hasn't expired yet.
    focuses = []
    for i in range(3):
        t += 0.05
        if i % 2 == 0:
            buffer.append(StreamEvent("body.health", "body", t, 100 + i, 19.0))
            buffer.append(StreamEvent("body.hunger", "body", t, 100 + i, 15.0))
        else:
            buffer.append(StreamEvent("body.health", "body", t, 100 + i, 20.0))
            buffer.append(StreamEvent("body.hunger", "body", t, 100 + i, 14.0))
        buffer.append(StreamEvent("reward.scalar", "reward", t, 100 + i, {"value": 0.0}))
        buffer.append(StreamEvent("internal.risk", "event", t, 100 + i, {"value": 0.05}))
        state = ctrl.compute(21 + i, buffer)
        focuses.append(state.focus_stream)

    assert focuses == [first_focus] * len(focuses)


def test_reset_clears_focus_and_dwell_state():
    ctrl = AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="budgeted")
    buffer = TemporalBuffer()
    _fill(buffer, 0.05, 0, health=5.0, reward=-1.0)
    ctrl.compute(0, buffer)
    assert ctrl._focus is not None
    ctrl.reset()
    assert ctrl._focus is None
    assert ctrl._dwell_remaining == 0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown attention mode"):
        AttentionController(_CATALOG, DEFAULT_STREAM_REGISTRY, mode="bogus")


# ---------------------------------------------------- simulated run (acceptance)


def test_simulated_run_records_attention_state_every_tick(tmp_path):
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=15,
        record_dir=str(tmp_path),
        session_id="attention-session",
        program_config=FAST_CONFIG,
        attention_mode="budgeted",
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        stream_registry=MINECRAFT_STREAM_REGISTRY,
    ).run()

    session_dir = os.path.join(str(tmp_path), "attention-session")
    ticks_seen = 0
    ticks_with_attention = 0
    saw_weights_stream = False
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        ticks_seen += 1
        attention = decision.get("attention")
        if attention and attention.get("mode") == "budgeted":
            ticks_with_attention += 1
            assert attention["reasons"]  # the debugging payoff
            assert 0.0 <= attention["budget_used"] <= attention["budget_total"] + 1e-9
        if any(r["stream_id"] == ATTENTION_WEIGHTS_STREAM for r in sensory):
            saw_weights_stream = True
    assert ticks_seen == 15
    assert ticks_with_attention == 15
    assert saw_weights_stream

    rendered = view_episode(session_dir, "episode_00000")
    assert "attention_mode: budgeted" in rendered

    dashboard_report = dashboard(str(tmp_path))
    assert "attention_mode" in dashboard_report


def test_default_attention_mode_is_off_and_matches_pre_59_behavior(tmp_path):
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=10,
        record_dir=str(tmp_path),
        session_id="attention-off-session",
        program_config=FAST_CONFIG,
    )
    assert runtime_config.attention_mode == "off"
    summary = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        stream_registry=MINECRAFT_STREAM_REGISTRY,
    ).run()[0]
    assert summary.attention_mode == "off"
    assert summary.avg_attention_budget_used is None
    assert summary.attention_focus_counts == {}


def test_every_minecraft_agent_input_stream_is_attended_in_budgeted_mode():
    catalog = build_survival_stream_specs()
    ctrl = AttentionController(catalog, MINECRAFT_STREAM_REGISTRY, mode="budgeted")
    agent_input_ids = set(MINECRAFT_STREAM_REGISTRY.ids_by_classification(catalog, "agent_input"))
    assert set(ctrl.stream_ids) == agent_input_ids
