"""Phase-4 tests: modality encoders, temporal fusion, and the latent layout.

Covers fixed vector width with silent/missing streams, determinism, the
layout-hash guard, online/offline feature parity, and the no-Minecraft-constants
boundary on the encoders package.
"""

import os
import pathlib

import pytest

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.streams import TemporalBuffer, TemporalFusion
from cognitive_runtime.core.streams.encoders import (
    CategoryEncoder,
    EntityEncoder,
    EventEncoder,
    GridVisionEncoder,
    ScalarEncoder,
    SpatialEncoder,
)
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.fusion import LatentState
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import load_session_metadata
from cognitive_runtime.training.datasets import build_dataset
from cognitive_runtime.training.imitation import BCModel

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32}


def _event(stream_id, modality, payload, seq=0, ts=0.0):
    return StreamEvent(stream_id, modality, ts, seq, payload)


# ------------------------------------------------------------ unit: encoders


def test_scalar_encoder_value_trend_mean_max():
    spec = StreamSpec("body.health", "body", range=(0.0, 20.0), neutral=20.0)
    events = [_event("body.health", "body", v, i, i) for i, v in enumerate([10.0, 20.0])]
    token = ScalarEncoder().encode(events, spec)
    assert token.vector == [1.0, 0.5, 0.75, 1.0]  # latest, trend, mean, max (normed)
    assert ScalarEncoder().neutral(spec) == [1.0, 0.0, 1.0, 1.0]


def test_spatial_encoder_position_and_rotation_share_width():
    pos_spec = StreamSpec("spatial.position", "spatial", range=(0.0, 32.0))
    rot_spec = StreamSpec("spatial.rotation", "spatial")
    pos = SpatialEncoder().encode(
        [_event("spatial.position", "spatial", {"x": 0.0, "y": 64.0, "z": 0.0}),
         _event("spatial.position", "spatial", {"x": 16.0, "y": 64.0, "z": 0.0})],
        pos_spec,
    )
    rot = SpatialEncoder().encode(
        [_event("spatial.rotation", "spatial", {"yaw": 0.0, "pitch": 0.0})], rot_spec
    )
    assert len(pos.vector) == len(rot.vector) == SpatialEncoder().width()
    assert pos.vector[0] == 0.5 and pos.vector[3] == 0.5  # latest x_norm, displacement dx
    assert rot.vector[6] == 1.0  # yaw_cos at 0 degrees


def test_grid_vision_encoder_fixed_width_and_histogram():
    legend = {0: "ground", 1: "solid", 9: "entity"}
    spec = StreamSpec("vision.frame.grid", "vision", legend=legend)
    grid = [[0, 1], [9, 0]]
    enc = GridVisionEncoder()
    token = enc.encode([_event("vision.frame.grid", "vision", grid)], spec)
    assert len(token.vector) == enc.width(spec)
    # histogram (classes sorted: entity, ground, solid) over 4 cells.
    assert token.vector[:3] == [0.25, 0.5, 0.25]


def test_event_encoder_marks_presence():
    spec = StreamSpec("event.died", "event")
    assert EventEncoder().encode([_event("event.died", "event", {})], spec).vector == [1.0]
    assert EventEncoder().neutral(spec) == [0.0]


def test_entity_encoder_nearest_and_empty():
    spec = StreamSpec("vision.entities", "vision", range=(0.0, 16.0))
    token = EntityEncoder().encode(
        [_event("vision.entities", "vision", [{"distance": 8.0, "angle": 0.0}])], spec
    )
    assert token.vector[0] == 0.5 and token.vector[3] == 0.25  # dist norm, count
    empty = EntityEncoder().encode([_event("vision.entities", "vision", [])], spec)
    assert empty.vector == [1.0, 0.0, 0.0, 0.0]


def test_category_encoder_one_hot_with_other_bucket():
    spec = StreamSpec("world.front_block", "world", categories=("grass", "tree"))
    enc = CategoryEncoder()
    assert enc.encode([_event("world.front_block", "world", "tree")], spec).vector == [0, 1, 0]
    assert enc.encode([_event("world.front_block", "world", "lava")], spec).vector == [0, 0, 1]
    assert enc.width(spec) == 3


# ------------------------------------------------------- fusion: width/determinism


def _catalog():
    return MinecraftSurvivalBox(config=FAST_CONFIG).stream_catalog()


def test_fusion_fixed_width_with_missing_and_silent_streams():
    fusion = TemporalFusion(_catalog())
    empty = fusion.fuse(None, TemporalBuffer())  # no streams at all
    assert empty.width == fusion.width
    assert empty.vector == [0.0] * fusion.width or len(empty.vector) == fusion.width

    # A buffer with only one stream still yields the full fixed width.
    buffer = TemporalBuffer()
    buffer.append(_event("body.health", "body", 12.0))
    partial = fusion.fuse(None, buffer)
    assert partial.width == fusion.width
    assert partial.slice("body.health")[0] != 0.0  # populated
    # A silent stream keeps its neutral fill.
    assert partial.slices["event.died"]  # slot exists


def test_fusion_is_deterministic():
    fusion = TemporalFusion(_catalog())
    buffer = TemporalBuffer()
    for i in range(5):
        buffer.append(_event("body.hunger", "body", 20.0 - i, i, i))
    a = fusion.fuse(None, buffer)
    b = fusion.fuse(None, buffer)
    assert a.vector == b.vector
    assert a.layout_hash == b.layout_hash


def test_layout_hash_changes_with_catalog():
    base = TemporalFusion(_catalog())
    # Drop a stream: different layout => different hash.
    trimmed = TemporalFusion([s for s in _catalog() if s.stream_id != "body.health"])
    assert base.layout_hash != trimmed.layout_hash
    assert trimmed.width < base.width


def test_exact_streams_and_new_events_do_not_change_default_fusion_layout():
    catalog = _catalog()
    legacy_catalog = [
        spec for spec in catalog
        if spec.stream_id not in {
            "body.inventory_exact",
            "world.front_block_exact",
            "world.nearby_blocks_exact",
            "event.created_light_source",
            "event.tool_used",
            "event.mob_killed",
            "event.bumped",
        }
    ]
    assert TemporalFusion(catalog).layout_hash == TemporalFusion(legacy_catalog).layout_hash


def test_layout_hash_changes_with_categorical_vocabulary():
    """A renamed categorical vocabulary of the same size changes the one-hot
    semantics, so it must change the layout hash (loud failure, not silent
    mis-prediction)."""

    def rename(spec):
        if spec.categories is None:
            return spec
        renamed = tuple(f"other_{c}" for c in spec.categories)
        return StreamSpec(
            stream_id=spec.stream_id, modality=spec.modality,
            description=spec.description, nominal_rate_hz=spec.nominal_rate_hz,
            payload_schema=spec.payload_schema, range=spec.range,
            legend=spec.legend, categories=renamed, neutral=spec.neutral,
        )

    base = TemporalFusion(_catalog())
    renamed = TemporalFusion([rename(s) for s in _catalog()])
    assert renamed.width == base.width  # same size vocabulary, same layout width
    assert renamed.layout_hash != base.layout_hash


# ---------------------------------------------------- layout-hash guard on load


def test_learned_policy_rejects_layout_mismatch():
    from cognitive_runtime.policies import LearnedPolicy

    fusion = TemporalFusion(_catalog())
    # Model trained on a *different* layout hash than the runtime produces.
    model = BCModel(
        feature_names=["f"], action_keys=["NULL"], weights=[[0.0]], bias=[0.0],
        meta={"representation": "latent", "layout_hash": "deadbeef"},
    )
    policy = LearnedPolicy(model)
    memory = Memory()
    memory.set_fused_latent(LatentState([0.0] * fusion.width, {}, fusion.layout_hash))
    with pytest.raises(ValueError, match="layout mismatch"):
        policy._latent_features(memory)

    # Matching hash: no error.
    ok = BCModel(
        feature_names=["f"], action_keys=["NULL"], weights=[[0.0]], bias=[0.0],
        meta={"representation": "latent", "layout_hash": fusion.layout_hash},
    )
    LearnedPolicy(ok)._latent_features(memory)  # does not raise


# ------------------------------------------------------ online/offline parity


class _CapturePolicy(SingleActionPolicy):
    name = "scripted"

    def __init__(self, seed):
        self.base = ScriptedSurvivalPolicy(seed=seed)
        self.latents = []

    def reset(self):
        self.base.reset()

    def decide(self, state, memory, prediction):
        self.latents.append(list(memory.fused_latent().vector))
        return self.base.decide(state, memory, prediction)


def test_online_offline_feature_parity(tmp_path):
    """The recorded window must encode to identical vectors online (loop) and
    offline (dataset) — same fusion code path over the same buffer."""
    capture = _CapturePolicy(seed=1)
    config = RuntimeConfig(
        episodes=1, seed=7, max_ticks_per_episode=200,
        record_dir=str(tmp_path), session_id="parity",
        program_config=FAST_CONFIG, record_frames=True,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG), policy=capture, config=config
    ).run()
    session_dir = os.path.join(str(tmp_path), "parity")

    fusion = TemporalFusion(
        [StreamSpec.from_dict(s) for s in load_session_metadata(session_dir)["stream_catalog"]]
    )
    dataset = build_dataset([session_dir], representation="latent")
    width = fusion.width
    assert len(dataset) == len(capture.latents) == 200
    for offline_feats, online_vec in zip(dataset.features, capture.latents):
        assert offline_feats[:width] == online_vec


# ----------------------------------------------- no Minecraft constants boundary


def test_encoders_package_has_no_minecraft_constants():
    from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
    from cognitive_runtime.programs.minecraft.world import BLOCK_IDS, BREAK_YIELD

    forbidden = (
        set(BLOCK_IDS) | set(BREAK_YIELD) | set(BREAK_YIELD.values())
        | {a.name for a in ACTION_SPACE if a.name != "NULL"}
    )
    forbidden -= {""}
    import cognitive_runtime.core.streams.encoders as pkg

    encoders_dir = pathlib.Path(pkg.__file__).parent
    offenders = {}
    for path in encoders_dir.glob("*.py"):
        text = path.read_text()
        hits = sorted(w for w in forbidden if w in text)
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"Minecraft constants leaked into encoders: {offenders}"
