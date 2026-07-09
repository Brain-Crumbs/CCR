"""Issue #21: per-stream schema registry.

Covers catalog completeness (every Minecraft stream has a declaration),
explicit fixed-stub marking, byte-compatibility of the fusion layout built
from the registry, and that the declarations land in session metadata.
"""

import pytest

from cognitive_runtime.core.streams import (
    DEFAULT_STREAM_REGISTRY,
    StreamDeclaration,
    StreamRegistry,
)
from cognitive_runtime.core.streams.encoders import ScalarEncoder
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.fusion import TemporalFusion, default_encoder_registry
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.programs.minecraft.streams import build_survival_stream_specs
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import load_session_metadata

FAST_CONFIG = {"episode_ticks": 20, "world_size": 32}

# Pinned before the registry refactor (see fusion.py's default_encoder_registry
# docstring): proves the declarative registry reproduces the exact same
# fusion layout, so saved online-Q checkpoints stay loadable.
EXPECTED_WIDTH = 105
EXPECTED_LAYOUT_HASH = "5143ff8ff49183705dbae311fb777541619edbcc"


# ------------------------------------------------------------- completeness


def test_minecraft_catalog_is_fully_declared():
    catalog = build_survival_stream_specs()
    assert MINECRAFT_STREAM_REGISTRY.missing(catalog) == []
    MINECRAFT_STREAM_REGISTRY.assert_complete(catalog)  # does not raise


def test_generic_registry_alone_is_missing_minecraft_specific_streams():
    """world.time/biome/etc. aren't generic -- only the Minecraft overlay
    declares them, which is what makes the completeness check meaningful."""
    catalog = build_survival_stream_specs()
    missing = DEFAULT_STREAM_REGISTRY.missing(catalog)
    assert "world.time" in missing
    assert "body.inventory_exact" in missing


def test_assert_complete_raises_with_missing_stream_ids():
    registry = StreamRegistry([StreamDeclaration("body.health", ScalarEncoder)])
    catalog = [
        StreamSpec("body.health", "body"),
        StreamSpec("body.hunger", "body"),
    ]
    with pytest.raises(ValueError, match="body.hunger"):
        registry.assert_complete(catalog)


# ------------------------------------------------------------- fixed stubs


def test_every_declared_stream_is_a_fixed_stub_today():
    """Phase B+ trainable encoders don't exist in this repo yet, so every
    current declaration must be explicit about being a fixed stub (target
    doc success criterion: 'trainable encoder or a deliberate fixed stub')."""
    catalog = build_survival_stream_specs()
    for spec in catalog:
        decl = MINECRAFT_STREAM_REGISTRY.declaration_for(spec.stream_id)
        assert decl is not None, spec.stream_id
        assert decl.trainable is False
        assert decl.is_fixed_stub() is True
        assert decl.train_eval_behavior in ("fixed", "raw")


def test_raw_streams_have_no_fusion_encoder_but_do_have_metadata():
    catalog = build_survival_stream_specs()
    by_id = {s.stream_id: s for s in catalog}
    for stream_id in ("vision.frame.pixels", "world.time", "body.inventory_exact"):
        decl = MINECRAFT_STREAM_REGISTRY.declaration_for(stream_id)
        assert decl.encoder() is None
        assert decl.train_eval_behavior == "raw"
        assert decl.note
        assert decl.resolve_checkpoint_key(stream_id).startswith("stream_encoder.")
        assert by_id[stream_id].stream_id == stream_id  # sanity: still in the catalog


def test_reserved_ids_for_audio_and_mouse_look():
    for stream_id in ("audio.ambient", "input.keypress", "input.mouse_look", "motor.history"):
        decl = DEFAULT_STREAM_REGISTRY.declaration_for(stream_id)
        assert decl is not None, stream_id
        assert decl.train_eval_behavior == "raw"


# ------------------------------------------------------------- fusion parity


def test_default_encoder_registry_matches_declarative_registry():
    catalog = build_survival_stream_specs()
    fusion = TemporalFusion(catalog, default_encoder_registry())
    assert fusion.width == EXPECTED_WIDTH
    assert fusion.layout_hash == EXPECTED_LAYOUT_HASH


def test_fusion_from_declarative_registry_is_byte_compatible():
    """The registry-built StreamEncoderRegistry drives the same fusion layout
    as before the refactor -- existing online-Q checkpoints keep loading."""
    catalog = build_survival_stream_specs()
    fusion = TemporalFusion(catalog, DEFAULT_STREAM_REGISTRY.to_encoder_registry())
    assert fusion.width == EXPECTED_WIDTH
    assert fusion.layout_hash == EXPECTED_LAYOUT_HASH


# ------------------------------------------------------------- describe()


def test_describe_reports_shape_rate_encoder_width_checkpoint_and_behavior():
    catalog = build_survival_stream_specs()
    described = {d["stream_id"]: d for d in MINECRAFT_STREAM_REGISTRY.describe(catalog)}

    pixels = described["vision.frame.pixels"]
    assert pixels["shape"] == [33, 33, 3]
    assert pixels["nominal_rate_hz"] == 20.0
    assert pixels["encoder"] is None
    assert pixels["latent_width"] == 0
    assert pixels["trainable"] is False
    assert pixels["fixed_stub"] is True
    assert pixels["train_eval_behavior"] == "raw"
    assert pixels["checkpoint_key"] == "stream_encoder.vision_frame_pixels"

    health = described["body.health"]
    assert health["encoder"] == "ScalarEncoder"
    assert health["latent_width"] == 4
    assert health["train_eval_behavior"] == "fixed"
    assert health["checkpoint_key"] == "stream_encoder.body_health"


# ------------------------------------------------------------- session metadata


def test_session_metadata_records_stream_registry(tmp_path):
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=FAST_CONFIG["episode_ticks"],
        record_dir=str(tmp_path),
        session_id="stream-registry-metadata",
        program_config=FAST_CONFIG,
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=1),
        config=runtime_config,
        stream_registry=MINECRAFT_STREAM_REGISTRY,
    )
    runtime.run()

    metadata = load_session_metadata(runtime.recorder.session_dir)
    declared = {d["stream_id"]: d for d in metadata["stream_registry"]}
    catalog_ids = {s["stream_id"] for s in metadata["stream_catalog"]}
    assert catalog_ids == set(declared)
    assert declared["body.health"]["checkpoint_key"] == "stream_encoder.body_health"
    assert declared["vision.frame.pixels"]["train_eval_behavior"] == "raw"
