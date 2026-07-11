"""Issue #32: raw/aux-debug/privileged stream classification + attention
metadata, and the "raw input" ablation profile it enables.
"""

import pytest

from cognitive_runtime.core.streams import (
    STREAM_CLASSIFICATIONS,
    AttentionMetadata,
    DEFAULT_STREAM_REGISTRY,
    StreamDeclaration,
    StreamRegistry,
)
from cognitive_runtime.core.streams.encoders import ScalarEncoder
from cognitive_runtime.core.streams.fusion import TemporalFusion
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.programs.minecraft.streams import build_survival_stream_specs

FAST_CONFIG = {"episode_ticks": 20, "world_size": 32}


# ------------------------------------------------------------- declaration validation


def test_classification_is_required_and_validated():
    with pytest.raises(ValueError, match="classification"):
        StreamDeclaration("body.health", ScalarEncoder, classification=None)
    with pytest.raises(ValueError, match="classification"):
        StreamDeclaration("body.health", ScalarEncoder, classification="not-a-real-one")


def test_agent_input_requires_attention_metadata():
    with pytest.raises(ValueError, match="AttentionMetadata"):
        StreamDeclaration("body.health", ScalarEncoder, classification="agent_input")
    # aux_debug/privileged do not require it.
    StreamDeclaration("world.sheltered", ScalarEncoder, classification="aux_debug")


def test_attention_metadata_validates_its_own_fields():
    with pytest.raises(ValueError, match="relative_compute_cost"):
        AttentionMetadata(modality="body", relative_compute_cost="extreme")
    with pytest.raises(ValueError, match="expected_sample_rate_hz"):
        AttentionMetadata(modality="body", expected_sample_rate_hz=-1.0)
    with pytest.raises(ValueError, match="modality"):
        AttentionMetadata(modality="")


# ------------------------------------------------------------- completeness (acceptance criteria)


def test_every_minecraft_catalog_stream_is_classified():
    catalog = build_survival_stream_specs()
    for spec in catalog:
        classification = MINECRAFT_STREAM_REGISTRY.classification_for(spec.stream_id)
        assert classification in STREAM_CLASSIFICATIONS, spec.stream_id


def test_attention_completeness_for_every_agent_input_stream():
    catalog = build_survival_stream_specs()
    MINECRAFT_STREAM_REGISTRY.assert_attention_complete(catalog)  # does not raise
    agent_input_ids = MINECRAFT_STREAM_REGISTRY.ids_by_classification(catalog, "agent_input")
    assert agent_input_ids  # non-empty
    for stream_id in agent_input_ids:
        decl = MINECRAFT_STREAM_REGISTRY.declaration_for(stream_id)
        assert decl.attention is not None, stream_id
        assert decl.attention.modality
        assert decl.attention.relative_compute_cost in ("low", "medium", "high")


def test_raw_ish_and_semantic_streams_classify_as_the_issue_describes():
    """The issue's own worked examples: vision.frame.pixels is raw-ish agent
    input; world.front_block/world.sheltered/vision.entities/event streams
    are semantic and demoted to aux_debug."""
    catalog = build_survival_stream_specs()
    by_id = MINECRAFT_STREAM_REGISTRY.classification_for
    assert by_id("vision.frame.pixels") == "agent_input"
    assert by_id("world.front_block") == "aux_debug"
    assert by_id("world.sheltered") == "aux_debug"
    assert by_id("vision.entities") == "aux_debug"
    assert by_id("event.damage_taken") == "aux_debug"
    assert by_id("event.item_collected") == "aux_debug"
    # proprioception/interoception stays agent input
    assert by_id("body.health") == "agent_input"
    assert by_id("spatial.position") == "agent_input"
    assert by_id("reward.scalar") == "agent_input"
    # exact world-fact mirrors are privileged (excluded from agent input
    # *and* aux-loss targets); the agent's own exact inventory is not
    assert by_id("world.nearby_blocks_exact") == "privileged"
    assert by_id("world.front_block_exact") == "privileged"
    assert by_id("event.block_broken_exact") == "privileged"
    assert by_id("event.block_placed_exact") == "privileged"
    assert by_id("body.inventory_exact") == "agent_input"
    assert {s.stream_id for s in catalog} >= {
        "world.nearby_blocks_exact", "world.front_block_exact",
        "event.block_broken_exact", "event.block_placed_exact", "body.inventory_exact",
    }


def test_internal_modulation_streams_are_reserved_agent_input():
    """Issue #32: the (not-yet-published, issue #58) internal.* streams get
    classified through the same registry mechanism ahead of time."""
    decl = DEFAULT_STREAM_REGISTRY.declaration_for("internal.prediction_error")
    assert decl is not None
    assert decl.classification == "agent_input"
    assert decl.attention is not None
    assert decl.attention.modality == "internal"


def test_mouse_look_is_agent_input_and_published_by_minecraft():
    decl = DEFAULT_STREAM_REGISTRY.declaration_for("input.mouse_look")
    assert decl is not None
    assert decl.classification == "agent_input"
    assert decl.attention is not None
    catalog = build_survival_stream_specs()
    assert "input.mouse_look" in {s.stream_id for s in catalog}


# ------------------------------------------------------------- describe() session metadata


def test_describe_reports_classification_and_attention_metadata():
    catalog = build_survival_stream_specs()
    described = {d["stream_id"]: d for d in MINECRAFT_STREAM_REGISTRY.describe(catalog)}

    pixels = described["vision.frame.pixels"]
    assert pixels["classification"] == "agent_input"
    assert pixels["attention_modality"] == "vision"
    assert pixels["attention_expected_sample_rate_hz"] == 20.0
    assert pixels["attention_relative_compute_cost"] == "high"
    assert pixels["attention_localization_hint"] is True

    sheltered = described["world.sheltered"]
    assert sheltered["classification"] == "aux_debug"
    assert sheltered["attention_modality"] is None

    exact = described["world.nearby_blocks_exact"]
    assert exact["classification"] == "privileged"


# ------------------------------------------------------------- the "raw input" profile


def test_to_encoder_registry_rejects_unknown_classification():
    with pytest.raises(ValueError, match="unknown classification"):
        MINECRAFT_STREAM_REGISTRY.to_encoder_registry(classifications={"not-a-real-one"})
    with pytest.raises(ValueError, match="unknown classification"):
        MINECRAFT_STREAM_REGISTRY.ids_by_classification(build_survival_stream_specs(), "bogus")


def test_raw_profile_fusion_drops_semantic_streams_but_keeps_proprioception():
    catalog = build_survival_stream_specs()
    raw_registry = MINECRAFT_STREAM_REGISTRY.to_encoder_registry(classifications={"agent_input"})
    fusion = TemporalFusion(catalog, raw_registry)
    fused_ids = {entry.stream_id for entry in fusion.layout}

    # semantic/hand-computed streams are gone from the policy's fused state
    for excluded in (
        "world.front_block", "world.sheltered", "vision.entities", "vision.frame.grid",
        "event.damage_taken", "event.item_collected", "spatial.distance_from_spawn",
    ):
        assert excluded not in fused_ids, excluded

    # proprioception/reward stay -- the profile changes what reaches the
    # policy, not the underlying legacy scalar-fusion machinery.
    for kept in ("body.health", "body.hunger", "spatial.position", "reward.scalar"):
        assert kept in fused_ids, kept

    assert fusion.width < TemporalFusion(catalog).width


def test_raw_profile_does_not_leak_through_a_broader_wildcard():
    """Regression: excluding a specific pattern (spatial.distance_from_spawn,
    aux_debug) must not let it fall through to a later, broader pattern
    (spatial.*, agent_input) that would otherwise match the same stream id."""
    catalog = build_survival_stream_specs()
    raw_registry = MINECRAFT_STREAM_REGISTRY.to_encoder_registry(classifications={"agent_input"})
    assert raw_registry.encoder_for("spatial.distance_from_spawn") is None
    fusion = TemporalFusion(catalog, raw_registry)
    assert "spatial.distance_from_spawn" not in {e.stream_id for e in fusion.layout}


def test_ids_by_classification_partitions_the_catalog():
    catalog = build_survival_stream_specs()
    agent_input = set(MINECRAFT_STREAM_REGISTRY.ids_by_classification(catalog, "agent_input"))
    aux_debug = set(MINECRAFT_STREAM_REGISTRY.ids_by_classification(catalog, "aux_debug"))
    privileged = set(MINECRAFT_STREAM_REGISTRY.ids_by_classification(catalog, "privileged"))

    assert agent_input & aux_debug == set()
    assert agent_input & privileged == set()
    assert aux_debug & privileged == set()
    assert agent_input | aux_debug | privileged == {s.stream_id for s in catalog}
