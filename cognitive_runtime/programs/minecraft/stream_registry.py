"""Minecraft-specific stream declarations (issue #21).

Extends `core.streams.registry.DEFAULT_STREAM_REGISTRY` with the concrete
stream ids the survival catalog (`streams.py:build_survival_stream_specs`)
publishes that don't fit a generic modality pattern: composite/dict payloads
no generic encoder handles, unbounded-vocabulary duplicates kept for replay
fidelity, and raw tensors reserved for a future neural encoder. Every one of
these is a deliberate fixed stub (`StreamDeclaration.trainable=False`) --
`test_stream_registry.py` asserts the survival catalog has no stream left
undeclared.
"""

from __future__ import annotations

from cognitive_runtime.core.streams.registry import (
    DEFAULT_STREAM_REGISTRY,
    StreamDeclaration,
    StreamRegistry,
)

MINECRAFT_STREAM_REGISTRY: StreamRegistry = DEFAULT_STREAM_REGISTRY.extend(
    [
        StreamDeclaration(
            "world.time",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "Composite {time_of_day, day_length, is_night} payload; no "
                "generic encoder fits it (ScalarEncoder would silently drop "
                "two of the three fields). Deliberate fixed stub pending a "
                "small dedicated encoder."
            ),
        ),
        StreamDeclaration(
            "world.biome",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "String biome name with no declared StreamSpec.categories "
                "vocabulary, so CategoryEncoder cannot bind. Deliberate "
                "fixed stub until a biome vocabulary is published."
            ),
        ),
        StreamDeclaration(
            "world.nearby_blocks",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "5x5 string block-name grid; GridVisionEncoder requires an "
                "int grid + StreamSpec.legend. Deliberate fixed stub pending "
                "a dedicated string-grid encoder."
            ),
        ),
        StreamDeclaration(
            "world.nearby_blocks_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Exact-name debug mirror of world.nearby_blocks; not fused, kept for replay fidelity.",
        ),
        StreamDeclaration(
            "world.front_block_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Exact-name debug mirror of world.front_block (which is fused); not itself fused.",
        ),
        StreamDeclaration(
            "body.inventory_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "Exact {minecraft_item_name: count} payload with an unbounded "
                "item vocabulary -- needs a learned embedding (target doc: "
                "'inventory/entities -> embedding or small MLP', Phase B+). "
                "Redundant today with the bounded body.inventory summary, "
                "which is fused."
            ),
        ),
        StreamDeclaration(
            "event.item_collected_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Exact item-count event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.block_broken_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Exact block-position event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.block_placed_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Exact block-position event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.crafted",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Structured crafting/smelting event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.advancement",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Milestone event with open-ended ids; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.dimension_changed",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Dimension transition event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.biome_entered",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Biome transition event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.structure_discovered",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Open-ended structure discovery event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.container_interaction",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Structured container interaction event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.created_light_source",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
        StreamDeclaration(
            "event.mob_killed",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
        StreamDeclaration(
            "event.bumped",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
    ]
)
