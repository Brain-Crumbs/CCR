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

from cognitive_runtime.core.streams.encoders import ScalarEncoder
from cognitive_runtime.core.streams.registry import (
    DEFAULT_STREAM_REGISTRY,
    AttentionMetadata,
    StreamDeclaration,
    StreamRegistry,
)

#: Issue #32 classification note: the "_exact" mirrors of *world* facts
#: (exact block identity/position, whether as a state stream or an event)
#: are **privileged** -- ground truth a raw-vision agent would have to earn
#: through perception, kept only for replay fidelity/debugging and excluded
#: from both policy input and auxiliary-loss targets. The "_exact" mirror of
#: the agent's *own* inventory is not privileged in that sense (it is
#: self-state the agent trivially has access to), so it stays agent input.
MINECRAFT_STREAM_REGISTRY: StreamRegistry = DEFAULT_STREAM_REGISTRY.extend(
    [
        StreamDeclaration(
            "world.entity_bearing",
            ScalarEncoder,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="world", relative_compute_cost="low", localization_hint=True,
            ),
            note=(
                "Nearest visible entity's salience + bearing (issue #60's stimulus "
                "localization contract): ScalarEncoder reads the {'value': ...} "
                "leaf, and the attention controller reads the {'direction': "
                "{'bearing_deg': ...}} leaf via `localization_hint=True`. Entity "
                "presence/proximity, not identity -- proprioceptive-adjacent "
                "salience, not the privileged ground truth vision.entities carries."
            ),
        ),
        StreamDeclaration(
            "world.time",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note=(
                "Composite {time_of_day, day_length, is_night} payload; no "
                "generic encoder fits it (ScalarEncoder would silently drop "
                "two of the three fields). Deliberate fixed stub pending a "
                "small dedicated encoder. Exact clock knowledge a raw-vision "
                "agent would otherwise infer from lighting -- debug/aux-loss "
                "target, not raw agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "world.biome",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note=(
                "String biome name with no declared StreamSpec.categories "
                "vocabulary, so CategoryEncoder cannot bind. Deliberate "
                "fixed stub until a biome vocabulary is published. Semantic "
                "world label, kept for debugging/aux loss (issue #32)."
            ),
        ),
        StreamDeclaration(
            "world.nearby_blocks",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note=(
                "5x5 string block-name grid; GridVisionEncoder requires an "
                "int grid + StreamSpec.legend. Deliberate fixed stub pending "
                "a dedicated string-grid encoder. Coarse semantic patch, "
                "debug/aux-loss target rather than raw agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "world.nearby_blocks_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="privileged",
            note=(
                "Exact-name debug mirror of world.nearby_blocks; not fused, "
                "kept for replay fidelity. Exact world block identity: "
                "privileged ground truth, excluded from agent input and from "
                "aux-loss targets (issue #32)."
            ),
        ),
        StreamDeclaration(
            "world.front_block_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="privileged",
            note=(
                "Exact-name debug mirror of world.front_block (which is "
                "fused); not itself fused. Exact world block identity: "
                "privileged ground truth (issue #32)."
            ),
        ),
        StreamDeclaration(
            "body.inventory_open",
            encoder_factory=None,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="low"),
            note=(
                "Inventory-open flag (issue #42's OPEN_INVENTORY/CLOSE_INVENTORY); "
                "no legacy fusion slot is bound (Minecraft-specific, unlike the "
                "generic body.alive/body.in_water booleans in the default "
                "registry), neural path uses BodyStateEncoder like those. "
                "Self-state (the agent's own UI state), not a privileged world "
                "fact -- agent input."
            ),
        ),
        StreamDeclaration(
            "body.inventory_exact",
            encoder_factory=None,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.EntityEncoder",
            neural_latent_width=16,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="medium"),
            note=(
                "Exact {minecraft_item_name: count} payload with an unbounded "
                "item vocabulary. Neural path uses EntityEncoder; no legacy "
                "fusion slot is bound because the bounded body.inventory "
                "summary remains the compatibility baseline. Self-state (the "
                "agent's own possessions), not a privileged world fact -- "
                "agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.item_collected_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note=(
                "Exact item-count event; semantic/replay payload, not fused "
                "today. Narrates a self-inventory change (not world ground "
                "truth); debug/aux-loss target (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.block_broken_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="privileged",
            note=(
                "Exact block-position event; semantic/replay payload, not "
                "fused today. Exact world block identity + position: "
                "privileged ground truth (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.block_placed_exact",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="privileged",
            note=(
                "Exact block-position event; semantic/replay payload, not "
                "fused today. Exact world block identity + position: "
                "privileged ground truth (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.crafted",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Structured crafting/smelting event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.advancement",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Milestone event with open-ended ids; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.dimension_changed",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Dimension transition event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.biome_entered",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Biome transition event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.structure_discovered",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Open-ended structure discovery event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.container_interaction",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Structured container interaction event; semantic/replay payload, not fused today.",
        ),
        StreamDeclaration(
            "event.created_light_source",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
        StreamDeclaration(
            "event.mob_killed",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
        StreamDeclaration(
            "event.tool_used",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
        StreamDeclaration(
            "event.bumped",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Semantic event with no fusion slot today; not one of the nine events EventEncoder is bound to.",
        ),
    ]
)
