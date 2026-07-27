"""Crafter-specific stream declarations (issue #89).

Extends ``core.streams.registry.DEFAULT_STREAM_REGISTRY`` with the concrete
ids ``streams.py:build_crafter_stream_specs`` publishes that the generic
registry doesn't already cover by pattern.  ``vision.frame.grid``,
``vision.frame.pixels``, ``body.health``, ``body.inventory``, ``body.alive``,
``spatial.*``, ``reward.*``, ``event.action_rejected`` and ``event.died``
already match generic declarations in ``DEFAULT_STREAM_REGISTRY`` and are
reused unchanged; only Crafter's own vitals/flags/event need a declaration
here.
"""

from __future__ import annotations

from cognitive_runtime.core.streams.encoders import ScalarEncoder
from cognitive_runtime.core.streams.registry import (
    DEFAULT_STREAM_REGISTRY,
    AttentionMetadata,
    StreamDeclaration,
    StreamRegistry,
)

CRAFTER_STREAM_REGISTRY: StreamRegistry = DEFAULT_STREAM_REGISTRY.extend(
    [
        StreamDeclaration(
            "body.food",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Food vital (Crafter's hunger analog); legacy fusion scalar-encoded, "
                 "same pattern as Minecraft's body.hunger.",
        ),
        StreamDeclaration(
            "body.drink",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Drink (thirst) vital; no Minecraft analog -- Crafter's own "
                 "interoceptive sense.",
        ),
        StreamDeclaration(
            "body.energy",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Energy vital (depletes over time, restored by SLEEP); Crafter's own "
                 "interoceptive sense.",
        ),
        StreamDeclaration(
            "body.sleeping",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="low"),
            note="Sleeping flag; self-state, mirrors Minecraft's body.in_water toggle.",
        ),
        StreamDeclaration(
            "event.achievement",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="aux_debug",
            note="Milestone counter (repeatable per episode); semantic/replay payload, "
                 "mirrors Minecraft's event.advancement, not fused today.",
        ),
    ]
)
