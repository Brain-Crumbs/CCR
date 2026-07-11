"""Per-stream schema registry (issue #21).

`StreamSpec` (`events.py`) already carries a stream's shape/schema and
sample rate; what was still scattered by hand across `fusion.py`,
`training/datasets.py` and policy code was the rest of the neural-path
contract: which encoder module binds to a stream, whether that encoder is
trainable or a deliberate fixed stub, its latent width, its checkpoint key,
and its train/eval behavior. A :class:`StreamDeclaration` bundles all of
that into one object; a :class:`StreamRegistry` is an ordered collection of
them (first pattern match wins, exactly like `StreamEncoderRegistry`) that
can check a catalog for completeness and hand back the `StreamEncoderRegistry`
`TemporalFusion` builds its layout from -- so adding a stream is one
declaration instead of edits spread across callers.

Generic, modality-shaped declarations (the ones every Program's streams of
that shape share) live in :data:`DEFAULT_STREAM_REGISTRY` below. A Program
with its own concrete stream ids that don't fit a generic pattern -- debug
mirrors, composite payloads, raw sensor tensors with no encoder yet --
extends it with :meth:`StreamRegistry.extend` (see
`programs/minecraft/stream_registry.py`). This module never imports
`cognitive_runtime.programs` (enforced by the same test that keeps the rest
of `core/streams/` environment-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from cognitive_runtime.core.streams.bus import stream_matches
from cognitive_runtime.core.streams.encoder_registry import StreamEncoder, StreamEncoderRegistry
from cognitive_runtime.core.streams.encoders import (
    CategoryEncoder,
    EntityEncoder,
    EventEncoder,
    GridVisionEncoder,
    ScalarEncoder,
    SpatialEncoder,
)
from cognitive_runtime.core.streams.events import StreamSpec

#: Train/eval behavior a declared stream's encoder has:
#:  - "fixed"      deterministic hand-written encoder, bound into the current
#:                  `TemporalFusion` layout; behaves identically in train/eval.
#:  - "trainable"  a learned module with weights that mutate online and
#:                  differ under train()/eval() (Phase B+; nothing in this
#:                  repo uses it yet -- see `cognitive_runtime.neural`).
#:  - "raw"        deliberately has no fusion-layout slot: the raw payload
#:                  itself is the interface (a future neural encoder's input,
#:                  or a debug/duplicate stream), not a fused scalar slice.
TRAIN_EVAL_BEHAVIORS = frozenset({"fixed", "trainable", "raw"})

#: Issue #32 classification of a stream's *role* for the neural agent, one
#: axis distinct from ``train_eval_behavior`` (encoder machinery):
#:  - "agent_input"  raw/near-raw sensory, proprioceptive, motor, reward and
#:                   interoceptive (`internal.*`) streams the policy should
#:                   actually consume.
#:  - "aux_debug"    hand-computed semantic summaries (world facts a player
#:                   effectively has access to, event narrations) useful for
#:                   dashboards/replay and as auxiliary-loss targets, but not
#:                   fed to the policy in the "raw input" profile.
#:  - "privileged"   exact ground-truth simulator state (unbounded-vocabulary
#:                   "_exact" mirrors of world facts) recorded for replay
#:                   fidelity only; excluded from both policy input and
#:                   auxiliary-loss targets so the agent can't read the answer.
STREAM_CLASSIFICATIONS = frozenset({"agent_input", "aux_debug", "privileged"})

#: Coarse relative cost of encoding a stream this tick, used by the attention
#: controller (#59) to weigh salience against compute budget.
ATTENTION_COMPUTE_COSTS = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class AttentionMetadata:
    """Per-stream metadata the attention controller (#59) scores against.

    Fields mirror the "Make Input Streams Explicit" list in
    `docs/neural-stream-agent.md`: modality, expected sample rate, relative
    compute cost of encoding, and whether the stream can carry a
    direction/region localization hint consumed by the orienting reflex
    (#60). This is plain data -- no attention *scoring* logic lives here.
    """

    modality: str
    expected_sample_rate_hz: Optional[float] = None
    relative_compute_cost: str = "low"
    localization_hint: bool = False

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("AttentionMetadata.modality must be non-empty")
        if self.relative_compute_cost not in ATTENTION_COMPUTE_COSTS:
            raise ValueError(
                f"invalid relative_compute_cost {self.relative_compute_cost!r}; "
                f"expected one of {sorted(ATTENTION_COMPUTE_COSTS)}"
            )
        if self.expected_sample_rate_hz is not None and self.expected_sample_rate_hz <= 0:
            raise ValueError(
                "expected_sample_rate_hz must be positive, got "
                f"{self.expected_sample_rate_hz!r}"
            )


@dataclass(frozen=True)
class StreamDeclaration:
    """One input stream's complete schema for the neural path.

    `pattern` matches stream ids the same way `StreamEncoderRegistry` does:
    an exact id (`"body.health"`) or a `"prefix.*"` glob (`"event.*"`).
    `checkpoint_key` is optional; when unset it is derived deterministically
    from the concrete stream id (`resolve_checkpoint_key`), so every stream
    gets one without hand-authoring a name for it.
    """

    pattern: str
    encoder_factory: Optional[Callable[[], StreamEncoder]]
    trainable: bool = False
    train_eval_behavior: str = "fixed"
    checkpoint_key: Optional[str] = None
    note: str = ""
    neural_encoder: Optional[str] = None
    neural_latent_width: Optional[int] = None
    #: Issue #32: agent input / aux-debug / privileged (excluded from agent
    #: input). Required -- there is no default, so every declaration must
    #: state its classification explicitly.
    classification: Optional[str] = None
    #: Required when ``classification == "agent_input"`` (#59/#60 consume
    #: this); optional otherwise.
    attention: Optional["AttentionMetadata"] = None

    def __post_init__(self) -> None:
        if self.train_eval_behavior not in TRAIN_EVAL_BEHAVIORS:
            raise ValueError(
                f"invalid train_eval_behavior {self.train_eval_behavior!r} for "
                f"{self.pattern!r}; expected one of {sorted(TRAIN_EVAL_BEHAVIORS)}"
            )
        if self.encoder_factory is None and self.train_eval_behavior not in ("raw", "trainable"):
            raise ValueError(
                f"{self.pattern!r} has no encoder_factory (no fusion-layout "
                f"slot) but train_eval_behavior={self.train_eval_behavior!r}; "
                "expected 'raw' or 'trainable'"
            )
        if self.trainable and self.train_eval_behavior != "trainable":
            raise ValueError(
                f"{self.pattern!r} is declared trainable but "
                f"train_eval_behavior={self.train_eval_behavior!r}; expected "
                "'trainable'"
            )
        if self.trainable and not self.neural_encoder:
            raise ValueError(f"{self.pattern!r} is trainable but has no neural_encoder")
        if self.neural_latent_width is not None and self.neural_latent_width <= 0:
            raise ValueError(
                f"{self.pattern!r} neural_latent_width must be positive, "
                f"got {self.neural_latent_width!r}"
            )
        if self.classification not in STREAM_CLASSIFICATIONS:
            raise ValueError(
                f"{self.pattern!r} has invalid classification {self.classification!r}; "
                f"expected one of {sorted(STREAM_CLASSIFICATIONS)} (issue #32)"
            )
        if self.classification == "agent_input" and self.attention is None:
            raise ValueError(
                f"{self.pattern!r} is classified agent_input but declares no "
                "AttentionMetadata (issue #32: every agent-input stream needs "
                "attention metadata for #59/#60)"
            )

    def encoder(self) -> Optional[StreamEncoder]:
        """A fresh encoder instance, or `None` for a "raw" (no fusion slot)
        declaration."""
        return self.encoder_factory() if self.encoder_factory is not None else None

    def is_fixed_stub(self) -> bool:
        """True for every non-trainable stream -- the target doc's
        "deliberate fixed stub" case, whether or not it has a fusion slot."""
        return not self.trainable

    def resolve_checkpoint_key(self, stream_id: str) -> str:
        return self.checkpoint_key or f"stream_encoder.{stream_id.replace('.', '_')}"


class StreamRegistry:
    """An ordered set of `StreamDeclaration`s; first pattern match wins."""

    def __init__(self, declarations: Optional[Iterable[StreamDeclaration]] = None) -> None:
        self._declarations: List[StreamDeclaration] = list(declarations or [])

    def extend(self, declarations: Iterable[StreamDeclaration]) -> "StreamRegistry":
        """A new registry with `declarations` appended after this one's --
        lower priority, and never overriding an existing pattern match."""
        return StreamRegistry([*self._declarations, *declarations])

    @property
    def declarations(self) -> Tuple[StreamDeclaration, ...]:
        """Every declaration in this registry, in match-priority order."""
        return tuple(self._declarations)

    def declaration_for(self, stream_id: str) -> Optional[StreamDeclaration]:
        for decl in self._declarations:
            if stream_matches(decl.pattern, stream_id):
                return decl
        return None

    def classification_for(self, stream_id: str) -> Optional[str]:
        decl = self.declaration_for(stream_id)
        return decl.classification if decl is not None else None

    def missing(self, catalog: Iterable[StreamSpec]) -> List[str]:
        """Stream ids in `catalog` with no matching declaration."""
        return sorted(
            spec.stream_id for spec in catalog if self.declaration_for(spec.stream_id) is None
        )

    def assert_complete(self, catalog: Iterable[StreamSpec]) -> None:
        catalog = list(catalog)
        missing = self.missing(catalog)
        if missing:
            raise ValueError(
                f"stream(s) missing a StreamDeclaration: {missing}; every input "
                "stream needs shape/rate/encoder/checkpoint/train-eval metadata "
                "declared in the registry (see docs/streams.md)"
            )

    def assert_attention_complete(self, catalog: Iterable[StreamSpec]) -> None:
        """Every catalog stream classified ``agent_input`` must declare
        `AttentionMetadata` (issue #32 acceptance: completeness test).

        `StreamDeclaration.__post_init__` already refuses to construct an
        `agent_input` declaration without attention metadata, so this can
        only fail if a catalog stream's *declaration* is missing entirely
        (call `assert_complete` first) or a caller inspects a registry that
        was assembled from raw dicts bypassing validation.
        """
        catalog = list(catalog)
        self.assert_complete(catalog)
        missing = sorted(
            spec.stream_id
            for spec in catalog
            if self.classification_for(spec.stream_id) == "agent_input"
            and self.declaration_for(spec.stream_id).attention is None  # type: ignore[union-attr]
        )
        if missing:
            raise ValueError(f"agent_input stream(s) missing AttentionMetadata: {missing}")

    def ids_by_classification(
        self, catalog: Iterable[StreamSpec], classification: str
    ) -> List[str]:
        """Catalog stream ids the registry classifies as `classification`."""
        if classification not in STREAM_CLASSIFICATIONS:
            raise ValueError(
                f"unknown classification {classification!r}; expected one of "
                f"{sorted(STREAM_CLASSIFICATIONS)}"
            )
        return sorted(
            spec.stream_id
            for spec in catalog
            if self.classification_for(spec.stream_id) == classification
        )

    def to_encoder_registry(
        self, classifications: Optional[Iterable[str]] = None
    ) -> StreamEncoderRegistry:
        """The plain `StreamEncoderRegistry` `TemporalFusion` builds its
        layout from: every declared pattern with a fusion-layout encoder, in
        declaration order (first match wins, same semantics as
        `StreamEncoderRegistry.encoder_for`).

        `classifications`, when given, restricts the result to declarations
        whose `classification` is in the set -- e.g. `{"agent_input"}` builds
        the "raw input" profile fusion registry (issue #32): the online
        policy's fused state then only reflects raw/near-raw and
        proprioceptive streams, while aux/debug and privileged streams stay
        published and recorded, just not fused into the policy's input.

        A declaration excluded by `classifications` still registers its
        pattern, with `None` in place of an encoder, so it keeps shadowing
        any *later, broader* pattern at its priority position (e.g.
        `spatial.distance_from_spawn` staying excluded rather than falling
        through to the following `spatial.*`) instead of silently leaking
        back in through the wildcard.
        """
        allowed: Optional[set] = None
        if classifications is not None:
            allowed = set(classifications)
            unknown = allowed - STREAM_CLASSIFICATIONS
            if unknown:
                raise ValueError(
                    f"unknown classification(s) {sorted(unknown)}; expected subset of "
                    f"{sorted(STREAM_CLASSIFICATIONS)}"
                )
        registry = StreamEncoderRegistry()
        for decl in self._declarations:
            if decl.encoder_factory is None:
                continue
            if allowed is not None and decl.classification not in allowed:
                registry.register(decl.pattern, None)
                continue
            registry.register(decl.pattern, decl.encoder())
        return registry

    def describe(self, catalog: Iterable[StreamSpec]) -> List[Dict[str, object]]:
        """Session-metadata-ready declarations for each stream in `catalog`,
        `stream_id`-ordered: shape/schema/rate from the `StreamSpec`, the rest
        from the matched `StreamDeclaration`."""
        out: List[Dict[str, object]] = []
        for spec in sorted(catalog, key=lambda s: s.stream_id):
            decl = self.declaration_for(spec.stream_id)
            encoder = decl.encoder() if decl is not None else None
            out.append(
                {
                    "stream_id": spec.stream_id,
                    "modality": spec.modality,
                    "shape": list(spec.shape) if spec.shape is not None else None,
                    "payload_schema": spec.payload_schema,
                    "nominal_rate_hz": spec.nominal_rate_hz,
                    "encoder": type(encoder).__name__ if encoder is not None else None,
                    "latent_width": encoder.width(spec) if encoder is not None else 0,
                    "neural_encoder": decl.neural_encoder if decl is not None else None,
                    "neural_latent_width": (
                        decl.neural_latent_width if decl is not None else None
                    ),
                    "trainable": decl.trainable if decl is not None else None,
                    "fixed_stub": decl.is_fixed_stub() if decl is not None else None,
                    "train_eval_behavior": decl.train_eval_behavior if decl is not None else None,
                    "checkpoint_key": (
                        decl.resolve_checkpoint_key(spec.stream_id) if decl is not None else None
                    ),
                    "classification": decl.classification if decl is not None else None,
                    "attention_modality": (
                        decl.attention.modality if decl is not None and decl.attention else None
                    ),
                    "attention_expected_sample_rate_hz": (
                        decl.attention.expected_sample_rate_hz
                        if decl is not None and decl.attention
                        else None
                    ),
                    "attention_relative_compute_cost": (
                        decl.attention.relative_compute_cost
                        if decl is not None and decl.attention
                        else None
                    ),
                    "attention_localization_hint": (
                        decl.attention.localization_hint
                        if decl is not None and decl.attention
                        else None
                    ),
                    "note": decl.note if decl is not None else "",
                }
            )
        return out


#: Generic, modality-shaped declarations shared by any Program. Order matters:
#: more specific patterns (exact ids) are listed before the wildcards they
#: would otherwise be shadowed by, matching the precedence
#: `default_encoder_registry()` has always had.
DEFAULT_STREAM_REGISTRY = StreamRegistry(
    [
        StreamDeclaration(
            "body.alive",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="low"),
            note="Alive flag; legacy fusion scalar-encoded, neural path uses BodyStateEncoder.",
        ),
        StreamDeclaration(
            "body.health",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Health vital; legacy fusion scalar-encoded, neural path uses BodyStateEncoder.",
        ),
        StreamDeclaration(
            "body.hotbar",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.EntityEncoder",
            neural_latent_width=16,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="medium"),
            note="Hotbar symbolic summary; legacy fusion scalar-encoded, neural path uses EntityEncoder.",
        ),
        StreamDeclaration(
            "body.hunger",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Hunger vital; legacy fusion scalar-encoded, neural path uses BodyStateEncoder.",
        ),
        StreamDeclaration(
            "body.in_water",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="low"),
            note="In-water flag; legacy fusion scalar-encoded, neural path uses BodyStateEncoder.",
        ),
        StreamDeclaration(
            "body.inventory",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.EntityEncoder",
            neural_latent_width=16,
            classification="agent_input",
            attention=AttentionMetadata(modality="body", relative_compute_cost="medium"),
            note="Inventory summary; legacy fusion scalar-encoded, neural path uses EntityEncoder.",
        ),
        StreamDeclaration(
            "body.oxygen",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.BodyStateEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="body", expected_sample_rate_hz=1.0, relative_compute_cost="low"
            ),
            note="Oxygen vital; legacy fusion scalar-encoded, neural path uses BodyStateEncoder.",
        ),
        StreamDeclaration(
            "reward.*",
            ScalarEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.RewardEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="reward", expected_sample_rate_hz=20.0, relative_compute_cost="low"
            ),
            note="Reward scalar(s); legacy fusion scalar-encoded, neural path uses RewardEncoder.",
        ),
        # A distance is one number, not a pose: this exact id must be checked
        # before the "spatial.*" pose pattern below (first match wins).
        StreamDeclaration(
            "spatial.distance_from_spawn",
            ScalarEncoder,
            classification="aux_debug",
            note=(
                "Scalar distance, not a pose. Requires knowing the absolute "
                "spawn point -- a hand-computed reward-shaping convenience, "
                "not a raw proprioceptive signal (issue #32)."
            ),
        ),
        StreamDeclaration(
            "spatial.*",
            SpatialEncoder,
            classification="agent_input",
            attention=AttentionMetadata(modality="spatial", relative_compute_cost="low"),
            note=(
                "Position/rotation pose streams; proprioception (the agent's own "
                "embodiment), not a privileged world fact (issue #32)."
            ),
        ),
        StreamDeclaration(
            "vision.frame.grid",
            GridVisionEncoder,
            classification="aux_debug",
            note=(
                "Coarse semantic grid frame: hand-classified per-cell tags "
                "(solid/water/resource/entity/agent), not raw pixels -- a "
                "debug/aux-loss target, demoted from agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "vision.entities",
            EntityEncoder,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.EntityEncoder",
            neural_latent_width=16,
            classification="aux_debug",
            note=(
                "Visible entities; legacy fusion uses fixed summary, neural path "
                "uses EntityEncoder. Ground-truth id/distance/angle without vision "
                "processing -- an object-detection aux-loss target, not raw "
                "agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "vision.frame.pixels",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="agent_input",
            attention=AttentionMetadata(
                modality="vision",
                expected_sample_rate_hz=20.0,
                relative_compute_cost="high",
                localization_hint=True,
            ),
            note=(
                "Raw RGB pixel tensor. Deliberate fixed stub: no encoder is "
                "bound in the legacy scalar TemporalFusion layout -- it is "
                "reserved for a trainable PixelStreamEncoder "
                "(docs/neural-stream-agent.md Phase B, cognitive_runtime.neural). "
                "The one raw-ish stream; a frame region maps to a look direction, "
                "so it is the only stream with a localization hint (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.action_rejected", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.block_broken", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.block_placed", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.damage_taken", EventEncoder, classification="aux_debug",
            note=(
                "Semantic event mark (the 'why'); the felt effect is "
                "body.health, which is agent input -- this narration is an "
                "aux-loss target (issue #32)."
            ),
        ),
        StreamDeclaration(
            "event.died", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.entered_shelter", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.food_eaten", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.item_collected", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "event.survived_night", EventEncoder, classification="aux_debug",
            note="Semantic event mark; narration/aux-loss target, not raw agent input (issue #32).",
        ),
        StreamDeclaration(
            "world.front_block", CategoryEncoder, classification="aux_debug",
            note=(
                "Faced-block category; hand-written survival-heuristic sense, "
                "kept for debugging/aux loss, not raw agent input (issue #32)."
            ),
        ),
        StreamDeclaration(
            "world.sheltered", ScalarEncoder, classification="aux_debug",
            note=(
                "Shelter flag; hand-written survival-heuristic sense, kept for "
                "debugging/aux loss, not raw agent input (issue #32)."
            ),
        ),
        # Reserved ids (docs/streams.md's modality table): no Program
        # publishes these yet. Declaring them now reserves the convention
        # ahead of the target input set (docs/neural-stream-agent.md
        # "Make Input Streams Explicit"): audio, keyboard, mouse/look and
        # internal modulation.
        StreamDeclaration(
            "audio.*",
            encoder_factory=None,
            train_eval_behavior="raw",
            neural_encoder="cognitive_runtime.neural.AudioEncoder",
            neural_latent_width=8,
            classification="agent_input",
            attention=AttentionMetadata(modality="audio", relative_compute_cost="medium"),
            note=(
                "Reserved audio stream ids. Deliberate fixed neural stub: no "
                "Program publishes audio yet and no capture backend exists."
            ),
        ),
        StreamDeclaration(
            "input.keypress",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="agent_input",
            attention=AttentionMetadata(modality="input", relative_compute_cost="low"),
            note=(
                "Reserved id: no Program publishes raw keyboard input yet "
                "(target input: keyboard/control history)."
            ),
        ),
        StreamDeclaration(
            "input.mouse_look",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="agent_input",
            attention=AttentionMetadata(
                modality="motor", expected_sample_rate_hz=20.0, relative_compute_cost="low"
            ),
            note=(
                "Mouse/look control history (issue #32): the Minecraft "
                "SurvivalBox now publishes {d_yaw, d_pitch} every tick from the "
                "LOOK_* action taken (adapter.py); a near-raw motor stream, "
                "deliberately still 'raw' (no fusion slot / encoder) pending a "
                "dedicated encoder."
            ),
        ),
        StreamDeclaration(
            "internal.*",
            encoder_factory=None,
            train_eval_behavior="raw",
            classification="agent_input",
            attention=AttentionMetadata(
                modality="internal", expected_sample_rate_hz=20.0, relative_compute_cost="low"
            ),
            note=(
                "Interoceptive modulation streams (prediction error, "
                "reward-prediction error, learning progress, novelty, risk; "
                "issue #58): computed by core.modulation and published by "
                "runtime.loop every cognitive tick, not part of any "
                "Program's stream catalog. Interoception is agent input "
                "(issue #32), even though it is the runtime's own signal "
                "rather than external sensing."
            ),
        ),
        StreamDeclaration(
            "motor.history",
            encoder_factory=None,
            trainable=True,
            train_eval_behavior="trainable",
            neural_encoder="cognitive_runtime.neural.MotorHistoryEncoder",
            neural_latent_width=16,
            classification="agent_input",
            attention=AttentionMetadata(
                modality="motor", expected_sample_rate_hz=20.0, relative_compute_cost="low"
            ),
            note=(
                "Reserved id: recent-action history is today computed ad hoc "
                "by training/features.py:motor_history_features from the "
                "recorded motor log, not published as a sensory stream. "
                "Neural path uses MotorHistoryEncoder; parity mode reproduces "
                "the one-hot baseline exactly."
            ),
        ),
    ]
)
