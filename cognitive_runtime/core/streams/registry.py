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
from typing import Callable, Dict, Iterable, List, Optional

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

    def __post_init__(self) -> None:
        if self.train_eval_behavior not in TRAIN_EVAL_BEHAVIORS:
            raise ValueError(
                f"invalid train_eval_behavior {self.train_eval_behavior!r} for "
                f"{self.pattern!r}; expected one of {sorted(TRAIN_EVAL_BEHAVIORS)}"
            )
        if self.encoder_factory is None and self.train_eval_behavior != "raw":
            raise ValueError(
                f"{self.pattern!r} has no encoder_factory (no fusion-layout "
                f"slot) but train_eval_behavior={self.train_eval_behavior!r}; "
                "expected 'raw'"
            )
        if self.trainable and self.train_eval_behavior != "trainable":
            raise ValueError(
                f"{self.pattern!r} is declared trainable but "
                f"train_eval_behavior={self.train_eval_behavior!r}; expected "
                "'trainable'"
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

    def declaration_for(self, stream_id: str) -> Optional[StreamDeclaration]:
        for decl in self._declarations:
            if stream_matches(decl.pattern, stream_id):
                return decl
        return None

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

    def to_encoder_registry(self) -> StreamEncoderRegistry:
        """The plain `StreamEncoderRegistry` `TemporalFusion` builds its
        layout from: every declared pattern with a fusion-layout encoder, in
        declaration order (first match wins, same semantics as
        `StreamEncoderRegistry.encoder_for`)."""
        registry = StreamEncoderRegistry()
        for decl in self._declarations:
            if decl.encoder_factory is not None:
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
                    "trainable": decl.trainable if decl is not None else None,
                    "fixed_stub": decl.is_fixed_stub() if decl is not None else None,
                    "train_eval_behavior": decl.train_eval_behavior if decl is not None else None,
                    "checkpoint_key": (
                        decl.resolve_checkpoint_key(spec.stream_id) if decl is not None else None
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
        StreamDeclaration("body.alive", ScalarEncoder, note="Alive flag, scalar-encoded."),
        StreamDeclaration("body.health", ScalarEncoder, note="Health vital."),
        StreamDeclaration("body.hotbar", ScalarEncoder, note="Hotbar summary."),
        StreamDeclaration("body.hunger", ScalarEncoder, note="Hunger vital."),
        StreamDeclaration("body.in_water", ScalarEncoder, note="In-water flag."),
        StreamDeclaration("body.inventory", ScalarEncoder, note="Inventory summary."),
        StreamDeclaration("body.oxygen", ScalarEncoder, note="Oxygen vital."),
        StreamDeclaration("reward.*", ScalarEncoder, note="Reward scalar(s)."),
        # A distance is one number, not a pose: this exact id must be checked
        # before the "spatial.*" pose pattern below (first match wins).
        StreamDeclaration(
            "spatial.distance_from_spawn", ScalarEncoder, note="Scalar distance, not a pose."
        ),
        StreamDeclaration("spatial.*", SpatialEncoder, note="Position/rotation pose streams."),
        StreamDeclaration("vision.frame.grid", GridVisionEncoder, note="Coarse semantic grid frame."),
        StreamDeclaration("vision.entities", EntityEncoder, note="Visible-entity summary."),
        StreamDeclaration(
            "vision.frame.pixels",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "Raw RGB pixel tensor. Deliberate fixed stub: no encoder is "
                "bound in the legacy scalar TemporalFusion layout -- it is "
                "reserved for a trainable PixelStreamEncoder "
                "(docs/neural-stream-agent.md Phase B, cognitive_runtime.neural)."
            ),
        ),
        StreamDeclaration("event.action_rejected", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.block_broken", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.block_placed", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.damage_taken", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.died", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.entered_shelter", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.food_eaten", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.item_collected", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("event.survived_night", EventEncoder, note="Semantic event mark."),
        StreamDeclaration("world.front_block", CategoryEncoder, note="Faced-block category."),
        StreamDeclaration("world.sheltered", ScalarEncoder, note="Shelter flag."),
        # Reserved ids (docs/streams.md's modality table): no Program
        # publishes these yet. Declaring them now reserves the convention
        # ahead of the target input set (docs/neural-stream-agent.md
        # "Make Input Streams Explicit"): audio, keyboard, and mouse/look.
        StreamDeclaration(
            "audio.ambient",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Reserved id: no Program publishes audio yet (target input: audio).",
        ),
        StreamDeclaration(
            "input.keypress",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "Reserved id: no Program publishes raw keyboard input yet "
                "(target input: keyboard/control history)."
            ),
        ),
        StreamDeclaration(
            "input.mouse_look",
            encoder_factory=None,
            train_eval_behavior="raw",
            note="Reserved id: no Program publishes mouse/look controls yet (target input: mouse/look).",
        ),
        StreamDeclaration(
            "motor.history",
            encoder_factory=None,
            train_eval_behavior="raw",
            note=(
                "Reserved id: recent-action history is today computed ad hoc "
                "by training/features.py:motor_history_features from the "
                "recorded motor log, not published as a sensory stream. "
                "Reserved for when it becomes a real input stream "
                "(target input: keyboard/control/motor history)."
            ),
        ),
    ]
)
