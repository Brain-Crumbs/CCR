"""Crafter stream catalog and publisher.

Maps CrafterWorld onto the generic stream taxonomy (mirrors
``programs.minecraft.streams``'s pattern): a real RGB pixel frame the
environment itself renders (not a synthetic colorized proxy), an egocentric
semantic-grid crop for the generic vision-grid encoder, body vitals, an
inventory summary, and achievement/event streams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognitive_runtime.core.goal import GOAL_STREAM, GOAL_STREAM_SPEC, GoalState
from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.pacer import RatePacer

#: Crafter's *raw* semantic material/object vocabulary:
#: ``crafter.constants.materials`` (ids 1..12, in declaration order) plus
#: ``engine.SemanticView``'s tracked object types (ids 13..18); id 0 is
#: void/out-of-bounds.  This is adapter metadata, not the vocabulary exposed
#: to the world model.  It stays hardcoded so this module -- and the generic
#: vision-grid encoder it feeds -- remains importable without ``crafter``.
#: ``tests/test_crafter_world.py`` checks it against the live package.
SEMANTIC_LEGEND_NAMES: Dict[int, str] = {
    0: "void", 1: "water", 2: "grass", 3: "stone", 4: "path", 5: "sand",
    6: "tree", 7: "lava", 8: "coal", 9: "iron", 10: "diamond",
    11: "table", 12: "furnace",
    13: "player", 14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow", 18: "plant",
}

_HAZARD = {"lava"}
_RESOURCE = {"tree", "coal", "iron", "diamond", "plant"}
_SOLID = {"stone", "table", "furnace"}
_HOSTILE = {"zombie", "skeleton", "arrow"}


def _legend_class(name: str) -> str:
    """Raw Crafter material/object -> compact action-relevant class.

    The world model should learn to act on geometry and affordances, not
    memorize Crafter's renderer-level material ids.  These six categories
    preserve the distinctions needed for local control: traversable space,
    water, blocking/hazardous terrain, gatherable resources, entities, and the
    agent.  Crafting recipes and item counts belong in body/inventory state,
    not in the visual semantic target.
    """
    if name == "player":
        return "agent"
    if name in _HOSTILE or name == "cow":
        return "entity"
    if name == "water":
        return "water"
    if name in _RESOURCE:
        return "resource"
    if name in _SOLID or name in _HAZARD:
        return "solid"
    return "ground"  # walkable terrain, or void: open, never blocks a view


#: The compact vocabulary emitted in ``vision.frame.grid`` and supervised by
#: the semantic decoder.  These ids are deliberately contiguous so they can
#: be used directly as cross-entropy targets.
SEMANTIC_CLASSES: Tuple[str, ...] = (
    "ground", "water", "solid", "resource", "entity", "agent",
)
#: Bump whenever an existing compact id changes meaning.  Nursery recording
#: caches use this to avoid training a new decoder against stale raw grids.
SEMANTIC_VOCABULARY_VERSION = "crafter-action-semantics-v1"
SEMANTIC_CLASS_IDS: Dict[str, int] = {
    name: index for index, name in enumerate(SEMANTIC_CLASSES)
}

#: Frame cell id -> compact class tag.  Unlike ``SEMANTIC_LEGEND_NAMES``, this
#: is the model vocabulary, not Crafter's raw simulator vocabulary.
FRAME_LEGEND: Dict[int, str] = dict(enumerate(SEMANTIC_CLASSES))

#: Raw Crafter semantic id -> compact model semantic id.
RAW_TO_SEMANTIC_ID: Dict[int, int] = {
    raw_id: SEMANTIC_CLASS_IDS[_legend_class(name)]
    for raw_id, name in SEMANTIC_LEGEND_NAMES.items()
}

VISION_STREAM = "vision.frame.grid"
PIXEL_STREAM = "vision.frame.pixels"
BODY_HEARTBEAT_KEY = "body.heartbeat"
#: Republish body vitals unchanged every this many ticks (matches
#: ``programs.minecraft.streams.BODY_HEARTBEAT_TICKS`` at the same nominal
#: 20 ticks/sec convention), so subscribers can distinguish "silent because
#: unchanged" from "silent because dead sensor".
BODY_HEARTBEAT_TICKS = 20
BODY_HEARTBEAT_HZ = 1.0

VITAL_RANGE = (0.0, 9.0)  # health/food/drink/energy scale


def crop_semantic_grid(
    semantic: np.ndarray, position: Tuple[int, int], radius: int
) -> List[List[int]]:
    """Return a compact semantic crop centered on the player.

    The crop is clamped at the world edge (Crafter has no wraparound).  Raw
    simulator ids are converted at this boundary so recordings, the fused
    grid encoder, and semantic-decoder targets all share the same small,
    action-relevant vocabulary.
    """
    x, y = position
    w, h = semantic.shape
    out: List[List[int]] = []
    for dx in range(-radius, radius + 1):
        row = []
        for dy in range(-radius, radius + 1):
            cx = min(max(x + dx, 0), w - 1)
            cy = min(max(y + dy, 0), h - 1)
            raw_id = int(semantic[cx, cy])
            # Unknown values are treated as ground rather than creating an
            # out-of-vocabulary target when a future Crafter release adds a
            # decorative material.
            row.append(RAW_TO_SEMANTIC_ID.get(raw_id, SEMANTIC_CLASS_IDS["ground"]))
        out.append(row)
    return out


def build_crafter_stream_specs(
    *,
    grid_radius: int,
    pixel_shape: Tuple[int, int, int],
    world_size: float,
    vision_hz: float = 20.0,
    heartbeat_hz: float = BODY_HEARTBEAT_HZ,
) -> List[StreamSpec]:
    """The Crafter catalog, with encoder metadata (ranges/legend).

    ``vision_hz``/``heartbeat_hz`` let a realtime run declare the rates its
    pacer actually publishes at, matching ``build_survival_stream_specs``'s
    convention.
    """
    grid_side = 2 * grid_radius + 1
    return [
        StreamSpec(VISION_STREAM, "vision", "Egocentric compact action-semantic grid crop.",
                   nominal_rate_hz=vision_hz,
                   payload_schema=f"{grid_side}x{grid_side} 6-class int grid",
                   legend=FRAME_LEGEND),
        StreamSpec(PIXEL_STREAM, "vision",
                   "RGB camera frame Crafter renders natively -- real pixel provenance, "
                   "not a synthetic colorized proxy.",
                   nominal_rate_hz=vision_hz,
                   payload_schema=f"{pixel_shape[0]}x{pixel_shape[1]}x3 uint8 image",
                   range=(0.0, 255.0), shape=pixel_shape, overflow="coalesce"),
        StreamSpec("body.health", "body", "Health, on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..9", range=VITAL_RANGE, neutral=9.0),
        StreamSpec("body.food", "body", "Food, on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..9", range=VITAL_RANGE, neutral=9.0),
        StreamSpec("body.drink", "body", "Drink (thirst), on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..9", range=VITAL_RANGE, neutral=9.0),
        StreamSpec("body.energy", "body", "Energy (sleep), on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..9", range=VITAL_RANGE, neutral=9.0),
        StreamSpec("body.inventory", "body", "Resource/tool counts, on change.",
                   payload_schema="{item: count}"),
        StreamSpec("body.sleeping", "body", "Sleeping flag, on change.",
                   payload_schema="bool"),
        StreamSpec("body.alive", "body", "Alive flag; flips once on death.",
                   payload_schema="bool", neutral=1.0),
        StreamSpec("spatial.position", "spatial", "Agent grid position, on change.",
                   payload_schema="{x, y}", range=(0.0, world_size), neutral=world_size / 2.0),
        StreamSpec("spatial.facing", "spatial",
                   "Agent facing direction, on change -- a discrete grid flip "
                   "((-1,0)/(1,0)/(0,-1)/(0,1)), not a continuous yaw; updates on every "
                   "directional move attempt, even one blocked by terrain.",
                   payload_schema="{x, y}"),
        StreamSpec("event.achievement", "event",
                   "Achievement counter incremented (repeatable per episode, unlike "
                   "Minecraft's once-only event.advancement).",
                   payload_schema='{"id": str, "count": int}'),
        StreamSpec("event.died", "event", payload_schema='{"reason": str|null}'),
        StreamSpec("event.action_rejected", "event", payload_schema='{"reason": str}'),
        StreamSpec("reward.scalar", "reward",
                   "Crafter reward: health delta plus a one-time bonus per newly "
                   "unlocked achievement.",
                   nominal_rate_hz=20.0,
                   payload_schema='{"value": float, "components": dict}'),
        GOAL_STREAM_SPEC,
    ]


class CrafterStreamPublisher:
    def __init__(
        self,
        bus: SensoryStreamBus,
        source: str = "",
        pacer: Optional[RatePacer] = None,
    ):
        self._bus = bus
        self._delta = DeltaPublisher(bus)
        self._source = source
        #: Disabled by default (fast-forward): every-tick/heartbeat cadence.
        self._pacer = pacer if pacer is not None else RatePacer(enabled=False)

    def reset(self) -> None:
        self._delta.reset()
        self._pacer.reset()

    def publish_tick(
        self,
        tick: int,
        state: Dict[str, Any],
        pixels: np.ndarray,
        timestamp: float,
        achievement_events: List[Tuple[str, int]],
        reward_signal: Optional[Any] = None,
        died: bool = False,
        paced: bool = True,
        goal_state: Optional[GoalState] = None,
    ) -> List[StreamEvent]:
        """Publish this tick's streams from the current state snapshot.

        With ``paced`` the realtime pacer gates vision frames and the body
        heartbeat to their wall-clock rates; ``paced=False`` bypasses it so
        the initial post-reset snapshot always publishes every stream (no
        subscriber starts blind).
        """
        published: List[StreamEvent] = []

        def pub(stream_id: str, payload: Any, force: bool = False) -> None:
            event = self._delta.publish(
                stream_id, payload, timestamp, force=force, source=self._source
            )
            if event is not None:
                published.append(event)

        if paced and self._pacer.enabled:
            show_frame = self._pacer.should_publish(VISION_STREAM, now=timestamp)
            heartbeat = self._pacer.should_publish(BODY_HEARTBEAT_KEY, now=timestamp)
        else:
            show_frame = True
            heartbeat = tick % BODY_HEARTBEAT_TICKS == 0

        if show_frame:
            pub(VISION_STREAM, state["grid"], force=True)
            pub(PIXEL_STREAM, pixels, force=True)
        pub("body.health", state["health"], force=heartbeat)
        pub("body.food", state["food"], force=heartbeat)
        pub("body.drink", state["drink"], force=heartbeat)
        pub("body.energy", state["energy"], force=heartbeat)
        pub("body.inventory", state["inventory"])
        pub("body.sleeping", state["sleeping"])
        pub("body.alive", state["alive"])
        pub("spatial.position", state["position"])
        pub("spatial.facing", state["facing"])
        if goal_state is not None:
            # Every tick, not on-change: `distance`/`relative_position` move
            # with the agent even while `active`/`source` stay constant, so
            # a delta publisher would republish nearly every tick anyway --
            # `force=True` makes that explicit rather than relying on
            # incidental float inequality.
            pub(GOAL_STREAM, goal_state.as_payload(), force=True)

        for name, count in achievement_events:
            published.append(
                self._bus.publish(
                    "event.achievement", {"id": name, "count": count}, timestamp,
                    source=self._source,
                )
            )
        if died:
            published.append(
                self._bus.publish(
                    "event.died", {"reason": "health"}, timestamp, source=self._source
                )
            )
        if reward_signal is not None:
            published.append(
                self._bus.publish(
                    "reward.scalar",
                    {"value": reward_signal.value, "components": dict(reward_signal.components)},
                    timestamp,
                    source=self._source,
                )
            )
        return published
