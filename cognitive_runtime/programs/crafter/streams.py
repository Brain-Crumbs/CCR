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

from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.pacer import RatePacer

#: Crafter's semantic material/object vocabulary: ``crafter.constants.materials``
#: (ids 1..12, in declaration order) plus ``engine.SemanticView``'s tracked
#: object types (ids 13..18); id 0 is void/out-of-bounds. Hardcoded (not
#: imported) so this module -- and the generic vision-grid encoder it feeds
#: -- stays importable without the optional ``crafter`` package installed;
#: ``tests/test_crafter_world.py`` checks this against the live package.
SEMANTIC_LEGEND_NAMES: Dict[int, str] = {
    0: "void", 1: "water", 2: "grass", 3: "stone", 4: "path", 5: "sand",
    6: "tree", 7: "lava", 8: "coal", 9: "iron", 10: "diamond",
    11: "table", 12: "furnace",
    13: "player", 14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow", 18: "plant",
}

_WALKABLE = {"grass", "path", "sand", "void"}
_HAZARD = {"lava"}
_RESOURCE = {"tree", "coal", "iron", "diamond", "plant"}
_SOLID = {"stone", "table", "furnace"}
_HOSTILE = {"zombie", "skeleton", "arrow"}


def _legend_class(name: str) -> str:
    """Cell name -> generic class tag the ``GridVisionEncoder`` pools on
    (issue #32's shared vocabulary): solid/water/resource/entity/agent/ground."""
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


#: Frame cell id -> generic class tag, so the vision encoder stays generic.
FRAME_LEGEND: Dict[int, str] = {
    code: _legend_class(name) for code, name in SEMANTIC_LEGEND_NAMES.items()
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
    """A ``(2*radius+1)`` square, egocentric crop of Crafter's full-world
    semantic grid, clamped at the world edge (Crafter has no wraparound;
    mirrors ``programs.minecraft.world.SimulatedWorld.render_frame``'s
    clamped patch)."""
    x, y = position
    w, h = semantic.shape
    out: List[List[int]] = []
    for dx in range(-radius, radius + 1):
        row = []
        for dy in range(-radius, radius + 1):
            cx = min(max(x + dx, 0), w - 1)
            cy = min(max(y + dy, 0), h - 1)
            row.append(int(semantic[cx, cy]))
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
        StreamSpec(VISION_STREAM, "vision", "Egocentric local semantic-grid crop.",
                   nominal_rate_hz=vision_hz, payload_schema=f"{grid_side}x{grid_side} int grid",
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
