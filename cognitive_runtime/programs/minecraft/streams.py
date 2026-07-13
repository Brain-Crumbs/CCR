"""SurvivalBox stream catalog and publisher.

Maps the survival world onto the generic stream taxonomy with **native
cadences** — the point of streams is that not everything publishes every
tick: vision does, body vitals publish on change plus a heartbeat, spatial
and world state publish on change, semantic events are irregular.

The publisher works from the legacy Observation built by the backend, so
stream payloads are exactly the values (and rounding) the pull-style path
exposes — determinism and parity come for free, and any `SurvivalBackend`
works unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.pacer import RatePacer
from cognitive_runtime.programs.minecraft.world import (
    AGENT_FRAME_ID,
    BLOCK_IDS,
    BREAK_YIELD,
    LOOK_STEP_DEG,
    MOB_FRAME_ID,
    PITCH_STEP_DEG,
    PIXEL_RADIUS,
    PIXEL_SCALE,
    SOLID,
)

#: Body vitals republish unchanged values every this many ticks, so
#: subscribers can distinguish "silent because unchanged" from "silent
#: because dead sensor".  At 20 tps this is a 1 Hz heartbeat, which is also
#: the vitals' nominal rate for stale-stream detection.
BODY_HEARTBEAT_TICKS = 20
BODY_HEARTBEAT_HZ = 1.0

#: Pacer keys: the vision frame is rate-paced in realtime, and body vitals
#: share one heartbeat token (they beat together).  The RGB pixel frame beats
#: with the grid frame (same vision cadence).
VISION_STREAM = "vision.frame.grid"
PIXEL_STREAM = "vision.frame.pixels"
BODY_HEARTBEAT_KEY = "body.heartbeat"

#: RGB pixel-frame dimensions (H, W, C), derived from the world's render geometry.
PIXEL_SHAPE = ((2 * PIXEL_RADIUS + 1) * PIXEL_SCALE, (2 * PIXEL_RADIUS + 1) * PIXEL_SCALE, 3)

VITAL_RANGE = (0.0, 20.0)  # health/hunger/oxygen scale

#: Mouse/look control history (issue #32): the {d_yaw, d_pitch} commanded by
#: this tick's LOOK_* action, or (0.0, 0.0) for every other action -- a
#: near-raw motor stream, distinct from `spatial.rotation` (the resulting
#: absolute pose). Same magnitudes the sim and the mineflayer bridge both
#: apply (`world.py` / `bridge/mineflayer/actions.js`), so the stream means
#: the same thing on either backend without any bridge changes.
MOUSE_LOOK_STREAM = "input.mouse_look"
LOOK_ACTION_DELTAS: Dict[str, Dict[str, float]] = {
    "LOOK_LEFT": {"d_yaw": -LOOK_STEP_DEG, "d_pitch": 0.0},
    "LOOK_RIGHT": {"d_yaw": LOOK_STEP_DEG, "d_pitch": 0.0},
    "LOOK_UP": {"d_yaw": 0.0, "d_pitch": -PITCH_STEP_DEG},
    "LOOK_DOWN": {"d_yaw": 0.0, "d_pitch": PITCH_STEP_DEG},
}
NULL_MOUSE_LOOK: Dict[str, float] = {"d_yaw": 0.0, "d_pitch": 0.0}


def mouse_look_delta(action_name: str) -> Dict[str, float]:
    """The {d_yaw, d_pitch} commanded by `action_name`, or zero for the rest."""
    return dict(LOOK_ACTION_DELTAS.get(action_name, NULL_MOUSE_LOOK))


def _entity_bearing_payload(mobs: List[Dict[str, float]]) -> Dict[str, Any]:
    """The orienting reflex's stimulus localization contract (issue #60):
    the nearest visible entity's salience (closer = higher) and bearing
    (signed degrees, positive = right, matching `world.mob_summary`'s
    `angle`), in the generic `{"value", "direction": {"bearing_deg"}}`
    shape `core.attention._direction` reads off any `localization_hint`
    stream. `mobs` is already nearest-first (`mob_summary`), so the first
    entry is the one to orient toward."""
    if not mobs:
        return {"value": 0.0, "direction": None}
    nearest = mobs[0]
    salience = round(1.0 / (1.0 + max(float(nearest["distance"]), 0.0)), 4)
    return {"value": salience, "direction": {"bearing_deg": nearest["angle"]}}


#: Naturally harvestable blocks (yield an item, not player-placed) map to the
#: generic "resource" class so the vision encoder senses resource density.
_RESOURCE_BLOCKS = set(BREAK_YIELD) - {"placed_block"}


def _frame_legend() -> Dict[int, str]:
    """Frame cell id -> generic class tag, so the vision encoder stays generic."""
    legend: Dict[int, str] = {}
    for name, code in BLOCK_IDS.items():
        if name == "water":
            tag = "water"
        elif name in _RESOURCE_BLOCKS:
            tag = "resource"
        elif name in SOLID:
            tag = "solid"
        else:
            tag = "ground"
        legend[code] = tag
    legend[MOB_FRAME_ID] = "entity"
    legend[AGENT_FRAME_ID] = "agent"
    return legend


#: Vocabulary the categorical encoders one-hot against (block-name payloads).
FRONT_BLOCK_CATEGORIES = tuple(sorted(BLOCK_IDS))
FRAME_LEGEND = _frame_legend()


def build_survival_stream_specs(
    world_size: int = 64,
    *,
    vision_hz: float = 20.0,
    heartbeat_hz: float = BODY_HEARTBEAT_HZ,
    bounded_position: bool = True,
) -> List[StreamSpec]:
    """The survival catalog, with encoder metadata (ranges/legend/categories).

    Position ranges depend on the world size, so the catalog is built per
    config rather than as a bare constant.  ``vision_hz``/``heartbeat_hz``
    let a realtime run declare the rates its pacer actually publishes at
    (``SurvivalBoxConfig.realtime_vision_hz`` / ``realtime_body_heartbeat_hz``)
    instead of the fast-forward tick cadence.  ``bounded_position=False``
    (remote backend) drops the ``[0, world_size]`` position/distance ranges --
    a live server's absolute coordinates aren't bounded by the config.
    """
    pos_range = (0.0, float(world_size)) if bounded_position else None
    pos_neutral = world_size / 2.0 if bounded_position else 0.0
    return [
        StreamSpec("vision.frame.grid", "vision", "Coarse top-down frame.",
                   nominal_rate_hz=vision_hz, payload_schema="11x11 int grid",
                   legend=FRAME_LEGEND),
        StreamSpec(PIXEL_STREAM, "vision",
                   "RGB camera frame: first-person viewer pixels on the remote "
                   "backend when available, deterministic colorized proxy in the sim.",
                   nominal_rate_hz=vision_hz,
                   payload_schema=f"{PIXEL_SHAPE[0]}x{PIXEL_SHAPE[1]}x3 uint8 image",
                   range=(0.0, 255.0), shape=PIXEL_SHAPE, overflow="coalesce"),
        StreamSpec("vision.entities", "vision",
                   "Visible mobs (id/distance/angle), every tick while any are visible; "
                   "occluded by walls (line-of-sight), not just distance.",
                   payload_schema="[{id, distance, angle}]", range=(0.0, 16.0)),
        StreamSpec("world.entity_bearing", "world",
                   "Nearest visible entity's salience + bearing, for the orienting "
                   "reflex's stimulus localization contract (issue #60): 0.0/no "
                   "direction when nothing is visible.",
                   payload_schema='{"value": float, "direction": {"bearing_deg": float}|null}',
                   range=(0.0, 1.0)),
        StreamSpec("body.health", "body", "Health, on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.hunger", "body", "Hunger, on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.oxygen", "body", "Oxygen, on change + heartbeat.",
                   nominal_rate_hz=heartbeat_hz,
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.inventory", "body", "Inventory summary, on change.",
                   payload_schema="{item: count}"),
        StreamSpec("body.inventory_exact", "body",
                   "Exact Minecraft inventory names, on change.",
                   payload_schema="{minecraft_item_name: count}"),
        StreamSpec("body.hotbar", "body", "Hotbar slots + selected, on change.",
                   payload_schema='{"slots": [item|null], "selected": int}'),
        StreamSpec("body.inventory_open", "body",
                   "Inventory-open flag (OPEN_INVENTORY/CLOSE_INVENTORY), on change.",
                   payload_schema="bool"),
        StreamSpec("body.in_water", "body", "In-water flag, on change.",
                   payload_schema="bool"),
        StreamSpec("body.alive", "body", "Alive flag; flips once on death.",
                   payload_schema="bool", neutral=1.0),
        StreamSpec("spatial.position", "spatial", "Agent position, on change.",
                   payload_schema="{x, y, z}", range=pos_range, neutral=pos_neutral),
        StreamSpec("spatial.rotation", "spatial", "Agent view direction, on change.",
                   payload_schema="{yaw, pitch}"),
        StreamSpec("spatial.distance_from_spawn", "spatial",
                   "Distance from the spawn point, on change.",
                   payload_schema="float",
                   range=(0.0, float(world_size)) if bounded_position else None),
        StreamSpec("world.time", "world", "Day/night clock, every tick.",
                   nominal_rate_hz=20.0,
                   payload_schema="{time_of_day, day_length, is_night}"),
        StreamSpec("world.biome", "world", "Biome under the agent, on change.",
                   payload_schema="str"),
        StreamSpec("world.nearby_blocks", "world", "5x5 block patch, on cell change.",
                   payload_schema="5x5 str grid"),
        StreamSpec("world.nearby_blocks_exact", "world",
                   "5x5 exact Minecraft block-name patch, on cell change.",
                   payload_schema="5x5 str grid"),
        StreamSpec("world.front_block", "world", "Block faced, on change.",
                   payload_schema="str", categories=FRONT_BLOCK_CATEGORIES),
        StreamSpec("world.front_block_exact", "world",
                   "Exact Minecraft block faced, on change.",
                   payload_schema="str"),
        StreamSpec("world.sheltered", "world", "Shelter state, on change.",
                   payload_schema="bool"),
        StreamSpec(MOUSE_LOOK_STREAM, "input",
                   "Mouse/look control history: the LOOK_* delta commanded this tick "
                   "(zero for every other action), every tick.",
                   nominal_rate_hz=20.0,
                   payload_schema='{"d_yaw": float, "d_pitch": float}'),
        StreamSpec("event.damage_taken", "event", payload_schema='{"reason": str}'),
        StreamSpec("event.item_collected", "event", payload_schema='{"item": str}'),
        StreamSpec("event.item_collected_exact", "event",
                   "Exact item id + count gained, every gain (not just the first).",
                   payload_schema='{"item": str, "count": int}'),
        StreamSpec("event.block_broken", "event", payload_schema='{"block": str}'),
        StreamSpec("event.block_broken_exact", "event",
                   "Exact block id + position mined.",
                   payload_schema='{"block": str, "position": {"x": float, "y": float, "z": float}}'),
        StreamSpec("event.block_placed", "event"),
        StreamSpec("event.block_placed_exact", "event",
                   "Exact block id + position placed.",
                   payload_schema='{"block": str, "position": {"x": float, "y": float, "z": float}}'),
        StreamSpec("event.crafted", "event",
                   "Crafting/smelting outcome: recipe id + exact inputs/outputs.",
                   payload_schema='{"recipe": str, "inputs": {"item": int}, "outputs": {"item": int}}'),
        StreamSpec("event.advancement", "event",
                   "Milestone/advancement earned (vanilla id on a live server; "
                   "sim.* on the simulated backend), once per episode.",
                   payload_schema='{"id": str}'),
        StreamSpec("event.dimension_changed", "event",
                   "Dimension transition, e.g. overworld <-> nether.",
                   payload_schema='{"from": str, "to": str}'),
        StreamSpec("event.biome_entered", "event",
                   "Biome underfoot changed (event view of world.biome).",
                   payload_schema='{"biome": str}'),
        StreamSpec("event.structure_discovered", "event",
                   "Named structure entered for the first time this episode.",
                   payload_schema='{"structure": str}'),
        StreamSpec("event.container_interaction", "event",
                   "Container / crafting-table / furnace opened.",
                   payload_schema='{"container": str, "position": {"x": float, "y": float, "z": float}}'),
        StreamSpec("event.created_light_source", "event"),
        StreamSpec("event.tool_used", "event",
                   "Tool/weapon swung while equipped (issue #30 tool-use goal).",
                   payload_schema='{"item": str}'),
        StreamSpec("event.mob_killed", "event"),
        StreamSpec("event.bumped", "event"),
        StreamSpec("event.food_eaten", "event"),
        StreamSpec("event.entered_shelter", "event"),
        StreamSpec("event.survived_night", "event"),
        StreamSpec("event.died", "event", payload_schema='{"reason": str|null}'),
        StreamSpec("event.action_rejected", "event", payload_schema='{"reason": str}'),
        StreamSpec("reward.scalar", "reward", "Survival reward, every tick.",
                   nominal_rate_hz=20.0,
                   payload_schema='{"value": float, "components": dict}',
                   range=(-2.0, 2.0)),
    ]


class SurvivalStreamPublisher:
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
        observation: Observation,
        world_events: List[str],
        death_reason: Optional[str] = None,
        paced: bool = True,
    ) -> List[StreamEvent]:
        """Publish this tick's streams from the post-step observation and the
        world's semantic event strings.  Returns everything published.

        With ``paced`` the realtime pacer gates vision frames and the body
        heartbeat to their wall-clock rates; ``paced=False`` bypasses it so
        the initial post-reset snapshot always publishes every stream (no
        subscriber starts blind).  In fast-forward mode the pacer is inert and
        ``paced`` has no effect.
        """
        data = observation.data
        timestamp = observation.timestamp
        published: List[StreamEvent] = []

        def pub(stream_id: str, payload, force: bool = False) -> None:
            event = self._delta.publish(
                stream_id, payload, timestamp, force=force, source=self._source
            )
            if event is not None:
                published.append(event)

        if paced and self._pacer.enabled:
            # Realtime: throttle vision + heartbeat to their target rates.  We
            # pace off *simulated* time (which the realtime scheduler holds
            # locked to wall clock) so pacing is deterministic and a realtime
            # recording replays bit-for-bit in fast-forward.
            show_frame = self._pacer.should_publish(VISION_STREAM, now=timestamp)
            heartbeat = self._pacer.should_publish(BODY_HEARTBEAT_KEY, now=timestamp)
        else:
            # Fast-forward, or the forced snapshot: every-tick vision, a
            # 20-tick (1 Hz) vitals heartbeat — the established Phase-1 cadence.
            show_frame = True
            heartbeat = observation.tick % BODY_HEARTBEAT_TICKS == 0

        if show_frame:
            pub("vision.frame.grid", observation.frame, force=True)
            if observation.pixels is not None:
                pub(PIXEL_STREAM, observation.pixels, force=True)
        mobs = data["mobs"]
        pub("vision.entities", mobs, force=bool(mobs))
        # force=bool(mobs): a stationary mob at melee range (dist <=
        # ZOMBIE_REACH stops closing, world.py:840) repeats the identical
        # salience/bearing payload tick after tick; without forcing, the
        # delta publisher would drop every event after the first, leaving
        # the attention controller scoring a stale timestamp and the
        # orienting reflex blind to an entity that is actively attacking.
        pub("world.entity_bearing", _entity_bearing_payload(mobs), force=bool(mobs))
        pub("body.health", data["health"], force=heartbeat)
        pub("body.hunger", data["hunger"], force=heartbeat)
        pub("body.oxygen", data["oxygen"], force=heartbeat)
        pub("body.inventory", data["inventory"])
        if "inventory_exact" in data:
            pub("body.inventory_exact", data["inventory_exact"])
        pub("body.hotbar", {"slots": data["hotbar"], "selected": data["selected_slot"]})
        pub("body.inventory_open", data["inventory_open"])
        pub("body.in_water", data["in_water"])
        pub("body.alive", not data["dead"])
        pub("spatial.position", data["position"])
        pub("spatial.rotation", {"yaw": data["yaw"], "pitch": data["pitch"]})
        pub("spatial.distance_from_spawn", data["distance_from_spawn"])
        pub("world.time", {
            "time_of_day": data["time_of_day"],
            "day_length": data["day_length"],
            "is_night": data["is_night"],
        }, force=True)
        pub("world.biome", data["biome"])
        pub("world.nearby_blocks", data["nearby_blocks"])
        if "nearby_blocks_exact" in data:
            pub("world.nearby_blocks_exact", data["nearby_blocks_exact"])
        pub("world.front_block", data["front_block"])
        if "front_block_exact" in data:
            pub("world.front_block_exact", data["front_block_exact"])
        pub("world.sheltered", data["sheltered"])

        for event_string in world_events:
            translated = self._translate_event(event_string, death_reason)
            if translated is not None:
                stream_id, payload = translated
                published.append(
                    self._bus.publish(stream_id, payload, timestamp, source=self._source)
                )
        return published

    @staticmethod
    def _translate_event(event_string: str, death_reason: Optional[str]):
        if event_string.startswith("damage:"):
            return "event.damage_taken", {"reason": event_string.split(":", 1)[1]}
        if event_string.startswith("new_item:"):
            return "event.item_collected", {"item": event_string.split(":", 1)[1]}
        if event_string.startswith("item_collected_exact:"):
            return "event.item_collected_exact", json.loads(event_string.split(":", 1)[1])
        if event_string.startswith("broke_block:"):
            return "event.block_broken", {"block": event_string.split(":", 1)[1]}
        if event_string.startswith("block_broken_exact:"):
            return "event.block_broken_exact", json.loads(event_string.split(":", 1)[1])
        if event_string == "placed_block":
            return "event.block_placed", {}
        if event_string.startswith("block_placed_exact:"):
            return "event.block_placed_exact", json.loads(event_string.split(":", 1)[1])
        if event_string.startswith("crafted:"):
            return "event.crafted", json.loads(event_string.split(":", 1)[1])
        if event_string.startswith("advancement:"):
            return "event.advancement", {"id": event_string.split(":", 1)[1]}
        if event_string.startswith("dimension_changed:"):
            _, from_dim, to_dim = event_string.split(":", 2)
            return "event.dimension_changed", {"from": from_dim, "to": to_dim}
        if event_string.startswith("biome_entered:"):
            return "event.biome_entered", {"biome": event_string.split(":", 1)[1]}
        if event_string.startswith("structure_discovered:"):
            return "event.structure_discovered", {"structure": event_string.split(":", 1)[1]}
        if event_string.startswith("container_interact:"):
            return "event.container_interaction", json.loads(event_string.split(":", 1)[1])
        if event_string.startswith("action_rejected:"):
            return "event.action_rejected", {"reason": event_string.split(":", 1)[1]}
        if event_string == "created_light_source":
            return "event.created_light_source", {}
        if event_string.startswith("used_tool:"):
            return "event.tool_used", {"item": event_string.split(":", 1)[1]}
        if event_string == "killed_mob":
            return "event.mob_killed", {}
        if event_string == "bumped":
            return "event.bumped", {}
        if event_string == "ate_food":
            return "event.food_eaten", {}
        if event_string == "entered_shelter":
            return "event.entered_shelter", {}
        if event_string == "survived_night":
            return "event.survived_night", {}
        if event_string == "died":
            return "event.died", {"reason": death_reason}
        return None  # hit_mob / acquired_food: semantic-only reward hints for now
