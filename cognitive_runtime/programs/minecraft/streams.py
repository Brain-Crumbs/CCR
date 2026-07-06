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

from typing import Dict, List, Optional

from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.programs.minecraft.world import (
    AGENT_FRAME_ID,
    BLOCK_IDS,
    BREAK_YIELD,
    MOB_FRAME_ID,
    SOLID,
)

#: Body vitals republish unchanged values every this many ticks, so
#: subscribers can distinguish "silent because unchanged" from "silent
#: because dead sensor".
BODY_HEARTBEAT_TICKS = 20

VITAL_RANGE = (0.0, 20.0)  # health/hunger/oxygen scale


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


def build_survival_stream_specs(world_size: int = 64) -> List[StreamSpec]:
    """The survival catalog, with encoder metadata (ranges/legend/categories).

    Position ranges depend on the world size, so the catalog is built per
    config rather than as a bare constant.
    """
    pos_range = (0.0, float(world_size))
    return [
        StreamSpec("vision.frame.grid", "vision", "Coarse top-down frame.",
                   nominal_rate_hz=20.0, payload_schema="11x11 int grid",
                   legend=FRAME_LEGEND),
        StreamSpec("vision.entities", "vision",
                   "Visible mobs (distance/angle), every tick while any are visible.",
                   payload_schema="[{distance, angle}]", range=(0.0, 16.0)),
        StreamSpec("body.health", "body", "Health, on change + heartbeat.",
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.hunger", "body", "Hunger, on change + heartbeat.",
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.oxygen", "body", "Oxygen, on change + heartbeat.",
                   payload_schema="float 0..20", range=VITAL_RANGE, neutral=20.0),
        StreamSpec("body.inventory", "body", "Inventory summary, on change.",
                   payload_schema="{item: count}"),
        StreamSpec("body.hotbar", "body", "Hotbar slots + selected, on change.",
                   payload_schema='{"slots": [item|null], "selected": int}'),
        StreamSpec("spatial.position", "spatial", "Agent position, on change.",
                   payload_schema="{x, y, z}", range=pos_range, neutral=world_size / 2.0),
        StreamSpec("spatial.rotation", "spatial", "Agent view direction, on change.",
                   payload_schema="{yaw, pitch}"),
        StreamSpec("world.time", "world", "Day/night clock, every tick.",
                   nominal_rate_hz=20.0,
                   payload_schema="{time_of_day, day_length, is_night}"),
        StreamSpec("world.biome", "world", "Biome under the agent, on change.",
                   payload_schema="str"),
        StreamSpec("world.nearby_blocks", "world", "5x5 block patch, on cell change.",
                   payload_schema="5x5 str grid"),
        StreamSpec("world.front_block", "world", "Block faced, on change.",
                   payload_schema="str", categories=FRONT_BLOCK_CATEGORIES),
        StreamSpec("world.sheltered", "world", "Shelter state, on change.",
                   payload_schema="bool"),
        StreamSpec("event.damage_taken", "event", payload_schema='{"reason": str}'),
        StreamSpec("event.item_collected", "event", payload_schema='{"item": str}'),
        StreamSpec("event.block_broken", "event", payload_schema='{"block": str}'),
        StreamSpec("event.block_placed", "event"),
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


#: Default catalog (world_size 64); adapters rebuild per config for ranges.
SURVIVAL_STREAM_SPECS: List[StreamSpec] = build_survival_stream_specs()


class SurvivalStreamPublisher:
    def __init__(self, bus: SensoryStreamBus, source: str = ""):
        self._bus = bus
        self._delta = DeltaPublisher(bus)
        self._source = source

    def reset(self) -> None:
        self._delta.reset()

    def publish_tick(
        self,
        observation: Observation,
        world_events: List[str],
        death_reason: Optional[str] = None,
    ) -> List[StreamEvent]:
        """Publish this tick's streams from the post-step observation and the
        world's semantic event strings.  Returns everything published."""
        data = observation.data
        timestamp = observation.timestamp
        published: List[StreamEvent] = []

        def pub(stream_id: str, payload, force: bool = False) -> None:
            event = self._delta.publish(
                stream_id, payload, timestamp, force=force, source=self._source
            )
            if event is not None:
                published.append(event)

        heartbeat = observation.tick % BODY_HEARTBEAT_TICKS == 0
        pub("vision.frame.grid", observation.frame, force=True)
        mobs = data["mobs"]
        pub("vision.entities", mobs, force=bool(mobs))
        pub("body.health", data["health"], force=heartbeat)
        pub("body.hunger", data["hunger"], force=heartbeat)
        pub("body.oxygen", data["oxygen"], force=heartbeat)
        pub("body.inventory", data["inventory"])
        pub("body.hotbar", {"slots": data["hotbar"], "selected": data["selected_slot"]})
        pub("spatial.position", data["position"])
        pub("spatial.rotation", {"yaw": data["yaw"], "pitch": data["pitch"]})
        pub("world.time", {
            "time_of_day": data["time_of_day"],
            "day_length": data["day_length"],
            "is_night": data["is_night"],
        }, force=True)
        pub("world.biome", data["biome"])
        pub("world.nearby_blocks", data["nearby_blocks"])
        pub("world.front_block", data["front_block"])
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
        if event_string.startswith("broke_block:"):
            return "event.block_broken", {"block": event_string.split(":", 1)[1]}
        if event_string == "placed_block":
            return "event.block_placed", {}
        if event_string == "ate_food":
            return "event.food_eaten", {}
        if event_string == "entered_shelter":
            return "event.entered_shelter", {}
        if event_string == "survived_night":
            return "event.survived_night", {}
        if event_string == "died":
            return "event.died", {"reason": death_reason}
        return None  # bumped / hit_mob / killed_mob / acquired_food: no stream yet
