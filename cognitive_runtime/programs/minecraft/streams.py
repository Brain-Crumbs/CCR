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

from typing import List, Optional

from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

#: Body vitals republish unchanged values every this many ticks, so
#: subscribers can distinguish "silent because unchanged" from "silent
#: because dead sensor".
BODY_HEARTBEAT_TICKS = 20

SURVIVAL_STREAM_SPECS: List[StreamSpec] = [
    StreamSpec("vision.frame.grid", "vision", "Coarse top-down frame.",
               nominal_rate_hz=20.0, payload_schema="11x11 int grid"),
    StreamSpec("vision.entities", "vision",
               "Visible mobs (distance/angle), every tick while any are visible.",
               payload_schema="[{distance, angle}]"),
    StreamSpec("body.health", "body", "Health, on change + heartbeat.",
               payload_schema="float 0..20"),
    StreamSpec("body.hunger", "body", "Hunger, on change + heartbeat.",
               payload_schema="float 0..20"),
    StreamSpec("body.oxygen", "body", "Oxygen, on change + heartbeat.",
               payload_schema="float 0..20"),
    StreamSpec("body.inventory", "body", "Inventory summary, on change.",
               payload_schema="{item: count}"),
    StreamSpec("body.hotbar", "body", "Hotbar slots + selected, on change.",
               payload_schema='{"slots": [item|null], "selected": int}'),
    StreamSpec("spatial.position", "spatial", "Agent position, on change.",
               payload_schema="{x, y, z}"),
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
               payload_schema="str"),
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
               payload_schema='{"value": float, "components": dict}'),
]


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
