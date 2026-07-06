"""Generic motor stream encoding.

The policy side publishes actions as ``motor.command`` events with payload
``{"action": <Action.key()>}``.  Reusing the :class:`Action` key machinery
keeps the action space program-defined and opaque to the runtime: the bus
carries strings, only the Program interprets them.
"""

from __future__ import annotations

from typing import Any

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.streams.bus import StreamBus
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

MOTOR_COMMAND_STREAM = "motor.command"

MOTOR_COMMAND_SPEC = StreamSpec(
    stream_id=MOTOR_COMMAND_STREAM,
    modality="motor",
    description="One action to apply this tick, encoded as an Action key.",
    nominal_rate_hz=None,
    payload_schema='{"action": "<Action.key()>"}',
)


def motor_command_payload(action: Action) -> dict:
    return {"action": action.key()}


def publish_motor_command(
    bus: StreamBus, action: Action, timestamp: float, source: str = ""
) -> StreamEvent:
    """Publish an action onto a motor bus as a ``motor.command`` event."""
    return bus.publish(
        MOTOR_COMMAND_STREAM, motor_command_payload(action), timestamp, source=source
    )


def action_from_motor_event(event: StreamEvent) -> Action:
    """Decode a ``motor.command`` event back into an :class:`Action`.

    Raises ``ValueError`` on malformed payloads; Programs turn that into an
    ``event.action_rejected`` publication rather than letting it propagate.
    """
    if event.stream_id != MOTOR_COMMAND_STREAM:
        raise ValueError(f"not a motor command stream: {event.stream_id!r}")
    payload: Any = event.payload
    if not isinstance(payload, dict):
        raise ValueError(f"motor command payload must be a dict, got {payload!r}")
    key = payload.get("action")
    if not isinstance(key, str) or not key:
        raise ValueError(f"motor command payload missing action key: {payload!r}")
    return Action.from_key(key)
