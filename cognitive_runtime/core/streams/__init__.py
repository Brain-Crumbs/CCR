"""Time-indexed sensory/motor stream primitives (Phase 0).

The stream substrate the sensory-stream architecture builds on: Programs
publish :class:`StreamEvent`s onto buses, the runtime collects them into
cognitive tick windows, buffers recent history per stream, and encodes
windows into latent tokens.

Environment-agnostic by construction: nothing in this package may import
from ``cognitive_runtime.programs``.  Not yet wired into the legacy loop —
see docs/streams.md and the migration tracking issue.
"""

from cognitive_runtime.core.streams.events import (
    MODALITIES,
    StreamEvent,
    StreamSpec,
    validate_stream_identity,
)
from cognitive_runtime.core.streams.bus import (
    MotorStreamBus,
    SensoryStreamBus,
    StreamBus,
    StreamSubscription,
    stream_matches,
)
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer
from cognitive_runtime.core.streams.synchronizer import TickSynchronizer, TickWindow
from cognitive_runtime.core.streams.encoder_registry import (
    LatentToken,
    PassthroughEncoder,
    StreamEncoder,
    StreamEncoderRegistry,
)
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.motor import (
    MOTOR_COMMAND_SPEC,
    MOTOR_COMMAND_STREAM,
    action_from_motor_event,
    motor_command_payload,
    publish_motor_command,
)
from cognitive_runtime.core.streams.shim import (
    LatestValueView,
    ObservationStreamShim,
)

__all__ = [
    "MODALITIES",
    "StreamEvent",
    "StreamSpec",
    "validate_stream_identity",
    "StreamBus",
    "SensoryStreamBus",
    "MotorStreamBus",
    "StreamSubscription",
    "stream_matches",
    "TemporalBuffer",
    "TickSynchronizer",
    "TickWindow",
    "LatentToken",
    "StreamEncoder",
    "PassthroughEncoder",
    "StreamEncoderRegistry",
    "DeltaPublisher",
    "MOTOR_COMMAND_STREAM",
    "MOTOR_COMMAND_SPEC",
    "motor_command_payload",
    "publish_motor_command",
    "action_from_motor_event",
    "ObservationStreamShim",
    "LatestValueView",
]
