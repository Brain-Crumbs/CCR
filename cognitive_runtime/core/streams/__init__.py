"""Time-indexed sensory/motor stream primitives.

The stream substrate the sensory-stream architecture is built on: Programs
publish :class:`StreamEvent`s onto buses, the runtime collects them into
cognitive tick windows, buffers recent history per stream, encodes streams
with per-modality encoders and fuses them into a fixed-width
:class:`LatentState`.  This is the primary data path of the runtime loop
(``runtime/loop.py``); see docs/streams.md.

Environment-agnostic by construction: nothing in this package may import
from ``cognitive_runtime.programs`` (enforced by a test).
"""

from cognitive_runtime.core.streams.events import (
    MODALITIES,
    OVERFLOW_POLICIES,
    StreamEvent,
    StreamSpec,
    validate_stream_identity,
)
from cognitive_runtime.core.streams.bus import (
    DEFAULT_QUEUE_CAPACITY,
    MotorStreamBus,
    SensoryStreamBus,
    StreamBus,
    StreamSubscription,
    stream_matches,
)
from cognitive_runtime.core.streams.pacer import RatePacer
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer
from cognitive_runtime.core.streams.synchronizer import TickSynchronizer, TickWindow
from cognitive_runtime.core.streams.encoder_registry import (
    LatentToken,
    PassthroughEncoder,
    StreamEncoder,
    StreamEncoderRegistry,
)
from cognitive_runtime.core.streams.registry import (
    ATTENTION_COMPUTE_COSTS,
    DEFAULT_STREAM_REGISTRY,
    STREAM_CLASSIFICATIONS,
    TRAIN_EVAL_BEHAVIORS,
    AttentionMetadata,
    StreamDeclaration,
    StreamRegistry,
)
from cognitive_runtime.core.streams.trainable import (
    FixedStreamModule,
    TrainableStreamModule,
    fixed_stream_module,
)
from cognitive_runtime.core.streams.encoders import (
    CategoryEncoder,
    EntityEncoder,
    EventEncoder,
    GridVisionEncoder,
    ScalarEncoder,
    SpatialEncoder,
)
from cognitive_runtime.core.streams.fusion import (
    LatentState,
    TemporalFusion,
    default_encoder_registry,
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
    "OVERFLOW_POLICIES",
    "StreamEvent",
    "StreamSpec",
    "validate_stream_identity",
    "StreamBus",
    "SensoryStreamBus",
    "MotorStreamBus",
    "StreamSubscription",
    "stream_matches",
    "DEFAULT_QUEUE_CAPACITY",
    "RatePacer",
    "TemporalBuffer",
    "TickSynchronizer",
    "TickWindow",
    "LatentToken",
    "StreamEncoder",
    "PassthroughEncoder",
    "StreamEncoderRegistry",
    "DEFAULT_STREAM_REGISTRY",
    "TRAIN_EVAL_BEHAVIORS",
    "STREAM_CLASSIFICATIONS",
    "ATTENTION_COMPUTE_COSTS",
    "AttentionMetadata",
    "StreamDeclaration",
    "StreamRegistry",
    "TrainableStreamModule",
    "FixedStreamModule",
    "fixed_stream_module",
    "ScalarEncoder",
    "SpatialEncoder",
    "GridVisionEncoder",
    "EventEncoder",
    "EntityEncoder",
    "CategoryEncoder",
    "LatentState",
    "TemporalFusion",
    "default_encoder_registry",
    "DeltaPublisher",
    "MOTOR_COMMAND_STREAM",
    "MOTOR_COMMAND_SPEC",
    "motor_command_payload",
    "publish_motor_command",
    "action_from_motor_event",
    "ObservationStreamShim",
    "LatestValueView",
]
