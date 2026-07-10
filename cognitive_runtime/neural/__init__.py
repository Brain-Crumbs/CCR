"""PyTorch-backed neural module contracts (Phase A: interfaces only).

Everything under ``cognitive_runtime.neural`` is isolated the same way as
``cognitive_runtime/models/vision.py``: torch is a hard dependency of this
package, imported eagerly here so a missing install fails once, loudly, with
an actionable message, instead of surfacing as a confusing ``AttributeError``
deep inside a submodule. The rest of the runtime (``cognitive_runtime.core``,
``cognitive_runtime.runtime``, replay, and the CLI's ``scripted``/``random``/
``null``/``online`` policies) never imports this package, so it stays
torch-free.

This phase defines contracts only — abstract base classes with docstrings
covering input/output tensor shapes and checkpoint keys for the pieces the
neural stream agent target (see ``docs/online-learning.md``) will need:

- :class:`StreamEncoderModule` -- trainable per-stream encoder contract.
- :class:`PixelStreamEncoder` -- CNN encoder for ``vision.frame.pixels``.
- :class:`LatentFusionModel` -- per-stream latents -> fused agent state.
- :class:`WorldModel` / :class:`WorldModelOutput` -- next-state/reward/
  terminal/risk/prediction-error head.
- :class:`PolicyModel` -- fused latent + world-model features -> action
  logits.
- :class:`ValueModel` -- expected-return critic.
- :class:`OnlineOptimizer` -- losses, gradient steps, clipping, target
  networks, and optimizer checkpoint state.

The pixel encoder is the first concrete stream module; learned fusion,
world-model, policy and value implementations still arrive in later phases.
The package also defines the unified checkpoint bundle format those concrete
modules use once wired into a learner.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by test w/o torch
    raise ImportError(
        "cognitive_runtime.neural requires PyTorch, which is not installed. "
        "Install the optional extra with: pip install -e .[neural]"
    ) from exc

from cognitive_runtime.neural.encoder import StreamEncoderModule
from cognitive_runtime.neural.pixel_stream_encoder import (
    PIXEL_CHECKPOINT_KEY,
    PIXEL_STREAM_ID,
    PixelStreamEncoder,
    pixels_to_chw,
)
from cognitive_runtime.neural.trainable_stream_encoders import (
    AUDIO_CHECKPOINT_KEY,
    AUDIO_STREAM_PATTERN,
    BODY_STATE_CHECKPOINT_KEY,
    ENTITY_CHECKPOINT_KEY,
    MOTOR_HISTORY_CHECKPOINT_KEY,
    MOTOR_HISTORY_STREAM_ID,
    REWARD_CHECKPOINT_KEY,
    AudioEncoder,
    BodyStateEncoder,
    EntityEncoder,
    MotorHistoryEncoder,
    RewardEncoder,
)
from cognitive_runtime.neural.fusion import (
    LatentFusionInputs,
    LatentFusionModel,
    latent_fusion_inputs_from_buffer,
)
from cognitive_runtime.neural.optimizer import OnlineOptimizer
from cognitive_runtime.neural.policy import PolicyModel
from cognitive_runtime.neural.value import ValueModel
from cognitive_runtime.neural.world_model import MLPWorldModel, WorldModel, WorldModelOutput
from cognitive_runtime.neural.entity_persistence import (
    DEFAULT_GAP_CAP_TICKS,
    ENTITY_PERSISTENCE_CHECKPOINT_KEY,
    EntityPersistenceModel,
    EntityPersistenceOutput,
    normalize_gap,
)
from cognitive_runtime.neural.checkpoint import (
    FORMAT_VERSION as NEURAL_CHECKPOINT_FORMAT,
    CheckpointCompatibilityError,
    NeuralAgentCheckpoint,
    action_space_hash,
    checkpoint_metadata_path,
    compatibility_hash,
    read_checkpoint_metadata,
)

__all__ = [
    "StreamEncoderModule",
    "PixelStreamEncoder",
    "PIXEL_STREAM_ID",
    "PIXEL_CHECKPOINT_KEY",
    "pixels_to_chw",
    "MotorHistoryEncoder",
    "BodyStateEncoder",
    "RewardEncoder",
    "EntityEncoder",
    "AudioEncoder",
    "MOTOR_HISTORY_STREAM_ID",
    "MOTOR_HISTORY_CHECKPOINT_KEY",
    "BODY_STATE_CHECKPOINT_KEY",
    "REWARD_CHECKPOINT_KEY",
    "ENTITY_CHECKPOINT_KEY",
    "AUDIO_STREAM_PATTERN",
    "AUDIO_CHECKPOINT_KEY",
    "LatentFusionModel",
    "LatentFusionInputs",
    "latent_fusion_inputs_from_buffer",
    "WorldModel",
    "WorldModelOutput",
    "MLPWorldModel",
    "EntityPersistenceModel",
    "EntityPersistenceOutput",
    "ENTITY_PERSISTENCE_CHECKPOINT_KEY",
    "DEFAULT_GAP_CAP_TICKS",
    "normalize_gap",
    "PolicyModel",
    "ValueModel",
    "OnlineOptimizer",
    "NEURAL_CHECKPOINT_FORMAT",
    "CheckpointCompatibilityError",
    "NeuralAgentCheckpoint",
    "action_space_hash",
    "checkpoint_metadata_path",
    "compatibility_hash",
    "read_checkpoint_metadata",
]
