"""Environment-agnostic core abstractions.

Nothing in this package may know about Minecraft (or any other Program).
"""

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.policy import Policy, SingleActionPolicy
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.world_model import Prediction, TrendWorldModel, WorldModel
from cognitive_runtime.core.entity_persistence import (
    EntityPersistence,
    EntityPersistencePrediction,
    NullEntityPersistence,
)
from cognitive_runtime.core.novelty import combine_novelty
from cognitive_runtime.core.learner import Learner, NullLearner
from cognitive_runtime.core.attention import (
    AttentionBudget,
    AttentionCoefficients,
    AttentionConfig,
    AttentionController,
    AttentionReason,
    AttentionSignal,
    AttentionState,
)

__all__ = [
    "Action",
    "NULL_ACTION",
    "Observation",
    "RewardSignal",
    "Program",
    "ProgramMetadata",
    "ActionResult",
    "Policy",
    "SingleActionPolicy",
    "State",
    "Memory",
    "WorldModel",
    "TrendWorldModel",
    "Prediction",
    "EntityPersistence",
    "EntityPersistencePrediction",
    "NullEntityPersistence",
    "combine_novelty",
    "Learner",
    "NullLearner",
    "AttentionSignal",
    "AttentionState",
    "AttentionController",
    "AttentionBudget",
    "AttentionCoefficients",
    "AttentionConfig",
    "AttentionReason",
]
