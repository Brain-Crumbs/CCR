"""Runtime-facing online Q policy and learner.

The policy chooses from a dependency-free :class:`OnlineQModel`; the learner
uses the current reward window to update the *previous* decision, preserving
the runtime's one-tick motor latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.learner import Learner, window_reward
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.models.online_q import OnlineQModel


@dataclass(frozen=True)
class OnlineQDecision:
    latent_vector: List[float]
    layout_hash: str
    recent_action_keys: List[str]
    action_key: str
    q_values: List[float]
    epsilon: float
    model_training_ticks: int


class OnlineQPolicy(Policy):
    """Epsilon-greedy policy over the fused latent state.

    ``NULL`` remains a normal action in the model's action space.  When it is
    selected the policy returns ``[]``, matching the runtime motor contract.
    """

    name = "online"

    def __init__(
        self,
        model: OnlineQModel | str,
        action_space: Optional[Sequence[Action]] = None,
        *,
        history: int = 8,
        training: bool = True,
    ):
        self.model = OnlineQModel.load(model) if isinstance(model, str) else model
        self.history = history
        self.training = training
        self._recent: Deque[str] = deque(maxlen=history)
        self._latest_decision: Optional[OnlineQDecision] = None
        self._actions_by_key = self._build_action_map(action_space)

    def _build_action_map(
        self, action_space: Optional[Sequence[Action]]
    ) -> Dict[str, Action]:
        if action_space is None:
            return {key: Action.from_key(key) for key in self.model.action_keys}
        actions = {action.key(): action for action in action_space}
        if list(actions) != self.model.action_keys:
            raise ValueError(
                "online Q policy action-space mismatch: model has "
                f"{self.model.action_keys}, runtime has {list(actions)}"
            )
        return actions

    def reset(self) -> None:
        self._recent.clear()
        self._latest_decision = None

    def train_mode(self) -> None:
        self.training = True

    def eval_mode(self) -> None:
        self.training = False

    @property
    def latest_decision(self) -> Optional[OnlineQDecision]:
        return self._latest_decision

    def emit(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> List[Action]:
        latent = memory.fused_latent()
        if latent is None:
            raise RuntimeError(
                "online Q policy needs the fused LatentState, but the runtime "
                "produced none this tick"
            )
        self.model.check_compatible(layout_hash=latent.layout_hash, latent_width=latent.width)
        recent = list(self._recent)
        epsilon = self.model.current_epsilon() if self.training else 0.0
        action_key = self.model.select_action_key_from_latent(
            latent, recent, epsilon=epsilon
        )
        q_values = self.model.q_values_from_latent(latent, recent)
        self._latest_decision = OnlineQDecision(
            latent_vector=list(latent.vector),
            layout_hash=latent.layout_hash,
            recent_action_keys=recent,
            action_key=action_key,
            q_values=q_values,
            epsilon=epsilon,
            model_training_ticks=self.model.training_ticks,
        )
        self._recent.append(action_key)
        if action_key == "NULL":
            return []
        return [self._actions_by_key[action_key]]


class OnlineQLearner(Learner):
    """Temporal-difference learner for :class:`OnlineQPolicy`.

    The current window's reward belongs to the action emitted on the previous
    cognitive tick.  Because the loop calls learner.update after the current
    policy emission, this learner keeps one pending decision and updates it
    against the policy's latest decision as the next state.
    """

    def __init__(
        self,
        model: OnlineQModel,
        policy: OnlineQPolicy,
        *,
        training: bool = True,
        checkpoint_path: Optional[str] = None,
        save_every_updates: Optional[int] = None,
    ):
        if policy.model is not model:
            raise ValueError("OnlineQLearner and OnlineQPolicy must share the same model")
        self.model = model
        self.policy = policy
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.save_every_updates = save_every_updates
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.observed_ticks = 0
        self.update_count = 0
        self.skipped_updates = 0
        self._previous_decision: Optional[OnlineQDecision] = None

    def reset(self) -> None:
        self.episode_reward = 0.0
        self._previous_decision = None

    def train_mode(self) -> None:
        self.training = True
        self.policy.train_mode()

    def eval_mode(self) -> None:
        self.training = False
        self.policy.eval_mode()

    def update(self, window: TickWindow) -> None:
        reward = window_reward(window)
        self.total_reward += reward
        self.episode_reward += reward
        self.observed_ticks += 1

        current = self.policy.latest_decision
        if current is None:
            self.skipped_updates += 1
            return

        if self._previous_decision is not None:
            if self.training:
                self.model.td_update(
                    self._previous_decision.latent_vector,
                    self._previous_decision.recent_action_keys,
                    self._previous_decision.action_key,
                    reward,
                    current.latent_vector,
                    current.recent_action_keys,
                    done=False,
                )
                self.update_count += 1
                self._save_if_due()
            else:
                self.skipped_updates += 1
        self._previous_decision = current

    def stats(self) -> Dict[str, Any]:
        return {
            "training": self.training,
            "training_ticks": self.model.training_ticks,
            "reward_total": round(self.total_reward, 6),
            "episode_reward": round(self.episode_reward, 6),
            "observed_ticks": self.observed_ticks,
            "td_updates": self.update_count,
            "skipped_updates": self.skipped_updates,
            "epsilon_state": self.model.to_dict()["epsilon_state"],
        }

    def save(self, path: Optional[str] = None) -> None:
        out = path or self.checkpoint_path
        if out is None:
            raise ValueError("no checkpoint path provided")
        self.model.meta["learner_stats"] = self.stats()
        self.model.save(out)

    def _save_if_due(self) -> None:
        if (
            self.checkpoint_path is not None
            and self.save_every_updates is not None
            and self.save_every_updates > 0
            and self.update_count % self.save_every_updates == 0
        ):
            self.save(self.checkpoint_path)

