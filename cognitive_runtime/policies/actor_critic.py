"""Runtime-facing neural actor/critic policy and learner (issue #29,
``docs/neural-stream-agent.md`` Phase E "Add Actor/Critic Online Learning").

Status: the target online learner, over the fixed fused latent state plus a
small set of world-model-derived features -- kept alongside ``--policy
online`` (:mod:`cognitive_runtime.policies.online_q`) as a baseline until
this one reliably beats it.

Mirrors :mod:`cognitive_runtime.policies.online_q`'s shape: a policy that
chooses actions from a :class:`~cognitive_runtime.neural.policy.PolicyModel`/
:class:`~cognitive_runtime.neural.value.ValueModel` pair, and a learner that
uses the current reward window to update the *previous* decision (one-tick
delayed reward attribution, the runtime's one-tick motor latency).

``world_features``
-------------------
:class:`~cognitive_runtime.neural.policy.PolicyModel`/:class:`~cognitive_runtime.neural.value.ValueModel`
take ``(fused_latent, world_features)``.  Rather than re-running a world
model over every candidate action each tick, ``world_features_vector`` reuses
the ``Prediction`` the runtime loop already computes once per tick (whatever
``self.world_model.predict(state, memory)`` produced -- the heuristic
``TrendWorldModel`` or a trained ``NeuralWorldModel`` bridge, issue #26) plus
this policy's own recent-action history, so no loop changes are needed:

    [risk, p_death, predicted_reward, prediction_error] + last_action_onehot

Fields ``Prediction`` leaves ``None`` (the heuristic model only fills
``risk``) degrade to ``0.0``, the same graceful-degradation convention
``cognitive_runtime.neural.replay_buffer.transition_priority`` uses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import torch

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.learner import Learner, window_training_reward
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.models.online_q import motor_history_features_for_actions
from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint
from cognitive_runtime.neural.experience_queue import SharedExperienceRing
from cognitive_runtime.neural.optimizer import ActorCriticOptimizer
from cognitive_runtime.neural.policy import PolicyModel
from cognitive_runtime.neural.replay_buffer import (
    MixedTrainingSchedule,
    ReplayBuffer,
    Transition,
)
from cognitive_runtime.neural.value import ValueModel
from cognitive_runtime.neural.weight_publisher import WeightSubscriber

WORLD_FEATURE_BASE_WIDTH = 4  # [risk, p_death, predicted_reward, prediction_error]


def world_feature_width(action_keys: Sequence[str]) -> int:
    """Width of :func:`world_features_vector`'s output for this action space."""
    return WORLD_FEATURE_BASE_WIDTH + len(action_keys)


def world_features_vector(
    prediction: Optional[Prediction],
    recent_action_keys: Sequence[str],
    action_keys: Sequence[str],
) -> List[float]:
    """``[risk, p_death, predicted_reward, prediction_error] + last_action_onehot``."""
    risk = float(prediction.risk) if prediction is not None else 0.0
    p_death = (
        float(prediction.p_death)
        if prediction is not None and prediction.p_death is not None
        else 0.0
    )
    predicted_reward = (
        float(prediction.predicted_reward)
        if prediction is not None and prediction.predicted_reward is not None
        else 0.0
    )
    prediction_error = (
        float(prediction.prediction_error)
        if prediction is not None and prediction.prediction_error is not None
        else 0.0
    )
    motor = motor_history_features_for_actions(recent_action_keys, action_keys)
    return [risk, p_death, predicted_reward, prediction_error] + motor


@dataclass(frozen=True)
class ActorCriticDecision:
    fused_latent: List[float]
    world_features: List[float]
    layout_hash: str
    recent_action_keys: List[str]
    action_index: int
    action_key: str
    log_prob: float
    value_estimate: float
    entropy: float


class ActorCriticPolicy(Policy):
    """Samples (training) or argmaxes (eval) over the actor head's logits.

    ``NULL`` remains a normal action in the action space; when it is chosen
    the policy returns ``[]``, matching the runtime motor contract.
    """

    name = "actor-critic"

    def __init__(
        self,
        policy_model: PolicyModel,
        critic_model: ValueModel,
        action_keys: Sequence[str],
        action_space: Optional[Sequence[Action]] = None,
        *,
        history: int = 8,
        training: bool = True,
        seed: int = 0,
    ):
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.action_keys = list(action_keys)
        self.history = history
        self.training = training
        self._recent: Deque[str] = deque(maxlen=history)
        self._latest_decision: Optional[ActorCriticDecision] = None
        self._actions_by_key = self._build_action_map(action_space)
        self._generator = torch.Generator().manual_seed(seed)

    def _build_action_map(
        self, action_space: Optional[Sequence[Action]]
    ) -> Dict[str, Action]:
        if action_space is None:
            return {key: Action.from_key(key) for key in self.action_keys}
        actions = {action.key(): action for action in action_space}
        if list(actions) != self.action_keys:
            raise ValueError(
                "actor-critic policy action-space mismatch: model has "
                f"{self.action_keys}, runtime has {list(actions)}"
            )
        return actions

    def reset(self) -> None:
        self._recent.clear()
        self._latest_decision = None

    def train_mode(self) -> None:
        self.training = True
        self.policy_model.train()
        self.critic_model.train()

    def eval_mode(self) -> None:
        self.training = False
        self.policy_model.eval()
        self.critic_model.eval()

    @property
    def latest_decision(self) -> Optional[ActorCriticDecision]:
        return self._latest_decision

    def model_metadata(self) -> Dict[str, Any]:
        policy_meta = getattr(self.policy_model, "checkpoint_metadata", None)
        critic_meta = getattr(self.critic_model, "checkpoint_metadata", None)
        return {
            "format": "actor-critic-v1",
            "action_keys": list(self.action_keys),
            "world_feature_width": world_feature_width(self.action_keys),
            "policy": policy_meta() if callable(policy_meta) else None,
            "critic": critic_meta() if callable(critic_meta) else None,
            "training": self.training,
        }

    def emit(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> List[Action]:
        latent = memory.fused_latent()
        if latent is None:
            raise RuntimeError(
                "actor-critic policy needs the fused LatentState, but the runtime "
                "produced none this tick"
            )
        recent = list(self._recent)
        features = world_features_vector(prediction, recent, self.action_keys)
        fused_t = torch.tensor([latent.vector], dtype=torch.float32)
        world_t = torch.tensor([features], dtype=torch.float32)

        was_training = self.policy_model.training
        self.policy_model.eval()
        self.critic_model.eval()
        with torch.no_grad():
            logits = self.policy_model(fused_t, world_t)[0]
            value = self.critic_model(fused_t, world_t)[0]
        if was_training:
            self.policy_model.train()
            self.critic_model.train()

        probs = torch.softmax(logits, dim=-1)
        if self.training:
            action_index = int(torch.multinomial(probs, 1, generator=self._generator).item())
        else:
            action_index = int(torch.argmax(probs).item())
        log_prob = float(torch.log(probs[action_index].clamp_min(1e-8)).item())
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
        action_key = self.action_keys[action_index]

        self._latest_decision = ActorCriticDecision(
            fused_latent=list(latent.vector),
            world_features=features,
            layout_hash=latent.layout_hash,
            recent_action_keys=recent,
            action_index=action_index,
            action_key=action_key,
            log_prob=log_prob,
            value_estimate=float(value.item()),
            entropy=entropy,
        )
        self._recent.append(action_key)
        if action_key == "NULL":
            return []
        return [self._actions_by_key[action_key]]


class ActorCriticLearner(Learner):
    """Online actor/critic learner: on-policy every tick, plus periodic
    replay minibatches when a :class:`ReplayBuffer` is attached (issue #28).

    The current window's reward belongs to the action emitted on the
    previous cognitive tick, the same one-tick delayed attribution
    :class:`~cognitive_runtime.policies.online_q.OnlineQLearner` uses.
    """

    def __init__(
        self,
        optimizer: ActorCriticOptimizer,
        policy: ActorCriticPolicy,
        *,
        training: bool = True,
        checkpoint: Optional[NeuralAgentCheckpoint] = None,
        save_every_ticks: Optional[int] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        mixed_schedule: Optional[MixedTrainingSchedule] = None,
        replay_batch_size: int = 32,
        live_fusion: Optional[Any] = None,
    ):
        if policy.policy_model is not optimizer.policy or policy.critic_model is not optimizer.critic:
            raise ValueError(
                "ActorCriticLearner and ActorCriticPolicy must share the same "
                "policy/critic modules"
            )
        self.optimizer = optimizer
        self.policy = policy
        self.training = training
        self._checkpoint_bundle = checkpoint
        self.save_every_ticks = save_every_ticks
        self.replay_buffer = replay_buffer
        self.mixed_schedule = mixed_schedule or MixedTrainingSchedule()
        self.replay_batch_size = replay_batch_size
        #: The live ``--fusion learned`` pipeline (issue #57), if any -- kept
        #: here only for stats/checkpoint visibility. The runtime loop
        #: (``runtime/loop.py``) owns calling ``.fuse()``/``.maybe_train_step()``
        #: on the *same* instance every tick; this learner never drives it.
        self.live_fusion = live_fusion
        self.n_actions = len(policy.action_keys)
        self.world_feature_width = world_feature_width(policy.action_keys)

        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.observed_ticks = 0
        self.update_count = 0
        self.replay_update_count = 0
        self.skipped_updates = 0
        self._previous_decision: Optional[ActorCriticDecision] = None
        self._last_checkpoint_reason: Optional[str] = None
        self._last_metrics: Dict[str, float] = {}

    def reset(self) -> None:
        self.episode_reward = 0.0
        self._previous_decision = None

    def train_mode(self) -> None:
        self.training = True
        self.policy.train_mode()
        if self.live_fusion is not None:
            self.live_fusion.train_mode()

    def eval_mode(self) -> None:
        self.training = False
        self.policy.eval_mode()
        if self.live_fusion is not None:
            self.live_fusion.eval_mode()

    def update(self, window: TickWindow) -> None:
        reward = window_training_reward(window)
        self.total_reward += reward
        self.episode_reward += reward
        self.observed_ticks += 1

        current = self.policy.latest_decision
        if current is None:
            self.skipped_updates += 1
            return

        if self._previous_decision is not None:
            if self.training:
                batch = self._decision_batch(self._previous_decision, reward, current, done=False)
                self._last_metrics = self.optimizer.step(batch)
                self.update_count += 1
                if self.replay_buffer is not None:
                    self.replay_buffer.add(
                        Transition(
                            latent=self._previous_decision.fused_latent,
                            action=self._previous_decision.action_index,
                            reward=reward,
                            next_latent=current.fused_latent,
                            done=False,
                            source="online",
                        )
                    )
                    due = self.mixed_schedule.on_tick(len(self.replay_buffer))
                    if due["replay"]:
                        self._replay_update()
                self._save_if_due(reason="interval")
            else:
                self.skipped_updates += 1
        self._previous_decision = current

    def _decision_batch(
        self,
        previous: ActorCriticDecision,
        reward: float,
        current: ActorCriticDecision,
        *,
        done: bool,
    ) -> Dict[str, torch.Tensor]:
        return {
            "fused_latent": torch.tensor([previous.fused_latent], dtype=torch.float32),
            "world_features": torch.tensor([previous.world_features], dtype=torch.float32),
            "action_onehot": torch.nn.functional.one_hot(
                torch.tensor([previous.action_index]), num_classes=self.n_actions
            ).float(),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "next_fused_latent": torch.tensor([current.fused_latent], dtype=torch.float32),
            "next_world_features": torch.tensor([current.world_features], dtype=torch.float32),
            "done": torch.tensor([1.0 if done else 0.0], dtype=torch.float32),
        }

    def _replay_update(self) -> None:
        if self.replay_buffer is None or len(self.replay_buffer) < self.mixed_schedule.min_buffer_size:
            return
        batch_size = min(self.replay_batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample_batch(batch_size, self.n_actions)
        # Replayed transitions only carry fused-latent references (issue #28),
        # not the live decision's world features; degrade to zero, the same
        # convention `transition_priority` uses for signals a transition
        # doesn't carry.
        zeros = torch.zeros(batch["fused_latent"].shape[0], self.world_feature_width)
        batch["world_features"] = zeros
        batch["next_world_features"] = zeros.clone()
        metrics = self.optimizer.step(batch)
        self._last_metrics = {f"replay_{key}": value for key, value in metrics.items()}
        self.replay_update_count += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "training": self.training,
            "training_ticks": self.optimizer.step_count,
            "reward_total": round(self.total_reward, 6),
            "episode_reward": round(self.episode_reward, 6),
            "observed_ticks": self.observed_ticks,
            "on_policy_updates": self.update_count,
            "replay_updates": self.replay_update_count,
            "skipped_updates": self.skipped_updates,
            "last_checkpoint_reason": self._last_checkpoint_reason,
            "last_metrics": dict(self._last_metrics),
            "live_fusion_metrics": (
                dict(self.live_fusion.last_metrics) if self.live_fusion is not None else None
            ),
        }

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self._checkpoint_bundle.path if self._checkpoint_bundle else None,
            "save_every_ticks": self.save_every_ticks,
            "optimizer": {
                "gamma": self.optimizer.gamma,
                "entropy_coef": self.optimizer.entropy_coef,
                "value_coef": self.optimizer.value_coef,
                "world_model_coef": self.optimizer.world_model_coef,
                "grad_clip_norm": self.optimizer.grad_clip_norm,
                "target_tau": self.optimizer.target_tau,
                "normalize_reward": self.optimizer.normalize_reward,
                "normalize_advantage": self.optimizer.normalize_advantage,
                "has_world_model": self.optimizer.world_model is not None,
            },
            "replay_buffer": self.replay_buffer.state_dict() if self.replay_buffer else None,
            "stats": self.stats(),
        }

    def save(self, *, reason: str = "manual") -> None:
        if self._checkpoint_bundle is None:
            raise ValueError("no checkpoint bundle configured")
        self._last_checkpoint_reason = reason
        self._checkpoint_bundle.training_ticks = self.optimizer.step_count
        self._checkpoint_bundle.training_stats = self.stats()
        if self.replay_buffer is not None:
            self._checkpoint_bundle.replay_metadata = self.replay_buffer.state_dict()
        self._checkpoint_bundle.save(reason=reason)

    def checkpoint(self, reason: str = "manual") -> None:
        if self._checkpoint_bundle is not None:
            self.save(reason=reason)

    def end_episode(self) -> None:
        self.checkpoint(reason="episode_end")

    def _save_if_due(self, *, reason: str) -> None:
        if (
            self._checkpoint_bundle is not None
            and self.save_every_ticks is not None
            and self.save_every_ticks > 0
            and self.optimizer.step_count % self.save_every_ticks == 0
        ):
            self.save(reason=reason)


class AsyncActorCriticLearner(Learner):
    """The actor-side half of the async actor/learner split (issue #37):
    "the live loop only does inference and data capture; training happens
    asynchronously in batches" -- unlike :class:`ActorCriticLearner`, this
    learner never calls ``optimizer.step``.  Every tick it does exactly two
    O(1), non-blocking things:

    1. Pushes the just-completed transition into ``experience_ring`` (a
       :class:`~cognitive_runtime.neural.experience_queue.SharedExperienceRing`)
       for a separate trainer process to consume on its own schedule.
       Backpressure is drop-oldest and lives inside the ring itself -- this
       call never blocks regardless of whether a trainer is running.
    2. Polls ``weight_subscriber`` (a
       :class:`~cognitive_runtime.neural.weight_publisher.WeightSubscriber`)
       for a newer published snapshot and, if one exists, hot-swaps it into
       the *same* policy/critic module objects the attached
       :class:`ActorCriticPolicy` already uses for inference -- "the actor
       swaps them in atomically between ticks": the loop calls
       ``learner.update(window)`` immediately after ``policy.emit()`` and
       before the next tick's ``policy.emit()``, so this is exactly between
       ticks. A missing/stale/crashed trainer just means the poll finds
       nothing new; the actor keeps acting on its last weights.

    Plugs into the existing ``Learner`` contract, so ``runtime/loop.py``
    needs no changes to run this instead of the synchronous
    :class:`ActorCriticLearner`.
    """

    def __init__(
        self,
        policy: ActorCriticPolicy,
        experience_ring: SharedExperienceRing,
        *,
        weight_subscriber: Optional[WeightSubscriber] = None,
        reload_every_ticks: int = 1,
    ):
        self.policy = policy
        self.experience_ring = experience_ring
        self.weight_subscriber = weight_subscriber
        self.reload_every_ticks = max(1, reload_every_ticks)

        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.observed_ticks = 0
        self.pushed_count = 0
        self.skipped_updates = 0
        self._previous_decision: Optional[ActorCriticDecision] = None
        self._tick = 0

    def reset(self) -> None:
        self.episode_reward = 0.0
        self._previous_decision = None

    def update(self, window: TickWindow) -> None:
        reward = window_training_reward(window)
        self.total_reward += reward
        self.episode_reward += reward
        self.observed_ticks += 1

        current = self.policy.latest_decision
        if current is None:
            self.skipped_updates += 1
        else:
            if self._previous_decision is not None:
                self.experience_ring.push(Transition(
                    latent=self._previous_decision.fused_latent,
                    action=self._previous_decision.action_index,
                    reward=reward,
                    next_latent=current.fused_latent,
                    done=False,
                ))
                self.pushed_count += 1
            self._previous_decision = current

        self._tick += 1
        if self.weight_subscriber is not None and self._tick % self.reload_every_ticks == 0:
            self.weight_subscriber.maybe_reload()

    def model_metadata(self) -> Dict[str, Any]:
        return self.policy.model_metadata()

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": "async-actor",
            "reward_total": round(self.total_reward, 6),
            "episode_reward": round(self.episode_reward, 6),
            "observed_ticks": self.observed_ticks,
            "pushed_transitions": self.pushed_count,
            "skipped_updates": self.skipped_updates,
            "experience_ring": self.experience_ring.stats().__dict__,
            "weight_subscriber": (
                self.weight_subscriber.stats() if self.weight_subscriber else None
            ),
        }

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {"format": "async-actor-critic-v1", "stats": self.stats()}

    def checkpoint(self, reason: str = "manual") -> None:
        # Checkpoint ownership belongs to the trainer process (issue #20 /
        # #37's "the trainer owns checkpoint writes; the actor only ever
        # loads"); the loop still calls this hook on shutdown/crash
        # (`CognitiveRuntime._checkpoint_online`), so it must exist, but the
        # actor has nothing of its own to persist.
        pass

    def end_episode(self) -> None:
        pass
