"""Prioritized replay buffer and mixed on-policy/replay training schedule
(issue #28, ``docs/neural-stream-agent.md`` "Add Replay Buffer And Mixed
Training").

:class:`ReplayBuffer` is the online transition buffer the future
``OnlineOptimizer`` (Phase E, issue #29) mixes with its short on-policy
update: a bounded ring buffer of :class:`Transition` records -- fused latent
*references* (plain float vectors already computed by the fusion pipeline,
not raw pixel frames) plus action/reward/next-latent/done and whichever
priority features (death, damage, novelty, world-model prediction error)
were available this tick -- sampled proportionally to a configurable
priority weighting.

Only the buffer's counters and priority configuration are meant to be
checkpointed (``state_dict``/``load_state_dict``), not its contents -- the
same way ``NeuralAgentCheckpoint.replay_metadata`` is documented as
"metadata; contents optional".  Contents are cheap to refill from recent
play or from :func:`load_session_into_buffer`, which replays a recorded
session (streams-v2) into a buffer for offline pretraining/regression.

No optimizer/policy training math lives here -- that is issue #29's
``OnlineOptimizer``.  :class:`MixedTrainingSchedule` only decides, per tick,
whether a replay minibatch pull is due; it does not take any gradient steps
itself.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from cognitive_runtime.core.modulation import (
    NOVELTY_STREAM as INTERNAL_NOVELTY_STREAM,
    REWARD_PREDICTION_ERROR_STREAM as INTERNAL_REWARD_PREDICTION_ERROR_STREAM,
)
from cognitive_runtime.core.streams import TemporalBuffer, TemporalFusion
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
    require_streams_v2,
)
from cognitive_runtime.core.streams.motor import MOTOR_COMMAND_STREAM
from cognitive_runtime.training.features import ACTION_KEYS

#: Priority is always at least this large, so a transition whose configured
#: signals all happen to be zero (no reward, no death, no damage, ...) can
#: still be sampled rather than permanently starved.
_PRIORITY_EPS = 1e-3

_WEIGHT_FIELDS = (
    "reward", "death", "damage", "novelty", "prediction_error", "reward_prediction_error",
)


@dataclass(frozen=True)
class PriorityWeights:
    """Per-signal weights combined into one transition priority.

    Any signal a transition doesn't carry (``novelty``/``prediction_error``/
    ``reward_prediction_error`` are ``None`` when unavailable -- e.g. a
    heuristic world model, or a recorded session that predates Phase D/#58)
    is dropped from the combination and the remaining weights are
    renormalized, so priority stays on a comparable scale whether or not
    every signal fired this tick.
    """

    reward: float = 1.0
    death: float = 1.0
    damage: float = 0.5
    novelty: float = 0.5
    prediction_error: float = 0.5
    #: The dopamine analog (issue #58): a large reward surprise -- the agent
    #: got much more or less reward than the world model predicted -- is
    #: exactly the kind of transition worth replaying.
    reward_prediction_error: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in _WEIGHT_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorityWeights":
        return cls(**{name: float(data[name]) for name in _WEIGHT_FIELDS if name in data})


@dataclass(frozen=True)
class Transition:
    """One online or replayed transition.

    ``latent``/``next_latent`` are the fused-state vectors themselves (small
    float lists), not references into a frame store -- bounded memory comes
    from storing this compact representation instead of raw sensory frames,
    per issue #28's "frames by reference/hash, not copies".  ``action`` is an
    index into the ordered action space (matching
    ``training.features.ACTION_KEYS``/``OnlineQModel.action_keys``).
    """

    latent: List[float]
    action: int
    reward: float
    next_latent: List[float]
    done: bool
    damage: bool = False
    novelty: Optional[float] = None
    prediction_error: Optional[float] = None
    reward_prediction_error: Optional[float] = None
    source: str = ""


def transition_priority(
    transition: Transition, weights: PriorityWeights, *, eps: float = _PRIORITY_EPS
) -> float:
    """Weighted combination of a transition's available priority signals.

    Degrades gracefully: ``novelty``/``prediction_error`` only contribute
    when present, and the weights actually used are renormalized to sum to
    the same total either way.
    """
    components: List[Tuple[float, float]] = [
        (weights.reward, abs(transition.reward)),
        (weights.death, 1.0 if transition.done else 0.0),
        (weights.damage, 1.0 if transition.damage else 0.0),
    ]
    if transition.novelty is not None:
        components.append((weights.novelty, max(0.0, transition.novelty)))
    if transition.prediction_error is not None:
        components.append((weights.prediction_error, max(0.0, transition.prediction_error)))
    if transition.reward_prediction_error is not None:
        components.append((
            weights.reward_prediction_error, abs(transition.reward_prediction_error),
        ))

    weight_total = sum(w for w, _ in components)
    if weight_total <= 0:
        return eps
    return sum(w * v for w, v in components) / weight_total + eps


@dataclass
class ReplayBufferConfig:
    capacity: int = 10_000
    #: Sharpens (>1) or flattens (<1) proportional priority sampling; 1.0 is
    #: plain proportional sampling, 0.0 is uniform regardless of priority.
    alpha: float = 0.6
    seed: int = 0
    weights: PriorityWeights = field(default_factory=PriorityWeights)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity!r}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "seed": self.seed,
            "weights": self.weights.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayBufferConfig":
        return cls(
            capacity=int(data.get("capacity", 10_000)),
            alpha=float(data.get("alpha", 0.6)),
            seed=int(data.get("seed", 0)),
            weights=PriorityWeights.from_dict(data.get("weights", {})),
        )


class ReplayBuffer:
    """Bounded ring buffer of :class:`Transition` with prioritized sampling.

    Oldest transitions are evicted once ``config.capacity`` is reached.
    Sampling is proportional to ``transition_priority(...) ** config.alpha``
    and draws from a buffer-owned :class:`random.Random` seeded by
    ``config.seed``, so two buffers fed the same transitions in the same
    order and sampled the same number of times produce identical batches.
    """

    def __init__(self, config: Optional[ReplayBufferConfig] = None):
        self.config = config or ReplayBufferConfig()
        self._transitions: List[Transition] = []
        self._write_index = 0
        self._rng = random.Random(self.config.seed)
        self.total_added = 0
        self.total_evicted = 0
        self.total_sampled = 0

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def capacity(self) -> int:
        return self.config.capacity

    def transitions(self) -> Tuple[Transition, ...]:
        """Read-only snapshot of the buffer's current contents."""
        return tuple(self._transitions)

    def add(self, transition: Transition) -> None:
        if len(self._transitions) < self.config.capacity:
            self._transitions.append(transition)
        else:
            self._transitions[self._write_index] = transition
            self._write_index = (self._write_index + 1) % self.config.capacity
            self.total_evicted += 1
        self.total_added += 1

    def priorities(self) -> List[float]:
        return [
            transition_priority(t, self.config.weights) ** self.config.alpha
            for t in self._transitions
        ]

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample ``batch_size`` transitions with replacement, proportional
        to priority."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if not self._transitions:
            raise ValueError("cannot sample from an empty replay buffer")
        sampled = self._rng.choices(
            self._transitions, weights=self.priorities(), k=batch_size
        )
        self.total_sampled += batch_size
        return sampled

    def as_batch(
        self, transitions: Sequence[Transition], n_actions: int
    ) -> Dict[str, torch.Tensor]:
        """Stack transitions into the tensor mapping
        :class:`~cognitive_runtime.neural.optimizer.OnlineOptimizer.step`
        expects: ``fused_latent``, ``action_onehot``, ``reward``,
        ``next_fused_latent``, ``done``."""
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        return {
            "fused_latent": torch.tensor(
                [t.latent for t in transitions], dtype=torch.float32
            ),
            "action_onehot": torch.nn.functional.one_hot(
                actions, num_classes=n_actions
            ).float(),
            "reward": torch.tensor([t.reward for t in transitions], dtype=torch.float32),
            "next_fused_latent": torch.tensor(
                [t.next_latent for t in transitions], dtype=torch.float32
            ),
            "done": torch.tensor(
                [1.0 if t.done else 0.0 for t in transitions], dtype=torch.float32
            ),
        }

    def sample_batch(self, batch_size: int, n_actions: int) -> Dict[str, torch.Tensor]:
        return self.as_batch(self.sample(batch_size), n_actions)

    def iter_minibatches(
        self, batch_size: int, n_actions: int, n_batches: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(n_batches):
            yield self.sample_batch(batch_size, n_actions)

    # -- checkpoint (counters + priority config only; see module docstring) --

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "total_added": self.total_added,
            "total_evicted": self.total_evicted,
            "total_sampled": self.total_sampled,
            "size": len(self._transitions),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.config = ReplayBufferConfig.from_dict(state.get("config", {}))
        self._rng = random.Random(self.config.seed)
        self.total_added = int(state.get("total_added", 0))
        self.total_evicted = int(state.get("total_evicted", 0))
        self.total_sampled = int(state.get("total_sampled", 0))


@dataclass
class MixedTrainingSchedule:
    """Ticks the cadence between the short on-policy update and a periodic
    replay minibatch pull; the actual updates are the future
    ``OnlineOptimizer``'s (issue #29), this only decides when each is due.
    """

    replay_every_n_ticks: int = 32
    min_buffer_size: int = 1
    _tick: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.replay_every_n_ticks <= 0:
            raise ValueError(
                f"replay_every_n_ticks must be positive, got {self.replay_every_n_ticks!r}"
            )

    def on_tick(self, buffer_size: int) -> Dict[str, bool]:
        """Call once per cognitive tick; returns which updates are due."""
        self._tick += 1
        replay_due = (
            self._tick % self.replay_every_n_ticks == 0
            and buffer_size >= self.min_buffer_size
        )
        return {"on_policy": True, "replay": replay_due}

    def reset(self) -> None:
        self._tick = 0


# --------------------------------------------------------------- session loader


def _motor_label(motor_records: Sequence[Mapping[str, Any]]) -> str:
    for record in motor_records:
        if record.get("stream_id") != MOTOR_COMMAND_STREAM:
            continue
        payload = record.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("action"), str):
            return payload["action"]
    return "NULL"


def load_session_into_buffer(
    buffer: ReplayBuffer,
    session_dirs: Union[str, Sequence[str]],
    *,
    max_transitions: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> int:
    """Replay recorded session(s) (streams-v2) into ``buffer`` for offline
    pretraining/regression, returning the number of transitions added.

    Reward/death/damage/prediction-error/novelty/reward-prediction-error are
    read off the tick *after* the one an action was taken on -- they are its
    causal consequence, the same convention
    ``training.datasets.build_world_model_dataset`` uses. ``novelty`` and
    ``reward_prediction_error`` are read from the recorded
    ``internal.novelty``/``internal.reward_prediction_error`` streams (issue
    #58) when present; sessions recorded before those streams existed (or a
    heuristic world model with no reward head) leave them ``None``, and
    priority degrades gracefully to whichever signals are available, per
    ``transition_priority``.
    """
    dirs = [session_dirs] if isinstance(session_dirs, str) else list(session_dirs)
    key_to_action = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    added = 0

    for session_dir in dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        catalog = [StreamSpec.from_dict(s) for s in metadata.get("stream_catalog", [])]
        session_fusion = TemporalFusion(catalog)
        if fusion is None:
            fusion = session_fusion
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); load "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            tick_buffer = TemporalBuffer()
            reward_total = 0.0
            episode_samples: List[
                Tuple[
                    List[float], int, float, bool, bool,
                    Optional[float], Optional[float], Optional[float],
                ]
            ] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward = float(decision.get("reward_window_total", 0.0))
                reward_total += reward
                died = False
                damaged = False
                novelty: Optional[float] = None
                reward_prediction_error: Optional[float] = None
                for record in sensory:
                    stream_id = record.get("stream_id", "")
                    if stream_id == "event.died":
                        died = True
                    elif stream_id == "event.damage_taken":
                        damaged = True
                    elif not record.get("elided") and stream_id in (
                        INTERNAL_NOVELTY_STREAM, INTERNAL_REWARD_PREDICTION_ERROR_STREAM,
                    ):
                        payload = record.get("payload")
                        if isinstance(payload, dict) and isinstance(
                            payload.get("value"), (int, float)
                        ):
                            if stream_id == INTERNAL_NOVELTY_STREAM:
                                novelty = float(payload["value"])
                            else:
                                reward_prediction_error = float(payload["value"])
                    if record.get("elided"):
                        continue
                    tick_buffer.append(stream_event_from_log(record, frame_store=frame_store))
                label_key = _motor_label(motor)
                if label_key in key_to_action:
                    assert fusion is not None
                    latent = fusion.fuse(None, tick_buffer).vector
                    prediction_error = decision.get("prediction_error")
                    episode_samples.append((
                        latent,
                        key_to_action[label_key],
                        reward,
                        died,
                        damaged,
                        float(prediction_error) if prediction_error is not None else None,
                        novelty,
                        reward_prediction_error,
                    ))
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for current, nxt in zip(episode_samples, episode_samples[1:]):
                latent, action, _reward, _died, _damaged, _pe, _novelty, _rpe = current
                (
                    next_latent, _next_action, next_reward, next_died, next_damaged,
                    next_pe, next_novelty, next_rpe,
                ) = nxt
                buffer.add(Transition(
                    latent=latent,
                    action=action,
                    reward=next_reward,
                    next_latent=next_latent,
                    done=next_died,
                    damage=next_damaged,
                    novelty=next_novelty,
                    prediction_error=next_pe,
                    reward_prediction_error=next_rpe,
                    source=f"{session_dir}/{episode_id}",
                ))
                added += 1
                if max_transitions is not None and added >= max_transitions:
                    if frame_store is not None:
                        frame_store.close()
                    return added
        if frame_store is not None:
            frame_store.close()

    return added
