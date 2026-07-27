"""Transition priority scoring (promoted out of
``cognitive_runtime.neural.replay_buffer``, issue #96).

``PriorityWeights``, ``Transition`` and :func:`transition_priority` are the
weighted combination of reward/death/damage/novelty/prediction-error/
reward-prediction-error signals that decides which transitions matter most
to keep or replay -- originally defined in ``neural.replay_buffer`` for the
``ReplayBuffer``'s proportional sampling (issue #28).

Torch-free by design (unlike the rest of ``cognitive_runtime.neural``, whose
``import torch`` at module scope makes merely *importing* it require the
optional ``neural`` extra): ``brain.hippocampus``'s episodic seed store
(issue #96) needs this exact scoring on every cognitive tick, including
programs and runs where torch is never installed, so it lives here instead.
``cognitive_runtime.neural.replay_buffer`` re-exports these three names
unchanged so every existing import keeps resolving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = ["PriorityWeights", "Transition", "transition_priority"]

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
