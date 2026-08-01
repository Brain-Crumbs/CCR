"""Action-burst policy: sample an action uniformly from a fixed subset and
hold it for a uniformly sampled number of ticks before resampling.

Motor babbling (epic #212 §12.2, issue #235): resampling a new random action
every tick produces a near-stationary agent whose position barely
decorrelates from one tick to the next. Holding each sampled action for a
short burst instead is what creates the multi-tick displacement ego-motion
learning needs. Shared by ``training.nursery``'s ``motor_babbling_open`` and
(per the epic doc) the future ``motor_babbling_walls``.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from cognitive_runtime.core.action import Action, NULL_ACTION
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction


class ActionBurstPolicy(SingleActionPolicy):
    """Uniformly sample an action from ``action_space``, hold it for a
    uniformly sampled ``[min_burst_ticks, max_burst_ticks]`` ticks, then
    resample -- deterministic given ``seed``."""

    name = "action-burst"

    def __init__(
        self,
        action_space: Sequence[Action],
        *,
        min_burst_ticks: int = 1,
        max_burst_ticks: int = 4,
        seed: int = 0,
    ):
        if not action_space:
            raise ValueError("action_space must not be empty")
        if min_burst_ticks < 1:
            raise ValueError(f"min_burst_ticks must be >= 1, got {min_burst_ticks!r}")
        if max_burst_ticks < min_burst_ticks:
            raise ValueError(
                f"max_burst_ticks ({max_burst_ticks!r}) must be >= "
                f"min_burst_ticks ({min_burst_ticks!r})"
            )
        self.action_space: List[Action] = list(action_space)
        self.min_burst_ticks = int(min_burst_ticks)
        self.max_burst_ticks = int(max_burst_ticks)
        self.seed = seed
        self.rng = random.Random(seed)
        self._current: Optional[Action] = None
        self._remaining = 0

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self._current = None
        self._remaining = 0

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        if self._remaining <= 0:
            self._current = self.rng.choice(self.action_space)
            self._remaining = self.rng.randint(self.min_burst_ticks, self.max_burst_ticks)
        self._remaining -= 1
        assert self._current is not None
        return self._current


class TerminalRecoveryActionBurstPolicy(ActionBurstPolicy):
    """An action-burst policy with an explicit, finite terminal recovery.

    ``motor_babbling_walls`` needs its random exploration to finish with a
    guaranteed opportunity to leave any wall it happens to be touching.  The
    normal prefix remains the same seeded, uniformly sampled 1--4 tick burst
    process as :class:`ActionBurstPolicy`; only the declared suffix is
    scripted.  This keeps the quality gate from mistaking an unlucky final
    burst for useful blocked-action evidence.
    """

    name = "terminal-recovery-action-burst"

    def __init__(
        self,
        action_space: Sequence[Action],
        *,
        episode_ticks: int,
        terminal_actions: Sequence[Action],
        min_burst_ticks: int = 1,
        max_burst_ticks: int = 4,
        seed: int = 0,
    ):
        super().__init__(
            action_space,
            min_burst_ticks=min_burst_ticks,
            max_burst_ticks=max_burst_ticks,
            seed=seed,
        )
        if episode_ticks <= 0:
            raise ValueError(f"episode_ticks must be positive, got {episode_ticks!r}")
        if not terminal_actions:
            raise ValueError("terminal_actions must be non-empty")
        if len(terminal_actions) >= episode_ticks:
            raise ValueError(
                "terminal_actions must leave one episode tick for the runtime's motor delay"
            )
        unknown = set(terminal_actions) - set(self.action_space)
        if unknown:
            raise ValueError(f"terminal_actions must belong to action_space, got {unknown!r}")
        self.episode_ticks = int(episode_ticks)
        self.terminal_actions = tuple(terminal_actions)
        self._tick = 0

    def reset(self) -> None:
        super().reset()
        self._tick = 0

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        # CognitiveRuntime steps the world before asking the policy to emit,
        # so an action emitted on tick N is applied on tick N + 1.  Start the
        # suffix one tick early and emit NULL on the final (unapplied) policy
        # tick.  This makes every declared recovery action affect the world
        # exactly once without recording a misleading duplicate command.
        terminal_start = self.episode_ticks - len(self.terminal_actions) - 1
        if self._tick >= self.episode_ticks - 1:
            action = NULL_ACTION
        elif self._tick >= terminal_start:
            index = self._tick - terminal_start
            action = self.terminal_actions[index]
        else:
            action = super().decide(state, memory, prediction)
        self._tick += 1
        return action
