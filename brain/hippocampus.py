"""The Hippocampus: a capacity-bounded, priority-weighted episodic seed
store (docs/v2/phases/phase-4-hippocampus-dreams.md, issue #96).

The missing episodic-memory organ: a fast store of sparse dream *seeds* --
``(z, action-sequence, neuromodulator tags)``, not full frames. Frames come
back later by *decoding* a dream rollout from a stored seed (the sibling
Dreams issue, #97); this module only builds and fills the store.

Built directly over :mod:`cognitive_runtime.core.priority`'s
``Transition``/``PriorityWeights``/``transition_priority`` -- the same
weighted, gracefully-degrading combination of reward/death/damage/novelty/
prediction-error/reward-prediction-error signals the replay buffer already
uses (issue #28) -- rather than reinventing a second priority scheme.
Encoding a seed maps this tick's neuromodulator tags onto a ``Transition``'s
fields: ``reward``/``novelty`` carry over directly, ``surprise`` (calibrated
or raw prediction error) fills ``prediction_error``, ``dopamine`` (the
reward-prediction-error analog, ``brain.neuromod.DOPAMINE_STREAM``) fills
``reward_prediction_error``, and continuous ``threat`` (the amygdala's
adrenaline level) is folded into the boolean ``damage`` flag by the same
threshold convention :class:`brain.arbiter.ArbiterConfig` already uses for
"high pain" -- a threat reading at or above the threshold reads as "painful"
for priority purposes even absent a distinct in-game damage event.

Capacity bounding is priority-based, not FIFO: once full, a new seed only
displaces the store's *lowest*-priority seed, and only if it scores higher
-- the min-heap "keep the top K" idiom, so a bland tick never evicts a
high-surprise/high-reward one just because it arrived later (the phase
doc's acceptance line). This is a different eviction policy from
``ReplayBuffer.add``'s plain ring buffer (which always overwrites the
*oldest* entry regardless of priority) -- the hippocampus is a *salience*
store, not a recency window.

Deliberately torch-free (unlike the rest of ``cognitive_runtime.neural``):
``cognitive_runtime.core.priority`` was promoted out of
``neural.replay_buffer`` for exactly this reason, so the loop can encode a
seed every waking tick -- including runs with no ``neural`` extra installed
-- without pulling in torch.

Context-cued *retrieval* is explicitly deferred (a follow-up per the phase
doc); this module only writes and bounds the store.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cognitive_runtime.core.priority import PriorityWeights, Transition, transition_priority

__all__ = [
    "SeedTags",
    "Seed",
    "HippocampusConfig",
    "Hippocampus",
]


@dataclass(frozen=True)
class SeedTags:
    """The neuromodulator/outcome tags recorded alongside a seed's ``z`` at
    encoding time -- everything :func:`transition_priority` needs, named
    after the streams they come from rather than the ``Transition`` fields
    they end up filling."""

    reward: float = 0.0
    done: bool = False
    damage: bool = False
    novelty: Optional[float] = None
    #: Calibrated (or raw) surprise / prediction error this tick.
    surprise: Optional[float] = None
    #: Dopamine analog (reward-prediction error, `brain.neuromod.DOPAMINE_STREAM`).
    dopamine: Optional[float] = None
    #: Amygdala adrenaline / appraised pain this tick, in `[0, 1)`.
    threat: Optional[float] = None


@dataclass(frozen=True)
class Seed:
    """One stored episodic seed: a sparse ``(z, actions, tags)`` triple plus
    the priority it was scored at when written."""

    z: List[float]
    #: The action-sequence: this tick's emitted actions (`Action.key()`
    #: strings), in emission order -- usually zero or one, occasionally more
    #: for a Program that emits multiple motor commands per cognitive tick.
    actions: List[str]
    tags: SeedTags
    priority: float
    tick_index: int
    source: str = ""


@dataclass(frozen=True)
class HippocampusConfig:
    capacity: int = 2000
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    #: A `SeedTags.threat` reading at or above this counts as "damage" for
    #: priority purposes -- the same cutover `brain.arbiter.ArbiterConfig.
    #: pain_threshold` uses for "high pain" (module docstring).
    threat_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity!r}")
        if not 0.0 <= self.threat_threshold <= 1.0:
            raise ValueError(
                f"threat_threshold must be in [0, 1], got {self.threat_threshold!r}"
            )


def _seed_priority(tags: SeedTags, config: HippocampusConfig) -> float:
    """Scores `tags` via `transition_priority`, folding continuous `threat`
    into the boolean `damage` flag (module docstring)."""
    threat_is_high = tags.threat is not None and tags.threat >= config.threat_threshold
    transition = Transition(
        latent=[],
        action=0,
        reward=tags.reward,
        next_latent=[],
        done=tags.done,
        damage=tags.damage or threat_is_high,
        novelty=tags.novelty,
        prediction_error=tags.surprise,
        reward_prediction_error=tags.dopamine,
    )
    return transition_priority(transition, config.weights)


class Hippocampus:
    """A bounded min-heap of the highest-priority seeds encoded so far.

    `encode()` is a cheap, one-shot write: scoring is a handful of
    arithmetic ops and insertion is `O(log capacity)`, so it is safe to call
    every cognitive tick from the loop's per-tick record path without
    stalling it.
    """

    def __init__(self, config: Optional[HippocampusConfig] = None):
        self.config = config or HippocampusConfig()
        # A min-heap of (priority, insertion_order, seed): the lowest-
        # priority entry is always at index 0, so eviction is "would this
        # new seed outrank the current minimum?".  `insertion_order` breaks
        # ties deterministically (Seed isn't itself ordered) and keeps
        # heapq from ever comparing two Seeds directly.
        self._heap: List[Tuple[float, int, Seed]] = []
        self._counter = itertools.count()
        self.total_encoded = 0
        self.total_evicted = 0
        self.total_skipped = 0

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def capacity(self) -> int:
        return self.config.capacity

    def seeds(self) -> Tuple[Seed, ...]:
        """Read-only snapshot of the store's contents, highest-priority
        first."""
        return tuple(s for _, _, s in sorted(self._heap, key=lambda entry: -entry[0]))

    def encode(
        self,
        z: Sequence[float],
        actions: Sequence[str],
        tags: SeedTags,
        *,
        tick_index: int = 0,
        source: str = "",
    ) -> Optional[Seed]:
        """One tick's `(z, actions, tags)`; scores and stores it, evicting
        the current lowest-priority seed if the store is full and this one
        outranks it. Returns the stored `Seed`, or `None` if the store was
        full and this seed didn't score high enough to displace anything
        (`total_skipped` still counts it)."""
        priority = _seed_priority(tags, self.config)
        seed = Seed(
            z=list(z),
            actions=list(actions),
            tags=tags,
            priority=priority,
            tick_index=tick_index,
            source=source,
        )
        self.total_encoded += 1
        entry = (priority, next(self._counter), seed)
        if len(self._heap) < self.config.capacity:
            heapq.heappush(self._heap, entry)
            return seed
        if priority > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            self.total_evicted += 1
            return seed
        self.total_skipped += 1
        return None

    def reset(self) -> None:
        self._heap.clear()
        self._counter = itertools.count()
        self.total_encoded = 0
        self.total_evicted = 0
        self.total_skipped = 0

    def state_dict(self) -> Dict[str, object]:
        """Counters only, matching `ReplayBuffer.state_dict`'s convention
        (issue #28's checkpoint doc) -- contents are cheap to refill from
        the next episode's wake ticks."""
        return {
            "capacity": self.config.capacity,
            "size": len(self._heap),
            "total_encoded": self.total_encoded,
            "total_evicted": self.total_evicted,
            "total_skipped": self.total_skipped,
        }
