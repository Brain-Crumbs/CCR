"""Short-term runtime memory (loop v2: stream-native).

Rebuilt around the Phase-0 :class:`TemporalBuffer`: instead of a window of
whole states, memory holds a bounded per-stream history of the sensory
events that have arrived, plus the latest latent tokens and the motor
emissions the policy has made.  It exposes the same generic signals the
policies and world model use (novelty over cognitive-tick windows,
repetition over motor emissions, per-stream numeric trends) with no
environment-specific logic.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Deque, Dict, List, Optional, Set

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.attention import AttentionState
from cognitive_runtime.core.hashing import canonical_json, hash_payload
from cognitive_runtime.core.streams.encoder_registry import LatentToken
from cognitive_runtime.core.streams.fusion import LatentState
from cognitive_runtime.core.streams.shim import LatestValueView
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer


def window_hash(window: TickWindow) -> str:
    """Content hash of a cognitive-tick window, for novelty detection.

    Hashes stream ids + payloads only (never sequence numbers or timestamps,
    which are unique every tick) so that a recurring *sensory situation*
    hashes the same — mirroring the intent of ``Observation.hash()``.  An
    ndarray payload (a pixel frame) contributes its content hash rather than
    a JSON dump, so a repeated frame is still recognized as the same
    situation without re-serializing it.
    """
    items = sorted(
        (event.stream_id, hash_payload(event.payload))
        for event in window.events
    )
    return hashlib.sha1(canonical_json(items).encode("utf-8")).hexdigest()


class Memory:
    def __init__(
        self,
        capacity: int = 512,
        capacity_by_modality: Optional[Dict[str, int]] = None,
    ):
        self.capacity = capacity
        self.buffer = TemporalBuffer(
            default_capacity=capacity, capacity_by_modality=capacity_by_modality
        )
        self.actions: Deque[Action] = deque(maxlen=capacity)
        self.window_hashes: Deque[str] = deque(maxlen=capacity)
        self._seen_hashes: Set[str] = set()
        self._novel_last_update = True
        self._latent_tokens: List[LatentToken] = []
        self._latent_state: Optional[LatentState] = None
        self._attention_state: Optional[AttentionState] = None

    def reset(self) -> None:
        self.buffer.reset()
        self.actions.clear()
        self.window_hashes.clear()
        self._seen_hashes.clear()
        self._novel_last_update = True
        self._latent_tokens = []
        self._latent_state = None
        self._attention_state = None

    # ------------------------------------------------------------- updates

    def update(self, window: TickWindow, tokens: Optional[List[LatentToken]] = None) -> None:
        """Absorb one cognitive-tick window and its encoded latent tokens."""
        self.buffer.extend(window.events)
        self._latent_tokens = list(tokens or [])
        digest = window_hash(window)
        self._novel_last_update = digest not in self._seen_hashes
        self._seen_hashes.add(digest)
        self.window_hashes.append(digest)

    def record_action(self, action: Action) -> None:
        self.actions.append(action)

    def record_actions(self, actions: List[Action]) -> None:
        """Record a cognitive tick's motor emissions; `[]` records a NULL."""
        self.actions.append(actions[0] if actions else NULL_ACTION)

    # ------------------------------------------------------------- readouts

    def latest_values(self) -> LatestValueView:
        """Observation-shaped view over the latest value of each stream."""
        return LatestValueView(self.buffer)

    def latent_tokens(self) -> List[LatentToken]:
        """The most recent per-stream encoded tokens for this window."""
        return self._latent_tokens

    def set_fused_latent(self, state: LatentState) -> None:
        """Store the fused latent state produced by ``TemporalFusion``."""
        self._latent_state = state

    def fused_latent(self) -> Optional[LatentState]:
        """The most recent fused :class:`LatentState`, if the loop computed one."""
        return self._latent_state

    def set_attention_state(self, state: AttentionState) -> None:
        """Store the tick's :class:`AttentionState` (issue #59)."""
        self._attention_state = state

    def attention_state(self) -> Optional[AttentionState]:
        """The most recent :class:`AttentionState`, if the loop computed one."""
        return self._attention_state

    def last_actions(self, n: int) -> List[Action]:
        return list(self.actions)[-n:]

    def repeated_action_streak(self) -> int:
        """Length of the trailing run of identical motor emissions."""
        streak = 0
        last = None
        for action in reversed(self.actions):
            if last is None:
                last = action
            if action != last:
                break
            streak += 1
        return streak

    def novelty_rate(self, window: int = 64) -> float:
        """Fraction of unique window hashes in the recent cognitive-tick window."""
        recent = list(self.window_hashes)[-window:]
        if not recent:
            return 1.0
        return len(set(recent)) / len(recent)

    @property
    def last_observation_was_novel(self) -> bool:
        return self._novel_last_update

    def stream_trend(self, stream_id: str, window: int = 16) -> float:
        """Slope (per event) of a numeric stream's payload over a short window.

        Non-numeric payloads contribute nothing; a stream with fewer than two
        numeric samples has trend 0.
        """
        events = self.buffer.window(stream_id, window)
        values = [
            float(e.payload)
            for e in events
            if isinstance(e.payload, (int, float)) and not isinstance(e.payload, bool)
        ]
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / (len(values) - 1)
