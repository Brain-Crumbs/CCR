"""Build behavioral-cloning datasets from recorded sessions (streams-v2).

Any streams-v2 session -- human demos, scripted traces, replayed successful
episodes -- becomes training data.  Instead of leaning on a recorded
observation dict, the builder reconstructs a ``LatestValueView`` incrementally
while scanning the stream log: at each cognitive tick it emits
``(features(view, motor_history), label)`` where the label is that tick's motor
emission (or ``NULL`` for an empty window).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_runtime.core.streams import LatestValueView, TemporalBuffer
from cognitive_runtime.core.streams.motor import MOTOR_COMMAND_STREAM
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
    require_streams_v2,
)
from cognitive_runtime.training.features import (
    ACTION_KEYS,
    FEATURE_NAMES,
    featurize,
    observation_data_from_streams,
)


@dataclass
class Dataset:
    features: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    sources: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            key = self.action_keys[label]
            counts[key] = counts.get(key, 0) + 1
        return counts


def _motor_label(motor_records: List[Dict[str, Any]]) -> str:
    """The action key emitted this tick, or ``NULL`` for an empty window."""
    for record in motor_records:
        if record.get("stream_id") != MOTOR_COMMAND_STREAM:
            continue
        payload = record.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("action"), str):
            return payload["action"]
    return "NULL"


def _spawn(session_dir: str, episode_id: str) -> Optional[Tuple[float, float]]:
    """The tick-0 position, used to recover ``distance_from_spawn``."""
    for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, episode_id):
        for record in sensory:
            if record.get("stream_id") == "spatial.position" and "payload" in record:
                pos = record["payload"]
                return (pos.get("x", 0.0), pos.get("z", 0.0))
        return None  # only inspect the first window
    return None


def build_dataset(
    session_dirs: List[str],
    history: int = 8,
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> Dataset:
    """Walk recorded sessions and emit (features, action) pairs.

    min_episode_reward filters out weak episodes (summed ``reward.scalar``),
    e.g. keep only successful replays when mixing data sources.
    """
    dataset = Dataset()
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        require_streams_v2(load_session_metadata(session_dir))
        for episode_id in list_episodes(session_dir):
            spawn = _spawn(session_dir, episode_id)
            buffer = TemporalBuffer()
            view = LatestValueView(buffer)
            recent: deque = deque(maxlen=history)
            reward_total = 0.0
            samples: List[Tuple[List[float], int]] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward_total += float(decision.get("reward_window_total", 0.0))
                for record in sensory:
                    if record.get("elided"):
                        continue  # hash-only line: no payload to fold into the view
                    buffer.append(stream_event_from_log(record))
                label_key = _motor_label(motor)
                if label_key in key_to_label:
                    obs_data = observation_data_from_streams(
                        view.to_observation().data, spawn
                    )
                    samples.append(
                        (featurize(obs_data, list(recent)), key_to_label[label_key])
                    )
                recent.append(label_key)
            if (
                min_episode_reward is not None
                and reward_total < min_episode_reward
            ):
                continue
            for feats, label in samples:
                dataset.features.append(feats)
                dataset.labels.append(label)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    return dataset
            dataset.sources.append(f"{session_dir}/{episode_id}")
    return dataset
