"""Build behavioral-cloning datasets from recorded sessions.

Any session recorded with `record_observations=True` -- human demos,
scripted traces, replayed successful episodes -- becomes training data:
(observation features, chosen action) pairs.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognitive_runtime.runtime.replay import list_episodes, load_episode
from cognitive_runtime.training.features import ACTION_KEYS, FEATURE_NAMES, featurize


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


def build_dataset(
    session_dirs: List[str],
    history: int = 8,
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> Dataset:
    """Walk recorded sessions and emit (features, action) pairs.

    min_episode_reward filters out weak episodes (e.g. keep only successful
    replays when mixing data sources).
    """
    dataset = Dataset()
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        for episode_id in list_episodes(session_dir):
            records, summary = load_episode(session_dir, episode_id)
            if (
                min_episode_reward is not None
                and float(summary.get("total_reward", 0.0)) < min_episode_reward
            ):
                continue
            recent: deque = deque(maxlen=history)
            for record in records:
                observation = record.get("observation")
                action_key = record.get("selected_action")
                if observation is None or action_key not in key_to_label:
                    continue
                dataset.features.append(featurize(observation.get("data", {}), list(recent)))
                dataset.labels.append(key_to_label[action_key])
                recent.append(action_key)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    return dataset
            dataset.sources.append(f"{session_dir}/{episode_id}")
    return dataset
