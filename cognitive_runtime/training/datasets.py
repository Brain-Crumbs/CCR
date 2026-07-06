"""Build behavioral-cloning datasets from recorded sessions (streams-v2).

Two feature representations share one scan of the stream log:

- ``latent`` (default) replays the **same** encoder + fusion pipeline the loop
  runs online, so train-time and inference-time features come from literally
  the same code path (`TemporalFusion.fuse`).
- ``handcrafted`` keeps the Minecraft featurizer available for A/B comparison.

Either way the label is the tick's motor emission (or ``NULL`` for an empty
window) and ``min_episode_reward`` filters on summed ``reward.scalar``.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_runtime.core.streams import (
    LatestValueView,
    TemporalBuffer,
    TemporalFusion,
)
from cognitive_runtime.core.streams.events import StreamSpec
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
    latent_feature_names,
    latent_features,
    observation_data_from_streams,
)

LATENT = "latent"
HANDCRAFTED = "handcrafted"


@dataclass
class Dataset:
    features: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    sources: List[str] = field(default_factory=list)
    representation: str = HANDCRAFTED
    layout_hash: Optional[str] = None

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
    """The tick-0 position, used to recover ``distance_from_spawn`` (handcrafted)."""
    for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, episode_id):
        for record in sensory:
            if record.get("stream_id") == "spatial.position" and "payload" in record:
                pos = record["payload"]
                return (pos.get("x", 0.0), pos.get("z", 0.0))
        return None
    return None


def _catalog(metadata: Dict[str, Any]) -> List[StreamSpec]:
    return [StreamSpec.from_dict(s) for s in metadata.get("stream_catalog", [])]


def build_dataset(
    session_dirs: List[str],
    history: int = 8,
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
    representation: str = LATENT,
) -> Dataset:
    """Walk recorded sessions and emit (features, action) pairs."""
    if representation not in (LATENT, HANDCRAFTED):
        raise ValueError(f"unknown representation {representation!r}")
    dataset = Dataset(representation=representation)
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        if representation == LATENT and fusion is None:
            fusion = TemporalFusion(_catalog(metadata))
            dataset.feature_names = latent_feature_names(fusion.feature_names())
            dataset.layout_hash = fusion.layout_hash

        for episode_id in list_episodes(session_dir):
            spawn = _spawn(session_dir, episode_id) if representation == HANDCRAFTED else None
            buffer = TemporalBuffer()
            view = LatestValueView(buffer)
            recent: deque = deque(maxlen=history)
            reward_total = 0.0
            samples: List[Tuple[List[float], int]] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward_total += float(decision.get("reward_window_total", 0.0))
                for record in sensory:
                    if record.get("elided"):
                        continue
                    buffer.append(stream_event_from_log(record))
                label_key = _motor_label(motor)
                if label_key in key_to_label:
                    if representation == LATENT:
                        assert fusion is not None
                        feats = latent_features(
                            fusion.fuse(None, buffer).vector, list(recent)
                        )
                    else:
                        obs = observation_data_from_streams(view.to_observation().data, spawn)
                        feats = featurize(obs, list(recent))
                    samples.append((feats, key_to_label[label_key]))
                recent.append(label_key)
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for feats, label in samples:
                dataset.features.append(feats)
                dataset.labels.append(label)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    return dataset
            dataset.sources.append(f"{session_dir}/{episode_id}")
    return dataset
