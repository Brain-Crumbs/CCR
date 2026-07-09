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
import sys
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
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
    require_streams_v2,
)
from cognitive_runtime.programs.minecraft.streams import PIXEL_STREAM
from cognitive_runtime.training.features import (
    ACTION_KEYS,
    FEATURE_NAMES,
    featurize,
    latent_feature_names,
    latent_features,
    motor_history_features,
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
    elided_layout_streams: set = set()

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        if representation == LATENT:
            if fusion is None:
                fusion = TemporalFusion(_catalog(metadata))
                dataset.feature_names = latent_feature_names(fusion.feature_names())
                dataset.layout_hash = fusion.layout_hash
            else:
                session_fusion = TemporalFusion(_catalog(metadata))
                if session_fusion.layout_hash != fusion.layout_hash:
                    raise ValueError(
                        f"session {session_dir} has an incompatible stream catalog "
                        f"(fusion layout {session_fusion.layout_hash} vs "
                        f"{fusion.layout_hash}); train on sessions recorded with "
                        "the same program config"
                    )

        frame_store = open_frame_store(session_dir)
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
                        stream_id = record.get("stream_id", "")
                        if fusion is not None and any(
                            e.stream_id == stream_id for e in fusion.layout
                        ):
                            elided_layout_streams.add(stream_id)
                        continue
                    buffer.append(stream_event_from_log(record, frame_store=frame_store))
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
        if frame_store is not None:
            frame_store.close()
    if elided_layout_streams:
        print(
            "warning: these streams were recorded hash-only (payload elided) and "
            "contribute nothing to the latent features: "
            + ", ".join(sorted(elided_layout_streams))
            + " — record training sessions with --record-frames / --record-streams "
            "to include them",
            file=sys.stderr,
        )
    return dataset


# --------------------------------------------------------------------------
# Neural (pixel) dataset: raw pixel frames + the fused NON-vision vector.
#
# The CNN is the whole visual pathway, so the fused-scalar half deliberately
# drops every ``vision.*`` stream (grid + entities + the pixel frame itself);
# the model learns its own vision from the pixels instead of leaning on the
# heuristic grid encoder.
# --------------------------------------------------------------------------

NEURAL_PIXELS = "neural_pixels"


def _non_vision_fusion(metadata: Dict[str, Any]) -> TemporalFusion:
    specs = [s for s in _catalog(metadata) if s.modality != "vision"]
    return TemporalFusion(specs)


@dataclass
class NeuralDataset:
    """Pixel frames + fused non-vision vectors + motor history + action labels."""

    pixels: List[Any] = field(default_factory=list)  # per sample, H x W x C ndarray
    non_vision: List[List[float]] = field(default_factory=list)
    motor: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    non_vision_names: List[str] = field(default_factory=list)
    motor_names: List[str] = field(default_factory=lambda: [f"last_action:{k}" for k in ACTION_KEYS])
    layout_hash: Optional[str] = None
    pixel_shape: Optional[Tuple[int, int, int]] = None
    sources: List[str] = field(default_factory=list)
    representation: str = NEURAL_PIXELS

    def __len__(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            key = self.action_keys[label]
            counts[key] = counts.get(key, 0) + 1
        return counts


def _pixel_shape_from_catalog(metadata: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    for spec in _catalog(metadata):
        if spec.stream_id == PIXEL_STREAM and spec.shape is not None:
            return tuple(spec.shape)  # type: ignore[return-value]
    return None


def build_neural_dataset(
    session_dirs: List[str],
    history: int = 8,
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> NeuralDataset:
    """Walk recorded sessions and emit (pixels, non_vision, motor, action) samples.

    Requires the pixel frames to be present in the log (record with
    ``--record-frames``); a session that elided them cannot train pixel vision
    and raises rather than training on blank frames.
    """
    dataset = NeuralDataset()
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    pixels_were_elided = False

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = _non_vision_fusion(metadata)
        if fusion is None:
            fusion = session_fusion
            dataset.non_vision_names = list(fusion.feature_names())
            dataset.layout_hash = fusion.layout_hash
            dataset.pixel_shape = _pixel_shape_from_catalog(metadata)
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible non-vision stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); train on "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            buffer = TemporalBuffer()
            recent: deque = deque(maxlen=history)
            reward_total = 0.0
            samples: List[Tuple[Any, List[float], List[float], int]] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward_total += float(decision.get("reward_window_total", 0.0))
                for record in sensory:
                    if record.get("elided"):
                        if record.get("stream_id") == PIXEL_STREAM:
                            pixels_were_elided = True
                        continue
                    buffer.append(stream_event_from_log(record, frame_store=frame_store))
                label_key = _motor_label(motor)
                latest_pixels = buffer.latest(PIXEL_STREAM)
                if label_key in key_to_label and latest_pixels is not None:
                    non_vision_vec = fusion.fuse(None, buffer).vector
                    motor_hist = motor_history_features(list(recent))
                    samples.append(
                        (latest_pixels.payload, non_vision_vec, motor_hist, key_to_label[label_key])
                    )
                recent.append(label_key)
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for pixels, non_vision_vec, motor_hist, label in samples:
                dataset.pixels.append(pixels)
                dataset.non_vision.append(non_vision_vec)
                dataset.motor.append(motor_hist)
                dataset.labels.append(label)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    return dataset
            dataset.sources.append(f"{session_dir}/{episode_id}")
        if frame_store is not None:
            frame_store.close()

    if len(dataset) == 0 and pixels_were_elided:
        raise ValueError(
            f"no pixel samples: the {PIXEL_STREAM} stream was recorded hash-only. "
            "Re-record the training sessions with --record-frames so pixel vision "
            "has frames to learn from."
        )
    return dataset
