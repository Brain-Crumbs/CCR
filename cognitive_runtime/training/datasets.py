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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def build_latent_fusion_dataset(
    session_dirs: List[str],
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> LatentFusionDataset:
    """Walk recorded sessions and emit samples for learned fusion training.

    The current and next inputs are produced with the same ``TemporalFusion``
    layout as the online baseline, plus current-window presence masks and
    recency/staleness scalars.  The final tick of each episode is omitted
    because it has no next latent target.
    """

    dataset = LatentFusionDataset()
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    elided_layout_streams: set = set()

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = TemporalFusion(_catalog(metadata))
        if fusion is None:
            fusion = session_fusion
            dataset.layout_hash = fusion.layout_hash
            dataset.feature_names = list(fusion.feature_names())
            dataset.stream_ids = [entry.stream_id for entry in fusion.layout]
            dataset.stream_slices = _stream_slices(fusion)
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); train on "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            buffer = TemporalBuffer()
            reward_total = 0.0
            episode_samples: List[
                Tuple[List[float], List[float], List[float], List[float], int, float]
            ] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward = float(decision.get("reward_window_total", 0.0))
                reward_total += reward
                present_streams = []
                for record in sensory:
                    stream_id = record.get("stream_id", "")
                    if record.get("elided"):
                        if fusion is not None and any(
                            e.stream_id == stream_id for e in fusion.layout
                        ):
                            elided_layout_streams.add(stream_id)
                        continue
                    buffer.append(stream_event_from_log(record, frame_store=frame_store))
                    present_streams.append(stream_id)
                label_key = _motor_label(motor)
                if label_key in key_to_label:
                    assert fusion is not None
                    latent = fusion.fuse(None, buffer).vector
                    mask, recent, stale = _fusion_aux_lists(fusion, buffer, present_streams)
                    episode_samples.append(
                        (
                            latent,
                            mask,
                            recent,
                            stale,
                            key_to_label[label_key],
                            reward,
                        )
                    )
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for current, nxt in zip(episode_samples, episode_samples[1:]):
                latent, mask, recent, stale, label, reward = current
                next_latent, next_mask, next_recent, next_stale, _next_label, _next_reward = nxt
                dataset.latents.append(latent)
                dataset.presence_masks.append(mask)
                dataset.recency.append(recent)
                dataset.staleness.append(stale)
                dataset.labels.append(label)
                dataset.rewards.append(reward)
                dataset.next_latents.append(next_latent)
                dataset.next_presence_masks.append(next_mask)
                dataset.next_recency.append(next_recent)
                dataset.next_staleness.append(next_stale)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    if frame_store is not None:
                        frame_store.close()
                    return dataset
            if len(episode_samples) >= 2:
                dataset.sources.append(f"{session_dir}/{episode_id}")
        if frame_store is not None:
            frame_store.close()

    if elided_layout_streams:
        print(
            "warning: these streams were recorded hash-only (payload elided) and "
            "contribute no learned-fusion input this tick: "
            + ", ".join(sorted(elided_layout_streams))
            + " -- record training sessions with --record-frames / --record-streams "
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
PIXEL_SEQUENCES = "pixel_sequences"
LATENT_FUSION = "latent_fusion"
WORLD_MODEL = "world_model"
MULTI_HORIZON_WORLD_MODEL = "multi_horizon_world_model"

#: Issue #32 ablation profiles for the neural (pixel) dataset's non-vision
#: companion vector:
#:  - "full"  every non-vision stream the (generic) default registry fuses,
#:            including hand-computed semantic scalars (front_block,
#:            sheltered) -- "pixels + semantics".
#:  - "raw"   only streams the Minecraft stream registry classifies
#:            agent_input (body/reward/spatial proprioception) -- "pixel
#:            only" (plus the minimal self-state a raw sensorimotor agent
#:            has, since pixels alone carry no vitals/reward/pose).
NEURAL_INPUT_PROFILES = frozenset({"full", "raw"})
NEURAL_FULL = "full"
NEURAL_RAW = "raw"


def _non_vision_fusion(metadata: Dict[str, Any], stream_profile: str = NEURAL_FULL) -> TemporalFusion:
    if stream_profile not in NEURAL_INPUT_PROFILES:
        raise ValueError(
            f"unknown stream_profile {stream_profile!r}; expected one of "
            f"{sorted(NEURAL_INPUT_PROFILES)}"
        )
    specs = [s for s in _catalog(metadata) if s.modality != "vision"]
    if stream_profile == NEURAL_RAW:
        from cognitive_runtime.programs.minecraft.stream_registry import (
            MINECRAFT_STREAM_REGISTRY,
        )

        return TemporalFusion(
            specs, MINECRAFT_STREAM_REGISTRY.to_encoder_registry(classifications={"agent_input"})
        )
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
    stream_profile: str = NEURAL_FULL

    def __len__(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            key = self.action_keys[label]
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class PixelSequenceDataset:
    """Adjacent pixel-frame pairs for offline visual representation learning."""

    pixels: List[Any] = field(default_factory=list)
    next_pixels: List[Any] = field(default_factory=list)
    pixel_shape: Optional[Tuple[int, int, int]] = None
    layout_hash: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    representation: str = PIXEL_SEQUENCES

    def __len__(self) -> int:
        return len(self.pixels)


@dataclass
class LatentFusionDataset:
    """Recorded-session samples for learned latent fusion.

    Each sample contains the fixed ``TemporalFusion`` vector plus the learned
    fusion side channels for the current tick, the demonstrated action label,
    the tick reward, and the next tick's inputs for next-fused-latent training.
    Stored as plain lists so importing this module never imports torch.
    """

    latents: List[List[float]] = field(default_factory=list)
    presence_masks: List[List[float]] = field(default_factory=list)
    recency: List[List[float]] = field(default_factory=list)
    staleness: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_latents: List[List[float]] = field(default_factory=list)
    next_presence_masks: List[List[float]] = field(default_factory=list)
    next_recency: List[List[float]] = field(default_factory=list)
    next_staleness: List[List[float]] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    stream_ids: List[str] = field(default_factory=list)
    stream_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    layout_hash: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    representation: str = LATENT_FUSION

    def __len__(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            key = self.action_keys[label]
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class WorldModelDataset:
    """Recorded-session samples for the action-conditioned neural world model
    (Phase D, issue #26).

    Each sample is one ``(fused_latent_t, action_t) -> (fused_latent_{t+1},
    reward_{t+1}, died_{t+1}, risk_{t+1})`` transition, using the same
    deterministic ``TemporalFusion`` vector the runtime loop already computes
    each tick (``memory.fused_latent()``) -- not the learned
    ``LatentFusionModel`` -- so the bridge into the loop needs no new plumbing.

    ``rewards``/``dones``/``risks`` are read off the *next* tick's decision
    record and sensory events, since that is the tick whose state and
    ``reward.scalar``/``event.died``/``event.damage_taken`` streams are the
    causal consequence of ``labels[i]`` applied from ``latents[i]`` (the
    runtime drains the motor bus for tick ``t``'s action during the physics
    steps leading into tick ``t+1``'s sensory window).
    """

    latents: List[List[float]] = field(default_factory=list)
    next_latents: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    risks: List[float] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    feature_names: List[str] = field(default_factory=list)
    layout_hash: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    representation: str = WORLD_MODEL

    def __len__(self) -> int:
        return len(self.labels)

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            key = self.action_keys[label]
            counts[key] = counts.get(key, 0) + 1
        return counts

    def death_count(self) -> int:
        return sum(1 for d in self.dones if d >= 0.5)


def build_world_model_dataset(
    session_dirs: List[str],
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> WorldModelDataset:
    """Walk recorded sessions and emit action-conditioned world-model samples.

    Ticks with an empty (``NULL``) motor window are dropped, same as the
    other latent dataset builders, so a sample pair may skip a few raw ticks;
    ``labels[i]`` is always the action that was actually taken between
    ``latents[i]`` and ``next_latents[i]``.
    """

    dataset = WorldModelDataset()
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    elided_layout_streams: set = set()

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = TemporalFusion(_catalog(metadata))
        if fusion is None:
            fusion = session_fusion
            dataset.layout_hash = fusion.layout_hash
            dataset.feature_names = list(fusion.feature_names())
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); train on "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            buffer = TemporalBuffer()
            reward_total = 0.0
            episode_samples: List[Tuple[List[float], int, float, bool, bool]] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward = float(decision.get("reward_window_total", 0.0))
                reward_total += reward
                died = False
                damaged = False
                for record in sensory:
                    stream_id = record.get("stream_id", "")
                    if stream_id == "event.died":
                        died = True
                    elif stream_id == "event.damage_taken":
                        damaged = True
                    if record.get("elided"):
                        if fusion is not None and any(
                            e.stream_id == stream_id for e in fusion.layout
                        ):
                            elided_layout_streams.add(stream_id)
                        continue
                    buffer.append(stream_event_from_log(record, frame_store=frame_store))
                label_key = _motor_label(motor)
                if label_key in key_to_label:
                    assert fusion is not None
                    latent = fusion.fuse(None, buffer).vector
                    episode_samples.append(
                        (latent, key_to_label[label_key], reward, died, damaged)
                    )
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for current, nxt in zip(episode_samples, episode_samples[1:]):
                latent, label, _reward, _died, _damaged = current
                next_latent, _next_label, next_reward, next_died, next_damaged = nxt
                dataset.latents.append(latent)
                dataset.next_latents.append(next_latent)
                dataset.labels.append(label)
                dataset.rewards.append(next_reward)
                dataset.dones.append(1.0 if next_died else 0.0)
                dataset.risks.append(1.0 if (next_died or next_damaged) else 0.0)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    if frame_store is not None:
                        frame_store.close()
                    return dataset
            if len(episode_samples) >= 2:
                dataset.sources.append(f"{session_dir}/{episode_id}")
        if frame_store is not None:
            frame_store.close()

    if elided_layout_streams:
        print(
            "warning: these streams were recorded hash-only (payload elided) and "
            "contribute nothing to the world-model input this tick: "
            + ", ".join(sorted(elided_layout_streams))
            + " -- record training sessions with --record-frames / --record-streams "
            "to include them",
            file=sys.stderr,
        )
    return dataset


@dataclass
class MultiHorizonWorldModelDataset:
    """Recorded-session samples for the multi-horizon world model (issue
    #39): each sample is one ``(fused_latent_t, action_t)`` input plus, for
    every ``h`` in ``horizons``, the realized target ``h`` ticks ahead --
    ``next_latent``, accumulated reward, "died by t+h", and "damaged-or-died
    by t+h".

    Horizons count *action ticks*, the same indexing
    :func:`build_world_model_dataset` uses for its single-step ``t+1``: ticks
    with an empty (``NULL``) motor window are dropped when building
    ``episode_samples``, so ``h=5`` means "5 ticks with a non-NULL action
    later in this episode", not literally 5 environment ticks. Only samples
    with enough remaining action ticks in their episode for the *largest*
    horizon are kept, so every sample has a target at every horizon.
    """

    horizons: List[int] = field(default_factory=lambda: [1, 4, 8])
    latents: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    future_latents: Dict[int, List[List[float]]] = field(default_factory=dict)
    future_rewards: Dict[int, List[float]] = field(default_factory=dict)
    future_dones: Dict[int, List[float]] = field(default_factory=dict)
    future_risks: Dict[int, List[float]] = field(default_factory=dict)
    action_keys: List[str] = field(default_factory=lambda: list(ACTION_KEYS))
    feature_names: List[str] = field(default_factory=list)
    layout_hash: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    representation: str = MULTI_HORIZON_WORLD_MODEL

    def __len__(self) -> int:
        return len(self.labels)

    def death_count(self, horizon: int) -> int:
        return sum(1 for d in self.future_dones[horizon] if d >= 0.5)


def build_multi_horizon_world_model_dataset(
    session_dirs: List[str],
    horizons: Iterable[int] = (1, 4, 8),
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> MultiHorizonWorldModelDataset:
    """Walk recorded sessions and emit multi-horizon world-model samples.

    Reuses the same per-episode ``(latent, label, reward, died, damaged)``
    scan as :func:`build_world_model_dataset`, then windows it at every
    configured horizon instead of only ``t+1``.
    """

    horizons_sorted = sorted({int(h) for h in horizons})
    if not horizons_sorted or horizons_sorted[0] <= 0:
        raise ValueError(f"horizons must be positive tick offsets, got {list(horizons)!r}")
    max_horizon = horizons_sorted[-1]

    dataset = MultiHorizonWorldModelDataset(horizons=horizons_sorted)
    for h in horizons_sorted:
        dataset.future_latents[h] = []
        dataset.future_rewards[h] = []
        dataset.future_dones[h] = []
        dataset.future_risks[h] = []

    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    elided_layout_streams: set = set()

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = TemporalFusion(_catalog(metadata))
        if fusion is None:
            fusion = session_fusion
            dataset.layout_hash = fusion.layout_hash
            dataset.feature_names = list(fusion.feature_names())
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); train on "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            buffer = TemporalBuffer()
            reward_total = 0.0
            episode_samples: List[Tuple[List[float], int, float, bool, bool]] = []
            for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                reward = float(decision.get("reward_window_total", 0.0))
                reward_total += reward
                died = False
                damaged = False
                for record in sensory:
                    stream_id = record.get("stream_id", "")
                    if stream_id == "event.died":
                        died = True
                    elif stream_id == "event.damage_taken":
                        damaged = True
                    if record.get("elided"):
                        if fusion is not None and any(
                            e.stream_id == stream_id for e in fusion.layout
                        ):
                            elided_layout_streams.add(stream_id)
                        continue
                    buffer.append(stream_event_from_log(record, frame_store=frame_store))
                label_key = _motor_label(motor)
                if label_key in key_to_label:
                    assert fusion is not None
                    latent = fusion.fuse(None, buffer).vector
                    episode_samples.append(
                        (latent, key_to_label[label_key], reward, died, damaged)
                    )
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue

            n = len(episode_samples)
            for i in range(n - max_horizon):
                latent, label, _reward, _died, _damaged = episode_samples[i]
                dataset.latents.append(latent)
                dataset.labels.append(label)
                for h in horizons_sorted:
                    window = episode_samples[i + 1 : i + 1 + h]
                    future_latent, _label, _r, _d, _dm = window[-1]
                    dataset.future_latents[h].append(future_latent)
                    dataset.future_rewards[h].append(sum(w[2] for w in window))
                    dataset.future_dones[h].append(1.0 if any(w[3] for w in window) else 0.0)
                    dataset.future_risks[h].append(
                        1.0 if any(w[3] or w[4] for w in window) else 0.0
                    )
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    if frame_store is not None:
                        frame_store.close()
                    return dataset
            if n > max_horizon:
                dataset.sources.append(f"{session_dir}/{episode_id}")
        if frame_store is not None:
            frame_store.close()

    if elided_layout_streams:
        print(
            "warning: these streams were recorded hash-only (payload elided) and "
            "contribute nothing to the world-model input this tick: "
            + ", ".join(sorted(elided_layout_streams))
            + " -- record training sessions with --record-frames / --record-streams "
            "to include them",
            file=sys.stderr,
        )
    return dataset


def load_episode_pixel_frames(session_dir: str, episode_id: str) -> List[Any]:
    """Raw ``vision.frame.pixels`` frames for one episode, in tick order.

    Unlike :func:`build_pixel_sequence_dataset`, which only pairs *adjacent*
    frames for representation-learning losses, this keeps the whole
    sequence so a caller can compare a multi-tick rollout against the
    actual frame at ``t + h`` for any horizon ``h`` -- what the ego-motion
    canary (issue #39, ``training/ego_motion_canary.py``) needs. Requires
    the pixel frames to be present in the log (record with
    ``--record-frames``).
    """
    metadata = load_session_metadata(session_dir)
    require_streams_v2(metadata)
    frame_store = open_frame_store(session_dir)
    frames: List[Any] = []
    try:
        for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, episode_id):
            for record in sensory:
                if record.get("stream_id") != PIXEL_STREAM:
                    continue
                if record.get("elided"):
                    raise ValueError(
                        f"{session_dir}/{episode_id}: {PIXEL_STREAM} was recorded hash-only; "
                        "re-record the session with --record-frames"
                    )
                frames.append(stream_event_from_log(record, frame_store=frame_store).payload)
    finally:
        if frame_store is not None:
            frame_store.close()
    return frames


def _pixel_shape_from_catalog(metadata: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    for spec in _catalog(metadata):
        if spec.stream_id == PIXEL_STREAM and spec.shape is not None:
            return tuple(spec.shape)  # type: ignore[return-value]
    return None


def _stream_slices(fusion: TemporalFusion) -> Dict[str, Tuple[int, int]]:
    offset = 0
    slices: Dict[str, Tuple[int, int]] = {}
    for entry in fusion.layout:
        slices[entry.stream_id] = (offset, offset + entry.width)
        offset += entry.width
    return slices


def _fusion_aux_lists(
    fusion: TemporalFusion,
    buffer: TemporalBuffer,
    present_stream_ids: Iterable[str],
    stale_streams: Iterable[str] = (),
) -> Tuple[List[float], List[float], List[float]]:
    present = set(present_stream_ids)
    stale = set(stale_streams)
    reference_time = fusion._reference_time(buffer)
    mask: List[float] = []
    recency: List[float] = []
    staleness: List[float] = []
    for entry in fusion.layout:
        latest = buffer.latest(entry.stream_id)
        recent = fusion._event_recency(latest.timestamp, reference_time) if latest else 0.0
        mask.append(1.0 if entry.stream_id in present else 0.0)
        recency.append(float(recent))
        staleness.append(1.0 if entry.stream_id in stale else 1.0 - float(recent))
    return mask, recency, staleness


def build_neural_dataset(
    session_dirs: List[str],
    history: int = 8,
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
    stream_profile: str = NEURAL_FULL,
) -> NeuralDataset:
    """Walk recorded sessions and emit (pixels, non_vision, motor, action) samples.

    Requires the pixel frames to be present in the log (record with
    ``--record-frames``); a session that elided them cannot train pixel vision
    and raises rather than training on blank frames.

    ``stream_profile`` (issue #32) selects the non-vision companion vector's
    ablation: ``"full"`` (default) is "pixels + semantics" -- every non-vision
    stream the generic registry fuses, including hand-computed semantic
    scalars; ``"raw"`` is "pixel only" -- the non-vision vector is restricted
    to streams the Minecraft stream registry classifies ``agent_input``
    (body/reward/spatial proprioception), dropping semantic aux/debug slots
    like ``world.front_block``/``world.sheltered``.
    """
    dataset = NeuralDataset(stream_profile=stream_profile)
    key_to_label = {key: i for i, key in enumerate(ACTION_KEYS)}
    fusion: Optional[TemporalFusion] = None
    pixels_were_elided = False

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = _non_vision_fusion(metadata, stream_profile)
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


def build_pixel_sequence_dataset(
    session_dirs: List[str],
    max_samples: Optional[int] = None,
    min_episode_reward: Optional[float] = None,
) -> PixelSequenceDataset:
    """Walk recorded sessions and emit adjacent pixel-frame pairs.

    Requires the pixel frames to be present in the log (record with
    ``--record-frames``).  Each sample is ``(frame_t, frame_t+1)`` within the
    same episode; episode boundaries are never crossed.
    """
    dataset = PixelSequenceDataset()
    fusion: Optional[TemporalFusion] = None
    pixels_were_elided = False

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = TemporalFusion(_catalog(metadata))
        if fusion is None:
            fusion = session_fusion
            dataset.layout_hash = fusion.layout_hash
            dataset.pixel_shape = _pixel_shape_from_catalog(metadata)
        elif session_fusion.layout_hash != fusion.layout_hash:
            raise ValueError(
                f"session {session_dir} has an incompatible stream catalog "
                f"({session_fusion.layout_hash} vs {fusion.layout_hash}); train on "
                "sessions recorded with the same program config"
            )

        frame_store = open_frame_store(session_dir)
        for episode_id in list_episodes(session_dir):
            reward_total = 0.0
            frames: List[Any] = []
            for decision, sensory, _motor in iter_cognitive_ticks(session_dir, episode_id):
                reward_total += float(decision.get("reward_window_total", 0.0))
                for record in sensory:
                    if record.get("stream_id") != PIXEL_STREAM:
                        continue
                    if record.get("elided"):
                        pixels_were_elided = True
                        continue
                    frames.append(stream_event_from_log(record, frame_store=frame_store).payload)
            if min_episode_reward is not None and reward_total < min_episode_reward:
                continue
            for left, right in zip(frames, frames[1:]):
                dataset.pixels.append(left)
                dataset.next_pixels.append(right)
                if max_samples is not None and len(dataset) >= max_samples:
                    dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                    if frame_store is not None:
                        frame_store.close()
                    return dataset
            if len(frames) >= 2:
                dataset.sources.append(f"{session_dir}/{episode_id}")
        if frame_store is not None:
            frame_store.close()

    if len(dataset) == 0 and pixels_were_elided:
        raise ValueError(
            f"no pixel sequence samples: the {PIXEL_STREAM} stream was recorded hash-only. "
            "Re-record the training sessions with --record-frames so pixel vision "
            "has adjacent frames to learn from."
        )
    return dataset
