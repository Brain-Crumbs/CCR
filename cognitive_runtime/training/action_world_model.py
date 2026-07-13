"""Action-conditioned recurrent world model (phases 1-3 of
docs/nursery-turn-in-place-analysis.md).

The nursery's original predictor (``visual_representation.NextLatentPredictor``)
is a memoryless latent->latent MLP with no action input: from a single frame
it cannot represent rotation speed or direction, and trained by long
backprop-through-composition it collapses to a fixed point -- identical
predictions at every horizon.  This module is the corrective:

- **Action-conditioned transition** (phase 1): ``z_{t+1} = f(z_t, a_t)`` --
  the recorded ``motor.command`` stream supplies the action driving every
  frame transition, so one model can serve every scenario instead of baking
  each scenario's scripted policy into the dynamics.
- **Recurrent state** (phase 1): a GRU over latents makes velocity/rotation
  rate representable; a single frame is not a Markov state of a turning
  agent.
- **Horizons in ticks** (phase 1): recorded vision may run below the tick
  rate (the first remote runs paced it at ~10 Hz against 20 Hz ticks), so
  horizons are declared in ticks and converted per-recording via the
  measured frames-per-tick.
- **Short-rollout scheduled sampling** (phase 2): train on 5-10 step
  rollouts from resampled start points instead of 100-step compositions
  whose cheapest solution is the identity.
- **Baseline-relative metrics + frozen-rollout detector** (phase 2):
  evaluation reports MSE(model)/MSE(copy-last) and MSE(model)/MSE(recurrence
  oracle) per horizon, and flags rollouts whose predictions do not vary
  across horizons while the actual frames do -- the exact failure signature
  of the collapsed predictor.
- **Joint training + probes** (phase 3): one dataset spans many scenarios'
  sessions; ``linear_probe_yaw`` checks whether the latent/hidden state
  linearly decodes the agent's heading (the cheap interpretability check for
  "did it actually capture rotation").

Dataset building is torch-free (plain lists/ndarrays, matching
``training.datasets``); only the model/training/evaluation half imports
torch.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
)
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.recorder import stream_event_from_log

PIXEL_STREAM = "vision.frame.pixels"
MOTOR_STREAM = "motor.command"
ROTATION_STREAM = "spatial.rotation"


# --------------------------------------------------------------------------- dataset


@dataclass
class EpisodeActionFrames:
    """One recorded episode as (frame, action, frame, action, ...).

    ``actions[i]`` is the action index driving the transition from
    ``frames[i]`` to ``frames[i+1]`` (the ``motor.command`` of the cognitive
    tick that observed ``frames[i+1]``).  ``yaw[i]`` is the agent heading in
    degrees at ``frames[i]`` when ``spatial.rotation`` was observed that
    tick, else ``None`` -- probe material, never a model input.
    """

    session_dir: str
    episode_id: str
    frames: List[Any] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    yaw: List[Optional[float]] = field(default_factory=list)
    ticks: List[int] = field(default_factory=list)
    playback_frame_count: Optional[int] = None

    @property
    def ticks_per_frame(self) -> float:
        if len(self.ticks) < 2:
            return 1.0
        span = self.ticks[-1] - self.ticks[0]
        return max(span / (len(self.ticks) - 1), 1e-9)


@dataclass
class ActionSequenceDataset:
    """Frame/action sequences from recorded sessions, across any number of
    scenarios -- the joint-training counterpart of ``PixelSequenceDataset``'s
    adjacent pairs."""

    episodes: List[EpisodeActionFrames] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=list)
    pixel_shape: Optional[Tuple[int, int, int]] = None
    sources: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return sum(len(e.actions) for e in self.episodes)

    @property
    def ticks_per_frame(self) -> float:
        """Median measured cognitive ticks per recorded vision frame -- 1.0
        on the simulated backend, ~2.0 on the first paced remote runs."""
        values = sorted(e.ticks_per_frame for e in self.episodes if len(e.ticks) >= 2)
        if not values:
            return 1.0
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return 0.5 * (values[mid - 1] + values[mid])


def horizons_ticks_to_frames(
    horizons_ticks: Sequence[int], ticks_per_frame: float
) -> List[int]:
    """Convert tick-denominated horizons to recorded-frame steps.

    The returned list preserves one entry per requested tick horizon. When
    vision runs below the tick rate, multiple tick horizons can map to the
    same recorded frame step (for example t+1 and t+2 at ~2 ticks/frame);
    callers that need unique evaluation work can de-duplicate separately.
    """
    if ticks_per_frame <= 0:
        raise ValueError(f"ticks_per_frame must be positive, got {ticks_per_frame!r}")
    frames: List[int] = []
    for h in horizons_ticks:
        f = max(1, int(round(h / ticks_per_frame)))
        frames.append(f)
    return frames


def _tick_action_name(motor_records: List[Dict[str, Any]]) -> Optional[str]:
    for record in reversed(motor_records):
        if record.get("stream_id") != MOTOR_STREAM:
            continue
        payload = record.get("payload") or {}
        name = payload.get("action")
        if isinstance(name, str) and name:
            return name
    return None


def _tick_yaw(sensory_records: List[Dict[str, Any]]) -> Optional[float]:
    for record in reversed(sensory_records):
        if record.get("stream_id") != ROTATION_STREAM:
            continue
        payload = record.get("payload") or {}
        yaw = payload.get("yaw")
        if isinstance(yaw, (int, float)):
            return float(yaw)
    return None


def build_action_sequence_dataset(
    session_dirs: Sequence[str],
    action_keys: Optional[Sequence[str]] = None,
) -> ActionSequenceDataset:
    """Walk recorded sessions and emit per-episode frame/action sequences.

    The action vocabulary defaults to every action name seen across the
    sessions (sorted, stable); pass ``action_keys`` to pin a fixed vocabulary
    (e.g. the full ``ACTION_SPACE``) so a model trained here can later see
    actions absent from its training scenarios.  Unknown names extend the
    vocabulary rather than fail: a held-out scenario must be encodable.
    """
    dataset = ActionSequenceDataset(
        action_keys=list(action_keys) if action_keys is not None else []
    )
    index: Dict[str, int] = {name: i for i, name in enumerate(dataset.action_keys)}

    def action_index(name: str) -> int:
        if name not in index:
            index[name] = len(dataset.action_keys)
            dataset.action_keys.append(name)
        return index[name]

    for session_dir in session_dirs:
        playback_frame_count: Optional[int] = None
        try:
            metadata = load_session_metadata(session_dir)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        nursery = (
            (metadata.get("program_config") or {}).get("nursery") or {}
            if isinstance(metadata, dict)
            else {}
        )
        active_ticks = nursery.get("active_episode_ticks")
        if isinstance(active_ticks, int) and active_ticks > 0:
            playback_frame_count = active_ticks
        frame_store = open_frame_store(session_dir)
        try:
            for episode_id in list_episodes(session_dir):
                episode = EpisodeActionFrames(session_dir=session_dir, episode_id=episode_id)
                episode.playback_frame_count = playback_frame_count
                last_action: Optional[str] = None
                for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                    tick = int(decision.get("tick_index", len(episode.ticks)))
                    action_name = _tick_action_name(motor) or last_action
                    last_action = action_name
                    yaw = _tick_yaw(sensory)
                    for record in sensory:
                        if record.get("stream_id") != PIXEL_STREAM:
                            continue
                        if record.get("elided"):
                            raise ValueError(
                                f"{session_dir}/{episode_id}: {PIXEL_STREAM} was recorded "
                                "hash-only; re-record the session with --record-frames"
                            )
                        frame = stream_event_from_log(record, frame_store=frame_store).payload
                        if episode.frames:
                            episode.actions.append(action_index(action_name or "NULL"))
                        episode.frames.append(frame)
                        episode.yaw.append(yaw)
                        episode.ticks.append(tick)
                if len(episode.frames) >= 2:
                    if dataset.pixel_shape is None:
                        first = episode.frames[0]
                        shape = getattr(first, "shape", None)
                        if shape is not None and len(shape) == 3:
                            dataset.pixel_shape = tuple(int(d) for d in shape)
                    dataset.episodes.append(episode)
                    dataset.sources.append(f"{session_dir}/{episode_id}")
        finally:
            if frame_store is not None:
                frame_store.close()
    return dataset


# --------------------------------------------------------------------------- model

# torch is imported lazily so the dataset half stays importable in
# torch-free installs, matching training.datasets.


def _torch():
    import torch  # noqa: PLC0415 -- optional heavy dependency
    import torch.nn.functional as F  # noqa: PLC0415

    return torch, F


@dataclass
class ActionWorldModelConfig:
    latent_width: int = 32
    hidden_dim: int = 64
    action_embed_dim: int = 8
    reconstruction_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    #: Teacher-forced frames before the training rollout starts -- enough
    #: history for the GRU to estimate motion (>= 2 to observe one delta).
    warmup_frames: int = 3
    #: Closed-loop steps per training window.  Short on purpose: long
    #: compositions of one transition under MSE select for the identity.
    rollout_frames: int = 8
    #: Probability a rollout step feeds its own prediction instead of the
    #: observed latent (scheduled sampling).
    scheduled_sampling_p: float = 0.5
    pixel_loss_weight: float = 1.0
    latent_loss_weight: float = 1.0
    seed: int = 0


def build_action_world_model(
    pixel_shape: Tuple[int, int, int],
    action_keys: Sequence[str],
    config: Optional[ActionWorldModelConfig] = None,
):
    """Construct an ``ActionConditionedWorldModel`` (requires torch)."""
    torch, _F = _torch()
    import torch.nn as nn  # noqa: PLC0415

    from cognitive_runtime.neural.pixel_stream_encoder import PixelStreamEncoder
    from cognitive_runtime.training.visual_representation import (
        PixelReconstructionDecoder,
        _reconstruction_shape,
    )

    cfg = config or ActionWorldModelConfig()

    class ActionConditionedWorldModel(nn.Module):
        """Encoder + action-conditioned GRU transition + decoder.

        The GRU hidden state is the model's world state: it accumulates
        observation history (so rotation *rate* is representable) and is
        advanced by ``(latent, action)`` pairs.  Closed-loop rollout feeds
        predicted latents back in, so evaluation exercises exactly the
        multi-horizon interface the nursery benchmarks.
        """

        def __init__(self) -> None:
            super().__init__()
            self.pixel_shape = tuple(int(d) for d in pixel_shape)
            self.action_keys = list(action_keys)
            self.latent_width = cfg.latent_width
            self.hidden_dim = cfg.hidden_dim
            self.reconstruction_shape = _reconstruction_shape(
                self.pixel_shape, cfg.reconstruction_size
            )
            self.encoder = PixelStreamEncoder(self.pixel_shape, latent_width=cfg.latent_width)
            self.action_embedding = nn.Embedding(len(self.action_keys), cfg.action_embed_dim)
            self.transition = nn.GRUCell(
                cfg.latent_width + cfg.action_embed_dim, cfg.hidden_dim
            )
            self.latent_head = nn.Linear(cfg.hidden_dim, cfg.latent_width)
            self.decoder = PixelReconstructionDecoder(
                cfg.latent_width, self.reconstruction_shape, hidden_dim=cfg.hidden_dim
            )

        def initial_state(self, batch: int):
            weight = self.latent_head.weight
            return weight.new_zeros(batch, self.hidden_dim)

        def step(self, latent, action_idx, hidden):
            """Advance the world state by one (latent, action) pair; returns
            (predicted next latent, next hidden)."""
            embedded = self.action_embedding(action_idx)
            hidden = self.transition(torch.cat([latent, embedded], dim=-1), hidden)
            return self.latent_head(hidden), hidden

        def rollout(self, start_latent, actions, hidden):
            """Closed-loop rollout: from ``start_latent`` and world state
            ``hidden``, apply ``actions`` ([B, R] indices) feeding each
            predicted latent back in; returns predicted latents [B, R, L]."""
            predictions = []
            latent = start_latent
            for i in range(actions.shape[1]):
                latent, hidden = self.step(latent, actions[:, i], hidden)
                predictions.append(latent)
            return torch.stack(predictions, dim=1), hidden

    return ActionConditionedWorldModel()


# --------------------------------------------------------------------------- training


def _episode_tensors(dataset: ActionSequenceDataset, reconstruction_shape):
    torch, _F = _torch()
    from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
    from cognitive_runtime.training.visual_representation import reconstruction_target

    tensors = []
    for episode in dataset.episodes:
        pixels = torch.stack([pixels_to_chw(f) for f in episode.frames])
        targets = reconstruction_target(pixels, reconstruction_shape)
        actions = torch.tensor(episode.actions, dtype=torch.long)
        tensors.append((episode, pixels, targets, actions))
    return tensors


def train_action_world_model(
    dataset: ActionSequenceDataset,
    config: Optional[ActionWorldModelConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train the action-conditioned world model with short-rollout scheduled
    sampling over every episode in ``dataset``.

    Loss per rollout step: pixel MSE of the decoded prediction against the
    downsampled true frame, plus latent MSE against the (detached) encoder
    latent of the true frame -- the pixel term keeps predictions decodable,
    the latent term keeps rollouts on the encoder's manifold without the
    100-step compositions that drove the old predictor to the identity.
    """
    torch, F = _torch()

    if len(dataset) == 0:
        raise ValueError("action sequence dataset is empty; record sessions with --record-frames")
    if dataset.pixel_shape is None:
        raise ValueError("dataset has no pixel shape; were frames recorded?")
    cfg = config or ActionWorldModelConfig()
    if cfg.warmup_frames < 1:
        raise ValueError(f"warmup_frames must be >= 1, got {cfg.warmup_frames}")
    if cfg.rollout_frames < 1:
        raise ValueError(f"rollout_frames must be >= 1, got {cfg.rollout_frames}")
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    model = build_action_world_model(dataset.pixel_shape, dataset.action_keys, cfg)
    episodes = _episode_tensors(dataset, model.reconstruction_shape)

    window = cfg.warmup_frames + cfg.rollout_frames
    starts: List[Tuple[int, int]] = []  # (episode index, start frame)
    for e_idx, (_episode, pixels, _targets, _actions) in enumerate(episodes):
        for t in range(pixels.shape[0] - window):
            starts.append((e_idx, t))
    if not starts:
        raise ValueError(
            f"no training windows: every episode is shorter than warmup+rollout "
            f"({window} frames); record longer episodes or shrink the window"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, List[float]] = {"total_loss": [], "pixel_loss": [], "latent_loss": []}

    model.train()
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(len(starts), generator=generator)
        epoch = {key: 0.0 for key in curves}
        seen = 0
        for batch_start in range(0, perm.numel(), cfg.batch_size):
            batch_ids = perm[batch_start : batch_start + cfg.batch_size]
            frames_batch = []
            actions_batch = []
            targets_batch = []
            for flat in batch_ids.tolist():
                e_idx, t = starts[flat]
                _episode, pixels, targets, actions = episodes[e_idx]
                frames_batch.append(pixels[t : t + window])
                targets_batch.append(targets[t : t + window])
                actions_batch.append(actions[t : t + window - 1])
            frames_b = torch.stack(frames_batch)  # [B, W+R, C, H, W]
            targets_b = torch.stack(targets_batch)
            actions_b = torch.stack(actions_batch)  # [B, W+R-1]
            batch_n = frames_b.shape[0]

            optimizer.zero_grad()
            flat_frames = frames_b.reshape(-1, *frames_b.shape[2:])
            latents = model.encoder(flat_frames).reshape(batch_n, window, -1)

            hidden = model.initial_state(batch_n)
            # Teacher-forced warmup: observed latents drive the state.
            predicted = None
            for i in range(cfg.warmup_frames - 1):
                predicted, hidden = model.step(latents[:, i], actions_b[:, i], hidden)

            pixel_loss = frames_b.new_zeros(())
            latent_loss = frames_b.new_zeros(())
            latent_in = latents[:, cfg.warmup_frames - 1]
            for step in range(cfg.rollout_frames):
                idx = cfg.warmup_frames - 1 + step
                predicted, hidden = model.step(latent_in, actions_b[:, idx], hidden)
                # Normalized like visual_representation.next_latent_prediction_loss:
                # raw encoder latents are unbounded (ReLU), and a raw-space MSE
                # swamps the pixel term.
                target_latent = latents[:, idx + 1].detach()
                latent_loss = latent_loss + F.mse_loss(
                    F.normalize(predicted, dim=1), F.normalize(target_latent, dim=1)
                )
                decoded = model.decoder(predicted)
                pixel_loss = pixel_loss + F.mse_loss(decoded, targets_b[:, idx + 1])
                # Scheduled sampling: feed the prediction back in with
                # probability p, else the observed latent.
                if float(torch.rand((), generator=generator)) < cfg.scheduled_sampling_p:
                    latent_in = predicted
                else:
                    latent_in = latents[:, idx + 1]
            pixel_loss = pixel_loss / cfg.rollout_frames
            latent_loss = latent_loss / cfg.rollout_frames
            total = cfg.pixel_loss_weight * pixel_loss + cfg.latent_loss_weight * latent_loss
            total.backward()
            optimizer.step()

            seen += batch_n
            epoch["total_loss"] += float(total.detach()) * batch_n
            epoch["pixel_loss"] += float(pixel_loss.detach()) * batch_n
            epoch["latent_loss"] += float(latent_loss.detach()) * batch_n
        for key in curves:
            curves[key].append(round(epoch[key] / max(seen, 1), 6))

    stats: Dict[str, Any] = {
        "samples": float(len(starts)),
        "episodes": float(len(episodes)),
        "epochs": float(cfg.epochs),
        "action_keys": list(dataset.action_keys),
        "ticks_per_frame": dataset.ticks_per_frame,
        "warmup_frames": cfg.warmup_frames,
        "rollout_frames": cfg.rollout_frames,
        "loss_curves": curves,
        "final_total_loss": curves["total_loss"][-1],
        "final_pixel_loss": curves["pixel_loss"][-1],
        "final_latent_loss": curves["latent_loss"][-1],
    }
    return model, stats


# --------------------------------------------------------------------------- evaluation


def _best_recurrence_lag(targets, max_lag: int = 60) -> Optional[int]:
    """The lag (>= 2) at which this episode's frames best repeat -- the
    'recurrence oracle' reference for periodic scenarios (a full turn of
    turn_in_place).  ``None`` when the episode is too short to test any lag."""
    torch, F = _torch()
    n = targets.shape[0]
    best_lag: Optional[int] = None
    best_mse = math.inf
    for lag in range(2, min(max_lag, n - 1) + 1):
        mse = float(F.mse_loss(targets[lag:], targets[:-lag]))
        if mse < best_mse:
            best_mse = mse
            best_lag = lag
    return best_lag


def evaluate_action_world_model(
    model: Any,
    dataset: ActionSequenceDataset,
    horizons_frames: Sequence[int],
    *,
    warmup_frames: int = 3,
    max_starts_per_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Closed-loop multi-horizon evaluation with baseline-relative metrics
    and the frozen-rollout detector.

    Returns ``{"horizons": {h: {...}}, "rollout_health": {...}}``.  Each
    horizon entry carries raw MSEs (model / copy-last / mean-frame /
    recurrence-oracle), PSNRs, and the ratios ``model_over_copy_last_mse``
    and ``model_over_oracle_mse``: < 1.0 means the model beats that
    reference.  ``rollout_health.frozen_rollout`` is True when predictions
    barely vary across horizons while the actual frames do -- the collapsed
    fixed-point signature this evaluation exists to catch.
    """
    torch, F = _torch()

    horizons_sorted = sorted(set(int(h) for h in horizons_frames))
    if not horizons_sorted or horizons_sorted[0] < 1:
        raise ValueError(f"horizons must be positive frame steps, got {horizons_frames!r}")
    max_horizon = horizons_sorted[-1]

    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    action_index = {name: i for i, name in enumerate(model.action_keys)}
    for name in dataset.action_keys:
        if name not in action_index:
            raise ValueError(
                f"action {name!r} is outside the model's vocabulary {model.action_keys!r}; "
                "build the training dataset with a pinned action_keys covering it"
            )

    samples: Dict[int, Dict[str, List[float]]] = {
        h: {"model": [], "copy_last": [], "mean_frame": [], "oracle": []}
        for h in horizons_sorted
    }
    prediction_dispersion: List[float] = []
    target_dispersion: List[float] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, targets, actions in episodes:
            n = pixels.shape[0]
            if n <= max_horizon + warmup_frames:
                continue
            # Remap this episode's action indices into the model's vocabulary.
            remap = torch.tensor(
                [action_index[dataset.action_keys[a]] for a in episode.actions],
                dtype=torch.long,
            )
            latents = model.encoder(pixels)
            mean_frame = targets.mean(dim=0)
            oracle_lag = _best_recurrence_lag(targets)

            # One teacher-forced pass gives the world state at every tick.
            hiddens = [model.initial_state(1)]
            hidden = hiddens[0]
            for i in range(n - 1):
                _pred, hidden = model.step(
                    latents[i : i + 1], remap[i : i + 1], hidden
                )
                hiddens.append(hidden)

            starts = range(warmup_frames, n - max_horizon)
            if max_starts_per_episode is not None:
                starts = list(starts)[:max_starts_per_episode]
            for t in starts:
                rolled, _h = model.rollout(
                    latents[t : t + 1],
                    remap[t : t + max_horizon].unsqueeze(0),
                    hiddens[t],
                )
                decoded_by_horizon = {}
                for h in horizons_sorted:
                    decoded = model.decoder(rolled[:, h - 1]).squeeze(0)
                    decoded_by_horizon[h] = decoded
                    target = targets[t + h]
                    samples[h]["model"].append(float(F.mse_loss(decoded, target)))
                    samples[h]["copy_last"].append(float(F.mse_loss(targets[t], target)))
                    samples[h]["mean_frame"].append(float(F.mse_loss(mean_frame, target)))
                    if oracle_lag is not None and t + h - oracle_lag >= 0:
                        samples[h]["oracle"].append(
                            float(F.mse_loss(targets[t + h - oracle_lag], target))
                        )
                if len(horizons_sorted) >= 2:
                    prediction_dispersion.append(
                        _pairwise_dispersion([decoded_by_horizon[h] for h in horizons_sorted])
                    )
                    target_dispersion.append(
                        _pairwise_dispersion([targets[t + h] for h in horizons_sorted])
                    )
    if was_training:
        model.train()

    report: Dict[int, Dict[str, Any]] = {}
    for h in horizons_sorted:
        entry = samples[h]
        if not entry["model"]:
            raise ValueError(
                f"no evaluation samples at horizon {h}; episodes too short for it"
            )
        model_mse = _mean(entry["model"])
        copy_mse = _mean(entry["copy_last"])
        mean_mse = _mean(entry["mean_frame"])
        oracle_mse = _mean(entry["oracle"]) if entry["oracle"] else None
        report[h] = {
            "n_samples": len(entry["model"]),
            "model_mse": model_mse,
            "copy_last_mse": copy_mse,
            "mean_frame_mse": mean_mse,
            "oracle_mse": oracle_mse,
            "psnr_model": _psnr(model_mse),
            "psnr_copy_last": _psnr(copy_mse),
            "psnr_mean_frame": _psnr(mean_mse),
            "model_over_copy_last_mse": _ratio(model_mse, copy_mse),
            "model_over_oracle_mse": _ratio(model_mse, oracle_mse),
            "beats_copy_last": bool(model_mse < copy_mse),
            "beats_mean_frame": bool(model_mse < mean_mse),
        }

    pred_disp = _mean(prediction_dispersion) if prediction_dispersion else 0.0
    tgt_disp = _mean(target_dispersion) if target_dispersion else 0.0
    rollout_health = {
        "prediction_dispersion": pred_disp,
        "target_dispersion": tgt_disp,
        # Predictions vary < 5% as much across horizons as reality does:
        # the rollout is frozen (identical frames at t+10 and t+100).
        "frozen_rollout": bool(tgt_disp > 1e-6 and pred_disp < 0.05 * tgt_disp),
    }
    return {"horizons": report, "rollout_health": rollout_health}


def _pairwise_dispersion(frames: List[Any]) -> float:
    _torch_mod, F = _torch()
    total = 0.0
    pairs = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            total += float(F.mse_loss(frames[i], frames[j]))
            pairs += 1
    return total / pairs if pairs else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


# --------------------------------------------------------------------------- probes


def linear_probe_yaw(model: Any, dataset: ActionSequenceDataset) -> Dict[str, Any]:
    """Can the representation linearly decode the agent's heading?

    Fits ridge regressions latent -> (sin yaw, cos yaw) and hidden ->
    (sin yaw, cos yaw) on every frame with a recorded ``spatial.rotation``,
    reporting R^2 and mean angular error.  A hidden state that decodes yaw
    where the raw latent cannot is direct evidence the recurrence is
    carrying orientation/motion state (phase 3's cheap interpretability
    check)."""
    torch, _F = _torch()

    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    action_index = {name: i for i, name in enumerate(model.action_keys)}
    latent_rows: List[Any] = []
    hidden_rows: List[Any] = []
    targets_rows: List[Any] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, _targets, _actions in episodes:
            latents = model.encoder(pixels)
            remap = torch.tensor(
                [action_index[dataset.action_keys[a]] for a in episode.actions],
                dtype=torch.long,
            )
            hidden = model.initial_state(1)
            for i, yaw in enumerate(episode.yaw):
                if yaw is not None:
                    rad = math.radians(yaw)
                    latent_rows.append(latents[i])
                    hidden_rows.append(hidden.squeeze(0))
                    targets_rows.append(
                        torch.tensor([math.sin(rad), math.cos(rad)])
                    )
                if i < len(episode.actions):
                    _pred, hidden = model.step(
                        latents[i : i + 1], remap[i : i + 1], hidden
                    )
    if was_training:
        model.train()

    if len(targets_rows) < 8:
        return {"n_samples": len(targets_rows), "note": "too few yaw-labelled frames to probe"}

    y = torch.stack(targets_rows)
    report: Dict[str, Any] = {"n_samples": int(y.shape[0])}
    for name, rows in (("latent", latent_rows), ("hidden", hidden_rows)):
        x = torch.stack(rows)
        report[name] = _ridge_probe(x, y)
    return report


def _ridge_probe(x, y, l2: float = 1e-3) -> Dict[str, float]:
    torch, _F = _torch()
    ones = torch.ones(x.shape[0], 1)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design + l2 * torch.eye(design.shape[1])
    weights = torch.linalg.solve(gram, design.T @ y)
    predicted = design @ weights
    residual = ((y - predicted) ** 2).sum()
    total = ((y - y.mean(dim=0)) ** 2).sum()
    r2 = float(1.0 - residual / total) if float(total) > 0 else 0.0
    pred_angle = torch.atan2(predicted[:, 0], predicted[:, 1])
    true_angle = torch.atan2(y[:, 0], y[:, 1])
    diff = torch.rad2deg(
        torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle))
    ).abs()
    return {"r2": r2, "mean_angular_error_deg": float(diff.mean())}


# --------------------------------------------------------------------------- checkpointing


def save_action_world_model(path: str, model: Any, stats: Dict[str, Any]) -> None:
    """Persist the full model (encoder + transition + decoder) so rollouts
    remain reproducible after the process exits -- the nursery's
    encoder-only checkpoint cannot regenerate predictions."""
    torch, _F = _torch()
    torch.save(
        {
            "format": "action-world-model-v1",
            "pixel_shape": list(model.pixel_shape),
            "action_keys": list(model.action_keys),
            "latent_width": model.latent_width,
            "hidden_dim": model.hidden_dim,
            "reconstruction_shape": list(model.reconstruction_shape),
            "state_dict": model.state_dict(),
            "training_stats": stats,
        },
        path,
    )


def load_action_world_model(path: str):
    torch, _F = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "action-world-model-v1":
        raise ValueError(f"unsupported action world model format {payload.get('format')!r}")
    cfg = ActionWorldModelConfig(
        latent_width=int(payload["latent_width"]),
        hidden_dim=int(payload["hidden_dim"]),
        reconstruction_size=int(payload["reconstruction_shape"][0]),
    )
    model = build_action_world_model(
        tuple(payload["pixel_shape"]), payload["action_keys"], cfg
    )
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("training_stats", {})
