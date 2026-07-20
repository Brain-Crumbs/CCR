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

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cognitive_runtime.runtime.replay import iter_cognitive_ticks, list_episodes
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.recorder import stream_event_from_log

PIXEL_STREAM = "vision.frame.pixels"
MOTOR_STREAM = "motor.command"
ROTATION_STREAM = "spatial.rotation"
#: Recorded ground-truth stream for the cortex's terminal head (issue #169).
#: Reward and risk targets instead come straight off the decision record
#: (``reward_window_total``/``risk``) -- not the ``internal.*`` streams
#: derived from them, which the loop deliberately publishes one tick late
#: (``runtime/loop.py``'s "primes the *next* tick's evaluation" comment) so
#: reading them out of a tick's *sensory* window would train the risk head
#: on the previous tick's value.
DEATH_STREAM = "event.died"


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
    #: Per-frame supervision for the cortex's reward/terminal/risk heads
    #: (issue #169) -- ``reward_window_total`` off the decision record,
    #: ``event.died`` presence in the tick's sensory window, and the
    #: recorded ``internal.risk`` stream value, aligned 1:1 with ``frames``
    #: like ``yaw``. Empty on hand-built episodes that predate these fields
    #: (e.g. synthetic test fixtures); ``_episode_head_targets`` falls back
    #: to all-zero targets in that case rather than raising.
    reward: List[float] = field(default_factory=list)
    terminal: List[bool] = field(default_factory=list)
    risk: List[float] = field(default_factory=list)

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
    """Convert tick-denominated horizons to recorded-frame steps, preserving
    order and de-duplicating collisions (two tick horizons can land on the
    same frame step when vision runs below the tick rate)."""
    if ticks_per_frame <= 0:
        raise ValueError(f"ticks_per_frame must be positive, got {ticks_per_frame!r}")
    frames: List[int] = []
    for h in horizons_ticks:
        f = max(1, int(round(h / ticks_per_frame)))
        if f not in frames:
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


def _tick_terminal(sensory_records: List[Dict[str, Any]]) -> bool:
    return any(record.get("stream_id") == DEATH_STREAM for record in sensory_records)


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
        frame_store = open_frame_store(session_dir)
        try:
            for episode_id in list_episodes(session_dir):
                episode = EpisodeActionFrames(session_dir=session_dir, episode_id=episode_id)
                last_action: Optional[str] = None
                for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                    tick = int(decision.get("tick_index", len(episode.ticks)))
                    action_name = _tick_action_name(motor) or last_action
                    last_action = action_name
                    yaw = _tick_yaw(sensory)
                    reward = float(decision.get("reward_window_total", 0.0))
                    terminal = _tick_terminal(sensory)
                    # Straight off the decision record (this tick's own
                    # `Prediction.risk`), not the `internal.risk` stream --
                    # see the module-level DEATH_STREAM comment for why.
                    risk = float(decision.get("risk", 0.0))
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
                        episode.reward.append(reward)
                        episode.terminal.append(terminal)
                        episode.risk.append(risk)
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
    #: Weights for the previously-untrained heads (issue #169): reward MSE
    #: vs. recorded ``reward_window_total``, terminal BCE vs. recorded
    #: ``event.died``, risk MSE vs. recorded ``internal.risk``. Default 1.0
    #: matches ``training.world_model.WorldModelTrainingConfig``'s
    #: reward/death/risk weights for the legacy memoryless model.
    reward_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0
    risk_loss_weight: float = 1.0
    #: Uncertainty-head weight: MSE against the (detached) realized latent
    #: squared error at each rollout step -- an auxiliary, self-supervised
    #: signal like the legacy model's ``prediction_error_loss_weight``, whose
    #: 0.1 default this matches so it calibrates without dominating the
    #: pixel/latent objective.
    uncertainty_loss_weight: float = 0.1
    #: Optional slow target encoder for the latent regression target. ``None``
    #: preserves the historical shared-encoder objective; a value in [0, 1)
    #: enables a Polyak/EMA copy updated after every optimizer step.
    ema_target_decay: Optional[float] = None
    #: Representation-collapse gate.  The gate fails only when the recurrent
    #: state has lost yaw *and* the encoder output is degenerate, avoiding a
    #: false alarm when one diagnostic alone is weak.
    collapse_gate_enabled: bool = True
    collapse_gate_min_hidden_yaw_r2: float = 0.1
    collapse_gate_min_latent_variance: float = 1e-6
    collapse_gate_min_effective_rank: float = 1.5
    seed: int = 0
    #: Horizons in ticks (not frames), persisted with the checkpoint so
    #: "T+8" means the same thing across worlds/sample rates. Convert to
    #: frame steps per-recording via ``horizons_ticks_to_frames``.
    horizons_ticks: Tuple[int, ...] = (1, 4, 8)
    #: Action-ablation harness (issue #92): when True, every action index fed
    #: to the model during training is overwritten with a constant (index 0)
    #: before the warmup/rollout loops see it, so the action stream carries
    #: zero information. Training two otherwise-identical models with this
    #: flag off/on and comparing held-out performance is the "does the model
    #: actually use its action input" proof -- a predictor that never sees
    #: its action can't tell "kept turning" from "stopped".
    withhold_actions: bool = False
    #: Temporal backbone (issue #93): ``"gru"`` (default), ``"dilated_conv"``,
    #: or ``"transformer"``. See ``brain.cortex.backbones``.
    backbone: str = "gru"
    #: Window size the windowed backbones attend over; ignored by ``"gru"``.
    context_length: int = 8
    #: Extra backbone-specific constructor kwargs, forwarded verbatim to
    #: ``brain.cortex.backbones.build_backbone``.
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: Context-length curriculum (issue #93, task 5): ramp a windowed
    #: backbone's attended-over window from 1 to ``context_length`` over the
    #: first ``context_length_curriculum_epochs`` epochs (default: all of
    #: them), rather than exposing it to the full window from epoch zero. No
    #: effect on ``"gru"`` (its context is unbounded, not windowed).
    context_length_curriculum: bool = True
    context_length_curriculum_epochs: Optional[int] = None


def build_action_world_model(
    pixel_shape: Tuple[int, int, int],
    action_keys: Sequence[str],
    config: Optional[ActionWorldModelConfig] = None,
):
    """Construct a :class:`brain.cortex.predictive.PredictiveCortex`
    (requires torch).

    This is a thin re-export shim (issue #91): the model itself -- encoder
    + action-conditioned GRU transition + latent head + decoder, plus the
    multi-horizon reward/terminal/risk/uncertainty heads -- now lives in
    ``brain.cortex.predictive``, promoted out of this module. Callers of
    this function are unaffected.
    """
    from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig

    cfg = config or ActionWorldModelConfig()
    cortex_cfg = PredictiveCortexConfig(
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        action_embed_dim=cfg.action_embed_dim,
        reconstruction_size=cfg.reconstruction_size,
        horizons_ticks=tuple(cfg.horizons_ticks),
        backbone=cfg.backbone,
        context_length=cfg.context_length,
        backbone_kwargs=dict(cfg.backbone_kwargs),
    )
    return PredictiveCortex(pixel_shape, action_keys, cortex_cfg)


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


def _episode_head_targets(episode: EpisodeActionFrames, n_frames: int) -> Tuple[Any, Any, Any, bool]:
    """``(reward, terminal, risk, has_targets)`` for one episode, the first
    three aligned with :func:`_episode_tensors`' pixel/latent indexing
    (issue #169). ``has_targets`` is ``False`` when the episode predates
    these fields (hand-built synthetic episodes in tests): the columns are
    then all-zero placeholders callers must mask out of the reward/
    terminal/risk losses rather than train the heads to imitate a fictitious
    "always zero" ground truth."""
    torch, _F = _torch()
    has_targets = (
        len(episode.reward) == n_frames
        and len(episode.terminal) == n_frames
        and len(episode.risk) == n_frames
    )

    def _column(values: Sequence[Any]) -> Any:
        if len(values) == n_frames:
            return torch.tensor([float(v) for v in values], dtype=torch.float32)
        return torch.zeros(n_frames, dtype=torch.float32)

    return _column(episode.reward), _column(episode.terminal), _column(episode.risk), has_targets


def train_action_world_model(
    dataset: ActionSequenceDataset,
    config: Optional[ActionWorldModelConfig] = None,
    *,
    initial_model: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train the action-conditioned world model with short-rollout scheduled
    sampling over every episode in ``dataset``.

    Loss per rollout step: pixel MSE of the decoded prediction against the
    downsampled true frame, plus latent MSE against the detached target
    latent of the true frame.  The target is the online encoder by default,
    or a slow Polyak copy when ``ema_target_decay`` is configured.  The pixel
    term keeps predictions decodable; the latent term keeps rollouts on the
    encoder's manifold without the 100-step compositions that drove the old
    predictor to the identity.

    ``initial_model`` (issue #134), when given, continues training that
    ``PredictiveCortex`` in place instead of building a fresh one -- a
    caller warm-starting from :func:`load_action_world_model` and saving
    the result back via :func:`save_action_world_model` actually improves
    the same persisted cortex across calls, rather than discarding it and
    training a disposable one every time. Its shape/vocabulary must match
    the dataset (a mismatch is a wiring bug, not something to silently
    reinitialize around).
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
    if cfg.ema_target_decay is not None and not 0.0 <= cfg.ema_target_decay < 1.0:
        raise ValueError(
            "ema_target_decay must be None or in [0, 1), got "
            f"{cfg.ema_target_decay!r}"
        )
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    if initial_model is not None:
        if tuple(initial_model.pixel_shape) != tuple(dataset.pixel_shape):
            raise ValueError(
                f"initial_model pixel shape {initial_model.pixel_shape} does not match "
                f"dataset pixel shape {dataset.pixel_shape}"
            )
        if list(initial_model.action_keys) != list(dataset.action_keys):
            raise ValueError(
                f"initial_model action_keys {initial_model.action_keys} does not match "
                f"dataset action_keys {dataset.action_keys}"
            )
        model = initial_model
    else:
        model = build_action_world_model(dataset.pixel_shape, dataset.action_keys, cfg)
    target_encoder = None
    if cfg.ema_target_decay is not None:
        target_encoder = copy.deepcopy(model.encoder)
        target_encoder.requires_grad_(False)
        target_encoder.eval()
    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    #: Per-episode reward/terminal/risk targets (issue #169), aligned 1:1
    #: with ``episodes`` by index.
    head_targets = [
        _episode_head_targets(episode, pixels.shape[0]) for episode, pixels, _t, _a in episodes
    ]

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
    curves: Dict[str, List[float]] = {
        "total_loss": [],
        "pixel_loss": [],
        "latent_loss": [],
        "reward_loss": [],
        "terminal_loss": [],
        "risk_loss": [],
        "uncertainty_loss": [],
    }

    #: Context-length curriculum (issue #93, task 5): windowed backbones
    #: (dilated-conv/transformer) get their attended-over window ramped from
    #: 1 to the configured maximum over training, rather than the full
    #: window from epoch zero -- ``None`` on "gru" (unbounded context, no
    #: window to ramp) leaves this a no-op.
    max_context = getattr(model, "context_length_max", None)
    curriculum_epochs = max(cfg.context_length_curriculum_epochs or cfg.epochs, 1)

    model.train()
    for epoch_idx in range(cfg.epochs):
        if cfg.context_length_curriculum and max_context:
            progress = min(epoch_idx / max(curriculum_epochs - 1, 1), 1.0)
            model.set_context_length(max(1, round(1 + progress * (max_context - 1))))
        perm = torch.randperm(len(starts), generator=generator)
        epoch = {key: 0.0 for key in curves}
        seen = 0
        for batch_start in range(0, perm.numel(), cfg.batch_size):
            batch_ids = perm[batch_start : batch_start + cfg.batch_size]
            frames_batch = []
            actions_batch = []
            targets_batch = []
            rewards_batch = []
            terminals_batch = []
            risks_batch = []
            has_targets_batch = []
            for flat in batch_ids.tolist():
                e_idx, t = starts[flat]
                _episode, pixels, targets, actions = episodes[e_idx]
                rewards, terminals, risks, has_targets = head_targets[e_idx]
                frames_batch.append(pixels[t : t + window])
                targets_batch.append(targets[t : t + window])
                actions_batch.append(actions[t : t + window - 1])
                rewards_batch.append(rewards[t : t + window])
                terminals_batch.append(terminals[t : t + window])
                risks_batch.append(risks[t : t + window])
                has_targets_batch.append(1.0 if has_targets else 0.0)
            frames_b = torch.stack(frames_batch)  # [B, W+R, C, H, W]
            targets_b = torch.stack(targets_batch)
            actions_b = torch.stack(actions_batch)  # [B, W+R-1]
            rewards_b = torch.stack(rewards_batch)
            terminals_b = torch.stack(terminals_batch)
            risks_b = torch.stack(risks_batch)
            #: Per-sample mask (issue #169): episodes without recorded
            #: reward/terminal/risk streams (e.g. hand-built synthetic test
            #: episodes) carry all-zero placeholder targets that must not
            #: train the heads to imitate a fictitious "always zero" ground
            #: truth -- only genuinely-supervised samples contribute to
            #: these three losses. Uncertainty is exempt: its target is the
            #: model's own realized latent error, always available.
            has_targets_b = torch.tensor(has_targets_batch, dtype=torch.float32)
            n_supervised = float(has_targets_b.sum())
            if cfg.withhold_actions:
                # Ablation: every action index becomes the same constant, so
                # backprop through the action embedding carries no signal
                # that distinguishes one action from another.
                actions_b = torch.zeros_like(actions_b)
            batch_n = frames_b.shape[0]

            optimizer.zero_grad()
            flat_frames = frames_b.reshape(-1, *frames_b.shape[2:])
            latents = model.encoder(flat_frames).reshape(batch_n, window, -1)
            target_latents = latents
            if target_encoder is not None:
                with torch.no_grad():
                    target_latents = target_encoder(flat_frames).reshape(batch_n, window, -1)

            hidden = model.initial_state(batch_n)
            # Teacher-forced warmup: observed latents drive the state.
            predicted = None
            for i in range(cfg.warmup_frames - 1):
                predicted, hidden = model.step(latents[:, i], actions_b[:, i], hidden)

            pixel_loss = frames_b.new_zeros(())
            latent_loss = frames_b.new_zeros(())
            reward_loss = frames_b.new_zeros(())
            terminal_loss = frames_b.new_zeros(())
            risk_loss = frames_b.new_zeros(())
            uncertainty_loss = frames_b.new_zeros(())
            latent_in = latents[:, cfg.warmup_frames - 1]
            for step in range(cfg.rollout_frames):
                idx = cfg.warmup_frames - 1 + step
                predicted, hidden = model.step(latent_in, actions_b[:, idx], hidden)
                # Normalized like visual_representation.next_latent_prediction_loss:
                # raw encoder latents are unbounded (ReLU), and a raw-space MSE
                # swamps the pixel term.
                target_latent = target_latents[:, idx + 1].detach()
                per_sample_latent_error = (
                    F.normalize(predicted, dim=1) - F.normalize(target_latent, dim=1)
                ).pow(2).mean(dim=1)
                latent_loss = latent_loss + per_sample_latent_error.mean()
                decoded = model.decoder(predicted)
                pixel_loss = pixel_loss + F.mse_loss(decoded, targets_b[:, idx + 1])

                # Heads (issue #169): reward/terminal/risk read off the same
                # post-step hidden state ``forward_horizons`` uses, supervised
                # against the recorded targets at this frame; uncertainty
                # against this step's own (detached) *normalized* latent
                # error above -- same bounded [0, 4] scale as latent_loss
                # (an ICM-style self-supervised signal, not backprop into the
                # latent prediction itself), rather than the raw unbounded
                # squared error, which can dwarf pixel/latent and destabilize
                # shared-backbone training even at a small loss weight.
                reward_pred, terminal_logit, risk_pred, uncertainty_pred = model.heads(hidden)
                if n_supervised > 0:
                    reward_sq = (reward_pred - rewards_b[:, idx + 1]).pow(2)
                    reward_loss = reward_loss + (reward_sq * has_targets_b).sum() / n_supervised
                    terminal_bce = F.binary_cross_entropy_with_logits(
                        terminal_logit, terminals_b[:, idx + 1], reduction="none"
                    )
                    terminal_loss = terminal_loss + (terminal_bce * has_targets_b).sum() / n_supervised
                    risk_sq = (risk_pred - risks_b[:, idx + 1]).pow(2)
                    risk_loss = risk_loss + (risk_sq * has_targets_b).sum() / n_supervised
                realized_latent_error = per_sample_latent_error.detach()
                uncertainty_loss = uncertainty_loss + F.mse_loss(uncertainty_pred, realized_latent_error)

                # Scheduled sampling: feed the prediction back in with
                # probability p, else the observed latent.
                if float(torch.rand((), generator=generator)) < cfg.scheduled_sampling_p:
                    latent_in = predicted
                else:
                    latent_in = latents[:, idx + 1]
            pixel_loss = pixel_loss / cfg.rollout_frames
            latent_loss = latent_loss / cfg.rollout_frames
            reward_loss = reward_loss / cfg.rollout_frames
            terminal_loss = terminal_loss / cfg.rollout_frames
            risk_loss = risk_loss / cfg.rollout_frames
            uncertainty_loss = uncertainty_loss / cfg.rollout_frames
            total = (
                cfg.pixel_loss_weight * pixel_loss
                + cfg.latent_loss_weight * latent_loss
                + cfg.reward_loss_weight * reward_loss
                + cfg.terminal_loss_weight * terminal_loss
                + cfg.risk_loss_weight * risk_loss
                + cfg.uncertainty_loss_weight * uncertainty_loss
            )
            total.backward()
            optimizer.step()
            if target_encoder is not None:
                decay = float(cfg.ema_target_decay)
                with torch.no_grad():
                    for target_param, online_param in zip(
                        target_encoder.parameters(), model.encoder.parameters()
                    ):
                        target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
                    for target_buffer, online_buffer in zip(
                        target_encoder.buffers(), model.encoder.buffers()
                    ):
                        target_buffer.copy_(online_buffer)

            seen += batch_n
            epoch["total_loss"] += float(total.detach()) * batch_n
            epoch["pixel_loss"] += float(pixel_loss.detach()) * batch_n
            epoch["latent_loss"] += float(latent_loss.detach()) * batch_n
            epoch["reward_loss"] += float(reward_loss.detach()) * batch_n
            epoch["terminal_loss"] += float(terminal_loss.detach()) * batch_n
            epoch["risk_loss"] += float(risk_loss.detach()) * batch_n
            epoch["uncertainty_loss"] += float(uncertainty_loss.detach()) * batch_n
        for key in curves:
            curves[key].append(round(epoch[key] / max(seen, 1), 6))

    if cfg.context_length_curriculum and max_context:
        # Leave the model at its full window post-training regardless of how
        # the curriculum schedule landed (e.g. a curriculum longer than
        # cfg.epochs would otherwise end mid-ramp): evaluation should see
        # the trained backbone's full context, not a truncated one.
        model.set_context_length(max_context)

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
        "final_reward_loss": curves["reward_loss"][-1],
        "final_terminal_loss": curves["terminal_loss"][-1],
        "final_risk_loss": curves["risk_loss"][-1],
        "final_uncertainty_loss": curves["uncertainty_loss"][-1],
        "ema_target_enabled": target_encoder is not None,
        "ema_target_decay": cfg.ema_target_decay,
    }
    diagnostics = representation_collapse_diagnostics(model, dataset, config=cfg)
    stats["representation_diagnostics"] = diagnostics
    if cfg.collapse_gate_enabled and diagnostics["gate_evaluable"] and not diagnostics["passed"]:
        raise RepresentationCollapseError(diagnostics)
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

    Returns ``{"horizons": {h: {...}}, "rollout_health": {...},
    "per_episode_model_mse": {h: [...]}}``.  Each horizon entry carries raw
    MSEs (model / copy-last / mean-frame / recurrence-oracle), PSNRs, and the
    ratios ``model_over_copy_last_mse`` and ``model_over_oracle_mse``: < 1.0
    means the model beats that reference.  ``rollout_health.frozen_rollout``
    is True when predictions barely vary across horizons while the actual
    frames do -- the collapsed fixed-point signature this evaluation exists
    to catch.  ``per_episode_model_mse`` holds one model-MSE mean per
    contributing episode/seed (as opposed to ``horizons[h]["model_mse"]``,
    pooled over every overlapping rollout window) -- the independent samples
    ``statistical_evaluation.cortex_horizon_statistics`` needs to report a
    mean +/- CI across held-out seeds rather than a single point estimate.
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
    #: Per-episode mean model MSE at each horizon (one value per contributing
    #: episode/seed, not per window) -- the statistical_evaluation.py CI
    #: machinery needs independent samples across held-out seeds, not the
    #: many overlapping rollout-window samples pooled into ``samples`` above.
    per_episode_model_mse: Dict[int, List[float]] = {h: [] for h in horizons_sorted}
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
            episode_model_mse: Dict[int, List[float]] = {h: [] for h in horizons_sorted}
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
                    model_mse_sample = float(F.mse_loss(decoded, target))
                    samples[h]["model"].append(model_mse_sample)
                    episode_model_mse[h].append(model_mse_sample)
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
            for h in horizons_sorted:
                if episode_model_mse[h]:
                    per_episode_model_mse[h].append(_mean(episode_model_mse[h]))
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
    return {
        "horizons": report,
        "rollout_health": rollout_health,
        "per_episode_model_mse": per_episode_model_mse,
    }


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


# --------------------------------------------------------------------------- heads diagnostic


def evaluate_cortex_heads(
    model: Any,
    dataset: ActionSequenceDataset,
    horizons_frames: Sequence[int],
    *,
    warmup_frames: int = 3,
    max_starts_per_episode: Optional[int] = None,
    n_bootstrap: int = 500,
    bootstrap_seed: int = 0,
) -> Dict[int, Dict[str, Any]]:
    """Held-out diagnostic for the reward/terminal/risk/uncertainty heads
    (issue #169) -- the closed-loop counterpart to
    ``evaluate_action_world_model``'s pixel/latent report, kept as a
    separate additive function rather than folded into that (heavily
    reused, currently pixel-only) report.

    Per horizon:

    - ``reward``/``terminal``/``risk``: model MSE against the recorded
      target vs. a constant-predictor baseline (the held-out target mean --
      the same baseline discipline ``evaluate_action_world_model``'s own
      ``mean_frame_mse`` already uses for pixels), plus whether the model
      beats it.
    - ``uncertainty``: the Pearson correlation between predicted
      ``uncertainty`` and the realized squared latent error at that
      horizon, with a percentile bootstrap confidence interval -- the
      calibration check ``training.world_model.uncertainty_calibration``
      runs for the legacy memoryless model, here for the recurrent cortex.
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
        h: {
            "reward_pred": [], "reward_target": [],
            "terminal_pred": [], "terminal_target": [],
            "risk_pred": [], "risk_target": [],
            "uncertainty": [], "latent_error": [],
        }
        for h in horizons_sorted
    }

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, _targets, actions in episodes:
            n = pixels.shape[0]
            if n <= max_horizon + warmup_frames:
                continue
            remap = torch.tensor(
                [action_index[dataset.action_keys[a]] for a in episode.actions],
                dtype=torch.long,
            )
            latents = model.encoder(pixels)
            rewards, terminals, risks, _has_targets = _episode_head_targets(episode, n)

            hiddens = [model.initial_state(1)]
            hidden = hiddens[0]
            for i in range(n - 1):
                _pred, hidden = model.step(latents[i : i + 1], remap[i : i + 1], hidden)
                hiddens.append(hidden)

            starts = range(warmup_frames, n - max_horizon)
            if max_starts_per_episode is not None:
                starts = list(starts)[:max_starts_per_episode]
            for t in starts:
                out = model.forward_horizons(
                    latents[t : t + 1],
                    remap[t : t + max_horizon].unsqueeze(0),
                    hiddens[t],
                    horizon_frames=horizons_sorted,
                )
                for h in horizons_sorted:
                    pred = out[h]
                    entry = samples[h]
                    entry["reward_pred"].append(float(pred.reward))
                    entry["reward_target"].append(float(rewards[t + h]))
                    entry["terminal_pred"].append(float(torch.sigmoid(pred.terminal_logit)))
                    entry["terminal_target"].append(float(terminals[t + h]))
                    entry["risk_pred"].append(float(pred.risk))
                    entry["risk_target"].append(float(risks[t + h]))
                    entry["uncertainty"].append(float(pred.uncertainty))
                    # Same normalized scale ``uncertainty_head`` is trained
                    # against in train_action_world_model, not raw latent MSE.
                    entry["latent_error"].append(
                        float(
                            F.mse_loss(
                                F.normalize(pred.latent, dim=1).squeeze(0),
                                F.normalize(latents[t + h : t + h + 1], dim=1).squeeze(0),
                            )
                        )
                    )
    if was_training:
        model.train()

    report: Dict[int, Dict[str, Any]] = {}
    for h in horizons_sorted:
        entry = samples[h]
        if not entry["reward_pred"]:
            raise ValueError(f"no evaluation samples at horizon {h}; episodes too short for it")
        reward_mse, reward_const_mse = _head_mse_vs_constant(entry["reward_pred"], entry["reward_target"])
        terminal_mse, terminal_const_mse = _head_mse_vs_constant(
            entry["terminal_pred"], entry["terminal_target"]
        )
        risk_mse, risk_const_mse = _head_mse_vs_constant(entry["risk_pred"], entry["risk_target"])
        correlation, ci = _bootstrap_correlation_ci(
            entry["uncertainty"], entry["latent_error"],
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        report[h] = {
            "n_samples": len(entry["reward_pred"]),
            "reward_mse": reward_mse,
            "reward_constant_mse": reward_const_mse,
            "reward_beats_constant": _beats_constant(reward_mse, reward_const_mse),
            "terminal_mse": terminal_mse,
            "terminal_constant_mse": terminal_const_mse,
            "terminal_beats_constant": _beats_constant(terminal_mse, terminal_const_mse),
            "risk_mse": risk_mse,
            "risk_constant_mse": risk_const_mse,
            "risk_beats_constant": _beats_constant(risk_mse, risk_const_mse),
            "uncertainty_error_correlation": correlation,
            "uncertainty_error_correlation_ci": ci,
        }
    return report


def _head_mse_vs_constant(preds: Sequence[float], targets: Sequence[float]) -> Tuple[float, float]:
    """``(model_mse, constant_predictor_mse)`` where the constant predictor
    always answers the held-out targets' own mean -- the same baseline
    discipline as ``evaluate_action_world_model``'s ``mean_frame_mse``."""
    n = len(targets)
    mean_target = sum(targets) / n
    model_mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n
    constant_mse = sum((mean_target - t) ** 2 for t in targets) / n
    return model_mse, constant_mse


def _beats_constant(model_mse: float, constant_mse: float) -> Optional[bool]:
    """``model_mse < constant_mse``, or ``None`` when the target column is
    degenerate (e.g. a holdout split with no death events, so every
    ``terminal`` target is 0.0): the constant predictor there scores an
    unbeatable exact 0.0, which is a property of the (missing) held-out
    coverage, not a signal about the head, so "does this beat the
    baseline" isn't a meaningful question to answer either way."""
    if constant_mse < 1e-9:
        return None
    return bool(model_mse < constant_mse)


def _pearson_correlation_floats(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation over plain float sequences; ``0.0`` for
    degenerate inputs (fewer than 2 samples, or a constant series) rather
    than a division-by-zero NaN."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom > 0 else 0.0


def _bootstrap_correlation_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """``(point_correlation, (ci_low, ci_high))`` via percentile bootstrap --
    issue #169's "uncertainty correlates positively with realized held-out
    latent error, CI clear of 0" acceptance criterion needs an interval, not
    just a point estimate."""
    point = _pearson_correlation_floats(xs, ys)
    n = len(xs)
    if n < 2:
        return point, (point, point)
    rng = random.Random(seed)
    resampled = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        resampled.append(_pearson_correlation_floats([xs[i] for i in idx], [ys[i] for i in idx]))
    resampled.sort()
    lo = (1.0 - confidence) / 2.0
    hi = 1.0 - lo
    lo_i = max(0, min(n_bootstrap - 1, int(lo * n_bootstrap)))
    hi_i = max(0, min(n_bootstrap - 1, int(hi * n_bootstrap)))
    return point, (resampled[lo_i], resampled[hi_i])


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


class RepresentationCollapseError(RuntimeError):
    """Raised when the yaw/variance representation gate detects collapse."""

    def __init__(self, diagnostics: Dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(
            "representation collapse gate failed: hidden yaw R^2 "
            f"{diagnostics.get('hidden_yaw_r2')} < "
            f"{diagnostics['thresholds']['min_hidden_yaw_r2']} while latent "
            f"variance/rank collapsed (variance={diagnostics['latent']['mean_variance']:.3e}, "
            f"effective_rank={diagnostics['latent']['effective_rank']:.3f})"
        )


def latent_variance_rank(model: Any, dataset: ActionSequenceDataset) -> Dict[str, Any]:
    """Measure encoder spread and effective rank over all recorded frames.

    Effective rank is ``exp(entropy(normalized singular-value energy))``;
    it is 0 for a constant representation and approaches ``latent_width``
    when variance is distributed evenly across dimensions.
    """
    torch, _F = _torch()
    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for _episode, pixels, _targets, _actions in episodes:
            rows.append(model.encoder(pixels))
    if was_training:
        model.train()
    if not rows:
        return {
            "n_samples": 0,
            "dimensions": int(model.latent_width),
            "mean_variance": 0.0,
            "effective_rank": 0.0,
            "matrix_rank": 0,
        }

    latents = torch.cat(rows, dim=0)
    centered = latents - latents.mean(dim=0, keepdim=True)
    variances = centered.pow(2).mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.pow(2)
    total_energy = energy.sum()
    if float(total_energy) > 0.0:
        probabilities = energy / total_energy
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    else:
        effective_rank = 0.0
    return {
        "n_samples": int(latents.shape[0]),
        "dimensions": int(latents.shape[1]),
        "mean_variance": float(variances.mean()),
        "min_variance": float(variances.min()),
        "max_variance": float(variances.max()),
        "effective_rank": effective_rank,
        "matrix_rank": int(torch.linalg.matrix_rank(centered)),
    }


def representation_collapse_diagnostics(
    model: Any,
    dataset: ActionSequenceDataset,
    *,
    config: Optional[ActionWorldModelConfig] = None,
) -> Dict[str, Any]:
    """Return the yaw/variance gate report used by training and promotion.

    A failure requires both a poor hidden-state yaw probe and a collapsed
    latent distribution.  Datasets without enough yaw labels are reported
    as non-evaluable rather than being silently called healthy or failed.
    """
    cfg = config or ActionWorldModelConfig()
    latent = latent_variance_rank(model, dataset)
    yaw_probe = linear_probe_yaw(model, dataset)
    hidden_yaw_r2 = (
        float(yaw_probe["hidden"]["r2"]) if "hidden" in yaw_probe else None
    )
    variance_collapsed = latent["mean_variance"] < cfg.collapse_gate_min_latent_variance
    rank_collapsed = latent["effective_rank"] < cfg.collapse_gate_min_effective_rank
    latent_collapsed = variance_collapsed or rank_collapsed
    gate_evaluable = hidden_yaw_r2 is not None
    yaw_collapsed = (
        gate_evaluable and hidden_yaw_r2 < cfg.collapse_gate_min_hidden_yaw_r2
    )
    passed = not (yaw_collapsed and latent_collapsed) if gate_evaluable else False
    return {
        "passed": bool(passed),
        "gate_evaluable": gate_evaluable,
        "hidden_yaw_r2": hidden_yaw_r2,
        "yaw_probe": yaw_probe,
        "latent": latent,
        "yaw_collapsed": bool(yaw_collapsed),
        "variance_collapsed": variance_collapsed,
        "rank_collapsed": rank_collapsed,
        "latent_collapsed": latent_collapsed,
        "thresholds": {
            "min_hidden_yaw_r2": cfg.collapse_gate_min_hidden_yaw_r2,
            "min_latent_variance": cfg.collapse_gate_min_latent_variance,
            "min_effective_rank": cfg.collapse_gate_min_effective_rank,
        },
    }


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
    encoder-only checkpoint cannot regenerate predictions.

    ``horizons_ticks`` (issue #91, task 4) is persisted here too, so a
    reloaded model's configured horizons survive the round trip unchanged.
    """
    torch, _F = _torch()
    torch.save(
        {
            "format": "action-world-model-v1",
            "pixel_shape": list(model.pixel_shape),
            "action_keys": list(model.action_keys),
            "latent_width": model.latent_width,
            "hidden_dim": model.hidden_dim,
            "reconstruction_shape": list(model.reconstruction_shape),
            "horizons_ticks": list(model.horizons_ticks),
            "backbone": model.config.backbone,
            "context_length": model.config.context_length,
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
        horizons_ticks=tuple(payload.get("horizons_ticks", ActionWorldModelConfig.horizons_ticks)),
        backbone=payload.get("backbone", ActionWorldModelConfig.backbone),
        context_length=int(payload.get("context_length", ActionWorldModelConfig.context_length)),
    )
    model = build_action_world_model(
        tuple(payload["pixel_shape"]), payload["action_keys"], cfg
    )
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("training_stats", {})
