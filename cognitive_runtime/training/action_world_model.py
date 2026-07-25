"""Action-conditioned recurrent world model (phases 1-3 of
docs/history/nursery-turn-in-place-analysis.md).

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
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

log = logging.getLogger("ccr.training.cortex")

from cognitive_runtime.observability import trace_event, trace_metrics
from cognitive_runtime.runtime.replay import iter_cognitive_ticks, list_episodes
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.core.streams import TemporalBuffer, TemporalFusion
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.runtime.replay import load_session_metadata, require_streams_v2

PIXEL_STREAM = "vision.frame.pixels"
MOTOR_STREAM = "motor.command"
ROTATION_STREAM = "spatial.rotation"
FACING_STREAM = "spatial.facing"
#: Recorded ground-truth stream for the cortex's terminal head (issue #169).
#: Reward and risk targets instead come straight off the decision record
#: (``reward_window_total``/``risk``) -- not the ``internal.*`` streams
#: derived from them, which the loop deliberately publishes one tick late
#: (``runtime/loop.py``'s "primes the *next* tick's evaluation" comment) so
#: reading them out of a tick's *sensory* window would train the risk head
#: on the previous tick's value.
DEATH_STREAM = "event.died"

# A prediction made by a direct multi-horizon head is not interchangeable
# with a repeatedly-applied one-step transition.  Keep this distinction in
# every report/export boundary rather than relying on a generic model_mse.
PredictionMode = Literal["direct", "rollout"]


# --------------------------------------------------------------------------- dataset


@dataclass
class EpisodeActionFrames:
    """One recorded episode as (frame, action, frame, action, ...).

    ``actions[i]`` is the action index driving the transition from
    ``frames[i]`` to ``frames[i+1]`` -- the ``motor.command`` emitted by the
    cognitive tick that observed ``frames[i]`` itself (issue #202), not the
    tick that observed ``frames[i+1]``: the runtime documents one-tick motor
    actuation latency (``runtime.loop``'s "Motor events emitted at cognitive
    tick *t* ... are applied by ``program.step()`` at the start of tick
    *t+1*"), so the command bundled with a tick's own log entry produces the
    *next* tick's frame, not the frame observed alongside it.  ``yaw[i]`` is
    the agent heading in degrees at ``frames[i]`` when ``spatial.rotation``
    was observed that tick, else ``None`` -- probe material, never a model
    input.
    """

    session_dir: str
    episode_id: str
    frames: List[Any] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    yaw: List[Optional[float]] = field(default_factory=list)
    #: Discrete grid facing ``(x, y)`` labels (Crafter).  The current value is
    #: forward-filled between delta-published facing events.
    facing: List[Optional[Tuple[float, float]]] = field(default_factory=list)
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
    #: The non-visual half of the bound workspace, replayed through the same
    #: ``TemporalFusion`` code path as the live runtime at each visual frame.
    workspace: List[List[float]] = field(default_factory=list)
    #: Latest recorded ``vision.frame.grid`` aligned to each pixel frame;
    #: ``None`` means the recording predates semantic supervision.
    semantic_grids: List[Optional[List[List[int]]]] = field(default_factory=list)

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
    workspace_layout_hash: Optional[str] = None
    workspace_feature_names: List[str] = field(default_factory=list)

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

    @property
    def workspace_width(self) -> int:
        return len(self.workspace_feature_names)


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


def _tick_facing(
    sensory_records: List[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """Read Crafter's discrete ``spatial.facing`` direction, when present."""
    for record in reversed(sensory_records):
        if record.get("stream_id") != FACING_STREAM:
            continue
        payload = record.get("payload") or {}
        x, y = payload.get("x"), payload.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
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
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        session_fusion = TemporalFusion(
            [StreamSpec.from_dict(spec) for spec in metadata.get("stream_catalog", [])]
        )
        if dataset.workspace_layout_hash is None:
            dataset.workspace_layout_hash = session_fusion.layout_hash
            dataset.workspace_feature_names = list(session_fusion.feature_names())
        elif session_fusion.layout_hash != dataset.workspace_layout_hash:
            raise ValueError(
                f"session {session_dir} has incompatible workspace fusion layout "
                f"({session_fusion.layout_hash} vs {dataset.workspace_layout_hash})"
            )
        frame_store = open_frame_store(session_dir)
        try:
            for episode_id in list_episodes(session_dir):
                episode = EpisodeActionFrames(session_dir=session_dir, episode_id=episode_id)
                workspace_buffer = TemporalBuffer()
                #: The motor command actually driving the transition into
                #: whichever frame(s) the *current* tick's sensory window
                #: contains -- i.e. the previous tick's own emission, held
                #: here until *this* tick's frames are recorded (issue #202:
                #: one-tick actuation latency, see the module-level comment
                #: below).
                last_committed_action: Optional[str] = None
                last_facing: Optional[Tuple[float, float]] = None
                last_semantic_grid: Optional[List[List[int]]] = None
                for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
                    workspace_buffer.extend([
                        stream_event_from_log(record, frame_store=frame_store) for record in sensory
                    ])
                    workspace_vector = session_fusion.fuse(None, workspace_buffer).vector
                    tick = int(decision.get("tick_index", len(episode.ticks)))
                    # One-tick motor actuation latency
                    # (``runtime.loop``'s documented contract): the motor
                    # command bundled with *this* tick's log entry is the
                    # action *this* tick just decided -- ``program.step()``
                    # applies it at the *start of the next* tick, so it
                    # produces the *next* frame, not any frame in this
                    # tick's own sensory window. Read it here, but do not
                    # use it to label this tick's frames below; only fold
                    # it into ``last_committed_action`` (used by the
                    # *following* tick's frames) after this tick's frames
                    # are recorded.
                    this_tick_action_name = _tick_action_name(motor)
                    yaw = _tick_yaw(sensory)
                    facing = _tick_facing(sensory) or last_facing
                    if facing is not None:
                        last_facing = facing
                    reward = float(decision.get("reward_window_total", 0.0))
                    terminal = _tick_terminal(sensory)
                    # Straight off the decision record (this tick's own
                    # `Prediction.risk`), not the `internal.risk` stream --
                    # see the module-level DEATH_STREAM comment for why.
                    risk = float(decision.get("risk", 0.0))
                    for record in sensory:
                        if record.get("stream_id") == "vision.frame.grid":
                            grid = record.get("payload")
                            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                                last_semantic_grid = grid
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
                            episode.actions.append(
                                action_index(last_committed_action or "NULL")
                            )
                        episode.frames.append(frame)
                        episode.yaw.append(yaw)
                        episode.facing.append(facing)
                        episode.ticks.append(tick)
                        episode.reward.append(reward)
                        episode.terminal.append(terminal)
                        episode.risk.append(risk)
                        episode.workspace.append(list(workspace_vector))
                        episode.semantic_grids.append(last_semantic_grid)
                    # Now that this tick's frames (if any) are recorded
                    # against the *previous* tick's command, fold this
                    # tick's own emission in for the *next* tick's frame(s)
                    # -- covers tick zero's two frames (the post-reset
                    # snapshot, then the first `program.step()`'s NULL-
                    # actuated frame) the same way: both are labelled
                    # ``last_committed_action`` as of *before* this loop
                    # iteration, i.e. still unset ("NULL") the first time
                    # through.
                    if this_tick_action_name is not None:
                        last_committed_action = this_tick_action_name
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
    visual_architecture: str = "spatial_residual_v1"
    change_mask_sparsity_weight: float = 0.01
    scene_height: Optional[int] = None
    hud_loss_weight: float = 0.25
    semantic_loss_weight: float = 1.0
    semantic_classes: int = 0
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
    scheduled_sampling_p: float = 0.25
    #: ``"windowed_rollout"`` preserves the original short scheduled-
    #: sampling trainer; ``"autoregressive"`` trains every causal prefix
    #: and every configured direct horizon in one sequence pass (C1).
    training_objective: str = "windowed_rollout"
    #: Small exposure-bias term retained by the autoregressive objective.
    #: Closed-loop auxiliary losses for the autoregressive objective.  The
    #: legacy combined name is retained as a compatibility alias: when set it
    #: supplies the latent weight unless the explicit field was changed.
    autoregressive_rollout_weight: Optional[float] = None
    autoregressive_rollout_latent_weight: float = 0.1
    autoregressive_rollout_pixel_weight: float = 0.5
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
    #: Build-time ring-buffer capacity for windowed backbones; ignored by
    #: ``"gru"``. The curriculum below changes only the effective attended
    #: width, not this capacity.
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
    #: Bind the CNN visual latent with the stream-native fused workspace and
    #: the motor efference copy (C2).  Kept configurable for pixel-only A/Bs
    #: and for loading historical checkpoints.
    workspace_enabled: bool = True


def build_action_world_model(
    pixel_shape: Tuple[int, int, int],
    action_keys: Sequence[str],
    config: Optional[ActionWorldModelConfig] = None,
    *,
    workspace_modalities: Optional[Dict[str, int]] = None,
    workspace_layout_hash: Optional[str] = None,
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
        visual_architecture=cfg.visual_architecture,
        change_mask_sparsity_weight=cfg.change_mask_sparsity_weight,
        semantic_classes=cfg.semantic_classes,
        horizons_ticks=tuple(cfg.horizons_ticks),
        backbone=cfg.backbone,
        context_length=cfg.context_length,
        backbone_kwargs=dict(cfg.backbone_kwargs),
        workspace_modalities=dict(workspace_modalities or {}),
        workspace_layout_hash=workspace_layout_hash,
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


def _workspace_modalities(dataset: ActionSequenceDataset, enabled: bool) -> Dict[str, int]:
    """C2's prediction-token layout: fused streams plus efference copy."""
    if not enabled:
        return {}
    # Hand-built legacy/synthetic datasets contain only pixel frames.  Keep
    # those fixtures and historical callers on the pixel-only path; recorded
    # streams-v2 datasets always supply the fused layout above.
    if dataset.workspace_width <= 0:
        return {}
    return {"workspace": dataset.workspace_width, "efference": len(dataset.action_keys)}


def _episode_workspace_tensors(
    episode: EpisodeActionFrames,
    dataset: ActionSequenceDataset,
    model: Any,
    *,
    actions: Optional[Any] = None,
) -> Dict[str, Any]:
    """Tensorize the bound workspace at every recorded visual frame."""
    torch, _F = _torch()
    if not getattr(model, "workspace_modalities", {}):
        return {}
    if model.workspace_layout_hash != dataset.workspace_layout_hash:
        raise ValueError(
            "workspace layout mismatch: model was trained for "
            f"{model.workspace_layout_hash!r}, dataset supplies {dataset.workspace_layout_hash!r}"
        )
    n_frames = len(episode.frames)
    if len(episode.workspace) != n_frames:
        raise ValueError(
            f"{episode.session_dir}/{episode.episode_id}: workspace/frame alignment is "
            f"{len(episode.workspace)}/{n_frames}"
        )
    result: Dict[str, Any] = {
        "workspace": torch.tensor(episode.workspace, dtype=torch.float32),
    }
    if "efference" in model.workspace_modalities:
        action_values = actions if actions is not None else torch.tensor(episode.actions, dtype=torch.long)
        efference = torch.zeros(n_frames, len(model.action_keys), dtype=torch.float32)
        if action_values.numel():
            efference[:-1].scatter_(1, action_values.reshape(-1, 1), 1.0)
        result["efference"] = efference
    return result


def _workspace_loss(model: Any, predicted: Any, targets: Dict[str, Any]) -> Any:
    """Mean reconstruction loss over every non-visual workspace modality."""
    torch, F = _torch()
    if not getattr(model, "workspace_modalities", {}):
        return predicted.new_zeros(())
    decoded = model.decode_workspace(predicted)
    terms = [F.mse_loss(decoded[name], targets[name]) for name in model.workspace_modalities]
    return torch.stack(terms).mean()


def _spatial_pixel_loss(model: Any, predicted: Any, reference: Any, target: Any, cfg: ActionWorldModelConfig,
                        reference_spatial: Optional[Any] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Scene/HUD weighted residual loss and its mask regularizer."""
    torch, F = _torch()
    flat_reference = reference.reshape(-1, *reference.shape[-3:])
    encoding = model.encode_visual(flat_reference) if reference_spatial is None else None
    decoded = model.decode_prediction(
        predicted.reshape(-1, predicted.shape[-1]),
        reference_frame=flat_reference,
        reference_spatial=(encoding.spatial if encoding is not None else reference_spatial.reshape(-1, *reference_spatial.shape[-3:])),
    )
    vision = decoded["vision"].reshape_as(target)
    scene_h = cfg.scene_height if cfg.scene_height is not None else vision.shape[-2]
    scene_h = max(0, min(int(scene_h), vision.shape[-2]))
    scene = F.mse_loss(vision[..., :scene_h, :], target[..., :scene_h, :]) if scene_h else vision.new_zeros(())
    hud = F.mse_loss(vision[..., scene_h:, :], target[..., scene_h:, :]) if scene_h < vision.shape[-2] else vision.new_zeros(())
    return scene + cfg.hud_loss_weight * hud, decoded["change_mask"].mean(), decoded


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


def _episode_semantic_targets(episode: EpisodeActionFrames, n_frames: int, classes: int) -> Optional[Any]:
    """Return aligned 9x9 class targets, or ``None`` for old recordings."""
    torch, F = _torch()
    if classes <= 0 or len(episode.semantic_grids) != n_frames or any(g is None for g in episode.semantic_grids):
        return None
    grids = torch.tensor(episode.semantic_grids, dtype=torch.long)
    if grids.ndim != 3 or grids.min() < 0 or grids.max() >= classes:
        raise ValueError(f"semantic grids must contain ids in [0, {classes}), got shape {tuple(grids.shape)}")
    return F.interpolate(grids.unsqueeze(1).float(), size=(9, 9), mode="nearest").squeeze(1).long()


def _update_ema_target_encoder(target_encoder: Any, model: Any, decay: float) -> None:
    torch, _F = _torch()
    with torch.no_grad():
        for target_param, online_param in zip(target_encoder.parameters(), model.parameters()):
            target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
        for target_buffer, online_buffer in zip(target_encoder.buffers(), model.buffers()):
            target_buffer.copy_(online_buffer)


def _train_autoregressive_objective(
    model: Any,
    dataset: ActionSequenceDataset,
    episodes: Sequence[Any],
    head_targets: Sequence[Any],
    cfg: ActionWorldModelConfig,
    target_encoder: Optional[Any],
    generator: Any,
    max_context: Optional[int],
    curriculum_epochs: int,
    ticks_per_frame: float,
) -> Dict[str, Any]:
    """Train all causal prefixes and configured direct horizons (C1)."""
    torch, F = _torch()
    horizons_ticks = tuple(model.horizons_ticks)
    if not horizons_ticks or horizons_ticks[0] < 1:
        raise ValueError(
            f"autoregressive objective needs positive horizons, got {horizons_ticks!r}"
        )
    if 1 not in horizons_ticks:
        raise ValueError(
            "autoregressive objective requires horizon 1 so the live one-step "
            "latent/head path remains trained"
        )
    # The model's direct heads remain keyed by tick horizons, while episode
    # tensors are indexed in recorded frames.  Convert each horizon separately
    # so two tick heads that share a frame offset are both trained.
    horizon_frames = tuple(
        horizons_ticks_to_frames((horizon,), ticks_per_frame)[0]
        for horizon in horizons_ticks
    )
    if all(
        pixels.shape[0] <= min(horizon_frames)
        for _episode, pixels, _targets, _actions in episodes
    ):
        raise ValueError("no autoregressive targets: episodes are shorter than the first horizon")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, List[float]] = {
        "total_loss": [], "pixel_loss": [], "latent_loss": [],
        "reward_loss": [], "terminal_loss": [], "risk_loss": [], "uncertainty_loss": [],
        "closed_loop_loss": [], "closed_loop_latent_loss": [],
        "closed_loop_pixel_loss": [], "workspace_loss": [], "semantic_loss": [],
    }
    log.info("training cortex (autoregressive)  episodes=%d  epochs=%d  horizons=%s  backbone=%s",
             len(episodes), cfg.epochs, list(horizons_ticks), _backbone_name(model))
    trace_event(
        "cortex.train.start", objective="autoregressive", episodes=len(episodes),
        epochs=cfg.epochs, horizons_ticks=list(horizons_ticks),
        horizon_frames=list(horizon_frames), backbone=_backbone_name(model),
        lr=cfg.lr, latent_width=cfg.latent_width, hidden_dim=cfg.hidden_dim,
        context_length=max_context, ema_target=target_encoder is not None,
    )
    training_started = time.perf_counter()
    rollout_latent_weight = (
        cfg.autoregressive_rollout_latent_weight
        if cfg.autoregressive_rollout_weight is None
        else cfg.autoregressive_rollout_weight
    )
    model.train()
    for epoch_idx in range(cfg.epochs):
        epoch_started = time.perf_counter()
        if cfg.context_length_curriculum and max_context:
            progress = min(epoch_idx / max(curriculum_epochs - 1, 1), 1.0)
            model.set_context_length(max(1, round(1 + progress * (max_context - 1))))
        epoch = {key: 0.0 for key in curves}
        seen = 0
        for episode_index in torch.randperm(len(episodes), generator=generator).tolist():
            _episode, pixels, targets, actions = episodes[episode_index]
            rewards, terminals, risks, has_targets = head_targets[episode_index]
            semantic_targets = _episode_semantic_targets(
                _episode, pixels.shape[0], cfg.semantic_classes
            )
            if pixels.shape[0] <= min(horizon_frames):
                continue
            optimizer.zero_grad()
            workspace = _episode_workspace_tensors(_episode, dataset, model, actions=actions)
            latents = model.encode_workspace(pixels, workspace).unsqueeze(0)
            target_latents = latents
            if target_encoder is not None:
                with torch.no_grad():
                    target_latents = target_encoder.encode_workspace(pixels, workspace).unsqueeze(0)
            actions_b = actions.unsqueeze(0)
            if cfg.withhold_actions:
                actions_b = torch.zeros_like(actions_b)
            hidden = model.forward_sequence(latents[:, :-1], actions_b)

            latent_terms = []
            pixel_terms = []
            mask_terms = []
            reward_terms = []
            terminal_terms = []
            risk_terms = []
            uncertainty_terms = []
            workspace_terms = []
            semantic_terms = []
            for horizon_tick, horizon_frame in zip(horizons_ticks, horizon_frames):
                positions = pixels.shape[0] - horizon_frame
                if positions <= 0:
                    continue
                reference_encoding = model.encode_visual(pixels[:positions])
                prediction = model.sequence_prediction(
                    hidden[:, :positions], horizon_tick,
                    reference_frame=targets[:positions].unsqueeze(0),
                    reference_spatial=reference_encoding.spatial.unsqueeze(0),
                )
                target_latent = target_latents[:, horizon_frame:].detach()
                per_error = (
                    F.normalize(prediction.latent, dim=-1) - F.normalize(target_latent, dim=-1)
                ).pow(2).mean(dim=-1)
                latent_terms.append(per_error.mean())
                pixel_term, mask_term, _decoded = _spatial_pixel_loss(
                    model, prediction.latent, pixels[:positions].unsqueeze(0),
                    targets[horizon_frame:].unsqueeze(0), cfg,
                )
                pixel_terms.append(pixel_term)
                mask_terms.append(mask_term)
                if semantic_targets is not None:
                    assert prediction.semantic_logits is not None
                    semantic_terms.append(F.cross_entropy(
                        prediction.semantic_logits.reshape(-1, cfg.semantic_classes, 9, 9),
                        semantic_targets[horizon_frame:].reshape(-1, 9, 9),
                    ))
                uncertainty_terms.append(F.mse_loss(prediction.uncertainty, per_error.detach()))
                workspace_terms.append(_workspace_loss(
                    model, prediction.latent,
                    {name: value[horizon_frame:].unsqueeze(0) for name, value in workspace.items()},
                ))
                if has_targets:
                    reward_terms.append(F.mse_loss(
                        prediction.reward, rewards[horizon_frame:].unsqueeze(0)
                    ))
                    terminal_terms.append(F.binary_cross_entropy_with_logits(
                        prediction.terminal_logit, terminals[horizon_frame:].unsqueeze(0)
                    ))
                    risk_terms.append(F.mse_loss(
                        prediction.risk, risks[horizon_frame:].unsqueeze(0)
                    ))

            def _mean_or_zero(terms: Sequence[Any]) -> Any:
                return torch.stack(list(terms)).mean() if terms else pixels.new_zeros(())

            latent_loss = _mean_or_zero(latent_terms)
            pixel_loss = _mean_or_zero(pixel_terms)
            mask_sparsity_loss = _mean_or_zero(mask_terms)
            semantic_loss = _mean_or_zero(semantic_terms)
            reward_loss = _mean_or_zero(reward_terms)
            terminal_loss = _mean_or_zero(terminal_terms)
            risk_loss = _mean_or_zero(risk_terms)
            uncertainty_loss = _mean_or_zero(uncertainty_terms)
            workspace_loss = _mean_or_zero(workspace_terms)

            # ``autoregressive_rollout_weight`` was the old latent-only
            # coefficient.  Prefer the explicit knobs, but interpret a
            # supplied legacy value faithfully for old callers/checkpoints.
            closed_loop_latent_loss = pixels.new_zeros(())
            closed_loop_pixel_loss = pixels.new_zeros(())
            if rollout_latent_weight > 0 or cfg.autoregressive_rollout_pixel_weight > 0:
                state = model.initial_state(1)
                warmup = min(max(cfg.warmup_frames - 1, 0), actions_b.shape[1] - 1)
                for index in range(warmup):
                    _prediction, state = model.step(latents[:, index], actions_b[:, index], state)
                latent_in = latents[:, warmup]
                reference_frame = targets[warmup].unsqueeze(0)
                reference_spatial = model.encode_visual(F.interpolate(
                    reference_frame, size=model.pixel_shape[:2], mode="nearest"
                )).spatial
                rollout_steps = min(cfg.rollout_frames, actions_b.shape[1] - warmup)
                rollout_latent_terms = []
                rollout_pixel_terms = []
                for offset in range(rollout_steps):
                    index = warmup + offset
                    predicted, state = model.step(latent_in, actions_b[:, index], state)
                    rollout_latent_terms.append((
                        F.normalize(predicted, dim=-1)
                        - F.normalize(target_latents[:, index + 1].detach(), dim=-1)
                    ).pow(2).mean())
                    # This is deliberately decoded from the *closed-loop*
                    # latent.  It gives gradients to both decoder and
                    # transition path, unlike the direct-head pixel term.
                    pixel_term, _mask_term, decoded = _spatial_pixel_loss(
                        model, predicted, reference_frame, targets[index + 1].unsqueeze(0), cfg,
                        reference_spatial=reference_spatial,
                    )
                    rollout_pixel_terms.append(pixel_term)
                    use_prediction = float(torch.rand((), generator=generator)) < cfg.scheduled_sampling_p
                    latent_in = predicted if use_prediction else latents[:, index + 1]
                    reference_frame = decoded["vision"] if use_prediction else targets[index + 1].unsqueeze(0)
                    reference_spatial = model.encode_visual(F.interpolate(
                        reference_frame, size=model.pixel_shape[:2], mode="nearest"
                    )).spatial
                closed_loop_latent_loss = _mean_or_zero(rollout_latent_terms)
                closed_loop_pixel_loss = _mean_or_zero(rollout_pixel_terms)
            closed_loop_loss = (
                rollout_latent_weight * closed_loop_latent_loss
                + cfg.autoregressive_rollout_pixel_weight * closed_loop_pixel_loss
            )

            total = (
                cfg.pixel_loss_weight * pixel_loss
                + cfg.latent_loss_weight * latent_loss
                + cfg.reward_loss_weight * reward_loss
                + cfg.terminal_loss_weight * terminal_loss
                + cfg.risk_loss_weight * risk_loss
                + cfg.uncertainty_loss_weight * uncertainty_loss
                + workspace_loss
                + cfg.change_mask_sparsity_weight * mask_sparsity_loss
                + cfg.semantic_loss_weight * semantic_loss
                + closed_loop_loss
            )
            total.backward()
            optimizer.step()
            if target_encoder is not None:
                _update_ema_target_encoder(target_encoder, model, float(cfg.ema_target_decay))

            seen += 1
            for key, value in (
                ("total_loss", total), ("pixel_loss", pixel_loss), ("latent_loss", latent_loss),
                ("reward_loss", reward_loss), ("terminal_loss", terminal_loss),
                ("risk_loss", risk_loss), ("uncertainty_loss", uncertainty_loss),
                ("closed_loop_loss", closed_loop_loss),
                ("closed_loop_latent_loss", closed_loop_latent_loss),
                ("closed_loop_pixel_loss", closed_loop_pixel_loss),
                ("workspace_loss", workspace_loss),
                ("semantic_loss", semantic_loss),
            ):
                epoch[key] += float(value.detach())
        for key in curves:
            curves[key].append(round(epoch[key] / max(seen, 1), 6))
        _report_epoch(
            epoch_idx, cfg.epochs, curves,
            epoch_seconds=time.perf_counter() - epoch_started,
            training_started=training_started,
            **{"train/episodes_seen": seen,
               # only meaningful for the windowed backbones, which ramp it
               # over the context-length curriculum; `None` for the GRU
               **({"train/context_length": model.context_length}
                  if getattr(model, "context_length", None) else {})},
        )

    log.info("training complete  final_total=%.4f", curves["total_loss"][-1])
    trace_event("cortex.train.done", objective="autoregressive",
                final_total_loss=curves["total_loss"][-1],
                duration_s=round(time.perf_counter() - training_started, 3))
    if cfg.context_length_curriculum and max_context:
        model.set_context_length(max_context)
    return {
        "samples": float(sum(
            max(0, pixels.shape[0] - min(horizon_frames))
            for _e, pixels, _t, _a in episodes
        )),
        "episodes": float(len(episodes)), "epochs": float(cfg.epochs),
        "action_keys": list(model.action_keys), "warmup_frames": cfg.warmup_frames,
        "rollout_frames": cfg.rollout_frames, "loss_curves": curves,
        "training_objective": "autoregressive",
        "autoregressive_horizons": list(horizon_frames),
        "autoregressive_horizons_ticks": list(horizons_ticks),
        "ticks_per_frame": ticks_per_frame,
        "autoregressive_rollout_latent_weight": rollout_latent_weight,
        "autoregressive_rollout_pixel_weight": cfg.autoregressive_rollout_pixel_weight,
        "ema_target_enabled": target_encoder is not None, "ema_target_decay": cfg.ema_target_decay,
        **{f"final_{key}": curves[key][-1] for key in curves},
    }


def _window_weights(
    transition_weights: Sequence[Sequence[float]],
    starts: Sequence[Tuple[int, int]],
    window: int,
) -> List[float]:
    """Per-window sampling weight (issue #202) from per-transition weights:
    the mean of the ``window - 1`` transition weights the window actually
    trains on (``actions[t : t + window - 1]``, mirroring the batch-building
    slice below) -- a window overlapping mostly-blocked/stationary
    transitions is weighted down, one overlapping a rare entity entry/exit
    is weighted up. Falls back to ``1.0`` (uniform) for a window with no
    transitions in range, which should not occur given ``starts`` is built
    from these same episodes."""
    weights: List[float] = []
    for e_idx, t in starts:
        span = transition_weights[e_idx][t : t + window - 1]
        weights.append(sum(span) / len(span) if span else 1.0)
    return weights


def train_action_world_model(
    dataset: ActionSequenceDataset,
    config: Optional[ActionWorldModelConfig] = None,
    *,
    initial_model: Optional[Any] = None,
    transition_weights: Optional[Sequence[Sequence[float]]] = None,
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

    ``transition_weights`` (issue #202), when given, is one list per
    episode of ``dataset.episodes`` -- ``transition_weights[e][i]`` is the
    per-transition sampling weight for that episode's ``actions[i]``
    (e.g. from ``training.nursery.balanced_transition_weights``), used
    under the ``"windowed_rollout"`` objective to draw training windows
    with replacement in proportion to their (mean) weight instead of
    uniformly. This reweights *sampling frequency* only -- the recorded
    dataset itself is never duplicated or altered; a window whose weight
    is high may simply be drawn more than once in a given epoch (and a
    low-weight one not at all), same as a standard
    ``torch.utils.data.WeightedRandomSampler``. ``None`` (default)
    preserves the original uniform-permutation behavior. Ignored under the
    ``"autoregressive"`` objective, which does not sample windows this way.
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
    if cfg.training_objective not in {"windowed_rollout", "autoregressive"}:
        raise ValueError(
            "training_objective must be 'windowed_rollout' or 'autoregressive', got "
            f"{cfg.training_objective!r}"
        )
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    expected_workspace_modalities = _workspace_modalities(dataset, cfg.workspace_enabled)
    expected_workspace_layout = dataset.workspace_layout_hash if expected_workspace_modalities else None

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
        if (
            dict(getattr(initial_model, "workspace_modalities", {})) != expected_workspace_modalities
            or getattr(initial_model, "workspace_layout_hash", None) != expected_workspace_layout
        ):
            raise ValueError(
                "initial_model workspace layout does not match this dataset; "
                "start a new C2 checkpoint rather than silently continuing a "
                "pixel-only or differently-fused cortex"
            )
        model = initial_model
    else:
        model = build_action_world_model(
            dataset.pixel_shape, dataset.action_keys, cfg,
            workspace_modalities=expected_workspace_modalities,
            workspace_layout_hash=expected_workspace_layout,
        )
    # Evaluation must know which prediction surface was optimized.  It is
    # serialized in training_stats as well for a reloaded checkpoint.
    model.training_objective = cfg.training_objective
    model.training_visual_config = {
        "scene_height": cfg.scene_height,
        "hud_loss_weight": cfg.hud_loss_weight,
        "semantic_loss_weight": cfg.semantic_loss_weight,
        "semantic_classes": cfg.semantic_classes,
        "change_mask_sparsity_weight": cfg.change_mask_sparsity_weight,
    }
    target_encoder = None
    if cfg.ema_target_decay is not None:
        # The workspace binding is part of z, so the EMA target must carry
        # the complete token encoder, not merely the visual CNN.
        target_encoder = copy.deepcopy(model)
        target_encoder.requires_grad_(False)
        target_encoder.eval()
    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    #: Per-episode reward/terminal/risk targets (issue #169), aligned 1:1
    #: with ``episodes`` by index.
    head_targets = [
        _episode_head_targets(episode, pixels.shape[0]) for episode, pixels, _t, _a in episodes
    ]

    max_context = getattr(model, "context_length_max", None)
    curriculum_epochs = max(cfg.context_length_curriculum_epochs or cfg.epochs, 1)
    if cfg.training_objective == "autoregressive":
        stats = _train_autoregressive_objective(
            model, dataset, episodes, head_targets, cfg, target_encoder, generator,
            max_context, curriculum_epochs, dataset.ticks_per_frame,
        )
        stats["ticks_per_frame"] = dataset.ticks_per_frame
        diagnostics = representation_collapse_diagnostics(model, dataset, config=cfg)
        stats["representation_diagnostics"] = diagnostics
        if cfg.collapse_gate_enabled and diagnostics["gate_evaluable"] and not diagnostics["passed"]:
            raise RepresentationCollapseError(diagnostics)
        return model, stats

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
    start_weights = None
    if transition_weights is not None:
        if len(transition_weights) != len(episodes):
            raise ValueError(
                f"transition_weights has {len(transition_weights)} episode(s), dataset has "
                f"{len(episodes)}"
            )
        for index, (weights, (_episode, _pixels, _targets, actions)) in enumerate(zip(transition_weights, episodes)):
            if len(weights) != len(actions):
                raise ValueError(
                    f"transition_weights has {len(weights)} transitions for episode {index}, "
                    f"dataset has {len(actions)}"
                )
        start_weights = torch.tensor(
            _window_weights(transition_weights, starts, window), dtype=torch.float
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
        "workspace_loss": [],
        "semantic_loss": [],
    }

    log.info("training cortex (windowed_rollout)  windows=%d  epochs=%d  backbone=%s",
             len(starts), cfg.epochs, _backbone_name(model))
    trace_event(
        "cortex.train.start", objective="windowed_rollout", windows=len(starts),
        epochs=cfg.epochs, backbone=_backbone_name(model),
        lr=cfg.lr, batch_size=cfg.batch_size, latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim, context_length=max_context,
    )
    training_started = time.perf_counter()
    model.train()
    for epoch_idx in range(cfg.epochs):
        epoch_started = time.perf_counter()
        if cfg.context_length_curriculum and max_context:
            progress = min(epoch_idx / max(curriculum_epochs - 1, 1), 1.0)
            model.set_context_length(max(1, round(1 + progress * (max_context - 1))))
        if start_weights is not None:
            # Balanced sampling (issue #202): draw the same number of
            # windows as a uniform epoch would, but weighted -- a
            # standard WeightedRandomSampler-style draw with replacement,
            # so an over-represented class (e.g. a long blocked/stationary
            # phase) is drawn less often in expectation and an
            # under-represented one (e.g. an entity entering/leaving) more
            # often, without altering the recorded dataset itself.
            perm = torch.multinomial(
                start_weights, num_samples=len(starts), replacement=True, generator=generator,
            )
        else:
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
            semantic_batch = []
            workspace_batch: Dict[str, List[Any]] = {name: [] for name in model.workspace_modalities}
            has_targets_batch = []
            for flat in batch_ids.tolist():
                e_idx, t = starts[flat]
                _episode, pixels, targets, actions = episodes[e_idx]
                rewards, terminals, risks, has_targets = head_targets[e_idx]
                frames_batch.append(pixels[t : t + window])
                targets_batch.append(targets[t : t + window])
                actions_batch.append(actions[t : t + window - 1])
                semantic_batch.append(_episode.semantic_grids[t : t + window])
                rewards_batch.append(rewards[t : t + window])
                terminals_batch.append(terminals[t : t + window])
                risks_batch.append(risks[t : t + window])
                episode_workspace = _episode_workspace_tensors(_episode, dataset, model, actions=actions)
                for name, values in episode_workspace.items():
                    workspace_batch[name].append(values[t : t + window])
                has_targets_batch.append(1.0 if has_targets else 0.0)
            frames_b = torch.stack(frames_batch)  # [B, W+R, C, H, W]
            targets_b = torch.stack(targets_batch)
            actions_b = torch.stack(actions_batch)  # [B, W+R-1]
            rewards_b = torch.stack(rewards_batch)
            terminals_b = torch.stack(terminals_batch)
            risks_b = torch.stack(risks_batch)
            semantic_b = None
            if cfg.semantic_classes and all(
                len(grids) == window and all(grid is not None for grid in grids)
                for grids in semantic_batch
            ):
                semantic_b = torch.tensor(semantic_batch, dtype=torch.long)
                semantic_b = F.interpolate(
                    semantic_b.reshape(-1, 1, *semantic_b.shape[-2:]).float(),
                    size=(9, 9), mode="nearest",
                ).reshape(len(semantic_batch), window, 9, 9).long()
                if semantic_b.min() < 0 or semantic_b.max() >= cfg.semantic_classes:
                    raise ValueError(f"semantic grids must contain ids in [0, {cfg.semantic_classes})")
            workspace_b = {name: torch.stack(values) for name, values in workspace_batch.items()}
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
            flat_workspace = {
                name: values.reshape(batch_n * window, values.shape[-1])
                for name, values in workspace_b.items()
            }
            latents = model.encode_workspace(flat_frames, flat_workspace).reshape(batch_n, window, -1)
            target_latents = latents
            if target_encoder is not None:
                with torch.no_grad():
                    target_latents = target_encoder.encode_workspace(
                        flat_frames, flat_workspace
                    ).reshape(batch_n, window, -1)

            hidden = model.initial_state(batch_n)
            # Teacher-forced warmup: observed latents drive the state.
            predicted = None
            for i in range(cfg.warmup_frames - 1):
                predicted, hidden = model.step(latents[:, i], actions_b[:, i], hidden)

            pixel_loss = frames_b.new_zeros(())
            mask_sparsity_loss = frames_b.new_zeros(())
            semantic_loss = frames_b.new_zeros(())
            latent_loss = frames_b.new_zeros(())
            reward_loss = frames_b.new_zeros(())
            terminal_loss = frames_b.new_zeros(())
            risk_loss = frames_b.new_zeros(())
            uncertainty_loss = frames_b.new_zeros(())
            workspace_loss = frames_b.new_zeros(())
            latent_in = latents[:, cfg.warmup_frames - 1]
            reference_frame = targets_b[:, cfg.warmup_frames - 1]
            reference_spatial = model.encode_visual(F.interpolate(
                reference_frame, size=model.pixel_shape[:2], mode="nearest"
            )).spatial
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
                step_pixel_loss, step_mask_loss, decoded = _spatial_pixel_loss(
                    model, predicted, reference_frame, targets_b[:, idx + 1], cfg,
                    reference_spatial=reference_spatial,
                )
                pixel_loss = pixel_loss + step_pixel_loss
                mask_sparsity_loss = mask_sparsity_loss + step_mask_loss
                if semantic_b is not None:
                    visual = decoded
                    semantic_loss = semantic_loss + F.cross_entropy(
                        visual["semantic_logits"], semantic_b[:, idx + 1]
                    )
                workspace_loss = workspace_loss + _workspace_loss(
                    model, predicted,
                    {name: values[:, idx + 1] for name, values in workspace_b.items()},
                )

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
                use_prediction = float(torch.rand((), generator=generator)) < cfg.scheduled_sampling_p
                latent_in = predicted if use_prediction else latents[:, idx + 1]
                reference_frame = decoded["vision"] if use_prediction else targets_b[:, idx + 1]
                reference_spatial = model.encode_visual(F.interpolate(
                    reference_frame, size=model.pixel_shape[:2], mode="nearest"
                )).spatial
            pixel_loss = pixel_loss / cfg.rollout_frames
            mask_sparsity_loss = mask_sparsity_loss / cfg.rollout_frames
            semantic_loss = semantic_loss / cfg.rollout_frames
            latent_loss = latent_loss / cfg.rollout_frames
            reward_loss = reward_loss / cfg.rollout_frames
            terminal_loss = terminal_loss / cfg.rollout_frames
            risk_loss = risk_loss / cfg.rollout_frames
            uncertainty_loss = uncertainty_loss / cfg.rollout_frames
            workspace_loss = workspace_loss / cfg.rollout_frames
            total = (
                cfg.pixel_loss_weight * pixel_loss
                + cfg.latent_loss_weight * latent_loss
                + cfg.reward_loss_weight * reward_loss
                + cfg.terminal_loss_weight * terminal_loss
                + cfg.risk_loss_weight * risk_loss
                + cfg.uncertainty_loss_weight * uncertainty_loss
                + workspace_loss
                + cfg.change_mask_sparsity_weight * mask_sparsity_loss
                + cfg.semantic_loss_weight * semantic_loss
            )
            total.backward()
            optimizer.step()
            if target_encoder is not None:
                _update_ema_target_encoder(target_encoder, model, float(cfg.ema_target_decay))

            seen += batch_n
            epoch["total_loss"] += float(total.detach()) * batch_n
            epoch["pixel_loss"] += float(pixel_loss.detach()) * batch_n
            epoch["latent_loss"] += float(latent_loss.detach()) * batch_n
            epoch["reward_loss"] += float(reward_loss.detach()) * batch_n
            epoch["terminal_loss"] += float(terminal_loss.detach()) * batch_n
            epoch["risk_loss"] += float(risk_loss.detach()) * batch_n
            epoch["uncertainty_loss"] += float(uncertainty_loss.detach()) * batch_n
            epoch["workspace_loss"] += float(workspace_loss.detach()) * batch_n
            epoch["semantic_loss"] += float(semantic_loss.detach()) * batch_n
        for key in curves:
            curves[key].append(round(epoch[key] / max(seen, 1), 6))
        _report_epoch(
            epoch_idx, cfg.epochs, curves,
            epoch_seconds=time.perf_counter() - epoch_started,
            training_started=training_started,
            **{"train/windows_seen": seen,
               # only meaningful for the windowed backbones, which ramp it
               # over the context-length curriculum; `None` for the GRU
               **({"train/context_length": model.context_length}
                  if getattr(model, "context_length", None) else {})},
        )

    log.info("training complete  final_total=%.4f", curves["total_loss"][-1])
    trace_event("cortex.train.done", objective="windowed_rollout",
                final_total_loss=curves["total_loss"][-1],
                duration_s=round(time.perf_counter() - training_started, 3))
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
        "final_workspace_loss": curves["workspace_loss"][-1],
        "ema_target_enabled": target_encoder is not None,
        "ema_target_decay": cfg.ema_target_decay,
        "training_objective": "windowed_rollout",
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


def evaluate_action_world_model_direct(
    model: Any,
    dataset: ActionSequenceDataset,
    horizons_ticks: Sequence[int],
    *,
    warmup_frames: int = 3,
    max_starts_per_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate direct autoregressive horizon heads in pixel space.

    Heads are addressed by tick horizon while recordings are indexed by frame
    offset.  Multiple valid tick horizons may map to one frame offset on a
    downsampled recording, so report keys deliberately remain tick horizons.
    """
    torch, F = _torch()
    ticks = tuple(sorted({int(h) for h in horizons_ticks}))
    if not ticks or ticks[0] < 1:
        raise ValueError(f"horizons_ticks must be positive, got {horizons_ticks!r}")
    frames = tuple(horizons_ticks_to_frames((h,), dataset.ticks_per_frame)[0] for h in ticks)
    max_horizon = max(frames)
    action_index = {name: i for i, name in enumerate(model.action_keys)}
    samples = {tick: {"model": [], "copy_last": [], "mean_frame": [], "oracle": []} for tick in ticks}
    workspace_samples = {
        name: {tick: {"model": [], "copy_last": []} for tick in ticks}
        for name in getattr(model, "workspace_modalities", {})
    }
    per_episode_model_mse = {tick: [] for tick in ticks}
    prediction_dispersion: List[float] = []
    target_dispersion: List[float] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, targets, actions in _episode_tensors(dataset, model.reconstruction_shape):
            n = pixels.shape[0]
            if n <= max_horizon + warmup_frames:
                continue
            try:
                remap = torch.tensor([action_index[dataset.action_keys[a]] for a in episode.actions], dtype=torch.long)
            except KeyError as exc:
                raise ValueError(f"action {exc.args[0]!r} is outside the model vocabulary") from None
            workspace = _episode_workspace_tensors(episode, dataset, model, actions=remap)
            latents = model.encode_workspace(pixels, workspace).unsqueeze(0)
            # One encoder/backbone pass yields all causal prefixes; applying a
            # direct head below is intentionally not a transition rollout.
            hidden = model.forward_sequence(latents[:, :-1], remap.unsqueeze(0))
            mean_frame = targets.mean(dim=0)
            oracle_lag = _best_recurrence_lag(targets)
            starts = range(warmup_frames, n - max_horizon)
            if max_starts_per_episode is not None:
                starts = list(starts)[:max_starts_per_episode]
            episode_mse = {tick: [] for tick in ticks}
            for t in starts:
                decoded_by_tick = {}
                for tick, frame in zip(ticks, frames):
                    prediction = model.sequence_prediction(hidden[:, t : t + 1], tick)
                    decoded = model.decode_prediction(
                        prediction.latent[0], reference_frame=targets[t:t + 1],
                        reference_spatial=model.encode_visual(pixels[t:t + 1]).spatial,
                    )["vision"][0]
                    decoded_by_tick[tick] = decoded
                    target = targets[t + frame]
                    mse = float(F.mse_loss(decoded, target))
                    samples[tick]["model"].append(mse)
                    episode_mse[tick].append(mse)
                    samples[tick]["copy_last"].append(float(F.mse_loss(targets[t], target)))
                    samples[tick]["mean_frame"].append(float(F.mse_loss(mean_frame, target)))
                    if oracle_lag is not None and t + frame - oracle_lag >= 0:
                        samples[tick]["oracle"].append(float(F.mse_loss(targets[t + frame - oracle_lag], target)))
                    for name, by_horizon in workspace_samples.items():
                        predicted = prediction.modalities[name][0, 0]
                        target_modality = workspace[name][t + frame]
                        by_horizon[tick]["model"].append(float(F.mse_loss(predicted, target_modality)))
                        by_horizon[tick]["copy_last"].append(float(F.mse_loss(workspace[name][t], target_modality)))
                if len(frames) >= 2:
                    prediction_dispersion.append(_pairwise_dispersion([decoded_by_tick[h] for h in ticks]))
                    target_dispersion.append(_pairwise_dispersion([targets[t + f] for f in frames]))
            for tick in ticks:
                if episode_mse[tick]:
                    per_episode_model_mse[tick].append(_mean(episode_mse[tick]))
    if was_training:
        model.train()

    report: Dict[int, Dict[str, Any]] = {}
    for tick, frame in zip(ticks, frames):
        entry = samples[tick]
        if not entry["model"]:
            raise ValueError(f"no direct evaluation samples at horizon {tick} ticks ({frame} frames)")
        model_mse, copy_mse, mean_mse = _mean(entry["model"]), _mean(entry["copy_last"]), _mean(entry["mean_frame"])
        oracle_mse = _mean(entry["oracle"]) if entry["oracle"] else None
        report[tick] = {
            "horizon_tick": tick, "horizon_frame": frame, "n_samples": len(entry["model"]),
            "model_mse": model_mse, "copy_last_mse": copy_mse, "mean_frame_mse": mean_mse,
            "oracle_mse": oracle_mse, "psnr_model": _psnr(model_mse),
            "psnr_copy_last": _psnr(copy_mse), "psnr_mean_frame": _psnr(mean_mse),
            "model_over_copy_last_mse": _ratio(model_mse, copy_mse),
            "model_over_oracle_mse": _ratio(model_mse, oracle_mse),
            "beats_copy_last": bool(model_mse < copy_mse), "beats_mean_frame": bool(model_mse < mean_mse),
        }
    workspace_report = {
        name: {tick: {"n_samples": len(values["model"]), "model_mse": _mean(values["model"]),
                       "copy_last_mse": _mean(values["copy_last"]),
                       "model_over_copy_last_mse": _ratio(_mean(values["model"]), _mean(values["copy_last"])),
                       "beats_copy_last": bool(_mean(values["model"]) < _mean(values["copy_last"]))}
               for tick, values in by_frame.items()}
        for name, by_frame in workspace_samples.items()
    }
    return {
        "prediction_mode": "direct", "horizons": report, "horizons_ticks": list(ticks),
        "horizons_frames": list(frames), "ticks_per_frame": dataset.ticks_per_frame,
        "workspace_modalities": workspace_report,
        "prediction_health": {"prediction_dispersion": _mean(prediction_dispersion) if prediction_dispersion else 0.0,
                              "target_dispersion": _mean(target_dispersion) if target_dispersion else 0.0},
        "per_episode_model_mse": per_episode_model_mse,
    }


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
    log.info("evaluating cortex  horizons=%s  warmup=%d  episodes=%d",
             horizons_sorted, warmup_frames, len(dataset))

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
    workspace_samples: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        name: {
            h: {"model": [], "copy_last": []}
            for h in horizons_sorted
        }
        for name in getattr(model, "workspace_modalities", {})
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
            workspace = _episode_workspace_tensors(episode, dataset, model, actions=remap)
            latents = model.encode_workspace(pixels, workspace)
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
                reference_encoding = model.encode_visual(pixels[t : t + 1])
                rollout = model.forward_horizons(
                    latents[t : t + 1],
                    remap[t : t + max_horizon].unsqueeze(0),
                    hiddens[t],
                    horizon_frames=horizons_sorted,
                    reference_frame=targets[t : t + 1],
                    reference_spatial=reference_encoding.spatial,
                )
                decoded_by_horizon = {}
                for h in horizons_sorted:
                    decoded = rollout[h].decoded.squeeze(0)
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
                    for name, by_horizon in workspace_samples.items():
                        predicted_modality = rollout[h].modalities[name].squeeze(0)
                        target_modality = workspace[name][t + h]
                        by_horizon[h]["model"].append(
                            float(F.mse_loss(predicted_modality, target_modality))
                        )
                        by_horizon[h]["copy_last"].append(
                            float(F.mse_loss(workspace[name][t], target_modality))
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
    workspace_report = {
        name: {
            h: {
                "n_samples": len(values["model"]),
                "model_mse": _mean(values["model"]),
                "copy_last_mse": _mean(values["copy_last"]),
                "model_over_copy_last_mse": _ratio(
                    _mean(values["model"]), _mean(values["copy_last"])
                ),
                "beats_copy_last": bool(_mean(values["model"]) < _mean(values["copy_last"])),
            }
            for h, values in horizons.items()
        }
        for name, horizons in workspace_samples.items()
    }
    for h in horizons_sorted:
        r = report[h]
        log.info("  t+%d: model=%.4f  copy_last=%.4f  ratio=%.3f  beats=%s",
                 h, r["model_mse"], r["copy_last_mse"],
                 r["model_over_copy_last_mse"] or 0.0, r["beats_copy_last"])
    log.info("rollout_health  frozen=%s  pred_disp=%.4f  tgt_disp=%.4f",
             rollout_health["frozen_rollout"],
             rollout_health["prediction_dispersion"],
             rollout_health["target_dispersion"])
    if rollout_health["frozen_rollout"]:
        # A collapsed model is the failure this evaluation exists to catch;
        # a WARNING makes it visible in the console *and* the trace.
        log.warning("frozen rollout detected: predictions barely vary across horizons "
                    "(pred_disp=%.5f vs target_disp=%.5f) -- the cortex has collapsed",
                    rollout_health["prediction_dispersion"],
                    rollout_health["target_dispersion"])
    return {
        "prediction_mode": "rollout",
        "horizons": report,
        "horizons_frames": list(horizons_sorted),
        "ticks_per_frame": dataset.ticks_per_frame,
        "workspace_modalities": workspace_report,
        "rollout_health": rollout_health,
        "per_episode_model_mse": per_episode_model_mse,
    }


def evaluate_action_world_model_milestone(
    model: Any,
    dataset: ActionSequenceDataset,
    horizons_ticks: Sequence[int],
    *,
    warmup_frames: int = 3,
) -> Dict[str, Any]:
    """Evaluate both prediction paths and select the trained path as primary.

    Compatibility aliases (``horizons`` and ``per_episode_model_mse``) point
    at the primary report; consumers which need an unambiguous artifact use
    ``direct``/``rollout`` and ``primary_mode``.
    """
    direct = evaluate_action_world_model_direct(
        model, dataset, horizons_ticks, warmup_frames=warmup_frames
    )
    # Rollout has one prediction per *frame* step, whereas direct heads keep
    # every tick identity even when two heads target the same recorded frame.
    frames = sorted(set(direct["horizons_frames"]))
    rollout = evaluate_action_world_model(
        model, dataset, frames, warmup_frames=warmup_frames
    )
    primary_mode: PredictionMode = (
        "direct" if getattr(model, "training_objective", None) == "autoregressive" else "rollout"
    )
    # The objective lives on the training config, not the module; callers can
    # set it below on the model so checkpoint-loaded models retain the choice.
    primary = direct if primary_mode == "direct" else rollout
    return {
        "primary_mode": primary_mode, "direct": direct, "rollout": rollout,
        "rollout_health": rollout["rollout_health"],
        "horizons": primary["horizons"],
        "per_episode_model_mse": primary["per_episode_model_mse"],
        "horizons_ticks": direct["horizons_ticks"], "horizons_frames": frames,
        "ticks_per_frame": dataset.ticks_per_frame,
    }


def _backbone_name(model: Any) -> str:
    """Which temporal backbone this cortex is running.

    ``model.backbone_name`` never existed, so every training log printed
    ``backbone=?`` -- which made the one line that identifies a
    backbone-benchmark arm useless.  The name lives on the cortex config.
    """
    config = getattr(model, "config", None)
    return str(getattr(config, "backbone", None) or getattr(model, "backbone_name", None) or "?")


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _report_epoch(
    epoch_idx: int,
    epochs: int,
    curves: Dict[str, List[float]],
    *,
    epoch_seconds: float,
    training_started: float,
    **extra: Any,
) -> None:
    """One console line per epoch (with per-epoch cost and ETA) and one
    metrics line in the run trace.

    The trace copy is what makes a long cortex run debuggable: the loss
    curve is on disk epoch by epoch, so a run killed at epoch 25/30 still
    shows where the loss was going instead of losing everything with the
    process.
    """
    latest = {key: values[-1] for key, values in curves.items() if values}
    done = epoch_idx + 1
    elapsed = time.perf_counter() - training_started
    eta = elapsed / max(done, 1) * max(epochs - done, 0)
    log.info(
        "  epoch %d/%d  total=%.4f  pixel=%.4f  latent=%.4f  (%.1fs/epoch, eta %s)",
        done, epochs,
        latest.get("total_loss", float("nan")),
        latest.get("pixel_loss", float("nan")),
        latest.get("latent_loss", float("nan")),
        epoch_seconds, _format_eta(eta),
    )
    payload: Dict[str, Any] = {f"train/{key}": value for key, value in latest.items()}
    payload["train/epoch_seconds"] = round(epoch_seconds, 4)
    payload.update(extra)
    trace_metrics(step=done, **payload)


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
            latents = model.encode_workspace(
                pixels, _episode_workspace_tensors(episode, dataset, model, actions=remap)
            )
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


def evaluate_static_holdout(model: Any, dataset: ActionSequenceDataset) -> Dict[str, Any]:
    """Measure spatial residual predictions on semantic-static transitions.

    Unlike a pixel-hash filter, this uses the recorded semantic grid: animated
    HUD pixels are not mistaken for gameplay motion. The returned ratio is the
    Fix 4 promotion metric (model MSE / copy-last MSE, with an absolute MSE
    fallback when the target is exactly equal to the reference).
    """
    torch, F = _torch()
    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    model_mse: List[float] = []
    copy_mse: List[float] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, targets, actions in episodes:
            if len(episode.semantic_grids) != len(episode.frames):
                continue
            workspace = _episode_workspace_tensors(episode, dataset, model, actions=actions)
            latents = model.encode_workspace(pixels, workspace)
            hiddens = [model.initial_state(1)]
            hidden = hiddens[0]
            for index in range(len(actions)):
                _predicted, hidden = model.step(
                    latents[index:index + 1], actions[index:index + 1], hidden
                )
                hiddens.append(hidden)
            for t in range(len(actions)):
                before, after = episode.semantic_grids[t], episode.semantic_grids[t + 1]
                if before is None or after is None or before != after:
                    continue
                output = model.forward_horizons(
                    latents[t:t + 1], actions[t:t + 1].unsqueeze(0), hiddens[t],
                    horizon_frames=[1], reference_frame=targets[t:t + 1],
                    reference_spatial=model.encode_visual(pixels[t:t + 1]).spatial,
                )[1].decoded.squeeze(0)
                target = targets[t + 1]
                model_mse.append(float(F.mse_loss(output, target)))
                copy_mse.append(float(F.mse_loss(targets[t], target)))
    if was_training:
        model.train()
    if not model_mse:
        raise ValueError("static holdout has no semantic-static transitions")
    mean_model = sum(model_mse) / len(model_mse)
    mean_copy = sum(copy_mse) / len(copy_mse)
    return {
        "n_samples": len(model_mse),
        "model_mse": mean_model,
        "copy_last_mse": mean_copy,
        "model_over_copy_last_mse": mean_model / mean_copy if mean_copy > 1e-12 else None,
        "passes_static_gate": mean_model <= 1.05 * mean_copy if mean_copy > 1e-12 else mean_model <= 1e-12,
    }


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


def _linear_probe_orientation(
    model: Any,
    dataset: ActionSequenceDataset,
    *,
    include_facing: bool,
) -> Dict[str, Any]:
    """Probe continuous yaw and, optionally, Crafter's discrete facing.

    Fits ridge regressions latent -> (sin yaw, cos yaw) and hidden ->
    (sin yaw, cos yaw) on every frame with a recorded ``spatial.rotation``,
    reporting R^2 and mean angular error.  A hidden state that decodes heading
    where the raw latent cannot is direct evidence the recurrence is
    carrying orientation/motion state (phase 3's cheap interpretability
    check)."""
    torch, _F = _torch()

    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    action_index = {name: i for i, name in enumerate(model.action_keys)}
    latent_rows: List[Any] = []
    hidden_rows: List[Any] = []
    targets_rows: List[Any] = []
    source_rows: List[str] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for episode, pixels, _targets, _actions in episodes:
            remap = torch.tensor(
                [action_index[dataset.action_keys[a]] for a in episode.actions],
                dtype=torch.long,
            )
            latents = model.encode_workspace(
                pixels, _episode_workspace_tensors(episode, dataset, model, actions=remap)
            )
            hidden = model.initial_state(1)
            for i, yaw in enumerate(episode.yaw):
                source: Optional[str] = None
                angle: Optional[float] = None
                if yaw is not None:
                    source, angle = "yaw", math.radians(yaw)
                elif include_facing and i < len(episode.facing):
                    facing = episode.facing[i]
                    if facing is not None and (facing[0] or facing[1]):
                        source, angle = "facing", math.atan2(facing[1], facing[0])
                if angle is not None:
                    latent_rows.append(latents[i])
                    hidden_rows.append(model.transition_backbone.readout(hidden).squeeze(0))
                    targets_rows.append(
                        torch.tensor([math.sin(angle), math.cos(angle)])
                    )
                    source_rows.append(source or "unknown")
                if i < len(episode.actions):
                    _pred, hidden = model.step(
                        latents[i : i + 1], remap[i : i + 1], hidden
                    )
    if was_training:
        model.train()

    if len(targets_rows) < 8:
        return {
            "n_samples": len(targets_rows),
            "sources": sorted(set(source_rows)),
            "note": "too few orientation-labelled frames to probe",
        }

    y = torch.stack(targets_rows)
    report: Dict[str, Any] = {
        "n_samples": int(y.shape[0]),
        "sources": sorted(set(source_rows)),
    }
    for name, rows in (("latent", latent_rows), ("hidden", hidden_rows)):
        x = torch.stack(rows)
        report[name] = _ridge_probe(x, y)
    return report


def linear_probe_yaw(model: Any, dataset: ActionSequenceDataset) -> Dict[str, Any]:
    """Backward-compatible probe using only continuous yaw labels."""
    return _linear_probe_orientation(model, dataset, include_facing=False)


def linear_probe_orientation(model: Any, dataset: ActionSequenceDataset) -> Dict[str, Any]:
    """Probe yaw or Crafter's discrete ``spatial.facing`` labels."""
    return _linear_probe_orientation(model, dataset, include_facing=True)


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
        for episode, pixels, _targets, actions in episodes:
            rows.append(model.encode_workspace(
                pixels, _episode_workspace_tensors(episode, dataset, model, actions=actions)
            ))
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

    A failure requires both a poor hidden-state orientation probe and a
    collapsed latent distribution.  Datasets without enough yaw/facing
    labels are reported as non-evaluable; promotion callers may reject that
    status when an orientation gate is mandatory.
    """
    cfg = config or ActionWorldModelConfig()
    latent = latent_variance_rank(model, dataset)
    orientation_probe = linear_probe_orientation(model, dataset)
    hidden_orientation_r2 = (
        float(orientation_probe["hidden"]["r2"])
        if "hidden" in orientation_probe else None
    )
    variance_collapsed = latent["mean_variance"] < cfg.collapse_gate_min_latent_variance
    rank_collapsed = latent["effective_rank"] < cfg.collapse_gate_min_effective_rank
    latent_collapsed = variance_collapsed or rank_collapsed
    gate_evaluable = hidden_orientation_r2 is not None
    yaw_collapsed = (
        gate_evaluable
        and hidden_orientation_r2 < cfg.collapse_gate_min_hidden_yaw_r2
    )
    passed = not (yaw_collapsed and latent_collapsed) if gate_evaluable else False
    return {
        "passed": bool(passed),
        "gate_evaluable": gate_evaluable,
        # Keep the old key for report consumers; it now means the selected
        # orientation source (yaw or Crafter facing).
        "hidden_yaw_r2": hidden_orientation_r2,
        "hidden_orientation_r2": hidden_orientation_r2,
        "yaw_probe": orientation_probe,
        "orientation_probe": orientation_probe,
        "orientation_sources": orientation_probe.get("sources", []),
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
            "format": "action-world-model-v2",
            "visual_architecture": model.visual_architecture,
            "visual_config": {
                "semantic_classes": model.config.semantic_classes,
                "change_mask_sparsity_weight": model.config.change_mask_sparsity_weight,
                **getattr(model, "training_visual_config", {}),
            },
            "pixel_shape": list(model.pixel_shape),
            "action_keys": list(model.action_keys),
            "latent_width": model.latent_width,
            "hidden_dim": model.hidden_dim,
            "reconstruction_shape": list(model.reconstruction_shape),
            "horizons_ticks": list(model.horizons_ticks),
            # Frame offsets are recording-rate dependent.  Persist the
            # training-time mapping alongside the tick identity so an export
            # can reproduce (or explicitly reject) the evaluated horizons.
            "horizons_frames": stats.get("autoregressive_horizons"),
            "ticks_per_frame": stats.get("ticks_per_frame"),
            "backbone": model.config.backbone,
            "context_length": model.config.context_length,
            "workspace_modalities": dict(model.workspace_modalities),
            "workspace_layout_hash": model.workspace_layout_hash,
            "training_objective": stats.get("training_objective", getattr(model, "training_objective", "windowed_rollout")),
            "state_dict": model.state_dict(),
            "training_stats": stats,
        },
        path,
    )


def load_action_world_model(path: str, *, inspection_only: bool = False):
    """Load a spatial checkpoint, or inspect a legacy checkpoint without migration.

    ``inspection_only`` intentionally returns ``(None, training_stats)`` for
    global-pool v1 files: their metadata remains available, but their dense
    decoder is never silently inserted into the spatial architecture.
    """
    torch, _F = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "action-world-model-v1":
        if inspection_only:
            stats = dict(payload.get("training_stats", {}))
            stats["legacy_visual_architecture"] = "global_pool_v1"
            stats["inspection_only"] = True
            return None, stats
        raise ValueError(
            "checkpoint uses global_pool_v1; start a new spatial_residual_v1 experiment"
        )
    if payload.get("format") != "action-world-model-v2":
        raise ValueError(f"unsupported action world model format {payload.get('format')!r}")
    if payload.get("visual_architecture") != "spatial_residual_v1":
        raise ValueError(
            f"checkpoint uses {payload.get('visual_architecture', 'global_pool_v1')}; "
            "start a new spatial_residual_v1 experiment"
        )
    visual_config = payload.get("visual_config", {})
    cfg = ActionWorldModelConfig(
        latent_width=int(payload["latent_width"]),
        hidden_dim=int(payload["hidden_dim"]),
        reconstruction_size=int(payload["reconstruction_shape"][0]),
        horizons_ticks=tuple(payload.get("horizons_ticks", ActionWorldModelConfig.horizons_ticks)),
        backbone=payload.get("backbone", ActionWorldModelConfig.backbone),
        context_length=int(payload.get("context_length", ActionWorldModelConfig.context_length)),
        workspace_enabled=bool(payload.get("workspace_modalities", {})),
        visual_architecture=payload["visual_architecture"],
        semantic_classes=int(visual_config.get("semantic_classes", 0)),
        change_mask_sparsity_weight=float(visual_config.get("change_mask_sparsity_weight", 0.01)),
        scene_height=visual_config.get("scene_height"),
        hud_loss_weight=float(visual_config.get("hud_loss_weight", 0.25)),
        semantic_loss_weight=float(visual_config.get("semantic_loss_weight", 1.0)),
    )
    model = build_action_world_model(
        tuple(payload["pixel_shape"]), payload["action_keys"], cfg,
        workspace_modalities=payload.get("workspace_modalities", {}),
        workspace_layout_hash=payload.get("workspace_layout_hash"),
    )
    model.training_objective = payload.get(
        "training_objective", payload.get("training_stats", {}).get("training_objective", "windowed_rollout")
    )
    model.training_visual_config = dict(visual_config)
    # C1 added direct heads for tick horizons > 1.  Older v1 checkpoints
    # naturally lack only these freshly initialized modules; allow precisely
    # that omission while retaining strict corruption detection everywhere
    # else.
    incompatibility = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key.startswith("multi_token_heads.")
    }
    # C3 replaced the transformer's learned absolute position table with
    # parameter-free ALiBi. It is safe to discard precisely that obsolete
    # tensor from pre-C3 checkpoints; all learned content weights still load.
    allowed_unexpected = (
        {"transition_backbone.position_embedding.weight"}
        if cfg.backbone == "transformer"
        else set()
    )
    unexpected = set(incompatibility.unexpected_keys) - allowed_unexpected
    missing = set(incompatibility.missing_keys) - allowed_missing
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise ValueError("incompatible action world model checkpoint (" + "; ".join(details) + ")")
    return model, payload.get("training_stats", {})
