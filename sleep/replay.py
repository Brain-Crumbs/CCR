"""Generative replay with the bootstrap guardrail (issue #99).

Dreaming from a half-trained model reinforces its own errors -- the "dream
bootstrap paradox" (``docs/v2/01-architecture.md``). Generative replay only
helps once two rules hold:

(a) keep a bounded reservoir of real transitions and never train on dreams
    alone;
(b) gate the dream fraction on measured model quality -- 0% until the model
    beats copy-last on held-out data by a margin, ramping with the quality
    ratio, capped at roughly 0.5.

The dream fraction is a function of a metric (``model_mse / copy_last_mse``
-- the same ratio ``action_world_model.evaluate_action_world_model`` and
``training.world_model.evaluate_multi_horizon_model`` report as
``model_over_copy_last_mse``/the ratio behind ``beats_copy_last``), never a
constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from cognitive_runtime.neural.replay_buffer import ReplayBuffer, Transition
from cognitive_runtime.neural.world_model import WorldModel

__all__ = [
    "DreamFractionGate",
    "dreamed_batch_size",
    "sample_generative_replay_batch",
    "evaluate_next_latent_quality",
    "train_with_generative_replay",
]


@dataclass(frozen=True)
class DreamFractionGate:
    """Maps a held-out quality ratio (``model_mse / copy_last_mse``, lower
    is better) to the fraction of a training batch drawn from dreams.

    ``0.0`` while the model has not beaten copy-last by ``margin`` (a ratio
    at or above ``margin`` means "not yet decent enough to dream from");
    ramps linearly to ``cap`` as the ratio falls to ``floor_ratio`` or
    below. The dream fraction is a function of this ratio, not a constant.
    """

    margin: float = 0.9
    floor_ratio: float = 0.4
    cap: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.floor_ratio < self.margin:
            raise ValueError(
                f"floor_ratio ({self.floor_ratio!r}) must be positive and below "
                f"margin ({self.margin!r})"
            )
        if not 0.0 < self.cap <= 1.0:
            raise ValueError(f"cap must be in (0, 1], got {self.cap!r}")

    def fraction(self, quality_ratio: float) -> float:
        if quality_ratio >= self.margin:
            return 0.0
        if quality_ratio <= self.floor_ratio:
            return self.cap
        progress = (self.margin - quality_ratio) / (self.margin - self.floor_ratio)
        return self.cap * progress


def dreamed_batch_size(batch_size: int, dream_fraction: float) -> int:
    """Dreamed-transition count for one batch of ``batch_size``, always
    leaving at least one real transition -- "never train on dreams alone"
    enforced at the batch level, not only via ``DreamFractionGate.cap``."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if not 0.0 <= dream_fraction <= 1.0:
        raise ValueError(f"dream_fraction must be in [0, 1], got {dream_fraction!r}")
    return min(round(batch_size * dream_fraction), batch_size - 1)


def sample_generative_replay_batch(
    real_buffer: ReplayBuffer,
    dream_source: Callable[[int], Sequence[Transition]],
    *,
    batch_size: int,
    n_actions: int,
    quality_ratio: float,
    gate: DreamFractionGate = DreamFractionGate(),
) -> Tuple[Dict[str, torch.Tensor], int]:
    """One generative-replay minibatch: real transitions sampled from the
    reservoir plus ``gate.fraction(quality_ratio)`` dreamed ones.

    ``real_buffer`` is only ever read from (``ReplayBuffer.sample``) --
    dreamed transitions are never added to it, so the reservoir stays
    exactly the real experience it was built from. Returns the batch dict
    ``ReplayBuffer.as_batch`` expects, plus how many transitions in it were
    dreamed.
    """
    n_dream = dreamed_batch_size(batch_size, gate.fraction(quality_ratio))
    n_real = batch_size - n_dream

    real_transitions = list(real_buffer.sample(n_real))
    dreamed_transitions = list(dream_source(n_dream)) if n_dream else []
    if len(dreamed_transitions) != n_dream:
        raise ValueError(
            f"dream_source returned {len(dreamed_transitions)} transitions, "
            f"requested {n_dream}"
        )
    combined = real_transitions + dreamed_transitions
    return real_buffer.as_batch(combined, n_actions), n_dream


def evaluate_next_latent_quality(
    model: WorldModel, transitions: Sequence[Transition], n_actions: int,
) -> Dict[str, float]:
    """``model_mse``/``copy_last_mse``/ratio/``beats_copy_last`` for
    ``model``'s ``next_latent`` head over ``transitions`` -- the same
    baseline-relative convention ``action_world_model.evaluate_action_world_model``
    and ``training.world_model.evaluate_multi_horizon_model`` report, at the
    reservoir's ``Transition`` granularity so it can drive
    ``DreamFractionGate.fraction`` directly from held-out replay data."""
    if not transitions:
        raise ValueError("cannot evaluate quality over an empty transition set")
    # Match `sleep.dream`'s convention: derive device/dtype from the model's
    # own parameters rather than defaulting to CPU float32, so this runs
    # against an accelerator-resident model without a device-mismatch error.
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    latents = torch.tensor([t.latent for t in transitions], dtype=dtype, device=device)
    next_latents = torch.tensor([t.next_latent for t in transitions], dtype=dtype, device=device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    action_onehot = F.one_hot(actions, num_classes=n_actions).to(dtype=dtype)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        predicted = model(latents, action_onehot).next_latent
    if was_training:
        model.train()

    model_mse = float(F.mse_loss(predicted, next_latents))
    copy_last_mse = float(F.mse_loss(latents, next_latents))
    ratio = model_mse / copy_last_mse if copy_last_mse > 0 else float("inf")
    return {
        "model_mse": model_mse,
        "copy_last_mse": copy_last_mse,
        "model_over_copy_last_mse": ratio,
        "beats_copy_last": bool(model_mse < copy_last_mse),
    }


def train_with_generative_replay(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    real_buffer: ReplayBuffer,
    dream_source: Callable[[int], Sequence[Transition]],
    dream_quality_ratio: Union[float, Callable[[], float]],
    *,
    steps: int,
    batch_size: int,
    n_actions: int,
    gate: DreamFractionGate = DreamFractionGate(),
) -> List[Dict[str, float]]:
    """A consolidation training loop mixing dreamed replay into
    ``real_buffer`` per :func:`sample_generative_replay_batch`.

    ``dream_quality_ratio`` gates the dream fraction on the *dream
    generator's* measured quality -- the bootstrap paradox is that dreams
    from a half-trained generator reinforce its own errors, so the guardrail
    cares whether ``dream_source``'s underlying model is any good, not
    whether ``model`` (the one being updated here, which may be a different
    object -- e.g. a frozen snapshot doing the dreaming while a continuing
    copy is trained) currently performs well on any particular held-out set.
    Pass a fixed float for a frozen dream generator (its quality does not
    change during this consolidation pass), or a zero-arg callable to
    re-measure a live generator (e.g. the concurrent schedule, where the
    same in-training cortex generates its own dreams) every step. Returns
    the per-step quality-ratio/dream-fraction log.
    """
    log: List[Dict[str, float]] = []
    for _ in range(steps):
        ratio = dream_quality_ratio() if callable(dream_quality_ratio) else dream_quality_ratio
        batch, n_dream = sample_generative_replay_batch(
            real_buffer, dream_source,
            batch_size=batch_size, n_actions=n_actions,
            quality_ratio=ratio, gate=gate,
        )
        model.train()
        optimizer.zero_grad()
        predicted = model(batch["fused_latent"], batch["action_onehot"]).next_latent
        loss = F.mse_loss(predicted, batch["next_fused_latent"])
        loss.backward()
        optimizer.step()
        log.append({
            "model_over_copy_last_mse": ratio,
            "dream_fraction": n_dream / batch_size,
            "loss": float(loss.detach()),
        })
    return log
