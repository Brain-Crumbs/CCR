"""Live cortex consolidation: micro-sleep + generative replay (issue #167).

This is the learner target of the wake/sleep cycle, re-pointed at the thing
the phase doc says sleep is *for*: the ``brain.cortex.PredictiveCortex`` world
model, not the legacy actor/critic RL stack in ``sleep.async_trainer``
(docs/v2/phases/phase-5-sleep-consolidation.md: "the heavy thing learned in
sleep is the *world model* ... not a bootstrapped policy chasing a moving value
target").

The wake half is already wired in the runtime loop (``runtime/loop.py``'s
per-tick ``Hippocampus.encode``); this module is the sleep half. During a
micro-sleep it:

- draws quality-gated, guardrailed minibatches from a
  :class:`~sleep.replay_mix.GenerativeReplayMixer` -- a reservoir of *real*
  live transitions mixed with dreams rolled from a **frozen cortex snapshot**
  (never the live model being trained, so a half-trained model can't compound
  its own errors into what it rehearses);
- takes cortex gradient steps on the latent-prediction loss (the same
  short-rollout latent MSE ``training.action_world_model`` trains on, and the
  loss ``sleep.replay_mix`` produces targets for -- pixel/head losses from B1
  layer on top of this once available);
- publishes the updated cortex weights back to the live world model (the A1
  ``CortexWorldModel`` adapter) between ticks via :meth:`publish_to`, reusing
  the raw-vs-EMA hand-off ``sleep.weight_publisher`` already draws: a raw
  snapshot for the phasic schedule (the actor is paused, no staleness), an
  EMA/Polyak-averaged one for the concurrent schedule (a slow-moving target
  that kills tick-to-tick oscillation).

The quality gate is measured from the **frozen dream source**'s own held-out
performance versus copy-last, not the live model's -- exactly the discipline
``tests/test_forgetting_metric.py`` and the ``sleep.replay_mix`` module
docstring describe. :meth:`refresh_dream_source` promotes the current live
cortex to the frozen snapshot and re-measures that margin, so dreaming stays at
0% until the cortex has actually cleared the bar and only ramps in as it earns
trust.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from brain.hippocampus import Hippocampus, Seed
from sleep.replay_mix import (
    GenerativeReplayMixer,
    ReplaySample,
    Reservoir,
    copy_last_quality_margin,
)

__all__ = ["CortexConsolidator"]


@dataclass(frozen=True)
class ConsolidationMetrics:
    """One micro-sleep pass's loss curve summary plus the guardrail bookkeeping
    a caller/test needs to check the quality gate held."""

    version: int
    steps: int
    mean_loss: float
    #: Mean *actual* dream fraction across the pass's batches -- 0.0 while the
    #: frozen snapshot has not cleared the quality bar, ramping toward ``cap``
    #: once it has (:func:`sleep.replay_mix.dream_fraction`).
    mean_dream_fraction: float
    quality_margin: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "version": float(self.version),
            "steps": float(self.steps),
            "mean_loss": self.mean_loss,
            "mean_dream_fraction": self.mean_dream_fraction,
            "quality_margin": self.quality_margin,
        }


class CortexConsolidator:
    """The sleep-phase learner for a live :class:`~brain.cortex.PredictiveCortex`.

    Owns the live cortex, its optimizer, the reservoir of real transitions the
    wake phase feeds, and a frozen dream snapshot. One consolidation pass
    (:meth:`consolidate`) draws quality-gated batches from a
    :class:`~sleep.replay_mix.GenerativeReplayMixer` and steps the cortex on
    the latent-prediction loss; :meth:`publish_to` hands the result back to the
    live world model between ticks.

    Designed to slot straight into
    :class:`~sleep.schedule.PhasicSleepSchedule`::

        schedule.consolidate(
            lambda: consolidator.consolidate(steps),
            reload_weights=lambda: consolidator.publish_to(world_model),
        )
    """

    def __init__(
        self,
        cortex: Any,  # PredictiveCortex (lazy torch type)
        hippocampus: Hippocampus,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        reservoir_capacity: int = 512,
        batch_size: int = 16,
        dream_length: int = 1,
        ramp_start: float = 0.0,
        ramp_end: float = 1.0,
        cap: float = 0.5,
        held_out_capacity: int = 64,
        held_out_every: int = 5,
        seed: int = 0,
        ema_decay: Optional[float] = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if dream_length <= 0:
            raise ValueError(f"dream_length must be positive, got {dream_length!r}")
        if held_out_every <= 0:
            raise ValueError(f"held_out_every must be positive, got {held_out_every!r}")
        if ema_decay is not None and not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay!r}")

        self.cortex = cortex
        self.hippocampus = hippocampus
        self.optimizer = optimizer or torch.optim.Adam(cortex.parameters(), lr=lr)
        self.batch_size = batch_size
        self.dream_length = dream_length
        self.held_out_capacity = held_out_capacity
        self.held_out_every = held_out_every

        self._action_index = {key: i for i, key in enumerate(cortex.action_keys)}
        self.reservoir = Reservoir(capacity=reservoir_capacity)
        #: Held-out real transitions -- ingested but *never* trained on, so
        #: both the live-cortex loss metric and the dream-source quality gate
        #: are measured on data the cortex has not fitted.
        self._held_out: List[ReplaySample] = []
        self._ingested = 0

        # Frozen dream source: seeded from the (untrained) live cortex, so its
        # quality margin starts at/below zero and the gate keeps dreams at 0%
        # until `refresh_dream_source` promotes a snapshot that has cleared the
        # bar. Rebuilt in place on every refresh (the mixer keeps its own rng).
        self.dream_source = self._frozen_snapshot()
        self.quality_margin = 0.0
        self.mixer = GenerativeReplayMixer(
            reservoir=self.reservoir,
            hippocampus=self.hippocampus,
            dream_cortex=self.dream_source,
            ramp_start=ramp_start,
            ramp_end=ramp_end,
            cap=cap,
            seed=seed,
        )

        self.ema_decay = ema_decay
        self._ema_shadow: Optional[Dict[str, torch.Tensor]] = None
        if ema_decay is not None:
            self._ema_shadow = {
                k: v.detach().clone() for k, v in cortex.state_dict().items()
            }

        self.version = 0
        self.consolidations = 0
        self.last_metrics: Optional[ConsolidationMetrics] = None

    # ------------------------------------------------------------- ingestion

    def record_transition(
        self,
        z0: Sequence[float],
        actions: Sequence[str],
        next_latents: torch.Tensor,
    ) -> None:
        """Feed one live wake transition into the reservoir (the "reservoir of
        real transitions" guardrail).

        ``z0`` is the latent at the start of the transition, ``actions`` the
        emitted action key(s) rolled forward from it (length must equal
        :attr:`dream_length`, so real and dreamed samples stack into one
        batch), and ``next_latents`` the observed latents that followed
        (``Tensor[dream_length, latent_width]``). A deterministic slice is
        diverted to the never-trained held-out set instead.
        """
        if len(actions) != self.dream_length:
            raise ValueError(
                f"transition has {len(actions)} actions, but dream_length is "
                f"{self.dream_length}; store transitions of length dream_length"
            )
        targets = next_latents.detach().to(torch.float32)
        if targets.shape[0] != self.dream_length:
            raise ValueError(
                f"next_latents has {targets.shape[0]} rows, expected dream_length "
                f"{self.dream_length}"
            )
        sample = ReplaySample(
            z0=[float(x) for x in z0],
            actions=list(actions),
            targets=targets,
            source="real",
        )
        # Route every `held_out_every`-th transition into held-out (bounded,
        # FIFO) and everything else into the training reservoir.
        if self._ingested % self.held_out_every == 0:
            self._held_out.append(sample)
            if len(self._held_out) > self.held_out_capacity:
                self._held_out.pop(0)
        else:
            self.reservoir.add(sample)
        self._ingested += 1

    def ingest_sample(self, sample: ReplaySample) -> None:
        """Add a pre-built real :class:`~sleep.replay_mix.ReplaySample` to the
        training reservoir (bypasses the held-out split -- for callers that
        manage their own held-out set)."""
        self.reservoir.add(sample)

    # --------------------------------------------------------- quality gate

    def _frozen_snapshot(self) -> Any:
        snapshot = copy.deepcopy(self.cortex)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        return snapshot

    def refresh_dream_source(
        self, *, held_out: Optional[Sequence[ReplaySample]] = None
    ) -> float:
        """Promote the current live cortex to the frozen dream source and
        re-measure the quality margin from *its* held-out performance.

        The bootstrap guardrail: dreams are only ever rolled from this frozen
        snapshot, and it is this snapshot's margin (not the live model's) that
        gates the dream fraction. Returns the new margin.

        ``held_out`` defaults to this consolidator's own rolling held-out set
        (recent real transitions). A caller whose dreams cover a *different*
        distribution than the current reservoir -- e.g. the frozen snapshot
        rehearses an older, already-mastered scenario whose seeds live in the
        hippocampus -- passes that scenario's held-out explicitly, so the gate
        is measured on what the dreams actually reconstruct.
        """
        self.dream_source = self._frozen_snapshot()
        self.mixer.dream_cortex = self.dream_source
        samples = list(held_out) if held_out is not None else self._held_out
        self.quality_margin = self._measure_quality_margin(self.dream_source, samples)
        return self.quality_margin

    def set_quality_margin(self, margin: float) -> None:
        """Override the gating margin directly (for a caller that measures the
        frozen snapshot's held-out quality with its own evaluation harness --
        e.g. the Milestone 5 forgetting-metric loop)."""
        self.quality_margin = float(margin)

    def _measure_quality_margin(
        self, model: Any, held_out: Sequence[ReplaySample]
    ) -> float:
        """``model``'s held-out latent-prediction margin over copy-last, the
        signal :func:`sleep.replay_mix.dream_fraction` gates on. 0.0 with no
        held-out data yet (no headroom to trust)."""
        if not held_out:
            return 0.0
        z0, actions, targets = self._stack(held_out)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred, _ = model.rollout(z0, actions, model.initial_state(z0.shape[0]))
            model_mse = float(F.mse_loss(pred, targets))
            copy_last = z0.unsqueeze(1).expand_as(targets)
            copy_last_mse = float(F.mse_loss(copy_last, targets))
        model.train(was_training)
        return copy_last_quality_margin(model_mse, copy_last_mse)

    def held_out_loss(self) -> Optional[float]:
        """The *live* cortex's mean latent MSE on the held-out set -- the
        Milestone 5 acceptance signal that "held-out cortex prediction
        improves during the run" (this drops across micro-sleeps). ``None``
        before any held-out transition has been recorded."""
        if not self._held_out:
            return None
        z0, actions, targets = self._stack(self._held_out)
        was_training = self.cortex.training
        self.cortex.eval()
        with torch.no_grad():
            pred, _ = self.cortex.rollout(z0, actions, self.cortex.initial_state(z0.shape[0]))
            loss = float(F.mse_loss(pred, targets))
        self.cortex.train(was_training)
        return loss

    def _stack(self, samples: Sequence[ReplaySample]):
        z0 = torch.tensor([s.z0 for s in samples], dtype=torch.float32)
        actions = torch.tensor(
            [[self._action_index[a] for a in s.actions] for s in samples],
            dtype=torch.long,
        )
        targets = torch.stack([s.targets for s in samples], dim=0)
        return z0, actions, targets

    # ------------------------------------------------------------- training

    def _train_step(self, z0: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> float:
        self.cortex.train()
        hidden = self.cortex.initial_state(z0.shape[0])
        pred, _ = self.cortex.rollout(z0, actions, hidden)
        loss = F.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def consolidate(self, steps: int) -> int:
        """Run one micro-sleep: ``steps`` cortex gradient steps on quality-gated
        replay batches. Returns the new weight version (monotonic).

        A no-op that leaves the version unchanged while the reservoir is still
        empty (nothing real to train on -- the guardrail never lets a pass run
        on dreams alone), so an early micro-sleep before any wake experience
        has arrived simply does nothing rather than raising.
        """
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps!r}")
        self.consolidations += 1
        if steps == 0 or len(self.reservoir) == 0:
            self.last_metrics = ConsolidationMetrics(
                version=self.version, steps=0, mean_loss=float("nan"),
                mean_dream_fraction=0.0, quality_margin=self.quality_margin,
            )
            return self.version

        losses: List[float] = []
        fractions: List[float] = []
        for _ in range(steps):
            batch = self.mixer.mix_batch(
                self.batch_size, self.quality_margin, dream_length=self.dream_length,
            )
            losses.append(self._train_step(batch.z0, batch.actions, batch.targets))
            fractions.append(batch.fraction_actual)

        if self._ema_shadow is not None:
            self._advance_ema()

        self.version += 1
        self.last_metrics = ConsolidationMetrics(
            version=self.version,
            steps=len(losses),
            mean_loss=sum(losses) / len(losses),
            mean_dream_fraction=sum(fractions) / len(fractions),
            quality_margin=self.quality_margin,
        )
        return self.version

    # --------------------------------------------------------- publish-back

    def _advance_ema(self) -> None:
        assert self._ema_shadow is not None
        decay = self.ema_decay
        for key, value in self.cortex.state_dict().items():
            shadow = self._ema_shadow[key]
            if torch.is_floating_point(shadow):
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self._ema_shadow[key] = value.detach().clone()

    def _publish_state(self, use_ema: bool) -> Dict[str, torch.Tensor]:
        if use_ema:
            if self._ema_shadow is None:
                raise ValueError("no EMA shadow: construct with ema_decay to publish EMA weights")
            source = self._ema_shadow
        else:
            source = self.cortex.state_dict()
        return {k: v.detach().clone() for k, v in source.items()}

    def publish_to(self, world_model: Any, *, use_ema: Optional[bool] = None) -> int:
        """Hand the consolidated weights back to the live world model (the A1
        ``CortexWorldModel`` adapter, or any object exposing a ``model``
        cortex) and reset its rolling world state so the fresh weights take
        effect cleanly on the next tick. Returns the published version.

        ``use_ema`` defaults to True when this consolidator was built with an
        ``ema_decay`` (the concurrent schedule's slow-moving target) and False
        otherwise (the phasic schedule's raw hand-off, safe because the actor
        is paused).
        """
        publish_ema = self._ema_shadow is not None if use_ema is None else use_ema
        state = self._publish_state(publish_ema)
        target = getattr(world_model, "model", world_model)
        target.load_state_dict(state)
        reset = getattr(world_model, "reset", None)
        if callable(reset):
            reset()
        return self.version

    # ----------------------------------------------------------------- misc

    def stats(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "consolidations": self.consolidations,
            "reservoir_size": len(self.reservoir),
            "held_out_size": len(self._held_out),
            "quality_margin": self.quality_margin,
            "last_metrics": self.last_metrics.as_dict() if self.last_metrics else None,
        }
