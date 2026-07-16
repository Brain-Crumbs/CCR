"""Versioned weight publication between the trainer process and the actor
(issue #37): "trainer publishes versioned policy snapshots; the actor swaps
them in atomically *between* ticks."

Rather than invent a second serialization format, publication reuses the
same :class:`~cognitive_runtime.neural.checkpoint.NeuralAgentCheckpoint`
bundle the trainer already writes for checkpoint ownership (issue #20) --
"the trainer owns checkpoint writes; the actor only ever loads" is exactly
:class:`WeightPublisher.publish` / :class:`WeightSubscriber.maybe_reload`.
The checkpoint's ``training_ticks`` (the optimizer's monotonic step count)
doubles as the snapshot version: it only advances when the trainer has
actually taken a gradient step, and ``NeuralAgentCheckpoint`` already writes
the tensor payload before the JSON sidecar and both via atomic
``os.replace`` (see ``neural/checkpoint.py``'s ``_atomic_torch_save``/
``_atomic_json_dump``), so a subscriber never observes a half-written
snapshot.

The actor-side bundle only needs to carry the modules it wants to hot-swap
(typically just ``policy``/``critic``) -- ``NeuralAgentCheckpoint.load``
only touches the modules passed into *its own* constructor, ignoring
whatever else (encoders, optimizer state, world model, ...) the trainer's
fuller bundle wrote to the same file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from cognitive_runtime.neural.checkpoint import (
    CheckpointCompatibilityError,
    NeuralAgentCheckpoint,
    read_checkpoint_metadata,
)


class WeightPublisher:
    """Trainer-side: publish the current weights as a new versioned
    snapshot.  Thin wrapper over ``NeuralAgentCheckpoint.save`` so the
    published file *is* the checkpoint -- one atomic write serves both
    "checkpoint ownership" and "weight publication"."""

    def __init__(self, checkpoint: NeuralAgentCheckpoint):
        self.checkpoint = checkpoint

    def publish(self, *, reason: str = "publish") -> int:
        """Write a new snapshot; returns its version (``training_ticks``).

        Callers must keep ``checkpoint.training_ticks`` in sync with the
        optimizer's step count before calling this (the same convention
        ``ActorCriticLearner.save`` uses) -- it is what makes the version
        monotonic and lets :meth:`WeightSubscriber.maybe_reload` tell a real
        update from a no-op republish.
        """
        metadata = self.checkpoint.save(reason=reason)
        return int(metadata.get("training_ticks", 0))


class EMAWeightPublisher(WeightPublisher):
    """Concurrent-schedule weight publication (issue #100).

    Phasic consolidation never publishes while the actor is acting, so a raw
    snapshot is safe -- the actor only ever sees a completed pass. The
    concurrent schedule has no such pause: the actor keeps polling and
    hot-swapping weights *while it acts*, so a raw in-training snapshot would
    hand it tick-to-tick oscillation straight from the optimizer's gradient
    noise. Publishing a Polyak/EMA-averaged copy instead -- a slow-moving
    target -- kills that oscillation without touching how training itself
    proceeds: the live modules keep training on the raw (fast) weights;
    only the file on disk carries the averaged ones, swapped in for the
    duration of the write and restored immediately after.

    The version stamped alongside each snapshot is still the checkpoint's
    own ``training_ticks`` (the optimizer's monotonic step count) -- EMA
    changes *which* weights are published, not the monotonic-version
    contract :class:`WeightSubscriber` already relies on.
    """

    def __init__(self, checkpoint: NeuralAgentCheckpoint, *, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay!r}")
        super().__init__(checkpoint)
        self.decay = decay
        self._shadow: Dict[str, Dict[str, Any]] = {}

    def _live_modules(self) -> Dict[str, Any]:
        modules: Dict[str, Any] = dict(self.checkpoint.encoders)
        for name in ("fusion", "world_model", "policy", "critic"):
            module = getattr(self.checkpoint, name)
            if module is not None:
                modules[name] = module
        return modules

    def _advance_shadow(self, modules: Dict[str, Any]) -> None:
        """``shadow <- decay * shadow + (1 - decay) * raw``, seeded with the
        raw weights on the first publish so an actor's first-ever reload
        isn't diluted toward an untrained init."""
        for name, module in modules.items():
            raw = module.state_dict()
            if name not in self._shadow:
                self._shadow[name] = {k: v.detach().clone() for k, v in raw.items()}
                continue
            shadow = self._shadow[name]
            for key, value in raw.items():
                target = shadow[key]
                if torch.is_floating_point(target):
                    target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    shadow[key] = value.detach().clone()

    def publish(self, *, reason: str = "publish") -> int:
        modules = self._live_modules()
        self._advance_shadow(modules)
        # `state_dict()` values alias the live parameter storage, so this
        # backup must clone -- `load_state_dict` below mutates in place.
        raw_state = {
            name: {k: v.detach().clone() for k, v in module.state_dict().items()}
            for name, module in modules.items()
        }
        try:
            for name, module in modules.items():
                module.load_state_dict(self._shadow[name])
            return super().publish(reason=reason)
        finally:
            for name, module in modules.items():
                module.load_state_dict(raw_state[name])


@dataclass
class WeightSubscriber:
    """Actor-side: poll ``path`` for a newer snapshot than the last one
    loaded, and hot-swap it into ``bundle``'s modules when found.

    ``bundle`` should be constructed with only the modules the actor wants
    to keep in sync (usually the live policy/critic instances the runtime's
    policy is already using) -- swapping is in place via
    ``nn.Module.load_state_dict``, so the actor's existing ``policy``/
    ``critic`` object references stay valid; only their weights change.
    """

    path: str
    bundle: NeuralAgentCheckpoint
    last_version: int = -1
    reload_count: int = 0
    skipped_count: int = 0
    max_staleness: int = 0

    def poll_version(self) -> Optional[int]:
        """Peek at the snapshot's version without loading it."""
        try:
            metadata = read_checkpoint_metadata(self.path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return int(metadata.get("training_ticks", -1))

    def staleness(self) -> Optional[int]:
        """How many versions the last-*loaded* snapshot is behind the
        newest one currently published on disk (concurrent schedule, issue
        #100): the actor can call this every tick -- even ticks that skip an
        actual reload -- to log and bound how stale its live weights are
        without paying for a reload. ``0`` once caught up; ``None`` before
        anything has ever been published. Also updates :attr:`max_staleness`,
        the peak staleness observed so far, so a run can assert its
        staleness stayed bounded rather than drifting upward."""
        version = self.poll_version()
        if version is None:
            return None
        gap = max(0, version - self.last_version)
        self.max_staleness = max(self.max_staleness, gap)
        return gap

    def maybe_reload(self) -> Optional[int]:
        """Load a newer snapshot if one is available; returns the new
        version, or ``None`` if nothing changed (or nothing has been
        published yet, or the snapshot could not be read this poll --
        transient races are simply retried on the next call)."""
        version = self.poll_version()
        if version is None or version <= self.last_version:
            self.skipped_count += 1
            return None
        try:
            metadata = self.bundle.load(
                self.path,
                expected_layout_hash=self.bundle.layout_hash,
                expected_action_keys=self.bundle.action_keys,
                restore_rng=False,
            )
        except (FileNotFoundError, EOFError, RuntimeError, CheckpointCompatibilityError):
            self.skipped_count += 1
            return None
        self.last_version = int(metadata.get("training_ticks", version))
        self.reload_count += 1
        return self.last_version

    def stats(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "last_version": self.last_version,
            "reload_count": self.reload_count,
            "skipped_count": self.skipped_count,
            "max_staleness": self.max_staleness,
        }
