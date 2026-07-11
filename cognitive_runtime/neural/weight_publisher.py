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

    def poll_version(self) -> Optional[int]:
        """Peek at the snapshot's version without loading it."""
        try:
            metadata = read_checkpoint_metadata(self.path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return int(metadata.get("training_ticks", -1))

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
        }
