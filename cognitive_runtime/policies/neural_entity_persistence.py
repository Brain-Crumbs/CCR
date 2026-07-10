"""Neural entity-persistence bridge: the trained ``EntityPersistenceModel``
behind the loop's ``core.entity_persistence.EntityPersistence`` seam
(issue #27).

Each tick, given whatever ``vision.entities`` currently holds (latest-value,
not "this tick's records" -- the stream is delta-published, see
``training.entity_persistence``), updates an ``EntityTracker`` and asks the
model to predict the feature of every currently-occluded tracked entity; the
combined novelty score (``core.novelty.combine_novelty``) uses the model's
own self-supervised ``surprise`` output, the max across occluded entities, as
its entity-persistence half.

Imports torch (via ``cognitive_runtime.neural``), so it is imported lazily by
the CLI, mirroring ``policies/neural_world_model.py``.
"""

from __future__ import annotations

from typing import Union

import torch

from cognitive_runtime.core.entity_features import VISION_ENTITIES_STREAM
from cognitive_runtime.core.entity_persistence import (
    EntityPersistence,
    EntityPersistencePrediction,
)
from cognitive_runtime.core.entity_tracker import EntityTracker
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.neural.entity_persistence import EntityPersistenceModel, normalize_gap
from cognitive_runtime.training.entity_persistence import load_entity_persistence_checkpoint


class NeuralEntityPersistence(EntityPersistence):
    """Bridges a trained :class:`EntityPersistenceModel` into the loop's
    ``EntityPersistence`` seam, tracking entities across ticks itself so the
    loop needs no changes beyond calling ``predict`` each tick."""

    def __init__(
        self,
        model: Union[EntityPersistenceModel, str],
        *,
        max_gap_ticks: int = 200,
    ) -> None:
        if isinstance(model, str):
            model, _metadata = load_entity_persistence_checkpoint(model)
        self.model = model
        self.model.eval()
        self.max_gap_ticks = int(max_gap_ticks)
        self._tracker = EntityTracker(max_gap_ticks=self.max_gap_ticks)

    def reset(self) -> None:
        self._tracker.reset()

    def predict(self, state: State, memory: Memory) -> EntityPersistencePrediction:
        latest = memory.buffer.latest(VISION_ENTITIES_STREAM)
        entities = (
            latest.payload if latest is not None and isinstance(latest.payload, list) else []
        )
        self._tracker.update(entities)

        occluded_ids = self._tracker.occluded()
        if not occluded_ids:
            return EntityPersistencePrediction()

        last_features = []
        gaps = []
        for eid in occluded_ids:
            tracked = self._tracker.state(eid)
            assert tracked is not None
            last_features.append(tracked.last_feature)
            gaps.append(normalize_gap(tracked.gap_ticks, self.model.gap_cap))

        with torch.no_grad():
            out = self.model(
                torch.tensor(last_features, dtype=torch.float32),
                torch.tensor(gaps, dtype=torch.float32),
            )
        return EntityPersistencePrediction(surprise=float(out.surprise.max()))
