"""Tracks per-entity identity/feature/gap state across ticks from raw
``vision.entities`` payloads (issue #27: object permanence).

An entity walking out of line-of-sight doesn't stop existing; the tracker is
what lets a persistence model be asked "what do you think entity X's state is
right now" during the gap, and lets a dataset builder find the
``(last-seen-before-gap, gap-length) -> state-at-reappearance`` training pairs
that recorded sessions provide for free every time a tracked entity leaves
and re-enters view.  Pure Python (no torch) so it is usable from the runtime
loop, replay, and dataset builders alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from cognitive_runtime.core.entity_features import entity_feature_vector, entity_id

#: An entity missing longer than this many ticks is assumed gone for good
#: (killed, despawned, wandered off) rather than tracked forever.
DEFAULT_MAX_GAP_TICKS = 200


@dataclass
class TrackedEntity:
    last_feature: List[float]
    last_seen_tick: int
    gap_ticks: int = 0  # 0 while visible; ticks since last seen while occluded


@dataclass
class Reappearance:
    """One entity going from occluded back to visible this tick."""

    entity_id: Any
    last_feature: List[float]  # feature just before the gap started
    gap_ticks: int  # how many ticks it was missing
    feature_now: List[float]  # realized feature this tick -- the training label


class EntityTracker:
    """Tracks entities across ticks by the stable id ``vision.entities``
    carries per record.

    ``max_gap_ticks`` bounds memory: an entity missing longer than this is
    dropped instead of tracked forever.
    """

    def __init__(self, max_gap_ticks: int = DEFAULT_MAX_GAP_TICKS) -> None:
        self.max_gap_ticks = int(max_gap_ticks)
        self._entities: Dict[Any, TrackedEntity] = {}
        self._tick = -1

    def reset(self) -> None:
        self._entities.clear()
        self._tick = -1

    def update(self, entities: Sequence[Mapping[str, Any]]) -> List[Reappearance]:
        """Advance one tick given this tick's visible entities.

        Returns the reappearance events this tick produced (empty most
        ticks): entities that were occluded and just became visible again.
        """
        self._tick += 1
        visible: Dict[Any, List[float]] = {}
        for entity in entities:
            eid = entity_id(entity)
            if eid is None:
                continue
            visible[eid] = entity_feature_vector(entity)

        reappearances: List[Reappearance] = []
        for eid, feature in visible.items():
            tracked = self._entities.get(eid)
            if tracked is not None and tracked.gap_ticks > 0:
                reappearances.append(
                    Reappearance(
                        entity_id=eid,
                        last_feature=list(tracked.last_feature),
                        gap_ticks=tracked.gap_ticks,
                        feature_now=list(feature),
                    )
                )
            self._entities[eid] = TrackedEntity(
                last_feature=feature, last_seen_tick=self._tick, gap_ticks=0
            )

        stale: List[Any] = []
        for eid, tracked in self._entities.items():
            if eid in visible:
                continue
            tracked.gap_ticks += 1
            if tracked.gap_ticks > self.max_gap_ticks:
                stale.append(eid)
        for eid in stale:
            del self._entities[eid]

        return reappearances

    def occluded(self) -> List[Any]:
        """Ids currently tracked but not visible this tick (``gap_ticks > 0``)."""
        return [eid for eid, tracked in self._entities.items() if tracked.gap_ticks > 0]

    def state(self, tracked_id: Any) -> Optional[TrackedEntity]:
        return self._entities.get(tracked_id)
