"""Offline cognition: dreams and staleness-free sleep consolidation.

The schedule is intentionally usable without the optional neural dependency;
dream and checkpoint helpers are imported only when requested.
"""

from sleep.schedule import ConsolidationResult, Phase, PhasicSleepSchedule


def __getattr__(name: str):
    if name in {"dream", "export_dream_file"}:
        from sleep.dream import dream, export_dream_file

        return {"dream": dream, "export_dream_file": export_dream_file}[name]
    if name in {"EMAWeightPublisher", "WeightPublisher", "WeightSubscriber"}:
        from sleep.weight_publisher import EMAWeightPublisher, WeightPublisher, WeightSubscriber

        return {
            "EMAWeightPublisher": EMAWeightPublisher,
            "WeightPublisher": WeightPublisher,
            "WeightSubscriber": WeightSubscriber,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ConsolidationResult",
    "Phase",
    "PhasicSleepSchedule",
    "EMAWeightPublisher",
    "WeightPublisher",
    "WeightSubscriber",
    "dream",
    "export_dream_file",
]
