"""Offline cognition: dreams and staleness-free sleep consolidation.

The schedule is intentionally usable without the optional neural dependency;
dream and checkpoint helpers are imported only when requested.
"""

from sleep.schedule import ConsolidationResult, Phase, PhasicSleepSchedule


def __getattr__(name: str):
    if name in {"dream", "dream_latents", "export_dream_file"}:
        from sleep.dream import dream, dream_latents, export_dream_file

        return {
            "dream": dream, "dream_latents": dream_latents, "export_dream_file": export_dream_file,
        }[name]
    if name in {"EMAWeightPublisher", "WeightPublisher", "WeightSubscriber"}:
        from sleep.weight_publisher import EMAWeightPublisher, WeightPublisher, WeightSubscriber

        return {
            "EMAWeightPublisher": EMAWeightPublisher,
            "WeightPublisher": WeightPublisher,
            "WeightSubscriber": WeightSubscriber,
        }[name]
    if name in {
        "dream_fraction", "copy_last_quality_margin", "ReplaySample", "Reservoir",
        "TrainingBatch", "GenerativeReplayMixer",
    }:
        from sleep import replay_mix

        return getattr(replay_mix, name)
    if name in {"ForgettingReport", "compute_forgetting_metric"}:
        from sleep import forgetting

        return getattr(forgetting, name)
    raise AttributeError(name)

__all__ = [
    "ConsolidationResult",
    "Phase",
    "PhasicSleepSchedule",
    "EMAWeightPublisher",
    "WeightPublisher",
    "WeightSubscriber",
    "dream",
    "dream_latents",
    "export_dream_file",
    "dream_fraction",
    "copy_last_quality_margin",
    "ReplaySample",
    "Reservoir",
    "TrainingBatch",
    "GenerativeReplayMixer",
    "ForgettingReport",
    "compute_forgetting_metric",
]
