"""Staged ontogeny: definitions are usable without the optional neural
dependency; the train/evaluate/promote runner needs it (issue #104).

Generalises ``cognitive_runtime.training.curriculum_runner`` (issue #43) --
that module is now a compatibility shim over this package. See
``docs/v2/phases/phase-7-development-ladder.md``.
"""

from development.definitions import (
    KNOWN_METRICS,
    KNOWN_WORLDS,
    MILESTONE_METRICS,
    MOTOR_FREEDOMS,
    CurriculumDefinition,
    CurriculumDefinitionError,
    CurriculumRunResult,
    CurriculumState,
    CurriculumStageSpec,
    PromotionCriteria,
    curriculum_definition_from_dict,
    load_curriculum_definition,
)


def __getattr__(name: str):
    if name in {"MilestoneMetricsFn", "run_curriculum"}:
        from development.runner import MilestoneMetricsFn, run_curriculum

        return {"MilestoneMetricsFn": MilestoneMetricsFn, "run_curriculum": run_curriculum}[name]
    raise AttributeError(name)


__all__ = [
    "KNOWN_METRICS",
    "KNOWN_WORLDS",
    "MILESTONE_METRICS",
    "MOTOR_FREEDOMS",
    "CurriculumDefinition",
    "CurriculumDefinitionError",
    "CurriculumRunResult",
    "CurriculumState",
    "CurriculumStageSpec",
    "PromotionCriteria",
    "curriculum_definition_from_dict",
    "load_curriculum_definition",
    "MilestoneMetricsFn",
    "run_curriculum",
]
