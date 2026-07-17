"""Compatibility shim for the Phase 7 development runner (issue #104).

The implementation moved to :mod:`development` (definitions in
``development.definitions``, the torch-dependent train/evaluate/promote loop
in ``development.runner``). This facade keeps pre-Phase-7 imports working;
new code should import :mod:`development` directly.
"""

from development.definitions import (
    KNOWN_METRICS,
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
    if name == "run_curriculum":
        from development.runner import run_curriculum

        return run_curriculum
    raise AttributeError(name)


__all__ = [
    "KNOWN_METRICS",
    "CurriculumDefinition",
    "CurriculumDefinitionError",
    "CurriculumRunResult",
    "CurriculumState",
    "CurriculumStageSpec",
    "PromotionCriteria",
    "curriculum_definition_from_dict",
    "load_curriculum_definition",
    "run_curriculum",
]
