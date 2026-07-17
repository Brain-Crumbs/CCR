"""Staged-ontogeny definitions: stage specs, promotion gates, curriculum
state (issue #104, generalising ``training/curriculum_runner.py``).

Torch-free by design (mirrors ``sleep/schedule.py`` vs. ``sleep/async_trainer.py``):
everything here is definition/validation/(de)serialization only, so a
curriculum or ladder definition can be loaded and checked without the
``neural`` extra installed. The actual train/evaluate/promote loop lives in
:mod:`development.runner`, which does need torch.

A stage originally declared just a world/reward config plus a single
:class:`PromotionCriteria` (issue #43). Phase 7 (issue #104) extends
:class:`CurriculumStageSpec` so a stage can instead declare the shape of a
staged-ontogeny rung: which ``World`` + scenario it runs, which senses are
active, its motor freedom (``frozen | overridden | learned``), which losses
are on, and one or more **milestone gates** -- promotion then requires
*every* gate to pass, not one scalar metric. The legacy single-``promotion``
shape keeps working unchanged (``gates`` defaults to empty), which is how old
curriculum definitions still load through the shim.

Wiring a stage's gates to the concrete Phase 2-6 milestone computation (the
cortex-beats-copy-last/action-ablation/forgetting/reflex-override metrics)
and authoring the actual Gestation->Foraging ladder is issue #105's job; this
module only carries the declarative shape and validates it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.reward_profile import (
    RewardProfile,
    RewardProfileError,
    load_reward_profile,
)

#: ``--world`` values a stage can declare (mirrors ``cli.py``'s ``WORLDS``
#: selector, issue #89); kept as a small local constant rather than importing
#: ``cli.py`` to avoid pulling argparse/CLI wiring into a definitions module.
KNOWN_WORLDS = ("minecraft", "crafter")

#: A stage's motor-freedom declaration (Phase 7 table): frozen (no motor
#: output), overridden (caregiver/scripted controls the body), or learned
#: (the voluntary path, Phase 6, is in charge).
MOTOR_FREEDOMS = ("frozen", "overridden", "learned")

#: Metrics :class:`PromotionCriteria` can read off a plain
#: ``EvaluationSummary`` (the pre-Phase-7 actor/critic eval): the raw
#: episode aggregates plus the synthetic ``survival_rate``.
_SUMMARY_METRICS = (
    "average_reward", "average_ticks", "total_reward", "total_ticks", "survival_rate",
)

#: Milestone metrics introduced by Phases 2-6 that a stage's gate can
#: reference once the corresponding evaluator computes it into the metrics
#: mapping passed to :meth:`PromotionCriteria.evaluate` (see
#: ``development.runner.run_curriculum``'s ``milestone_metrics`` hook).
MILESTONE_METRICS = (
    "cortex_beats_copy_last",      # Phase 2 walk_forward vs. copy-last baseline (#92)
    "action_ablation_margin",      # Phase 2 action-conditioning ablation gap (#92)
    "calibrated_surprise_ece",     # Phase 3 Arbiter's calibrated surprise (#95)
    "forgetting_score",            # Phase 5 generative-replay forgetting metric (#99)
    "reflex_activation_rate",      # Phase 6 reflex-activation-rate metric (#103)
    "reflex_override_precedence",  # Phase 6 caregiver-override precedence (#102)
)

#: The full vocabulary a :class:`PromotionCriteria` (whether the legacy
#: single ``promotion`` field or a Phase 7 milestone ``gates`` entry) may
#: name.
KNOWN_METRICS = _SUMMARY_METRICS + MILESTONE_METRICS


class CurriculumDefinitionError(ValueError):
    """A curriculum/ladder definition file (or dict) is malformed."""


def _err(source: str, message: str) -> CurriculumDefinitionError:
    return CurriculumDefinitionError(f"curriculum definition {source!r}: {message}")


@dataclass(frozen=True)
class PromotionCriteria:
    """One promotion gate: ``metric`` must reach ``threshold`` (mean over
    ``sample_size`` eval episodes, not a single episode) to pass.

    ``evaluate``/``value_of`` accept either the legacy ``EvaluationSummary``
    (issue #43's single-metric actor/critic eval) or a plain
    ``Mapping[str, float]`` of milestone metrics (issue #104) -- a stage's
    ``gates`` are a list of these, and promotion requires *all* of them to
    pass, not one scalar.
    """

    metric: str = "average_ticks"
    threshold: float = 0.0
    sample_size: int = 3

    def __post_init__(self) -> None:
        if self.metric not in KNOWN_METRICS:
            raise ValueError(
                f"unknown promotion metric {self.metric!r}; expected one of {KNOWN_METRICS}"
            )
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {self.sample_size}")

    def value_of(self, source: Union[Any, Mapping[str, float]]) -> float:
        if isinstance(source, Mapping):
            if self.metric not in source:
                raise CurriculumDefinitionError(
                    f"milestone metric {self.metric!r} not present in evaluation "
                    f"metrics {sorted(source)}; the stage's evaluator did not "
                    "compute it"
                )
            return float(source[self.metric])
        if self.metric == "survival_rate":
            reasons = source.termination_reasons
            if not reasons:
                return 0.0
            survived = sum(1 for r in reasons if not r.startswith("death"))
            return survived / len(reasons)
        return float(getattr(source, self.metric))

    def evaluate(self, source: Union[Any, Mapping[str, float]]) -> bool:
        return self.value_of(source) >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "threshold": self.threshold, "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class CurriculumStageSpec:
    """One curriculum/ladder stage.

    Legacy shape (issue #43): world/reward config for the Minecraft
    actor/critic loop, gated by a single ``promotion`` criterion.

    Phase 7 shape (issue #104), all optional and additive so old definitions
    keep loading unchanged: which ``World`` + ``scenario`` the stage runs,
    its active ``senses``, its ``motor_freedom``, its active ``losses``, and
    one or more milestone ``gates`` -- when ``gates`` is non-empty it
    replaces ``promotion`` as the thing a runner checks (all gates must pass;
    see :mod:`development.runner`).
    """

    name: str
    world_config: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, Any] = field(default_factory=dict)
    reward_profile_path: Optional[str] = None
    train_episodes: int = 10
    promotion: PromotionCriteria = field(default_factory=PromotionCriteria)
    #: Demotion/plateau rule: how many train+evaluate attempts this stage
    #: gets before the runner holds instead of retrying forever ("no silent
    #: spin").
    max_attempts: int = 3
    #: Which ``Program`` (``--world`` selector, issue #89) this stage runs;
    #: ``None`` keeps the pre-Phase-7 default (Minecraft).
    world: Optional[str] = None
    #: Which nursery scenario (``training/nursery.py``'s registry) this
    #: stage trains/evaluates against, e.g. ``"walk_forward"``.
    scenario: Optional[str] = None
    #: Active sense/stream names for this stage (declarative; Phase 7 does
    #: not itself gate which streams the runtime feeds the fusion -- that is
    #: the ladder's job, issue #105).
    senses: Sequence[str] = ()
    #: One of :data:`MOTOR_FREEDOMS`; ``None`` keeps the pre-Phase-7 default
    #: (the actor/critic loop always learns its own motor output).
    motor_freedom: Optional[str] = None
    #: Active loss terms for this stage, e.g. ``("prediction", "action_conditioning")``.
    losses: Sequence[str] = ()
    #: Milestone gates (issue #104): when non-empty, promotion requires
    #: *every* gate to pass against the stage's computed metrics, replacing
    #: the single-scalar ``promotion`` field.
    gates: Sequence[PromotionCriteria] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if self.reward_config and self.reward_profile_path:
            raise ValueError(
                f"stage {self.name!r}: reward_config and reward_profile_path are "
                "mutually exclusive (issue #41's profile path replaces the legacy "
                "reward_config weights entirely)"
            )
        if self.train_episodes < 1:
            raise ValueError(f"stage {self.name!r}: train_episodes must be >= 1")
        if self.max_attempts < 1:
            raise ValueError(f"stage {self.name!r}: max_attempts must be >= 1")
        if self.world is not None and self.world not in KNOWN_WORLDS:
            raise ValueError(
                f"stage {self.name!r}: unknown world {self.world!r}; expected one "
                f"of {KNOWN_WORLDS}"
            )
        if self.motor_freedom is not None and self.motor_freedom not in MOTOR_FREEDOMS:
            raise ValueError(
                f"stage {self.name!r}: unknown motor_freedom {self.motor_freedom!r}; "
                f"expected one of {MOTOR_FREEDOMS}"
            )
        self._check_names("senses", self.senses)
        self._check_names("losses", self.losses)
        gate_metrics = [gate.metric for gate in self.gates]
        if len(set(gate_metrics)) != len(gate_metrics):
            raise ValueError(
                f"stage {self.name!r}: duplicate milestone gate metrics {gate_metrics}"
            )

    def _check_names(self, field_name: str, values: Sequence[str]) -> None:
        if any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"stage {self.name!r}: {field_name} must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"stage {self.name!r}: duplicate {field_name} {list(values)}")

    @property
    def is_staged_ontogeny(self) -> bool:
        """True once a stage declares Phase 7's milestone gates instead of
        the legacy single-``promotion`` shape."""
        return bool(self.gates)

    def evaluate_gates(self, metrics: Mapping[str, float]) -> Dict[str, bool]:
        """Evaluate every milestone gate against ``metrics``; a stage
        promotes only when every gate passes (Phase 7: "not a single
        scalar")."""
        if not self.gates:
            raise ValueError(f"stage {self.name!r} has no milestone gates to evaluate")
        return {gate.metric: gate.evaluate(metrics) for gate in self.gates}

    def build_reward_profile(self) -> Optional[RewardProfile]:
        if not self.reward_profile_path:
            return None
        try:
            return load_reward_profile(self.reward_profile_path)
        except RewardProfileError as exc:
            raise CurriculumDefinitionError(
                f"stage {self.name!r}: {exc}"
            ) from exc

    def build_reward_config(self):
        if self.reward_profile_path or not self.reward_config:
            return None
        import dataclasses

        from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig

        return dataclasses.replace(SurvivalRewardConfig(), **self.reward_config)


@dataclass(frozen=True)
class CurriculumDefinition:
    """An ordered list of stages a checkpoint's brain progresses through."""

    name: str
    stages: Sequence[CurriculumStageSpec]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError(f"curriculum {self.name!r} has no stages")
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"curriculum {self.name!r} has duplicate stage names: {names}")

    def index_of(self, stage_name: str) -> int:
        for index, stage in enumerate(self.stages):
            if stage.name == stage_name:
                return index
        raise KeyError(f"unknown stage {stage_name!r} in curriculum {self.name!r}")


def _promotion_from_dict(source: str, stage_name: str, raw: Mapping[str, Any]) -> PromotionCriteria:
    if not isinstance(raw, Mapping):
        raise _err(source, f"stage {stage_name!r}: 'promotion'/'gates' entry must be a mapping")
    unknown = set(raw) - {"metric", "threshold", "sample_size"}
    if unknown:
        raise _err(source, f"stage {stage_name!r}: unknown promotion field(s) {sorted(unknown)}")
    try:
        return PromotionCriteria(
            metric=raw.get("metric", "average_ticks"),
            threshold=float(raw.get("threshold", 0.0)),
            sample_size=int(raw.get("sample_size", 3)),
        )
    except (TypeError, ValueError) as exc:
        raise _err(source, f"stage {stage_name!r}: invalid promotion criteria: {exc}") from exc


def _stage_from_dict(source: str, raw: Mapping[str, Any]) -> CurriculumStageSpec:
    if not isinstance(raw, Mapping):
        raise _err(source, f"each stage must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {
        "name", "world_config", "reward_config", "reward_profile_path",
        "train_episodes", "promotion", "max_attempts",
        "world", "scenario", "senses", "motor_freedom", "losses", "gates",
    }
    if unknown:
        raise _err(source, f"unknown stage field(s): {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise _err(source, "each stage requires a non-empty string 'name'")
    promotion_raw = raw.get("promotion", {})
    senses_raw = raw.get("senses", [])
    if not isinstance(senses_raw, list):
        raise _err(source, f"stage {name!r}: 'senses' must be a list")
    losses_raw = raw.get("losses", [])
    if not isinstance(losses_raw, list):
        raise _err(source, f"stage {name!r}: 'losses' must be a list")
    gates_raw = raw.get("gates", [])
    if not isinstance(gates_raw, list):
        raise _err(source, f"stage {name!r}: 'gates' must be a list")
    try:
        return CurriculumStageSpec(
            name=name,
            world_config=dict(raw.get("world_config", {})),
            reward_config=dict(raw.get("reward_config", {})),
            reward_profile_path=raw.get("reward_profile_path"),
            train_episodes=int(raw.get("train_episodes", 10)),
            promotion=_promotion_from_dict(source, name, promotion_raw),
            max_attempts=int(raw.get("max_attempts", 3)),
            world=raw.get("world"),
            scenario=raw.get("scenario"),
            senses=tuple(senses_raw),
            motor_freedom=raw.get("motor_freedom"),
            losses=tuple(losses_raw),
            gates=tuple(_promotion_from_dict(source, name, g) for g in gates_raw),
        )
    except ValueError as exc:
        raise _err(source, str(exc)) from exc


def curriculum_definition_from_dict(
    data: Mapping[str, Any], source: str = "<dict>",
) -> CurriculumDefinition:
    """Validate and build a :class:`CurriculumDefinition` from a parsed mapping.

    Also checks every stage shares the same simulated-world stream layout
    (``TemporalFusion.layout_hash`` over the stage's ``Program.stream_catalog()``)
    -- a curriculum that changes ``world_size`` (or any other knob that
    reshapes the stream catalog), or mixes ``World``s with incompatible
    catalogs, between stages cannot carry one policy/critic checkpoint across
    the stage boundary, so this fails at definition-load time instead of a
    confusing runtime ``CheckpointCompatibilityError``.
    """
    if not isinstance(data, Mapping):
        raise _err(source, f"top level must be a mapping, got {type(data).__name__}")
    unknown = set(data) - {"name", "description", "stages"}
    if unknown:
        raise _err(source, f"unknown top-level field(s): {sorted(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise _err(source, "top-level 'name' is required and must be a non-empty string")
    stages_raw = data.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise _err(source, "top-level 'stages' must be a non-empty list")
    stages = [_stage_from_dict(source, raw) for raw in stages_raw]
    try:
        definition = CurriculumDefinition(name=name, stages=stages)
    except ValueError as exc:
        raise _err(source, str(exc)) from exc
    _validate_shared_layout(source, definition)
    return definition


def _program_for_layout(world: Optional[str], world_config: Dict[str, Any]):
    """Construct the ``Program`` used to compute a stage's stream layout hash
    -- mirrors ``cli.py``'s ``--world`` factory (issue #89) but only needs
    the ``Program``, not the full stream/action registries."""
    if world in (None, "minecraft"):
        return MinecraftSurvivalBox(config=world_config)
    if world == "crafter":
        from cognitive_runtime.programs.crafter.adapter import CrafterWorld

        return CrafterWorld(config=world_config)
    raise AssertionError(f"unreachable: unknown world {world!r} (validated in __post_init__)")


def _validate_shared_layout(source: str, definition: CurriculumDefinition) -> None:
    layout_hashes = {}
    for stage in definition.stages:
        program = _program_for_layout(stage.world, stage.world_config)
        fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
        layout_hashes[stage.name] = fusion.layout_hash
    distinct = set(layout_hashes.values())
    if len(distinct) > 1:
        raise _err(
            source,
            "stages disagree on stream layout (likely a differing 'world_size', "
            "a mismatched 'world', or other stream-catalog knob), so one "
            f"checkpoint cannot carry across them: {layout_hashes}",
        )


def load_curriculum_definition(path: str) -> CurriculumDefinition:
    """Load and validate a curriculum definition from a `.yaml`/`.yml`/`.json` file."""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise _err(path, f"could not read curriculum file: {exc}") from exc

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a core dep
            raise _err(
                path, "PyYAML is required to load .yaml/.yml curriculum definitions"
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _err(path, f"invalid YAML: {exc}") from exc
    elif ext == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _err(path, f"invalid JSON: {exc}") from exc
    else:
        raise _err(path, f"unsupported extension {ext!r}; expected .yaml, .yml or .json")

    if data is None:
        raise _err(path, "curriculum file is empty")
    return curriculum_definition_from_dict(data, source=path)


@dataclass
class CurriculumState:
    """Curriculum progress, persisted into the checkpoint bundle's
    ``training_stats["curriculum"]`` (issue #20) so the runner can resume."""

    definition_name: str
    stage_index: int = 0
    attempts_at_stage: int = 0
    completed: bool = False
    held: bool = False
    hold_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "definition_name": self.definition_name,
            "stage_index": self.stage_index,
            "attempts_at_stage": self.attempts_at_stage,
            "completed": self.completed,
            "held": self.held,
            "hold_reason": self.hold_reason,
            "history": list(self.history),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CurriculumState":
        return CurriculumState(
            definition_name=data["definition_name"],
            stage_index=int(data.get("stage_index", 0)),
            attempts_at_stage=int(data.get("attempts_at_stage", 0)),
            completed=bool(data.get("completed", False)),
            held=bool(data.get("held", False)),
            hold_reason=data.get("hold_reason"),
            history=list(data.get("history", [])),
        )


@dataclass(frozen=True)
class CurriculumRunResult:
    status: str  # "completed" | "held"
    state: CurriculumState
    resumed: bool

    @property
    def held(self) -> bool:
        return self.status == "held"

    @property
    def completed(self) -> bool:
        return self.status == "completed"
