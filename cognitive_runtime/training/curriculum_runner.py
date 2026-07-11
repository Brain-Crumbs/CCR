"""Curriculum runner: staged goals with metric-gated promotion (issue #43).

Extends the world/config presets (issue #30, ``programs/minecraft/curriculum.py``)
into an unattended orchestrator: an ordered list of *stages* (world config +
reward config/profile (issue #41) + promotion criteria), where the runner
trains an actor/critic stack on a stage, evaluates it over N episodes, and
either promotes to the next stage (metric passed the threshold) or holds the
stage and logs why. The same policy/critic/optimizer weights -- the "brain" --
carry across stage boundaries; only the world/reward config changes.

Promotion here gates on the plain mean of one summary metric
(:class:`PromotionCriteria`) over a fixed sample size -- deliberately the
simplest thing that is still an N-episode statistical aggregate, not a
single-episode fluke. Issue #44's richer statistical harness
(:mod:`cognitive_runtime.training.statistical_evaluation`: confidence
intervals across survival/reward-by-tier/coverage/prediction-error, with
regression/improvement flagging) has since landed and can inspect a
curriculum run's recorded sessions directly
(``statistical-evaluate --from-sessions``), but this module's own promotion
gate intentionally keeps the simple mean criterion rather than growing a
second, harder-to-predict gating rule.

Curriculum state (current stage, attempt count, promotion history) lives in
the checkpoint bundle's ``training_stats["curriculum"]`` (issue #20), so
interrupting and restarting the runner resumes at the correct stage from the
same checkpoint. ``training_stats`` is a whole-dict replace on
:meth:`NeuralAgentCheckpoint.save`, so this module always reads the existing
sidecar first and merges its own key in, rather than clobbering whatever else
(e.g. evaluation-gate reports) a checkpoint already carries.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.neural import (
    ActorCriticOptimizer,
    MLPPolicyModel,
    MLPValueModel,
    NeuralAgentCheckpoint,
)
from cognitive_runtime.neural.checkpoint import (
    checkpoint_metadata_path,
    read_checkpoint_metadata,
)
from cognitive_runtime.policies.actor_critic import (
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.reward_profile import (
    RewardProfile,
    RewardProfileError,
    load_reward_profile,
)
from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary

#: Metrics :class:`PromotionCriteria` knows how to read off an
#: :class:`EvaluationSummary` (plus the synthetic ``survival_rate``, computed
#: from ``termination_reasons``).
KNOWN_METRICS = (
    "average_reward", "average_ticks", "total_reward", "total_ticks", "survival_rate",
)


class CurriculumDefinitionError(ValueError):
    """A curriculum definition file (or dict) is malformed."""


def _err(source: str, message: str) -> CurriculumDefinitionError:
    return CurriculumDefinitionError(f"curriculum definition {source!r}: {message}")


@dataclass(frozen=True)
class PromotionCriteria:
    """Promotion gate for one stage: ``metric`` over ``sample_size`` eval
    episodes must reach ``threshold`` (mean, not per-episode) to promote."""

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

    def value_of(self, summary: EvaluationSummary) -> float:
        if self.metric == "survival_rate":
            reasons = summary.termination_reasons
            if not reasons:
                return 0.0
            survived = sum(1 for r in reasons if not r.startswith("death"))
            return survived / len(reasons)
        return float(getattr(summary, self.metric))

    def evaluate(self, summary: EvaluationSummary) -> bool:
        return self.value_of(summary) >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "threshold": self.threshold, "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class CurriculumStageSpec:
    """One curriculum stage: world/backend config (#30) + reward config or
    profile (#41) + promotion criteria + a plateau rule (``max_attempts``)."""

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

    def build_reward_profile(self) -> Optional[RewardProfile]:
        if not self.reward_profile_path:
            return None
        try:
            return load_reward_profile(self.reward_profile_path)
        except RewardProfileError as exc:
            raise CurriculumDefinitionError(
                f"stage {self.name!r}: {exc}"
            ) from exc

    def build_reward_config(self) -> Optional[SurvivalRewardConfig]:
        if self.reward_profile_path or not self.reward_config:
            return None
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
        raise _err(source, f"stage {stage_name!r}: 'promotion' must be a mapping")
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
    }
    if unknown:
        raise _err(source, f"unknown stage field(s): {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise _err(source, "each stage requires a non-empty string 'name'")
    promotion_raw = raw.get("promotion", {})
    try:
        return CurriculumStageSpec(
            name=name,
            world_config=dict(raw.get("world_config", {})),
            reward_config=dict(raw.get("reward_config", {})),
            reward_profile_path=raw.get("reward_profile_path"),
            train_episodes=int(raw.get("train_episodes", 10)),
            promotion=_promotion_from_dict(source, name, promotion_raw),
            max_attempts=int(raw.get("max_attempts", 3)),
        )
    except ValueError as exc:
        raise _err(source, str(exc)) from exc


def curriculum_definition_from_dict(
    data: Mapping[str, Any], source: str = "<dict>",
) -> CurriculumDefinition:
    """Validate and build a :class:`CurriculumDefinition` from a parsed mapping.

    Also checks every stage shares the same simulated-world stream layout
    (``TemporalFusion.layout_hash`` over ``MinecraftSurvivalBox.stream_catalog()``)
    -- a curriculum that changes ``world_size`` (or any other knob that
    reshapes the stream catalog) between stages cannot carry one policy/critic
    checkpoint across the stage boundary, so this fails at definition-load
    time instead of a confusing runtime ``CheckpointCompatibilityError``.
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


def _validate_shared_layout(source: str, definition: CurriculumDefinition) -> None:
    layout_hashes = {}
    for stage in definition.stages:
        program = MinecraftSurvivalBox(config=stage.world_config)
        fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
        layout_hashes[stage.name] = fusion.layout_hash
    distinct = set(layout_hashes.values())
    if len(distinct) > 1:
        raise _err(
            source,
            "stages disagree on stream layout (likely a differing 'world_size' or "
            f"other stream-catalog knob), so one checkpoint cannot carry across "
            f"them: {layout_hashes}",
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


def _seed_for(base: int, stage_index: int, attempt: int, unit: int) -> int:
    """Deterministic, non-colliding seed per (stage, attempt) -- each retry
    sees fresh episode content instead of replaying an identical eval set."""
    return base + stage_index * 1_000_000 + attempt * max(unit, 1)


def _new_actor_critic_stack(world_config: Dict[str, Any], seed: int, *, lr: float, entropy_coef: float):
    program = MinecraftSurvivalBox(config=world_config)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_keys = [action.key() for action in program.metadata().action_space]
    wf_width = world_feature_width(action_keys)
    torch.manual_seed(seed)
    policy_model = MLPPolicyModel(
        fusion.width, wf_width, len(action_keys),
        hidden_dim=32, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    critic_model = MLPValueModel(
        fusion.width, wf_width,
        hidden_dim=32, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    optimizer = ActorCriticOptimizer(
        policy_model, critic_model, lr=lr, entropy_coef=entropy_coef, seed=seed,
    )
    arch = {
        "fused_width": fusion.width,
        "world_feature_width": wf_width,
        "n_actions": len(action_keys),
        "hidden_dim": 32,
        "has_world_model": False,
    }
    return fusion, action_keys, policy_model, critic_model, optimizer, arch


def _run_stage_episodes(
    stage: CurriculumStageSpec,
    policy_model, critic_model, optimizer, action_keys,
    episodes: int, seed: int, *, train: bool,
    record_dir: Optional[str], session_id: Optional[str], stage_index: int,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(
        config=stage.world_config,
        reward_config=stage.build_reward_config(),
        reward_profile=stage.build_reward_profile(),
    )
    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys,
        action_space=program.metadata().action_space, training=train, seed=seed,
    )
    learner = ActorCriticLearner(optimizer, policy, training=train)
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=stage.world_config["episode_ticks"],
        record=record_dir is not None,
        record_dir=record_dir or "sessions",
        session_id=session_id,
        program_config=stage.world_config,
        curriculum=stage.name,
        curriculum_stage_index=stage_index,
    )
    recorder = None if record_dir is not None else NullRecorder()
    return CognitiveRuntime(
        program=program, policy=policy, learner=learner,
        config=runtime_config, recorder=recorder,
    ).run()


def _merged_training_stats(checkpoint_path: str, curriculum_stats: Dict[str, Any]) -> Dict[str, Any]:
    """`training_stats["curriculum"]` merged over whatever the checkpoint's
    sidecar already carries (e.g. an evaluation-gates report), since
    :meth:`NeuralAgentCheckpoint.save` replaces `training_stats` wholesale."""
    existing: Dict[str, Any] = {}
    sidecar = checkpoint_metadata_path(checkpoint_path)
    if os.path.exists(sidecar):
        try:
            existing = dict(read_checkpoint_metadata(checkpoint_path).get("training_stats", {}))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
    existing["curriculum"] = curriculum_stats
    return existing


def run_curriculum(
    definition: CurriculumDefinition,
    *,
    checkpoint_path: str,
    model_seed: int = 1,
    train_seed: int = 100,
    eval_seed: int = 500,
    ac_lr: float = 1e-2,
    ac_entropy_coef: float = 0.05,
    start_stage: Optional[int] = None,
    force_promote: bool = False,
    fresh: bool = False,
    record_dir: Optional[str] = None,
) -> CurriculumRunResult:
    """Run (or resume) ``definition`` against ``checkpoint_path``.

    Trains each stage, evaluates its :class:`PromotionCriteria` over one or
    more attempts (bounded by the stage's ``max_attempts`` -- "no silent
    spin"), and promotes or holds. The same policy/critic/optimizer carry
    across every stage; only the world/reward config changes per stage.

    ``start_stage`` overrides where to begin (``--stage``); ``force_promote``
    promotes on the first attempt of the starting stage regardless of the
    metric (``--force-promote``, for manual experimentation). ``fresh``
    ignores any existing checkpoint and starts stage 0 with fresh weights.
    """
    fusion, action_keys, policy_model, critic_model, optimizer, arch = (
        _new_actor_critic_stack(definition.stages[0].world_config, model_seed, lr=ac_lr, entropy_coef=ac_entropy_coef)
    )
    checkpoint = NeuralAgentCheckpoint(
        checkpoint_path,
        layout_hash=fusion.layout_hash,
        action_keys=action_keys,
        policy=policy_model,
        critic=critic_model,
        online_optimizer=optimizer,
        extra_metadata={"actor_critic": arch},
    )

    resumed = False
    state = CurriculumState(definition_name=definition.name)
    if not fresh and os.path.exists(checkpoint_path):
        checkpoint.load(allow_action_space_growth=True)
        resumed = True
        curriculum_stats = checkpoint.training_stats.get("curriculum")
        if curriculum_stats:
            state = CurriculumState.from_dict(curriculum_stats)
            if state.definition_name != definition.name:
                raise CurriculumDefinitionError(
                    f"checkpoint {checkpoint_path!r} was progressing curriculum "
                    f"{state.definition_name!r}, not {definition.name!r}; pass --fresh "
                    "to start this curriculum over with a new checkpoint"
                )

    if start_stage is not None:
        if not 0 <= start_stage < len(definition.stages):
            raise ValueError(
                f"--stage {start_stage} out of range for curriculum "
                f"{definition.name!r} ({len(definition.stages)} stages)"
            )
        state.stage_index = start_stage
        state.attempts_at_stage = 0
        state.completed = False
        state.held = False
        state.hold_reason = None

    def _save(reason: str) -> None:
        checkpoint.training_ticks = optimizer.step_count
        checkpoint.training_stats = _merged_training_stats(checkpoint_path, state.to_dict())
        checkpoint.save(reason=reason)

    if state.completed:
        return CurriculumRunResult(status="completed", state=state, resumed=resumed)

    force_this_attempt = force_promote
    while state.stage_index < len(definition.stages):
        stage = definition.stages[state.stage_index]
        while True:
            attempt = state.attempts_at_stage
            train_seed_i = _seed_for(train_seed, state.stage_index, attempt, stage.train_episodes)
            eval_seed_i = _seed_for(eval_seed, state.stage_index, attempt, stage.promotion.sample_size)

            _run_stage_episodes(
                stage, policy_model, critic_model, optimizer, action_keys,
                stage.train_episodes, train_seed_i, train=True,
                record_dir=record_dir, session_id=None, stage_index=state.stage_index,
            )
            eval_episodes = _run_stage_episodes(
                stage, policy_model, critic_model, optimizer, action_keys,
                stage.promotion.sample_size, eval_seed_i, train=False,
                record_dir=record_dir, session_id=None, stage_index=state.stage_index,
            )
            summary = EvaluationSummary.from_episodes(stage.name, eval_episodes)
            value = stage.promotion.value_of(summary)
            met_criteria = stage.promotion.evaluate(summary)
            state.attempts_at_stage += 1
            forced = force_this_attempt and not met_criteria
            promoted = met_criteria or force_this_attempt
            force_this_attempt = False

            state.history.append({
                "stage": stage.name,
                "stage_index": state.stage_index,
                "attempt": state.attempts_at_stage,
                "metric": stage.promotion.metric,
                "threshold": stage.promotion.threshold,
                "value": value,
                "promoted": promoted,
                "forced": forced,
            })

            if promoted:
                state.held = False
                state.hold_reason = None
                state.stage_index += 1
                state.attempts_at_stage = 0
                _save(reason=f"curriculum-promote:{stage.name}")
                break

            if state.attempts_at_stage >= stage.max_attempts:
                state.held = True
                state.hold_reason = (
                    f"stage {stage.name!r} did not meet promotion criteria after "
                    f"{state.attempts_at_stage} attempt(s): {stage.promotion.metric}="
                    f"{value!r} < threshold={stage.promotion.threshold!r}"
                )
                _save(reason=f"curriculum-hold:{stage.name}")
                return CurriculumRunResult(status="held", state=state, resumed=resumed)

            _save(reason=f"curriculum-attempt:{stage.name}")

    state.completed = True
    _save(reason="curriculum-complete")
    return CurriculumRunResult(status="completed", state=state, resumed=resumed)
