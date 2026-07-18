"""Development runner: train/evaluate/promote a checkpoint through an ordered
list of staged goals, unattended (issue #43, generalised into ``development/``
by issue #104).

Extends the world/config presets (issue #30, ``programs/minecraft/curriculum.py``)
into an unattended orchestrator: an ordered list of *stages* (world config +
reward config/profile (issue #41) + promotion criteria), where the runner
trains a stage, evaluates it over N episodes, and either promotes to the
next stage (its gate(s) passed) or holds the stage and logs why. Legacy
stages (no declared ``motor_freedom``, issue #43's original shape) train and
carry an actor/critic stack -- the same policy/critic/optimizer weights --
across stage boundaries; only the world/reward config changes. A stage that
declares a ``motor_freedom`` (issue #104/#105's ladder) instead runs under
that freedom (frozen / caregiver-scripted-overridden / voluntary-learned,
see ``run_curriculum``'s docstring) and does not train the actor/critic
stack, which those stages never act through.

Promotion here still gates on the plain mean of one or more summary metrics
over a fixed sample size (:class:`~development.definitions.PromotionCriteria`)
-- deliberately the simplest thing that is still an N-episode statistical
aggregate, not a single-episode fluke. Issue #44's richer statistical harness
(:mod:`cognitive_runtime.training.statistical_evaluation`: confidence
intervals across survival/reward-by-tier/coverage/prediction-error, with
regression/improvement flagging) has since landed and can inspect a
curriculum run's recorded sessions directly
(``statistical-evaluate --from-sessions``), but this module's own promotion
gate intentionally keeps the simple mean criterion rather than growing a
second, harder-to-predict gating rule.

Phase 7 (issue #104) generalises promotion beyond that one scalar: a stage
may instead declare one or more milestone ``gates`` (its ``World``/senses/
motor-freedom/losses live on the same :class:`~development.definitions.CurriculumStageSpec`,
declaratively). When a stage has ``gates``, *every* gate must pass -- wired
through the ``milestone_metrics`` hook below, which lets a caller supply the
Phase 2-6 milestone computation (action-ablation, forgetting, reflex-override,
...) alongside the plain evaluation-episode metrics this module already
computes. A stage with no ``gates`` keeps the pre-Phase-7 single-``promotion``
behaviour unchanged, which is how old curriculum definitions still run
through the shim.

Curriculum state (current stage, attempt count, promotion history) lives in
the checkpoint bundle's ``training_stats["curriculum"]`` (issue #20), so
interrupting and restarting the runner resumes at the correct stage from the
same checkpoint. ``training_stats`` is a whole-dict replace on
:meth:`NeuralAgentCheckpoint.save`, so this module always reads the existing
sidecar first and merges its own key in, rather than clobbering whatever else
(e.g. evaluation-gate reports) a checkpoint already carries.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.learner import Learner, NullLearner
from cognitive_runtime.core.policy import Policy
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
from cognitive_runtime.policies.random_policy import RandomPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary
from development.definitions import (
    CurriculumDefinition,
    CurriculumDefinitionError,
    CurriculumRunResult,
    CurriculumStageSpec,
    CurriculumState,
    sense_stream_mask,
)
from motor.organism_policy import build_stage_policy
from motor.voluntary import VoluntaryController

#: Optional per-run factory for a ``"learned"`` stage's voluntary controller
#: (``run_curriculum``'s ``voluntary_controller`` hook): given the stage and
#: its action space, return the Phase 6 controller to hand control to. There
#: is no generic default -- unlike ``"overridden"`` (any scripted/caregiver
#: policy satisfies the freedom), ``"learned"`` specifically means the real
#: voluntary path (MPC-over-cortex or an alternative controller, issue
#: #103), which needs a trained predictive cortex the runner does not itself
#: build. A stage that declares ``"learned"`` without one is a wiring bug,
#: not something to default around (see ``build_stage_policy``'s own
#: docstring) -- so it raises rather than silently falling back to the
#: actor/critic loop.
VoluntaryControllerFactory = Callable[[CurriculumStageSpec, Sequence[Action]], VoluntaryController]

#: Type of the optional per-stage milestone-metrics provider: given the
#: stage and its plain evaluation summary, return the *additional* metrics
#: (e.g. ``{"cortex_beats_copy_last": 1.2}``) needed by that stage's
#: ``gates``. Wiring the real Phase 2-6 computations behind this hook is
#: issue #105's job (encoding the actual ladder); this module only needs the
#: seam.
MilestoneMetricsFn = Callable[[CurriculumStageSpec, EvaluationSummary], Mapping[str, float]]


def _seed_for(base: int, stage_index: int, attempt: int, unit: int) -> int:
    """Deterministic, non-colliding seed per (stage, attempt) -- each retry
    sees fresh episode content instead of replaying an identical eval set."""
    return base + stage_index * 1_000_000 + attempt * max(unit, 1)


def _program_for_stage(stage: CurriculumStageSpec):
    """Construct the ``Program`` a stage trains/evaluates against, honoring
    ``stage.world`` (issue #105: the pre-#105 runner hardcoded
    ``MinecraftSurvivalBox`` here and silently ignored a Crafter-world
    stage's ``world``/``scenario``/``motor_freedom`` declarations). Mirrors
    ``cli.py``'s ``--world`` factory and
    ``development.definitions._program_for_layout``, which computes the same
    Program (minus reward wiring, Minecraft-only) for the layout-hash check
    at definition-load time -- kept as a separate copy here rather than
    imported, since this one also wires up reward config/profile."""
    if stage.world in (None, "minecraft"):
        return MinecraftSurvivalBox(
            config=stage.world_config,
            reward_config=stage.build_reward_config(),
            reward_profile=stage.build_reward_profile(),
        )
    if stage.world == "crafter":
        from cognitive_runtime.programs.crafter.adapter import CrafterWorld

        return CrafterWorld(config=stage.world_config)
    raise AssertionError(f"unreachable: unknown world {stage.world!r} (validated in __post_init__)")


def _new_actor_critic_stack(stage: CurriculumStageSpec, seed: int, *, lr: float, entropy_coef: float):
    program = _program_for_stage(stage)
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


def _scripted_policy_for_stage(stage: CurriculumStageSpec, action_space: Sequence[Action], *, seed: int):
    """The scripted motor policy an ``"overridden"`` stage's episodes
    actually run under: the *same* scripted policy ``stage.scenario``'s
    nursery recording uses (``cognitive_runtime.training.nursery``'s
    ``CRAFTER_SCENARIOS``/``NURSERY_SCENARIOS`` registries), e.g. Babbling's
    ``turn`` cycles the four directional actions and Crawling's
    ``walk_forward`` walks a constant direction -- not a uniform-random
    substitute. Without this, the runner could record unrelated random
    experience while ``milestone_metrics`` promotes the stage against
    separately-generated, scenario-correct nursery data (issue #133 review),
    letting a stage pass without the organism ever having run its declared
    developmental task.

    Falls back to a stage-agnostic :class:`RandomPolicy` only when the stage
    has no scenario registered against a nursery world (e.g. a non-ladder
    curriculum that declares ``motor_freedom="overridden"`` without a
    nursery scenario) -- some scripted/caregiver-driven policy still
    satisfies the freedom generically.
    """
    if stage.scenario is not None:
        from cognitive_runtime.training.nursery import (
            CRAFTER_SCENARIOS,
            NURSERY_SCENARIOS,
            NurseryConfig,
        )

        world = stage.world or "minecraft"
        registry = CRAFTER_SCENARIOS if world == "crafter" else NURSERY_SCENARIOS
        scenario = registry.get(stage.scenario)
        if scenario is not None:
            cfg = NurseryConfig(
                episode_ticks=stage.world_config.get("episode_ticks", NurseryConfig.episode_ticks),
                world=world, seed=seed,
            )
            return scenario.build(seed, cfg).policy
    return RandomPolicy(list(action_space), seed=seed)


def _stage_policy_and_learner(
    stage: CurriculumStageSpec,
    action_space: Sequence[Action],
    policy_model, critic_model, optimizer, action_keys,
    *, train: bool, seed: int,
    voluntary_controller: Optional[VoluntaryControllerFactory],
) -> tuple[Policy, Learner]:
    """The ``Policy``/``Learner`` pair a stage's episodes actually run
    against (issue #133 bug fix): stages that declare a ``motor_freedom``
    (Phase 7) run under :func:`build_stage_policy`, not the actor/critic
    loop -- Gestation must genuinely freeze, Babbling/Crawling must be
    caregiver/scripted-driven, and none of the three trains the actor/critic
    stack (there is nothing for it to learn from: no action came from its
    own logits). Legacy stages (``motor_freedom is None``) keep the
    pre-Phase-7 behaviour unchanged.
    """
    if stage.motor_freedom is None:
        policy = ActorCriticPolicy(
            policy_model, critic_model, action_keys,
            action_space=action_space, training=train, seed=seed,
        )
        return policy, ActorCriticLearner(optimizer, policy, training=train)

    if stage.motor_freedom == "frozen":
        return build_stage_policy(stage, action_space), NullLearner()

    if stage.motor_freedom == "overridden":
        scripted = _scripted_policy_for_stage(stage, action_space, seed=seed)
        return build_stage_policy(stage, action_space, scripted=scripted), NullLearner()

    assert stage.motor_freedom == "learned"
    if voluntary_controller is None:
        raise CurriculumDefinitionError(
            f"stage {stage.name!r} declares motor_freedom='learned', but no "
            "voluntary_controller factory was passed to run_curriculum() -- "
            "development.runner does not build a real Phase 6 voluntary "
            "controller on its own (that needs a trained predictive cortex); "
            "pass one via run_curriculum(..., voluntary_controller=...) "
            "rather than silently falling back to the actor/critic loop"
        )
    voluntary = voluntary_controller(stage, action_space)
    return build_stage_policy(stage, action_space, voluntary=voluntary), NullLearner()


def _run_stage_episodes(
    stage: CurriculumStageSpec,
    policy_model, critic_model, optimizer, action_keys,
    episodes: int, seed: int, *, train: bool,
    record_dir: Optional[str], session_id: Optional[str], stage_index: int,
    name: Optional[str] = None,
    voluntary_controller: Optional[VoluntaryControllerFactory] = None,
) -> List[EpisodeSummary]:
    program = _program_for_stage(stage)
    policy, learner = _stage_policy_and_learner(
        stage, program.metadata().action_space,
        policy_model, critic_model, optimizer, action_keys,
        train=train, seed=seed, voluntary_controller=voluntary_controller,
    )
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
        name=name,
        # issue #135: a stage's declared `senses` used to be a label nobody
        # consulted -- every stage fused the full stream catalog regardless.
        # Zeroing the streams outside the declared senses (fixed layout/width,
        # so the checkpoint carried across stages, task 3, is unaffected)
        # makes changing `senses` actually change what the organism runs on.
        sense_stream_weights=sense_stream_mask(stage.senses, program.stream_catalog()) or None,
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


def _summary_metrics(summary: EvaluationSummary) -> Dict[str, float]:
    """The legacy metrics :class:`PromotionCriteria` already knew how to read
    off an :class:`EvaluationSummary`, as a plain mapping -- the base a
    stage's ``milestone_metrics`` provider extends with Phase 2-6 metrics."""
    reasons = summary.termination_reasons
    survival_rate = 0.0
    if reasons:
        survived = sum(1 for r in reasons if not r.startswith("death"))
        survival_rate = survived / len(reasons)
    return {
        "average_reward": summary.average_reward,
        "average_ticks": summary.average_ticks,
        "total_reward": summary.total_reward,
        "total_ticks": summary.total_ticks,
        "survival_rate": survival_rate,
    }


def _eval_sample_size(stage: CurriculumStageSpec) -> int:
    """How many eval episodes an attempt runs to satisfy this stage's
    promotion gate(s) (issue #138: the runner used to always take this from
    ``stage.promotion.sample_size``, ignoring milestone ``gates``' own
    ``sample_size`` entirely -- a gate declared as an N-episode aggregate
    silently promoted from whatever the legacy ``promotion`` field happened
    to carry instead).

    One evaluation pass produces one :class:`EvaluationSummary`/metrics
    mapping shared by every gate a stage declares, so there is one sample
    size per attempt, not one per gate -- the largest ``sample_size`` any
    gate requires, so each gate's own floor is met (running a few extra
    episodes for a smaller-sample-size gate only strengthens its aggregate).
    Stages with no ``gates`` keep reading the legacy ``promotion.sample_size``
    unchanged.
    """
    if stage.gates:
        return max(gate.sample_size for gate in stage.gates)
    return stage.promotion.sample_size


def _eval_seed_stride(stage: CurriculumStageSpec) -> int:
    """The ``unit`` :func:`_seed_for` spaces attempts by for this stage's
    eval pass -- at least ``stage.promotion.sample_size`` even when
    :func:`_eval_sample_size` (the actual episode count) is smaller (PR #159
    review): a checkpoint progressed under the pre-#138 runner has already
    consumed seeds in contiguous ``promotion.sample_size``-wide blocks per
    attempt (that was the *only* stride the old code ever used). Shrinking
    the stride to match a smaller gate ``sample_size`` after upgrading would
    let a resumed attempt replay a seed an earlier attempt already evaluated
    on, violating :func:`_seed_for`'s non-colliding-retry contract. Since the
    new stride is always >= the old one, each attempt's new block starts at
    or past where every prior attempt's old-stride block ended, regardless
    of how many attempts ran before the upgrade.
    """
    return max(_eval_sample_size(stage), stage.promotion.sample_size)


def _evaluate_stage(
    stage: CurriculumStageSpec,
    summary: EvaluationSummary,
    milestone_metrics: Optional[MilestoneMetricsFn],
):
    """Evaluate one stage's promotion gate(s) against its eval episodes.

    Returns ``(met_criteria, value, threshold, metric)`` where ``value``/
    ``threshold``/``metric`` are scalars for the legacy single-``promotion``
    path, or dicts keyed by metric name when the stage declares milestone
    ``gates`` (Phase 7: "not a single scalar").
    """
    if not stage.gates:
        value = stage.promotion.value_of(summary)
        met_criteria = stage.promotion.evaluate(summary)
        return met_criteria, value, stage.promotion.threshold, stage.promotion.metric

    metrics = _summary_metrics(summary)
    if milestone_metrics is not None:
        extra = milestone_metrics(stage, summary)
        overlap = set(extra) & set(metrics)
        if overlap:
            raise CurriculumDefinitionError(
                f"stage {stage.name!r}: milestone_metrics provider returned "
                f"metric(s) {sorted(overlap)} that collide with the built-in "
                "evaluation metrics"
            )
        metrics.update(extra)

    missing = [gate.metric for gate in stage.gates if gate.metric not in metrics]
    if missing:
        raise CurriculumDefinitionError(
            f"stage {stage.name!r}: milestone gate metric(s) {missing} were not "
            f"computed (have {sorted(metrics)}); pass a milestone_metrics "
            "provider that returns them"
        )
    gate_results = stage.evaluate_gates(metrics)
    met_criteria = all(gate_results.values())
    value = {metric: metrics[metric] for metric in gate_results}
    threshold = {gate.metric: gate.threshold for gate in stage.gates}
    metric = list(gate_results)
    return met_criteria, value, threshold, metric


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
    name: Optional[str] = None,
    milestone_metrics: Optional[MilestoneMetricsFn] = None,
    voluntary_controller: Optional[VoluntaryControllerFactory] = None,
    world_model_checkpoint_paths: Sequence[str] = (),
) -> CurriculumRunResult:
    """Run (or resume) ``definition`` against ``checkpoint_path``.

    Trains each stage, evaluates its promotion gate(s) over one or more
    attempts (bounded by the stage's ``max_attempts`` -- "no silent spin"),
    and promotes or holds.

    A stage with no declared ``motor_freedom`` (the pre-Phase-7 shape) keeps
    the legacy behaviour: the same actor/critic policy/critic/optimizer
    carry across every stage and learn online, only the world/reward config
    changing per stage. A stage that *does* declare a ``motor_freedom``
    (issue #105's ladder) instead runs under
    :func:`motor.organism_policy.build_stage_policy` -- frozen, caregiver/
    scripted-overridden, or (given ``voluntary_controller``) the Phase 6
    voluntary path -- and does not train the actor/critic stack, which those
    stages never act through (issue #133: ``_run_stage_episodes`` used to
    ignore ``motor_freedom`` entirely and always ran the actor/critic loop).

    ``start_stage`` overrides where to begin (``--stage``); ``force_promote``
    promotes on the first attempt of the starting stage regardless of the
    metric (``--force-promote``, for manual experimentation). ``fresh``
    ignores any existing checkpoint and starts stage 0 with fresh weights.

    ``milestone_metrics`` (issue #104) is called once per attempt for any
    stage that declares milestone ``gates``: it receives the stage and its
    plain :class:`EvaluationSummary` and returns the additional metrics (e.g.
    ``cortex_beats_copy_last``) those gates reference. Stages with no
    ``gates`` ignore it and keep the legacy single-``promotion`` behaviour.

    ``voluntary_controller`` (issue #133) is called once per attempt for any
    stage declaring ``motor_freedom="learned"``: given the stage and its
    action space, it must return the :class:`~motor.voluntary.VoluntaryController`
    that stage hands control to. There is no generic default (unlike
    ``"overridden"``, "learned" specifically means the real Phase 6 voluntary
    path, which needs a trained predictive cortex this module does not build
    on its own) -- a ``"learned"`` stage run without one raises rather than
    silently defaulting to the actor/critic loop.

    ``world_model_checkpoint_paths`` (issue #134) declares where a caller's
    ``milestone_metrics`` provider persists any predictive-cortex checkpoint
    it trains against (e.g.
    ``development.ladder.ladder_cortex_checkpoint_paths(checkpoint_path).values()``)
    -- purely for this checkpoint's own provenance. Before Phase 7, a
    milestone gate could genuinely train and persist a world model
    elsewhere with no trace of that in this checkpoint's own metadata at
    all -- the gate's model lived entirely outside it. After every attempt,
    any of these paths that now exist on disk are recorded under
    ``extra_metadata["ladder_world_model_checkpoints"]``, so this
    checkpoint's own metadata honestly reflects whether (and where) a world
    model backs the milestones it was promoted on. This is deliberately a
    *separate* field from ``extra_metadata["actor_critic"]["has_world_model"]``
    -- that flag means this checkpoint itself embeds an ``MLPWorldModel``
    (``cli.py``/``sleep/async_trainer.py`` construct one and load its
    optimizer state whenever it's set), which a ladder run's external,
    differently-shaped cortex checkpoint is not.
    """
    fusion, action_keys, policy_model, critic_model, optimizer, arch = (
        _new_actor_critic_stack(definition.stages[0], model_seed, lr=ac_lr, entropy_coef=ac_entropy_coef)
    )
    checkpoint = NeuralAgentCheckpoint(
        checkpoint_path,
        layout_hash=fusion.layout_hash,
        action_keys=action_keys,
        policy=policy_model,
        critic=critic_model,
        online_optimizer=optimizer,
        extra_metadata={"actor_critic": arch},
        name=name,
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
            eval_sample_size = _eval_sample_size(stage)
            eval_seed_i = _seed_for(eval_seed, state.stage_index, attempt, _eval_seed_stride(stage))

            _run_stage_episodes(
                stage, policy_model, critic_model, optimizer, action_keys,
                stage.train_episodes, train_seed_i, train=True,
                record_dir=record_dir, session_id=None, stage_index=state.stage_index,
                name=name, voluntary_controller=voluntary_controller,
            )
            eval_episodes = _run_stage_episodes(
                stage, policy_model, critic_model, optimizer, action_keys,
                eval_sample_size, eval_seed_i, train=False,
                record_dir=record_dir, session_id=None, stage_index=state.stage_index,
                name=name, voluntary_controller=voluntary_controller,
            )
            summary = EvaluationSummary.from_episodes(stage.name, eval_episodes)
            met_criteria, value, threshold, metric = _evaluate_stage(stage, summary, milestone_metrics)
            if world_model_checkpoint_paths:
                # A *separate* field from arch["has_world_model"] (issue
                # #134 review): that flag means this checkpoint embeds an
                # MLPWorldModel, which an external ladder cortex checkpoint
                # is not -- conflating the two would make cli.py/
                # sleep/async_trainer.py try to construct and load one that
                # was never actually saved here.
                checkpoint.extra_metadata["ladder_world_model_checkpoints"] = sorted(
                    p for p in world_model_checkpoint_paths if os.path.exists(p)
                )
            state.attempts_at_stage += 1
            forced = force_this_attempt and not met_criteria
            promoted = met_criteria or force_this_attempt
            force_this_attempt = False

            state.history.append({
                "stage": stage.name,
                "stage_index": state.stage_index,
                "attempt": state.attempts_at_stage,
                "metric": metric,
                "threshold": threshold,
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
                    f"{state.attempts_at_stage} attempt(s): {metric}="
                    f"{value!r} < threshold={threshold!r}"
                )
                _save(reason=f"curriculum-hold:{stage.name}")
                return CurriculumRunResult(status="held", state=state, resumed=resumed)

            _save(reason=f"curriculum-attempt:{stage.name}")

    state.completed = True
    _save(reason="curriculum-complete")
    return CurriculumRunResult(status="completed", state=state, resumed=resumed)
