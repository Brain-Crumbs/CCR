"""The Gestation -> Foraging ladder (issue #105): the concrete
:class:`~development.definitions.CurriculumDefinition` for
``docs/v2/phases/phase-7-development-ladder.md``'s five-stage table, plus the
:data:`~development.runner.MilestoneMetricsFn` that wires each stage's
``gates`` to a *real* Phase 2/6 computation instead of a stub.

Speaking (the table's 6th, deferred stage) is intentionally absent -- see the
phase doc's "Risks / notes".

Torch-free at import time, like :mod:`development.definitions`:
:data:`GESTATION_TO_FORAGING` is built from plain
:class:`~development.definitions.CurriculumStageSpec` dataclasses, which
needs neither ``torch`` nor the ``crafter`` extra installed. Only calling
:func:`ladder_milestone_metrics` (which trains/evaluates real nursery
scenarios via :mod:`cognitive_runtime.training.nursery`) pulls those in, and
it does so lazily, function-local, so a bare ``import development.ladder``
stays dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from development.definitions import (
    CurriculumDefinition,
    CurriculumDefinitionError,
    CurriculumStageSpec,
    PromotionCriteria,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from cognitive_runtime.training.online_q_acceptance import EvaluationSummary

#: Every stage shares this world config so the ladder's stream layout (and
#: therefore the one checkpoint carried across every stage, task 3) is
#: identical from Gestation through Foraging -- see
#: ``development.definitions._validate_shared_layout``. Only ``scenario``
#: and the declarative ``senses``/``motor_freedom``/``losses``/``gates``
#: fields change per stage; ``episode_ticks`` is intentionally small
#: (CI-runnable, Milestone 7's "unattended run ... resumes cleanly" is meant
#: to be exercised in CI, not just by a human at production scale).
_LADDER_WORLD_CONFIG: Dict[str, Any] = {"episode_ticks": 40}

#: Shared per-attempt sizing: one training episode and one evaluation
#: episode per attempt keeps a real, unattended `run_curriculum` pass over
#: several stages fast; `max_attempts` bounds "no silent spin" (task 4's
#: hold-on-failing-gate contract) without needing to be large -- a stage
#: either clears its milestone gate(s) within a few attempts or genuinely
#: holds, which is the point.
_TRAIN_EPISODES = 1
_EVAL_SAMPLE_SIZE = 1
_MAX_ATTEMPTS = 3


def _stage(
    name: str, scenario: str, senses, motor_freedom: str, losses, gates,
) -> CurriculumStageSpec:
    return CurriculumStageSpec(
        name=name,
        world="crafter",
        world_config=dict(_LADDER_WORLD_CONFIG),
        scenario=scenario,
        senses=senses,
        motor_freedom=motor_freedom,
        losses=losses,
        train_episodes=_TRAIN_EPISODES,
        promotion=PromotionCriteria(metric="average_ticks", threshold=0.0, sample_size=_EVAL_SAMPLE_SIZE),
        max_attempts=_MAX_ATTEMPTS,
        gates=gates,
    )


#: Gestation: "see & hear, habituate" -- a frozen-motor stage, so its
#: recording policy must itself be passive. ``object_permanence`` is the one
#: Crafter scenario recorded with ``NullPolicy`` (see
#: ``cognitive_runtime.training.nursery.CRAFTER_SCENARIOS``), matching a
#: frozen organism exactly: it never emits a motor command, it only watches.
#:
#: Gate: ``cortex_beats_copy_last``. ``ego_motion_canary.evaluate_ego_motion_
#: holdout`` reports a *per-horizon* bool (``report[h]["beats_copy_last"]``);
#: ``ladder_milestone_metrics`` reduces that to ``1.0`` only when the model
#: beats copy-last at *every* evaluated horizon, else ``0.0`` -- so a
#: threshold of ``1.0`` means "no horizon regresses to memorizing the last
#: frame", not "beats it on average".
_GESTATION = _stage(
    "gestation", "object_permanence",
    senses=("vision",),
    motor_freedom="frozen",
    losses=("prediction",),
    gates=(PromotionCriteria(metric="cortex_beats_copy_last", threshold=1.0, sample_size=_EVAL_SAMPLE_SIZE),),
)

#: Babbling: "its own body: action->sensory change". ``turn`` is Crafter's
#: simplest action->sensory regularity (boxed in on all sides, every move
#: blocked -- only the discrete ``facing`` changes), a better fit for a
#: caregiver/scripted "babbling" stage than ``walk_forward``'s locomotion,
#: which this ladder reserves for Crawling.
#:
#: Gate: ``action_ablation_margin``, the Milestone-2 action-conditioning
#: ablation gap (``nursery.run_action_ablation_eval``): mean held-out MSE
#: *without* the action stream minus mean MSE *with* it, averaged over the
#: evaluated horizons. Positive means action-conditioning measurably helps
#: prediction -- exactly "learned action->sensory change". The threshold is
#: a small positive epsilon (not ``0.0``): a margin that is merely
#: non-negative can be measurement noise around "the action stream does
#: nothing", which is not evidence of babbling having learned anything.
_BABBLING = _stage(
    "babbling", "turn",
    senses=("vision", "proprioception"),
    motor_freedom="overridden",
    losses=("prediction", "action_conditioning"),
    gates=(PromotionCriteria(metric="action_ablation_margin", threshold=1e-4, sample_size=_EVAL_SAMPLE_SIZE),),
)

#: Crawling: "moving changes the view predictably (walk_forward, discrete
#: turn)" -- the phase doc names both scenarios for this stage; ``scenario``
#: only holds one, so ``walk_forward`` is the stage's *recorded* scenario
#: (locomotion, the harder of the two) while both of its gates are the same
#: two milestones Babbling/Gestation each proved individually, now required
#: together on the harder scenario ("not a single scalar" -- task 4).
_CRAWLING = _stage(
    "crawling", "walk_forward",
    senses=("vision", "proprioception"),
    motor_freedom="overridden",
    losses=("prediction", "action_conditioning"),
    gates=(
        PromotionCriteria(metric="cortex_beats_copy_last", threshold=1.0, sample_size=_EVAL_SAMPLE_SIZE),
        PromotionCriteria(metric="action_ablation_margin", threshold=1e-4, sample_size=_EVAL_SAMPLE_SIZE),
    ),
)

#: Objects: "permanence, affordances, approach & scale" -- ``approach_entity``
#: (scale-with-distance) is Crafter's closest scenario to "affordances";
#: object permanence itself is already Gestation's recorded scenario, so
#: this stage's *new* content is approach.
#:
#: Gate: ``reflex_override_precedence`` rather than ``forgetting_score``.
#: The phase doc's own example list ("later stages on the forgetting
#: metric, reflex-override behaviour, etc.") offers both, but issue #99's
#: tracking comment records that the forgetting metric has no real,
#: computable implementation in this repo yet (no
#: ``test_generative_replay.py``/``test_forgetting_metric.py``) -- wiring a
#: gate to a metric nothing computes would violate task 4's "not a stub"
#: intent worse than picking a different real Phase 6 milestone. Objects is
#: the ladder's first ``learned``-motor stage with caregiver-guided stages
#: behind it, so proving the caregiver-override precedence contract
#: (``motor.reflexes.ReflexStack``: caregiver always wins) still holds is a
#: real, honestly-computable stand-in.
_OBJECTS = _stage(
    "objects", "approach_entity",
    senses=("vision", "proprioception"),
    motor_freedom="learned",
    losses=("prediction", "action_conditioning"),
    gates=(PromotionCriteria(metric="reflex_override_precedence", threshold=1.0, sample_size=_EVAL_SAMPLE_SIZE),),
)

#: Foraging: "goal-directed reward-seeking" -- also ``approach_entity``, the
#: closest thing to goal-directed approach in the Crafter scenario registry
#: (there is no dedicated foraging/food scenario yet); what changes from
#: Objects is motor freedom (fully ``learned``, no caregiver anywhere in the
#: ladder from here on) and the gate.
#:
#: Gate: ``reflex_activation_rate`` -- but the raw metric is *lower-is-
#: better* (a maturing organism relies less on reflex withdrawal, see
#: ``tests/test_reflexes.py::test_reflex_activation_rate_falls_across_
#: development_on_locomotion_and_threat_scenario``), while
#: ``PromotionCriteria.evaluate`` is always ``value >= threshold``. To keep
#: that polarity honest without a second metric name,
#: ``ladder_milestone_metrics`` stores ``1.0 - reflex_activation_rate``
#: under this same key -- a "voluntary-reliance score" where higher is
#: better -- and the threshold (``0.85``) requires the *raw* rate to sit at
#: or below ``0.15``, the same ceiling that test asserts a matured session
#: settles under.
_FORAGING = _stage(
    "foraging", "approach_entity",
    senses=("vision", "proprioception"),
    motor_freedom="learned",
    losses=("prediction", "action_conditioning"),
    gates=(PromotionCriteria(metric="reflex_activation_rate", threshold=0.85, sample_size=_EVAL_SAMPLE_SIZE),),
)

#: The five-stage Gestation->Foraging ladder (Speaking deferred). One named
#: organism's checkpoint is meant to walk this end to end via
#: ``development.runner.run_curriculum(GESTATION_TO_FORAGING, ...,
#: milestone_metrics=ladder_milestone_metrics)``.
GESTATION_TO_FORAGING = CurriculumDefinition(
    name="gestation-to-foraging",
    stages=(_GESTATION, _BABBLING, _CRAWLING, _OBJECTS, _FORAGING),
)


def _ladder_nursery_config(**overrides: Any) -> Any:
    """CI-fast ``NurseryConfig`` (mirrors ``tests/test_crafter_scenarios.py``'s
    ``_crafter_config``/``tests/test_predictive_cortex.py``'s
    ``_small_nursery_config``): small enough to record+train+evaluate in a
    few seconds, not tuned to reproduce Milestone 2's full effect size (see
    ``tests/test_nursery.py``'s note that ``beats_copy_last`` is a
    training-budget-scale property no fast unit test should hard-assert)."""
    from cognitive_runtime.training.nursery import NurseryConfig

    base: Dict[str, Any] = dict(
        world="crafter", episode_ticks=40, train_seeds=(0, 1), holdout_seeds=(1000,),
        horizons=(1,), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=4, consistency_epochs=1, batch_size=16,
    )
    base.update(overrides)
    return NurseryConfig(**base)


def _ladder_model_config(**overrides: Any) -> Any:
    """CI-fast ``ActionWorldModelConfig`` for the action-ablation gate."""
    from cognitive_runtime.training.action_world_model import ActionWorldModelConfig

    base: Dict[str, Any] = dict(
        latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=4, batch_size=16, warmup_frames=2, rollout_frames=3,
    )
    base.update(overrides)
    return ActionWorldModelConfig(**base)


def _require_losses(stage: CurriculumStageSpec, *required: str) -> None:
    """A milestone metric below trains/evaluates a cortex under a specific
    loss objective -- computing it for a stage that doesn't declare that
    loss active would promote/hold the stage on a metric its own ``losses``
    never asked for (issue #135: changing a stage's ``losses`` used to have
    no effect on anything). Raises *before* the lazy torch/nursery import
    each caller makes next, so a mis-declared curriculum fails fast and
    torch-free rather than silently computing a metric its losses disclaim.
    """
    missing = [loss for loss in required if loss not in stage.losses]
    if missing:
        raise CurriculumDefinitionError(
            f"stage {stage.name!r}: computing this gate's metric needs "
            f"{missing} declared in the stage's losses (have {list(stage.losses)})"
        )


def _cortex_beats_copy_last(
    stage: CurriculumStageSpec, record_dir: str, *, cortex_checkpoint_path: Optional[str] = None,
) -> float:
    """``1.0`` iff the stage's nursery scenario's held-out next-frame
    prediction beats the copy-last-frame baseline at *every* evaluated
    horizon, else ``0.0`` (see :data:`_GESTATION`'s docstring for why this
    encoding, not the raw per-horizon bool mapping).

    ``cortex_checkpoint_path`` (issue #134), when given, warm-starts this
    call from (and saves back to) that path instead of training a fresh,
    disposable model every attempt -- see ``run_nursery_scenario``'s own
    docstring.
    """
    _require_losses(stage, "prediction")
    from cognitive_runtime.training.nursery import run_nursery_scenario

    assert stage.scenario is not None
    _model, report = run_nursery_scenario(
        record_dir, stage.scenario, config=_ladder_nursery_config(world=stage.world or "crafter"),
        cortex_checkpoint_path=cortex_checkpoint_path,
    )
    beats = [entry["beats_copy_last"] for entry in report.horizon_metrics.values()]
    return 1.0 if beats and all(beats) else 0.0


def _action_ablation_margin(
    stage: CurriculumStageSpec, record_dir: str, *, cortex_checkpoint_path: Optional[str] = None,
) -> float:
    """Mean held-out MSE degradation from withholding the action stream
    during training, averaged over the evaluated horizons -- positive means
    action-conditioning measurably helps (see :data:`_BABBLING`'s
    docstring).

    ``cortex_checkpoint_path`` (issue #134), when given, warm-starts the
    with-actions cortex from (and saves it back to) that path instead of
    training a fresh, disposable model every attempt -- see
    ``run_action_ablation_eval``'s own docstring, which also warm-starts the
    without-actions control from its own sibling path so both runs keep an
    equal accumulated training budget across attempts (PR #155 review).
    """
    _require_losses(stage, "prediction", "action_conditioning")
    from cognitive_runtime.training.nursery import run_action_ablation_eval

    assert stage.scenario is not None
    report = run_action_ablation_eval(
        record_dir,
        train_scenarios=[stage.scenario],
        eval_scenario=stage.scenario,
        config=_ladder_nursery_config(world=stage.world or "crafter"),
        model_config=_ladder_model_config(),
        cortex_checkpoint_path=cortex_checkpoint_path,
    )
    margins = [
        report.without_actions_stats[h].mean - report.with_actions_stats[h].mean
        for h in report.with_actions_stats
    ]
    return sum(margins) / len(margins) if margins else 0.0


def _reflex_override_precedence() -> float:
    """Structural proof, not a measured trend: ``ReflexStack``'s contract
    (``motor/reflexes.py``: ``caregiver > priority reflex > voluntary``)
    means an injected caregiver override always wins. This runs a short
    scripted session that injects an override on alternating ticks while a
    reflex-triggering stimulus is present every tick (so a passing result
    proves the override beat a live reflex, not merely an idle stack), and
    returns the fraction of overridden ticks where the actuated action was
    indeed the injected one -- always ``1.0`` by contract; the value exists
    to prove the contract holds for real, not because it varies."""
    from cognitive_runtime.core.action import Action
    from motor.reflexes import CaregiverChannel, ReflexConfig, ReflexStack, Stimulus

    reflexes = ReflexStack([ReflexConfig("withdraw", "threat", Action("BACK"), threshold=0.5, priority=10)])
    channel = CaregiverChannel()
    overridden_ticks = 0
    correct_ticks = 0
    for tick in range(20):
        stimuli = [Stimulus("threat", 1.0)]  # would otherwise fire `withdraw` every tick
        if tick % 2 == 0:
            channel.inject(Action("GUIDED"), reason="ladder-objects-stage")
            overridden_ticks += 1
        decision = reflexes.decide(Action("FORWARD"), stimuli, channel.drain())
        if tick % 2 == 0 and decision.actuated == Action("GUIDED"):
            correct_ticks += 1
    return correct_ticks / overridden_ticks if overridden_ticks else 0.0


def _voluntary_reliance_score(
    *, threat_probability: float = 0.05, ticks: int = 200, seed: int = 0,
) -> float:
    """``1.0 - ReflexStack.activation_rate`` over a short synthetic session
    with a *low* threat-stimulus rate -- the matured-organism regime
    ``tests/test_reflexes.py::test_reflex_activation_rate_falls_across_
    development_on_locomotion_and_threat_scenario`` exercises (its lowest
    ``threat_probability`` there is also ``0.05``, asserting the resulting
    rate settles under ``0.15``). Returns the *inverted* rate so
    :data:`_FORAGING`'s ``>= threshold`` gate reads naturally as "voluntary
    reliance" rather than needing a less-than comparison."""
    import random

    from cognitive_runtime.core.action import Action
    from motor.reflexes import ReflexConfig, ReflexStack, Stimulus

    reflexes = ReflexStack([ReflexConfig("withdraw", "threat", Action("BACK"), threshold=0.5, priority=10)])
    rng = random.Random(seed)
    for _ in range(ticks):
        stimuli = [Stimulus("threat", 1.0)] if rng.random() < threat_probability else []
        reflexes.decide(Action("FORWARD"), stimuli)
    return 1.0 - reflexes.activation_rate


#: Which gate each carries a persisted world-model checkpoint for (issue
#: #134). Keyed by milestone metric name so :func:`ladder_cortex_checkpoint_paths`
#: and :func:`ladder_milestone_metrics` derive the same filenames from one
#: base path, and ``development.runner.run_curriculum`` can watch the same
#: paths to record which ones actually exist under its own
#: ``extra_metadata["ladder_world_model_checkpoints"]``.
_CORTEX_CHECKPOINT_SUFFIXES = {
    "cortex_beats_copy_last": ".ladder-visual-cortex.pt",
    "action_ablation_margin": ".ladder-action-cortex.pt",
}


def ladder_cortex_checkpoint_paths(base_path: str) -> Dict[str, str]:
    """The world-model checkpoint paths :func:`ladder_milestone_metrics`
    warm-starts from and saves to, derived from one ``base_path`` (e.g. the
    ladder's own ``checkpoint_path``) so a caller passes the *same* base
    into both ``functools.partial(ladder_milestone_metrics,
    cortex_checkpoint_base=base_path)`` and
    ``development.runner.run_curriculum(...,
    world_model_checkpoint_paths=ladder_cortex_checkpoint_paths(base_path).values())``
    -- one source of truth for the filenames instead of two call sites
    guessing the same suffix independently.
    """
    return {metric: base_path + suffix for metric, suffix in _CORTEX_CHECKPOINT_SUFFIXES.items()}


def ladder_milestone_metrics(
    stage: CurriculumStageSpec, summary: "EvaluationSummary", *, record_dir: str,
    cortex_checkpoint_base: Optional[str] = None,
) -> Mapping[str, float]:
    """The ladder's real ``milestone_metrics`` provider
    (``development.runner.run_curriculum``'s hook): computes only the
    metric(s) ``stage.gates`` actually references, using the real Phase 2/6
    computations documented on each stage constant above. ``record_dir``
    must be a real directory -- the nursery-backed metrics record and train
    against it.

    ``cortex_checkpoint_base`` (issue #134), when given, is expanded via
    :func:`ladder_cortex_checkpoint_paths` into per-gate checkpoint paths:
    each predictive gate then warm-starts from (and saves back to) *its
    own* persisted model across attempts/stages instead of training a
    fresh, disposable one every call -- so the organism's own learning can
    actually improve the gate, and a later attempt's promotion reflects
    that history rather than an unrelated temporary model.
    """
    gate_names = {gate.metric for gate in stage.gates}
    checkpoint_paths = (
        ladder_cortex_checkpoint_paths(cortex_checkpoint_base) if cortex_checkpoint_base else {}
    )
    metrics: Dict[str, float] = {}
    if "cortex_beats_copy_last" in gate_names:
        metrics["cortex_beats_copy_last"] = _cortex_beats_copy_last(
            stage, record_dir, cortex_checkpoint_path=checkpoint_paths.get("cortex_beats_copy_last"),
        )
    if "action_ablation_margin" in gate_names:
        metrics["action_ablation_margin"] = _action_ablation_margin(
            stage, record_dir, cortex_checkpoint_path=checkpoint_paths.get("action_ablation_margin"),
        )
    if "reflex_override_precedence" in gate_names:
        metrics["reflex_override_precedence"] = _reflex_override_precedence()
    if "reflex_activation_rate" in gate_names:
        metrics["reflex_activation_rate"] = _voluntary_reliance_score()
    return metrics
