"""The Gestation -> Foraging ladder (issue #105): structure/validation of
``development.ladder.GESTATION_TO_FORAGING``, ``motor.organism_policy``'s
motor-freedom-driven policy, and (torch+crafter-gated) the real runner/
milestone-metrics wiring end to end.

Mirrors ``tests/test_curriculum_runner.py``'s split: definition-shape tests
are dependency-light; the actual train/evaluate/promote/resume runner is
gated behind the extras it needs.
"""

from __future__ import annotations

import functools
import os

import pytest

from development.definitions import (
    MILESTONE_METRICS,
    CurriculumDefinition,
    CurriculumDefinitionError,
    CurriculumStageSpec,
    PromotionCriteria,
)
from development.ladder import GESTATION_TO_FORAGING
from motor.organism_policy import MotorFreedomPolicy, build_stage_policy
from motor.reflexes import CaregiverChannel, CaregiverOverride, ReflexConfig, ReflexStack, Stimulus
from motor.voluntary import CallableController
from cognitive_runtime.core.action import NULL_ACTION, Action

_STAGE_NAMES = ["gestation", "babbling", "crawling", "objects", "foraging"]
_STAGE_MOTOR_FREEDOMS = {
    "gestation": "frozen",
    "babbling": "overridden",
    "crawling": "overridden",
    "objects": "learned",
    "foraging": "learned",
}


# --------------------------------------------------------------------------
# Ladder structure (torch/crafter-free: GESTATION_TO_FORAGING is built from
# plain dataclasses).


def test_ladder_has_exactly_five_stages_in_order_no_speaking():
    assert [s.name for s in GESTATION_TO_FORAGING.stages] == _STAGE_NAMES


def test_every_stage_declares_crafter_world_and_a_scenario():
    for stage in GESTATION_TO_FORAGING.stages:
        assert stage.world == "crafter"
        assert stage.scenario
        assert stage.senses
        assert stage.losses
        assert stage.gates


def test_every_stage_motor_freedom_matches_the_phase_doc_table():
    for stage in GESTATION_TO_FORAGING.stages:
        assert stage.motor_freedom == _STAGE_MOTOR_FREEDOMS[stage.name]


def test_every_gate_metric_is_drawn_from_milestone_metrics():
    """``PromotionCriteria.__post_init__`` already enforces membership in
    ``KNOWN_METRICS`` (summary metrics + milestone metrics); this asserts
    the *stricter* table explicitly, so a future edit that quietly swaps in
    a plain summary metric (e.g. ``average_ticks``) instead of a real
    Phase 2-6 milestone fails loudly here."""
    for stage in GESTATION_TO_FORAGING.stages:
        for gate in stage.gates:
            assert gate.metric in MILESTONE_METRICS, (stage.name, gate.metric)


def test_gestation_is_frozen_with_a_beats_copy_last_gate():
    stage = GESTATION_TO_FORAGING.stages[0]
    assert stage.motor_freedom == "frozen"
    assert [g.metric for g in stage.gates] == ["cortex_beats_copy_last"]


def test_crawling_gates_on_both_copy_last_and_action_ablation():
    """The phase doc, verbatim: "Crawling gates on the cortex beating
    copy-last on walk_forward + action-ablation" -- both, not either."""
    stage = GESTATION_TO_FORAGING.stages[2]
    assert stage.scenario == "walk_forward"
    assert {g.metric for g in stage.gates} == {"cortex_beats_copy_last", "action_ablation_margin"}


def test_foraging_gate_expresses_low_reflex_activation_as_a_lower_bound():
    """``reflex_activation_rate`` is lower-is-better but ``PromotionCriteria``
    is always ``value >= threshold`` -- the ladder's encoding (documented on
    ``development.ladder._FORAGING``) stores ``1 - rate`` under the same
    metric name, so the threshold itself must read as a *high* bar (close to
    1.0), not a low one -- a low threshold here would silently invert the
    intended gate."""
    stage = GESTATION_TO_FORAGING.stages[4]
    gate = stage.gates[0]
    assert gate.metric == "reflex_activation_rate"
    assert gate.threshold > 0.5


@pytest.mark.parametrize("stage", GESTATION_TO_FORAGING.stages, ids=lambda s: s.name)
def test_curriculum_stage_spec_accepts_every_ladder_stage(stage):
    assert isinstance(stage, CurriculumStageSpec)


def test_ladder_stages_share_world_config_for_one_checkpoint():
    """Task 3's "one checkpoint carried across every stage" needs an
    identical stream layout across stages -- guaranteed here by every stage
    sharing the exact same ``world_config``."""
    configs = {stage.name: stage.world_config for stage in GESTATION_TO_FORAGING.stages}
    distinct = {tuple(sorted(cfg.items())) for cfg in configs.values()}
    assert len(distinct) == 1, configs


def test_ladder_validates_via_shared_layout_check():
    """The real validation (task 2's acceptance: "the ladder loads and
    validates") -- constructs a ``CrafterWorld`` per stage to hash its
    stream catalog, so this needs the ``crafter`` extra."""
    pytest.importorskip("crafter")
    from development.definitions import _validate_shared_layout

    _validate_shared_layout("development.ladder.GESTATION_TO_FORAGING", GESTATION_TO_FORAGING)


# --------------------------------------------------------------------------
# issue #135: `stage.losses` used to be a label `ladder_milestone_metrics`
# never consulted -- a stage could gate on `cortex_beats_copy_last`/
# `action_ablation_margin` regardless of whether it declared the loss those
# metrics actually train under. `_require_losses` (called by
# `_cortex_beats_copy_last`/`_action_ablation_margin` before their lazy
# torch/nursery import) raises instead, torch-free.

def test_beats_copy_last_gate_requires_prediction_loss():
    from development.ladder import ladder_milestone_metrics

    stage = CurriculumStageSpec(
        name="gestation-like", world="crafter", scenario="object_permanence",
        senses=("vision",), motor_freedom="frozen", losses=(),
        gates=(PromotionCriteria(metric="cortex_beats_copy_last", threshold=1.0),),
    )
    with pytest.raises(CurriculumDefinitionError, match="prediction"):
        ladder_milestone_metrics(stage, summary=None, record_dir="unused")


def test_action_ablation_margin_gate_requires_action_conditioning_loss():
    from development.ladder import ladder_milestone_metrics

    stage = CurriculumStageSpec(
        name="babbling-like", world="crafter", scenario="turn",
        senses=("vision", "proprioception"), motor_freedom="overridden", losses=("prediction",),
        gates=(PromotionCriteria(metric="action_ablation_margin", threshold=1e-4),),
    )
    with pytest.raises(CurriculumDefinitionError, match="action_conditioning"):
        ladder_milestone_metrics(stage, summary=None, record_dir="unused")


def test_ladder_stages_declare_losses_their_own_gates_require():
    """The real ladder table must itself satisfy the same precondition --
    otherwise Milestone 7's real run would hit this exact error."""
    from development.ladder import _require_losses

    for stage in GESTATION_TO_FORAGING.stages:
        gate_metrics = {gate.metric for gate in stage.gates}
        if "cortex_beats_copy_last" in gate_metrics:
            _require_losses(stage, "prediction")
        if "action_ablation_margin" in gate_metrics:
            _require_losses(stage, "prediction", "action_conditioning")


# --------------------------------------------------------------------------
# motor.organism_policy: unit-level, no torch/crafter needed.


_ACTIONS = [Action("FORWARD"), Action("BACK"), Action("GUIDED"), NULL_ACTION]


def _reflex_stack() -> ReflexStack:
    return ReflexStack([ReflexConfig("withdraw", "threat", Action("BACK"), threshold=0.5, priority=10)])


def test_frozen_always_emits_null_action_regardless_of_stimuli():
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[0], _ACTIONS,
        reflexes=_reflex_stack(), stimuli=[Stimulus("threat", 1.0)],
    )
    assert policy.decide(None, None, None) == NULL_ACTION
    assert policy.emit(None, None, None) == []


def test_frozen_requires_no_collaborators():
    """Gestation's freeze works entirely standalone."""
    policy = build_stage_policy(GESTATION_TO_FORAGING.stages[0], _ACTIONS)
    assert policy.decide(None, None, None) == NULL_ACTION


def test_overridden_falls_back_to_scripted_when_nothing_injected():
    from cognitive_runtime.policies.constant_action import ConstantActionPolicy

    channel = CaregiverChannel()
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[1], _ACTIONS,
        scripted=ConstantActionPolicy(Action("FORWARD")),
        reflexes=_reflex_stack(), caregiver=channel,
    )
    assert policy.decide(None, None, None) == Action("FORWARD")


def test_overridden_caregiver_override_takes_precedence_when_injected():
    from cognitive_runtime.policies.constant_action import ConstantActionPolicy

    channel = CaregiverChannel()
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[1], _ACTIONS,
        scripted=ConstantActionPolicy(Action("FORWARD")),
        reflexes=_reflex_stack(), caregiver=channel,
        stimuli=[Stimulus("threat", 1.0)],  # would otherwise fire `withdraw`
    )
    channel.inject(Action("GUIDED"), reason="babbling")
    assert policy.decide(None, None, None) == Action("GUIDED")
    # One-shot: the next tick has nothing pending, falls back to the reflex
    # (still outranking the scripted voluntary action).
    assert policy.decide(None, None, None) == Action("BACK")


def test_overridden_requires_a_scripted_policy():
    with pytest.raises(ValueError, match="overridden"):
        build_stage_policy(GESTATION_TO_FORAGING.stages[1], _ACTIONS)


def test_learned_is_driven_by_the_voluntary_controller():
    voluntary = CallableController("stub", lambda state, actions, goal: Action("FORWARD"))
    policy = build_stage_policy(GESTATION_TO_FORAGING.stages[3], _ACTIONS, voluntary=voluntary)
    assert policy.decide(None, None, None) == Action("FORWARD")


def test_learned_reflex_can_still_override_the_voluntary_choice():
    voluntary = CallableController("stub", lambda state, actions, goal: Action("FORWARD"))
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[3], _ACTIONS,
        voluntary=voluntary, reflexes=_reflex_stack(), stimuli=[Stimulus("threat", 1.0)],
    )
    assert policy.decide(None, None, None) == Action("BACK")


def test_learned_requires_a_voluntary_controller():
    with pytest.raises(ValueError, match="learned"):
        build_stage_policy(GESTATION_TO_FORAGING.stages[3], _ACTIONS)


def test_frozen_never_drains_a_supplied_caregiver_channel():
    channel = CaregiverChannel()
    channel.inject(Action("GUIDED"), reason="should-not-be-touched")
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[0], _ACTIONS, reflexes=_reflex_stack(), caregiver=channel,
    )
    policy.decide(None, None, None)
    # Still pending: `frozen` never called `.drain()`.
    assert channel.drain() == CaregiverOverride(Action("GUIDED"), reason="should-not-be-touched")


def test_learned_never_drains_a_supplied_caregiver_channel():
    channel = CaregiverChannel()
    channel.inject(Action("GUIDED"), reason="should-not-be-touched")
    voluntary = CallableController("stub", lambda state, actions, goal: Action("FORWARD"))
    policy = build_stage_policy(
        GESTATION_TO_FORAGING.stages[3], _ACTIONS,
        voluntary=voluntary, reflexes=_reflex_stack(), caregiver=channel,
    )
    result = policy.decide(None, None, None)
    assert result == Action("FORWARD")  # no threat stimulus: reflex doesn't fire either
    # Still pending: `learned` never called `.drain()` on the channel.
    assert channel.drain() == CaregiverOverride(Action("GUIDED"), reason="should-not-be-touched")


def test_build_stage_policy_rejects_a_stage_with_no_declared_motor_freedom():
    bare = CurriculumStageSpec(name="undeclared")
    with pytest.raises(ValueError, match="motor_freedom"):
        build_stage_policy(bare, _ACTIONS)


# --------------------------------------------------------------------------
# issue #136: `_reflex_override_precedence`/`_voluntary_reliance_score` used
# to run a hand-rolled `ReflexStack` disconnected from the organism, the
# stage, and the attempt's real outcome -- structurally guaranteed to return
# the same passing value before and after training. These prove the real
# gates now run the production `build_stage_policy` seam and vary with real
# per-attempt data. Torch/crafter-free: `motor.organism_policy`/
# `motor.reflexes`/`motor.voluntary` need neither.

from development.ladder import _reflex_override_precedence, _voluntary_reliance_score  # noqa: E402
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary  # noqa: E402


def test_reflex_override_precedence_is_a_genuine_bidirectional_contract_check():
    """Runs the real ``objects`` stage's ``build_stage_policy`` seam, not a
    bypass -- proves the value is computed (``1.0`` only because the real
    precedence contract genuinely holds), not returned by construction."""
    objects = GESTATION_TO_FORAGING.stages[3]
    assert objects.motor_freedom == "learned"
    assert _reflex_override_precedence(objects) == 1.0


def test_reflex_override_precedence_would_fail_if_the_real_contract_broke():
    """Same check, but against a stage whose real ``motor_freedom`` is
    ``"overridden"`` with no scripted policy declared -- ``build_stage_policy``
    can't build a working ``"learned"`` seam for it, proving this metric
    actually depends on the real stage it's given rather than always
    fabricating ``1.0``."""
    babbling = GESTATION_TO_FORAGING.stages[1]
    with pytest.raises(ValueError, match="overridden"):
        _reflex_override_precedence(babbling)


def _summary(termination_reasons):
    return EvaluationSummary(
        policy="foraging", total_reward=0.0, total_ticks=0,
        average_reward=0.0, average_ticks=0.0, termination_reasons=termination_reasons,
    )


def test_voluntary_reliance_score_tracks_the_attempts_real_outcome_not_a_constant():
    """issue #136: the old version returned the identical passing value
    regardless of ``summary`` (it never even accepted one). The real gate
    must genuinely differ between an attempt whose held-out episode
    survived and one that died -- and a death-heavy attempt must be able to
    fail :data:`~development.ladder._FORAGING`'s ``0.85`` threshold."""
    foraging = GESTATION_TO_FORAGING.stages[4]
    survived = _voluntary_reliance_score(foraging, _summary(["timeout"]))
    died = _voluntary_reliance_score(foraging, _summary(["death:health"]))

    assert survived != died
    assert survived >= 0.85
    assert died < 0.85


def test_ladder_milestone_metrics_computes_real_reflex_gates_for_objects_and_foraging(tmp_path):
    """End-to-end through the public ``ladder_milestone_metrics`` seam
    (not the private helpers directly): both gates resolve to real,
    attempt-dependent values, and Foraging's failing case genuinely holds
    the stage (task 4's acceptance: "a failing metric holds the organism")."""
    from development.ladder import ladder_milestone_metrics

    objects = GESTATION_TO_FORAGING.stages[3]
    foraging = GESTATION_TO_FORAGING.stages[4]
    record_dir = str(tmp_path / "nursery")

    objects_metrics = ladder_milestone_metrics(objects, _summary(["timeout"]), record_dir=record_dir)
    assert objects_metrics == {"reflex_override_precedence": 1.0}
    assert objects.evaluate_gates(objects_metrics)["reflex_override_precedence"] is True

    surviving_metrics = ladder_milestone_metrics(foraging, _summary(["timeout"]), record_dir=record_dir)
    dying_metrics = ladder_milestone_metrics(foraging, _summary(["death:health"]), record_dir=record_dir)
    assert foraging.evaluate_gates(surviving_metrics)["reflex_activation_rate"] is True
    assert foraging.evaluate_gates(dying_metrics)["reflex_activation_rate"] is False


# --------------------------------------------------------------------------
# Torch+crafter-gated: the real train/evaluate/promote/resume runner.

pytest.importorskip("torch")
pytest.importorskip("crafter")

from cognitive_runtime.neural import read_checkpoint_metadata  # noqa: E402
from cognitive_runtime.policies.actor_critic import ActorCriticLearner, ActorCriticPolicy  # noqa: E402
from development.ladder import ladder_milestone_metrics  # noqa: E402
from development.runner import run_curriculum  # noqa: E402


def _first_n_stages(n: int) -> CurriculumDefinition:
    return CurriculumDefinition(
        name="gestation-to-foraging-prefix", stages=GESTATION_TO_FORAGING.stages[:n],
    )


def _stub_milestone_metrics(stage, summary):
    """Canned passing values for every metric the first three ladder
    stages' gates reference -- fast, deterministic, mirrors
    ``test_curriculum_runner.py``'s
    ``test_milestone_metrics_provider_supplies_phase2to6_metrics_to_gates``.
    Not the real computation (that's ``ladder_milestone_metrics``, exercised
    separately below); this is for the resume-mechanics tests, which care
    about checkpoint/stage-index bookkeeping, not gate correctness."""
    values = {"cortex_beats_copy_last": 1.5, "action_ablation_margin": 1.0}
    return {gate.metric: values[gate.metric] for gate in stage.gates}


def test_resume_continues_at_the_correct_stage_without_redoing_earlier_stages(tmp_path):
    """Task 3's acceptance: promoting stage N -> N+1 resumes the *same*
    checkpoint (no re-init); interrupting and resuming continues from the
    held stage_index, not stage 0."""
    import dataclasses

    stages = list(_first_n_stages(2).stages)
    # Force a hold at "babbling" (index 1) so there is a stage boundary to
    # resume across, mirroring test_curriculum_runner.py's
    # test_resume_continues_at_held_stage_without_redoing_earlier_stages.
    stuck_babbling = dataclasses.replace(
        stages[1],
        gates=(type(stages[1].gates[0])(metric="action_ablation_margin", threshold=1e9, sample_size=1),),
        max_attempts=1,
    )
    definition = CurriculumDefinition(name="stuck-babbling", stages=(stages[0], stuck_babbling))
    checkpoint_path = str(tmp_path / "curriculum.pt")

    first = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )
    assert first.status == "held"
    assert first.state.stage_index == 1  # "gestation" promoted, "babbling" holds
    gestation_entries = [e for e in first.state.history if e["stage"] == "gestation"]
    assert len(gestation_entries) == 1
    assert gestation_entries[0]["promoted"] is True

    second = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )
    assert second.resumed is True
    assert second.status == "held"
    # No new "gestation" attempts: resumed at the held stage, not stage 0.
    assert len([e for e in second.state.history if e["stage"] == "gestation"]) == 1
    new_entries = second.state.history[len(first.state.history):]
    assert new_entries and all(e["stage"] == "babbling" for e in new_entries)

    # (c) the resumed stage's declared motor_freedom still matches the
    # ladder's.
    resumed_stage = definition.stages[second.state.stage_index]
    assert resumed_stage.name == "babbling"
    assert resumed_stage.motor_freedom == _STAGE_MOTOR_FREEDOMS["babbling"]

    meta = read_checkpoint_metadata(checkpoint_path)
    assert meta["training_stats"]["curriculum"]["stage_index"] == 1
    assert meta["training_stats"]["curriculum"]["held"] is True


def test_promotes_through_gestation_babbling_crawling_on_one_checkpoint(tmp_path):
    """Milestone 7's exit gate, with stubbed-but-shaped-like-the-real-thing
    metrics (fast): one checkpoint promotes Gestation->Babbling->Crawling
    unattended, never re-initializing weights between stages."""
    definition = _first_n_stages(3)
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )

    assert result.status == "completed"
    assert result.resumed is False
    stages_seen = [entry["stage"] for entry in result.state.history]
    assert stages_seen == ["gestation", "babbling", "crawling"]
    assert all(entry["promoted"] for entry in result.state.history)

    meta = read_checkpoint_metadata(checkpoint_path)
    assert meta["training_stats"]["curriculum"]["completed"] is True
    assert meta["training_stats"]["curriculum"]["definition_name"] == "gestation-to-foraging-prefix"


def test_ladder_runner_trains_against_crafter_not_minecraft(tmp_path, monkeypatch):
    """The issue #105 gap this fix closes: before it, ``development.runner``
    hardcoded ``MinecraftSurvivalBox`` and silently ignored ``stage.world``.
    Spies on ``CrafterWorld.__init__`` to prove a Crafter-world ladder stage
    genuinely constructs a ``CrafterWorld``, not a
    ``MinecraftSurvivalBox``."""
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    calls = []
    original_init = CrafterWorld.__init__

    def spy_init(self, config=None):
        calls.append(config)
        return original_init(self, config)

    monkeypatch.setattr(CrafterWorld, "__init__", spy_init)

    definition = _first_n_stages(1)  # "gestation" alone
    checkpoint_path = str(tmp_path / "curriculum.pt")
    run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )

    assert calls, "CrafterWorld was never constructed for a world='crafter' stage"


def test_real_milestone_metrics_provider_promotes_babbling_on_genuine_gate_values(tmp_path):
    """The closest thing to the Milestone 7 CI gate this issue can prove
    fast: runs the ladder's real ``babbling`` stage (not a stub) with the
    real ``ladder_milestone_metrics`` provider -- action-ablation trained
    and evaluated for real against recorded Crafter episodes -- and asserts
    it promotes on a genuinely computed ``action_ablation_margin``.

    Deliberately not "crawling"/"gestation": both gate (partly) on
    ``cortex_beats_copy_last``, which -- like
    ``tests/test_nursery.py``'s own long-standing note -- is a training-
    budget-scale property a CI-fast pixel encoder cannot reliably clear
    (confirmed empirically while implementing this test: even ~30 epochs
    did not beat the copy-last baseline on a 40-tick Crafter recording).
    ``action_ablation_margin`` on "babbling" (``turn``), by contrast, is
    reliably positive even at a few epochs -- action-conditioning helps
    prediction long before the encoder is any good at prediction itself.
    """
    babbling = GESTATION_TO_FORAGING.stages[1]
    definition = CurriculumDefinition(name="babbling-only", stages=(babbling,))
    checkpoint_path = str(tmp_path / "curriculum.pt")
    nursery_dir = str(tmp_path / "nursery")
    provider = functools.partial(ladder_milestone_metrics, record_dir=nursery_dir)

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None, milestone_metrics=provider,
    )

    assert result.status == "completed"
    entry = result.state.history[0]
    assert entry["promoted"] is True
    assert entry["metric"] == ["action_ablation_margin"]
    assert entry["value"]["action_ablation_margin"] >= entry["threshold"]["action_ablation_margin"]


def test_real_milestone_metrics_provider_persists_and_warm_starts_its_cortex(tmp_path, monkeypatch):
    """issue #134: the gate used to train a fresh, disposable cortex on
    every call, entirely outside the ladder's own checkpoint (which carried
    no trace of it at all). Wiring ``cortex_checkpoint_base`` through the
    real ``ladder_milestone_metrics`` provider and
    ``world_model_checkpoint_paths`` through ``run_curriculum`` fixes both:
    a second attempt warm-starts from the first's cortex, and the ladder's
    own checkpoint metadata records that a world model actually backs it."""
    import cognitive_runtime.training.nursery as nursery_module
    from development.ladder import ladder_cortex_checkpoint_paths

    babbling = GESTATION_TO_FORAGING.stages[1]
    definition = CurriculumDefinition(name="babbling-only", stages=(babbling,))
    nursery_dir = str(tmp_path / "nursery")
    cortex_base = str(tmp_path / "organism")
    cortex_paths = list(ladder_cortex_checkpoint_paths(cortex_base).values())
    provider = functools.partial(
        ladder_milestone_metrics, record_dir=nursery_dir, cortex_checkpoint_base=cortex_base,
    )

    captured = []
    original = nursery_module.train_action_world_model

    def spy(dataset, config=None, *, initial_model=None):
        captured.append(initial_model)
        return original(dataset, config, initial_model=initial_model)

    monkeypatch.setattr(nursery_module, "train_action_world_model", spy)

    checkpoint_path_1 = str(tmp_path / "curriculum-1.pt")
    result_1 = run_curriculum(
        definition, checkpoint_path=checkpoint_path_1, record_dir=None,
        milestone_metrics=provider, world_model_checkpoint_paths=cortex_paths,
    )
    assert result_1.status == "completed"
    assert any(os.path.exists(p) for p in cortex_paths), "a cortex checkpoint must be persisted"
    recorded = read_checkpoint_metadata(checkpoint_path_1)["extra"]["ladder_world_model_checkpoints"]
    assert recorded and all(os.path.exists(p) for p in recorded)
    assert captured[0] is None, "nothing to warm-start from on the very first attempt"

    calls_before_second_run = len(captured)
    checkpoint_path_2 = str(tmp_path / "curriculum-2.pt")
    result_2 = run_curriculum(
        definition, checkpoint_path=checkpoint_path_2, record_dir=None,
        milestone_metrics=provider, world_model_checkpoint_paths=cortex_paths,
    )
    assert result_2.status == "completed"
    assert captured[calls_before_second_run] is not None, (
        "a fresh curriculum run reusing the same cortex_checkpoint_base must warm-start"
    )


# --------------------------------------------------------------------------
# issue #133: `_run_stage_episodes` used to ignore `stage.motor_freedom`
# entirely and always ran the actor/critic loop, so Gestation was never
# genuinely frozen and Babbling/Crawling never ran their scripted/caregiver
# path. These prove the real runner now drives each stage's declared freedom.


def _spy_init(monkeypatch, cls) -> list:
    """Wraps ``cls.__init__`` to record one entry per real construction,
    still delegating to the original -- lets a test assert "never built"
    without disturbing behaviour when it is."""
    calls: list = []
    original_init = cls.__init__

    def wrapper(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", wrapper)
    return calls


def test_frozen_stage_never_builds_the_actor_critic_policy(tmp_path, monkeypatch):
    """Gestation is ``motor_freedom="frozen"``: before the fix, its episodes
    still ran through a freshly-built ``ActorCriticPolicy``/``ActorCriticLearner``
    exactly like every other stage."""
    ac_policy_calls = _spy_init(monkeypatch, ActorCriticPolicy)
    ac_learner_calls = _spy_init(monkeypatch, ActorCriticLearner)

    definition = _first_n_stages(1)  # "gestation" alone
    checkpoint_path = str(tmp_path / "curriculum.pt")
    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )

    assert result.status == "completed"
    assert not ac_policy_calls, "frozen gestation must not construct an ActorCriticPolicy"
    assert not ac_learner_calls, "frozen gestation must not construct an ActorCriticLearner"


def test_overridden_stage_never_builds_the_actor_critic_policy(tmp_path, monkeypatch):
    """Babbling is ``motor_freedom="overridden"``: before the fix it ran
    through the learned actor/critic loop instead of a scripted/random
    motor input."""
    ac_policy_calls = _spy_init(monkeypatch, ActorCriticPolicy)
    ac_learner_calls = _spy_init(monkeypatch, ActorCriticLearner)

    babbling = GESTATION_TO_FORAGING.stages[1]
    definition = CurriculumDefinition(name="babbling-only", stages=(babbling,))
    checkpoint_path = str(tmp_path / "curriculum.pt")
    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=_stub_milestone_metrics,
    )

    assert result.status == "completed"
    assert not ac_policy_calls, "overridden babbling must not construct an ActorCriticPolicy"
    assert not ac_learner_calls, "overridden babbling must not construct an ActorCriticLearner"


def test_overridden_stage_runs_the_scenarios_own_scripted_policy_not_random(monkeypatch):
    """A uniform-random substitute would let Babbling/Crawling promote
    without ever running their registered ``turn``/``walk_forward`` nursery
    scenario -- the runner must reuse the *same* scripted policy that
    scenario's own recording uses (``cognitive_runtime.training.nursery``),
    not a stage-agnostic random policy."""
    from cognitive_runtime.policies.constant_action import ConstantActionPolicy
    from cognitive_runtime.policies.scripted_sequence import ScriptedSequencePolicy
    from development.runner import _scripted_policy_for_stage

    babbling, crawling = GESTATION_TO_FORAGING.stages[1], GESTATION_TO_FORAGING.stages[2]
    action_space = [Action("MOVE_UP"), Action("MOVE_RIGHT"), Action("MOVE_DOWN"), Action("MOVE_LEFT")]

    babbling_policy = _scripted_policy_for_stage(babbling, action_space, seed=0)
    assert isinstance(babbling_policy, ScriptedSequencePolicy), (
        "babbling's 'turn' scenario is a scripted direction-cycling sequence, not random"
    )

    crawling_policy = _scripted_policy_for_stage(crawling, action_space, seed=0)
    assert isinstance(crawling_policy, ConstantActionPolicy), (
        "crawling's 'walk_forward' scenario is a constant-direction walk, not random"
    )


def test_learned_stage_raises_without_a_voluntary_controller_factory(tmp_path):
    """"learned" (Objects/Foraging) needs the real Phase 6 voluntary path,
    which development.runner cannot build on its own (no trained predictive
    cortex) -- it must raise, not silently fall back to the actor/critic
    loop the way it used to for every stage."""
    objects_stage = GESTATION_TO_FORAGING.stages[3]
    definition = CurriculumDefinition(name="objects-only", stages=(objects_stage,))
    checkpoint_path = str(tmp_path / "curriculum.pt")

    with pytest.raises(CurriculumDefinitionError, match="voluntary_controller"):
        run_curriculum(
            definition, checkpoint_path=checkpoint_path, record_dir=None,
            milestone_metrics=lambda stage, summary: {"reflex_override_precedence": 1.0},
        )


def test_learned_stage_runs_under_a_supplied_voluntary_controller(tmp_path, monkeypatch):
    """Given a real ``voluntary_controller`` factory, a "learned" stage
    promotes through it instead of raising -- and, like frozen/overridden,
    never touches the actor/critic loop."""
    ac_policy_calls = _spy_init(monkeypatch, ActorCriticPolicy)

    objects_stage = GESTATION_TO_FORAGING.stages[3]
    definition = CurriculumDefinition(name="objects-only", stages=(objects_stage,))
    checkpoint_path = str(tmp_path / "curriculum.pt")

    def controller_factory(stage, action_space):
        return CallableController("stub", lambda state, actions, goal: actions[0])

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=lambda stage, summary: {"reflex_override_precedence": 1.0},
        voluntary_controller=controller_factory,
    )

    assert result.status == "completed"
    assert not ac_policy_calls, "learned objects stage must not construct an ActorCriticPolicy"
