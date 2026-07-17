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

import pytest

from development.definitions import MILESTONE_METRICS, CurriculumDefinition, CurriculumStageSpec
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
# Torch+crafter-gated: the real train/evaluate/promote/resume runner.

pytest.importorskip("torch")
pytest.importorskip("crafter")

from cognitive_runtime.neural import read_checkpoint_metadata  # noqa: E402
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
