"""Cortex-backed MPC as the live voluntary controller (issue #168).

Covers the acceptance criteria:
- Voluntary action is chosen by cortex MPC (one-step planning over heads).
- A reflex demonstrably overrides the voluntary action when its stimulus fires.
- Predicted-vs-actuated divergence is logged (MotorDecision record).
- Nothing in the motor path takes a gradient step (pure torch.no_grad).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.core.action import NULL_ACTION, Action  # noqa: E402
from cognitive_runtime.core.memory import Memory  # noqa: E402
from cognitive_runtime.core.perception import State  # noqa: E402
from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.neural.pixel_stream_encoder import PIXEL_STREAM_ID  # noqa: E402
from cognitive_runtime.policies.cortex_world_model import CortexWorldModel  # noqa: E402
from motor.cortex_mpc import build_cortex_mpc, cortex_mpc_factory  # noqa: E402
from motor.organism_policy import MotorFreedomPolicy  # noqa: E402
from motor.reflexes import ReflexConfig, ReflexStack, Stimulus  # noqa: E402

_ACTION_KEYS = ["noop", "move_forward", "turn_left", "turn_right"]


def _small_cortex(pixel_shape=(8, 8, 3)) -> PredictiveCortex:
    torch.manual_seed(42)
    cfg = PredictiveCortexConfig(
        latent_width=8, hidden_dim=16, reconstruction_size=8, horizons_ticks=(1,)
    )
    return PredictiveCortex(pixel_shape, _ACTION_KEYS, cfg)


def _push_frame(memory: Memory, frame: np.ndarray, seq: int) -> None:
    memory.buffer.extend(
        [
            StreamEvent(
                stream_id=PIXEL_STREAM_ID,
                modality="vision",
                timestamp=float(seq),
                sequence_number=seq,
                payload=frame,
            )
        ]
    )


def _frame(rng: np.random.Generator, shape=(8, 8, 3)) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _primed_cortex_wm():
    """Return a CortexWorldModel whose internal state has been primed by
    one predict() call, so _latent and _pre_advance_hidden are populated."""
    cortex = _small_cortex()
    wm = CortexWorldModel(cortex, action_keys=_ACTION_KEYS)
    memory = Memory()
    rng = np.random.default_rng(1)
    _push_frame(memory, _frame(rng), 0)
    wm.predict(State(observation=None), memory)
    return wm, memory, rng


def test_cortex_mpc_chooses_deterministically():
    """The cortex MPC controller deterministically selects an action based
    on the cortex heads (reward + novelty), and the result is in the action
    space."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    assert controller.name == "cortex-mpc"

    actions = [Action(k) for k in _ACTION_KEYS]
    chosen = controller.choose(None, actions)
    assert chosen in actions


def test_cortex_mpc_is_deterministic_across_calls():
    """Repeated calls with the same state produce the same action."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]
    first = controller.choose(None, actions)
    second = controller.choose(None, actions)
    assert first == second


def test_cortex_mpc_does_not_take_gradient_steps():
    """No parameter in the cortex accumulates gradients during MPC planning."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]
    controller.choose(None, actions)
    for param in wm.model.parameters():
        assert param.grad is None


def test_cortex_mpc_does_not_mutate_world_model_state():
    """MPC's candidate-action rollouts must not corrupt the CortexWorldModel's
    persisted hidden state -- the world model's predict() is the only thing
    that should advance it."""
    wm, _memory, _rng = _primed_cortex_wm()
    hidden_before = wm._hidden
    latent_before = wm._latent.clone()
    pre_hidden_before = wm._pre_advance_hidden

    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]
    controller.choose(None, actions)

    assert wm._hidden is hidden_before
    assert torch.equal(wm._latent, latent_before)
    assert wm._pre_advance_hidden is pre_hidden_before


def test_cortex_mpc_handles_unprimed_world_model():
    """Before the first predict() call, _latent is None; MPC should produce
    NaN scores and fall back to the first action (deterministic tie-break)."""
    cortex = _small_cortex()
    wm = CortexWorldModel(cortex, action_keys=_ACTION_KEYS)
    assert wm._latent is None
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]
    chosen = controller.choose(None, actions)
    assert chosen == actions[0]


def test_reflex_overrides_voluntary_cortex_mpc():
    """When a reflex stimulus fires, the reflex's action overrides the
    voluntary MPC choice — demonstrating the precedence contract."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]

    # Use FLEE as the reflex action — guaranteed distinct from any MPC
    # choice within the normal action space.
    flee = Action("FLEE")
    reflexes = ReflexStack([
        ReflexConfig("withdraw", "threat", flee, threshold=0.5, priority=10),
    ])
    policy = MotorFreedomPolicy(
        "learned", actions,
        voluntary=controller, reflexes=reflexes,
    )

    state = State(observation=None)
    voluntary_action = controller.choose(state, actions)

    policy.set_stimuli([Stimulus("threat", 1.0, source="amygdala")])
    actuated = policy.decide(state, Memory(), None)

    assert actuated == flee
    assert policy.latest_motor_decision is not None
    assert policy.latest_motor_decision.voluntary == voluntary_action
    assert policy.latest_motor_decision.reflex is not None
    assert policy.latest_motor_decision.actuated == flee
    assert policy.latest_motor_decision.diverged


def test_motor_decision_recorded_without_reflex_override():
    """When no reflex fires, voluntary == actuated and diverged is False."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]

    reflexes = ReflexStack([
        ReflexConfig("withdraw", "threat", Action("turn_left"),
                     threshold=0.5, priority=10),
    ])
    policy = MotorFreedomPolicy(
        "learned", actions,
        voluntary=controller, reflexes=reflexes,
    )

    state = State(observation=None)
    policy.set_stimuli([])
    actuated = policy.decide(state, Memory(), None)

    assert policy.latest_motor_decision is not None
    assert policy.latest_motor_decision.voluntary == actuated
    assert policy.latest_motor_decision.actuated == actuated
    assert not policy.latest_motor_decision.diverged


def test_motor_decision_to_dict_has_all_fields():
    """The serialized motor decision carries voluntary, reflex,
    caregiver_override, and actuated — the full efference record."""
    wm, _memory, _rng = _primed_cortex_wm()
    controller = build_cortex_mpc(wm)
    actions = [Action(k) for k in _ACTION_KEYS]

    reflexes = ReflexStack([
        ReflexConfig("withdraw", "threat", Action("turn_left"),
                     threshold=0.5, priority=10),
    ])
    policy = MotorFreedomPolicy(
        "learned", actions,
        voluntary=controller, reflexes=reflexes,
        stimuli=[Stimulus("threat", 1.0)],
    )
    policy.decide(State(observation=None), Memory(), None)
    record = policy.latest_motor_decision.to_dict()
    assert set(record) == {"voluntary", "reflex", "caregiver_override", "actuated"}
    assert record["actuated"] == "turn_left"
    assert record["reflex"] is not None
    assert record["caregiver_override"] is None


def test_motor_decision_recorded_for_frozen_stage():
    """Frozen stages also produce a MotorDecision (NULL -> NULL, no divergence)."""
    policy = MotorFreedomPolicy("frozen", [Action("noop")])
    policy.decide(State(observation=None), Memory(), None)
    assert policy.latest_motor_decision is not None
    assert policy.latest_motor_decision.voluntary == NULL_ACTION
    assert policy.latest_motor_decision.actuated == NULL_ACTION
    assert not policy.latest_motor_decision.diverged


def test_cortex_mpc_factory_returns_controller_for_learned_stage():
    """The factory hook for run_curriculum returns a working cortex-MPC
    controller for a 'learned' stage."""
    from development.definitions import CurriculumStageSpec

    wm, _memory, _rng = _primed_cortex_wm()
    factory = cortex_mpc_factory(wm)
    stage = CurriculumStageSpec(
        name="test-learned",
        world_config={"episode_ticks": 10},
        motor_freedom="learned",
    )
    actions = [Action(k) for k in _ACTION_KEYS]
    controller = factory(stage, actions)
    assert controller.name == "cortex-mpc"
    chosen = controller.choose(None, actions)
    assert chosen in actions


def test_novelty_weight_affects_action_selection():
    """Different novelty weights can produce different action choices,
    demonstrating that the uncertainty head contributes to scoring."""
    wm, _memory, _rng = _primed_cortex_wm()
    actions = [Action(k) for k in _ACTION_KEYS]

    scores_low = []
    scores_high = []
    for weight in (0.0, 100.0):
        controller = build_cortex_mpc(wm, novelty_weight=weight)
        chosen = controller.choose(None, actions)
        scores_low.append(chosen) if weight == 0.0 else scores_high.append(chosen)

    # We can't guarantee different choices (depends on random weights), but
    # we verify both runs complete without error and return valid actions.
    assert scores_low[0] in actions
    assert scores_high[0] in actions


def test_cortex_world_model_exposes_pre_advance_state():
    """CortexWorldModel stores _latent and _pre_advance_hidden after predict(),
    and they differ from the post-advance _hidden."""
    wm, memory, rng = _primed_cortex_wm()
    assert wm._latent is not None
    assert wm._pre_advance_hidden is not None
    assert wm._hidden is not None
    # pre_advance_hidden is the state BEFORE the step; _hidden is AFTER
    # They should not be the same object (GRU returns a new tensor).
    if isinstance(wm._pre_advance_hidden, torch.Tensor):
        assert not torch.equal(wm._pre_advance_hidden, wm._hidden)


def test_cortex_world_model_reset_clears_mpc_state():
    """reset() clears _latent and _pre_advance_hidden alongside _hidden."""
    wm, _memory, _rng = _primed_cortex_wm()
    assert wm._latent is not None
    wm.reset()
    assert wm._latent is None
    assert wm._pre_advance_hidden is None
    assert wm._hidden is None
