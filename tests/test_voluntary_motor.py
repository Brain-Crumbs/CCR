from cognitive_runtime.core.action import NULL_ACTION, Action
from motor.voluntary import MPCController, build_voluntary_controller


def test_mpc_deterministically_selects_highest_scoring_prediction():
    actions = [NULL_ACTION, Action("LEFT"), Action("RIGHT")]
    values = {action.name: value for action, value in zip(actions, [0, 1, 4])}
    controller = MPCController(lambda state, action: values[action.name],
                               lambda prediction, goal: prediction)
    assert controller.choose(None, actions) == Action("RIGHT")
    assert controller.choose(None, [Action("LEFT"), Action("RIGHT")]) == Action("RIGHT")


def test_all_controller_variants_share_the_seam_and_mpc_is_default():
    actions = [NULL_ACTION, Action("GO")]
    assert build_voluntary_controller(
        predictor=lambda state, action: action.name == "GO",
        scorer=lambda prediction, goal: prediction,
    ).name == "mpc"
    alternatives = {name: (lambda state, choices, goal: choices[0])
                    for name in ("active", "imagination", "policy")}
    for name in alternatives:
        controller = build_voluntary_controller(name, alternatives=alternatives)
        assert controller.name == name
        assert controller.choose(None, actions) == NULL_ACTION


def test_mpc_does_not_construct_a_gradient_graph():
    import pytest

    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    controller = MPCController(lambda state, action: parameter * action.param("score", 0),
                               lambda prediction, goal: prediction.item())
    controller.choose(None, [Action.make("A", score=1), Action.make("B", score=2)])
    assert parameter.grad is None

