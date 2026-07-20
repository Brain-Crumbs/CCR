from cognitive_runtime.core.action import NULL_ACTION, Action
from motor.voluntary import MPCController, build_voluntary_controller


def test_mpc_deterministically_selects_highest_scoring_prediction():
    actions = [NULL_ACTION, Action("LEFT"), Action("RIGHT")]
    values = {action.name: value for action, value in zip(actions, [0, 1, 4])}
    controller = MPCController(lambda state, action: values[action.name],
                               lambda prediction, goal: prediction)
    assert controller.choose(None, actions) == Action("RIGHT")
    assert controller.choose(None, [Action("LEFT"), Action("RIGHT")]) == Action("RIGHT")


def test_mpc_treats_nan_as_worst_score_and_keeps_deterministic_order():
    controller = MPCController(
        lambda state, action: {"BAD": float("nan"), "GOOD": 1.0}[action.name],
        lambda prediction, goal: prediction,
    )
    assert controller.choose(None, [Action("BAD"), Action("GOOD")]) == Action("GOOD")

    all_nan = MPCController(lambda state, action: float("nan"), lambda prediction, goal: prediction)
    assert all_nan.choose(None, [Action("FIRST"), Action("SECOND")]) == Action("FIRST")


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


# --------------------------------------------------------------- issue #103: real alt controllers


def test_active_inference_controller_decodes_the_forecast_that_matches_the_goal():
    import pytest

    torch = pytest.importorskip("torch")
    from motor.policy import ActiveInferenceState, build_active_inference_controller

    class FakeCortex:
        latent_width = 2

        def step(self, latent, action_idx, hidden):
            targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            return targets[action_idx], hidden

    controller = build_active_inference_controller(FakeCortex(), ["LEFT", "RIGHT"])
    assert controller.name == "active"

    state = ActiveInferenceState(latent=torch.zeros(1, 2), hidden=None)
    actions = [Action("LEFT"), Action("RIGHT")]
    goal = torch.tensor([0.1, 0.9])  # closer to RIGHT's (0, 1) forecast
    assert controller.choose(state, actions, goal=goal) == Action("RIGHT")

    goal = torch.tensor([0.9, 0.1])  # closer to LEFT's (1, 0) forecast
    assert controller.choose(state, actions, goal=goal) == Action("LEFT")


def test_active_inference_controller_encodes_a_raw_pixel_goal_in_nchw_layout():
    import pytest

    torch = pytest.importorskip("torch")
    from cognitive_runtime.neural.pixel_stream_encoder import PixelStreamEncoder
    from motor.policy import ActiveInferenceState, build_active_inference_controller

    pixel_shape = (8, 8, 3)  # (H, W, C)
    encoder = PixelStreamEncoder(pixel_shape, latent_width=4)

    class FakeCortex:
        latent_width = 4

        def __init__(self, encoder):
            self.encoder = encoder

        def step(self, latent, action_idx, hidden):
            return latent, hidden

    controller = build_active_inference_controller(FakeCortex(encoder), ["LEFT", "RIGHT"])
    state = ActiveInferenceState(latent=torch.zeros(1, 4), hidden=None)
    goal = torch.randint(0, 256, pixel_shape, dtype=torch.uint8)  # raw H x W x C frame

    # Would previously raise inside PixelStreamEncoder.forward: unsqueeze(0)
    # on a raw H x W x C goal produces N x H x W x C, not the N x C x H x W
    # the encoder requires.
    chosen = controller.choose(state, [Action("LEFT"), Action("RIGHT")], goal=goal)
    assert chosen in (Action("LEFT"), Action("RIGHT"))


def test_active_inference_goal_rejects_invalid_dimensions_and_batched_latents():
    import pytest

    torch = pytest.importorskip("torch")
    from motor.policy import _encode_goal

    class FakeCortex:
        latent_width = 4

    for goal, message in [
        (torch.tensor(1.0), "at least 1D"),
        (torch.zeros(8, 8), "H x W x C"),
        (torch.zeros(2, 4), "one latent goal"),
    ]:
        with pytest.raises(ValueError, match=message):
            _encode_goal(FakeCortex(), goal)


def test_imagination_actor_handles_unbatched_latent():
    import pytest

    torch = pytest.importorskip("torch")
    from motor.policy import ImaginationActor

    actor = ImaginationActor(latent_width=2, n_actions=2, hidden_dim=2)
    with torch.no_grad():
        for parameter in actor.actor.parameters():
            parameter.zero_()
        actor.actor[-1].bias.copy_(torch.tensor([0.0, 1.0]))
    assert actor.act(torch.zeros(2)) == 1


def test_callable_policy_controller_allows_unoffered_universal_null_action():
    from motor.policy import build_policy_controller

    class EmptyPolicy:
        def emit(self, state, memory, prediction):
            return []

    controller = build_policy_controller(EmptyPolicy())
    assert controller.choose((None, None, None), [Action("MOVE")]) == NULL_ACTION


def test_active_inference_controller_requires_a_goal():
    import pytest

    pytest.importorskip("torch")
    from motor.policy import ActiveInferenceState, build_active_inference_controller

    class FakeCortex:
        latent_width = 2

        def step(self, latent, action_idx, hidden):
            return latent, hidden

    controller = build_active_inference_controller(FakeCortex(), ["A", "B"])
    state = ActiveInferenceState(latent=None, hidden=None)
    with pytest.raises(ValueError):
        controller.choose(state, [Action("A"), Action("B")])


def test_imagination_actor_learns_from_dreamed_rollouts_and_satisfies_seam():
    import pytest

    torch = pytest.importorskip("torch")
    from motor.policy import ImaginationActor, build_imagination_controller

    class FakeDreamCortex:
        """Reward equals the imagined action's own index, so the actor has
        a clear, deterministic incentive to prefer action 1 over action 0."""

        latent_width = 2

        def step(self, latent, action_idx, hidden):
            return latent, action_idx

        def heads(self, hidden):
            reward = hidden.float()
            zeros = torch.zeros_like(reward)
            return reward, zeros, zeros, zeros

    torch.manual_seed(0)
    actor = ImaginationActor(latent_width=2, n_actions=2, hidden_dim=8)
    cortex = FakeDreamCortex()
    seed_latent = torch.zeros(1, 2)
    hidden = torch.zeros(1, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    for _ in range(150):
        actor.train_on_dream(cortex, seed_latent, hidden, horizon=3, generator=generator)

    controller = build_imagination_controller(actor, ["LEFT", "RIGHT"])
    assert controller.name == "imagination"
    chosen = controller.choose(seed_latent, [Action("LEFT"), Action("RIGHT")])
    assert chosen == Action("RIGHT")


def test_policy_controller_wires_the_actor_critic_policy_head():
    import pytest

    pytest.importorskip("torch")
    from cognitive_runtime.core.memory import Memory
    from cognitive_runtime.core.observation import Observation
    from cognitive_runtime.core.perception import State
    from cognitive_runtime.core.streams.fusion import LatentState
    from cognitive_runtime.neural import MLPPolicyModel, MLPValueModel
    from cognitive_runtime.policies.actor_critic import ActorCriticPolicy, world_feature_width
    from motor.policy import build_policy_controller

    action_keys = ["NULL", "MOVE_FORWARD", "ATTACK"]
    wf_width = world_feature_width(action_keys)
    policy_model = MLPPolicyModel(2, wf_width, len(action_keys), hidden_dim=8)
    critic_model = MLPValueModel(2, wf_width, hidden_dim=8)
    policy = ActorCriticPolicy(policy_model, critic_model, action_keys, training=False)

    controller = build_policy_controller(policy)
    assert controller.name == "policy"

    memory = Memory()
    memory.set_fused_latent(LatentState(vector=[0.0, 0.0], slices={}, layout_hash="layout-a"))
    state = State(Observation(timestamp=0.0, tick=0, data={}))
    action_space = [Action.from_key(key) for key in action_keys]

    chosen = controller.choose((state, memory, None), action_space, goal=None)
    assert chosen.key() in action_keys
    if policy.latest_decision.action_key == "NULL":
        assert chosen == NULL_ACTION

