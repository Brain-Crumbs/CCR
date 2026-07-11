from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.neural import (  # noqa: E402
    CheckpointCompatibilityError,
    LatentFusionModel,
    NeuralAgentCheckpoint,
    PolicyModel,
    StreamEncoderModule,
    ValueModel,
    WorldModel,
    WorldModelOutput,
    action_space_hash,
    checkpoint_metadata_path,
)


LAYOUT_HASH = "layout-a"
ACTION_KEYS = ["NULL", "JUMP"]


class ToyEncoder(StreamEncoderModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def width(self, spec=None):
        return 2

    def encode_latent(self, events, spec=None):
        if not events:
            return None
        payload = events[-1].payload
        return self.forward(torch.tensor(payload, dtype=torch.float32))

    def predict_next_latent(self, latent_slice):
        return {"next": self.linear(latent_slice)}

    def forward(self, x):
        return self.linear(x)


class ToyFusion(LatentFusionModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def fused_width(self):
        return 3

    def forward(self, latents, presence_mask, recency):
        return self.linear(latents)


class ToyWorld(WorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 7)

    def forward(self, fused_latent, action_onehot):
        raw = self.linear(torch.cat([fused_latent, action_onehot], dim=1))
        return WorldModelOutput(
            next_latent=raw[:, :3],
            reward=raw[:, 3],
            terminal_logit=raw[:, 4],
            risk=raw[:, 5],
            prediction_error=raw[:, 6],
        )


class ToyPolicy(PolicyModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

    def action_space_size(self):
        return 2

    def forward(self, fused_latent, world_features):
        return self.linear(torch.cat([fused_latent, world_features], dim=1))


class ToyCritic(ValueModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, fused_latent, world_features):
        return self.linear(torch.cat([fused_latent, world_features], dim=1)).squeeze(-1)


def _modules(seed: int = 0):
    torch.manual_seed(seed)
    modules = {
        "encoder": ToyEncoder(),
        "fusion": ToyFusion(),
        "world": ToyWorld(),
        "policy": ToyPolicy(),
        "critic": ToyCritic(),
    }
    return modules


def _all_params(modules):
    params = []
    for module in modules.values():
        params.extend(module.parameters())
    return params


def _manager(path, modules, optimizer=None, *, layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS):
    return NeuralAgentCheckpoint(
        str(path),
        layout_hash=layout_hash,
        action_keys=action_keys,
        encoders={"stream_encoder.body_health": modules["encoder"]},
        fusion=modules["fusion"],
        world_model=modules["world"],
        policy=modules["policy"],
        critic=modules["critic"],
        optimizers={"main": optimizer} if optimizer is not None else {},
        replay_metadata={
            "buffer_size": 128,
            "priority": "reward/death/damage",
            "transitions_seen": 42,
        },
        training_stats={"loss": 0.25},
        training_ticks=3,
    )


def _fixed_outputs(modules):
    event = StreamEvent("body.health", "body", 0.0, 0, [0.25, -0.5])
    latent = modules["encoder"].encode_latent([event]).unsqueeze(0)
    fused = modules["fusion"](latent, torch.ones(1, 1), torch.ones(1, 1))
    action = torch.tensor([[0.0, 1.0]])
    world = modules["world"](fused, action)
    world_features = torch.stack([world.reward, world.risk], dim=1)
    return {
        "encoder": latent,
        "fusion": fused,
        "world_next": world.next_latent,
        "world_reward": world.reward,
        "policy": modules["policy"](fused, world_features),
        "critic": modules["critic"](fused, world_features),
    }


def _training_step(modules, optimizer):
    optimizer.zero_grad()
    outputs = _fixed_outputs(modules)
    loss = (
        outputs["policy"].pow(2).mean()
        + outputs["critic"].pow(2).mean()
        + outputs["world_next"].pow(2).mean()
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def test_neural_checkpoint_round_trip_restores_identical_module_outputs(tmp_path):
    modules = _modules(seed=1)
    before = _fixed_outputs(modules)
    path = tmp_path / "agent.pt"

    metadata = _manager(path, modules).save(reason="roundtrip")

    restored = _modules(seed=999)
    loaded = _manager(path, restored).load()
    after = _fixed_outputs(restored)

    assert metadata["format"] == "neural-agent-checkpoint-v1"
    assert loaded["reason"] == "roundtrip"
    for key in before:
        assert torch.allclose(before[key], after[key])


def test_neural_checkpoint_resume_continues_ticks_and_optimizer_state(tmp_path):
    path = tmp_path / "resume.pt"
    original = _modules(seed=2)
    original_optimizer = torch.optim.Adam(_all_params(original), lr=0.01)
    _training_step(original, original_optimizer)
    manager = _manager(path, original, original_optimizer)
    manager.training_ticks = 11
    manager.save(reason="interrupt")

    resumed = _modules(seed=999)
    resumed_optimizer = torch.optim.Adam(_all_params(resumed), lr=0.01)
    resumed_manager = _manager(path, resumed, resumed_optimizer)
    metadata = resumed_manager.resume()

    assert metadata["training_ticks"] == 11
    assert resumed_manager.training_ticks == 11
    assert resumed_optimizer.state_dict()["state"]

    _training_step(original, original_optimizer)
    _training_step(resumed, resumed_optimizer)

    for left, right in zip(_all_params(original), _all_params(resumed)):
        assert torch.allclose(left, right)


def test_neural_checkpoint_mismatched_layout_hash_raises_actionable_error(tmp_path):
    path = tmp_path / "agent.pt"
    _manager(path, _modules(seed=3)).save()

    with pytest.raises(CheckpointCompatibilityError, match="layout.*stream catalog"):
        _manager(path, _modules(seed=4), layout_hash="layout-b").load()


def test_neural_checkpoint_mismatched_action_space_hash_raises(tmp_path):
    path = tmp_path / "agent.pt"
    _manager(path, _modules(seed=5)).save()

    with pytest.raises(CheckpointCompatibilityError, match="action-space.*Program action space"):
        _manager(path, _modules(seed=6), action_keys=["NULL", "ATTACK"]).load()


# ------------------------------------------------ action-space growth (#42)


def _mlp_pair(action_keys, *, fused_width=3, hidden_dim=6, seed=0):
    from cognitive_runtime.neural import MLPPolicyModel, MLPValueModel

    torch.manual_seed(seed)
    world_feature_width = 4 + len(action_keys)  # WORLD_FEATURE_BASE_WIDTH + n_actions
    policy = MLPPolicyModel(
        fused_width, world_feature_width, len(action_keys),
        hidden_dim=hidden_dim, depth=1, action_keys=action_keys,
    )
    critic = MLPValueModel(
        fused_width, world_feature_width, hidden_dim=hidden_dim, depth=1,
        action_keys=action_keys,
    )
    return policy, critic, fused_width, world_feature_width


def test_neural_checkpoint_action_space_growth_preserves_old_weights_and_inits_new(tmp_path):
    old_keys = ["NULL", "JUMP"]
    new_keys = ["NULL", "JUMP", "CRAFT:recipe=log_to_planks"]
    path = tmp_path / "agent.pt"

    old_policy, old_critic, fused_width, old_wfw = _mlp_pair(old_keys, seed=1)
    NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=old_keys,
        policy=old_policy, critic=old_critic,
    ).save(reason="pre-growth")

    new_policy, new_critic, _, new_wfw = _mlp_pair(new_keys, fused_width=fused_width, seed=2)
    # Snapshot the live model's own fresh init for the newly-added rows/columns
    # *before* loading, so we can prove the migration leaves them alone.
    fresh_logits_tail = new_policy.logits_head.weight[len(old_keys):].clone()
    fresh_trunk_tail = new_policy.trunk[0].weight[:, fused_width + old_wfw:].clone()

    new_manager = NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=new_keys,
        policy=new_policy, critic=new_critic,
    )
    metadata = new_manager.load(allow_action_space_growth=True)

    assert metadata["action_keys"] == old_keys  # what the checkpoint itself held
    assert new_manager.action_keys == new_keys  # migrated to the live action space

    # Existing action rows/columns carry over exactly.
    assert torch.allclose(new_policy.logits_head.weight[: len(old_keys)], old_policy.logits_head.weight)
    assert torch.allclose(new_policy.logits_head.bias[: len(old_keys)], old_policy.logits_head.bias)
    assert torch.allclose(
        new_policy.trunk[0].weight[:, : fused_width + old_wfw], old_policy.trunk[0].weight
    )
    assert torch.allclose(
        new_critic.trunk[0].weight[:, : fused_width + old_wfw], old_critic.trunk[0].weight
    )

    # New action rows/columns keep the live model's own fresh init untouched
    # (not zeroed, not garbage from a shape-mismatched load).
    assert torch.allclose(new_policy.logits_head.weight[len(old_keys):], fresh_logits_tail)
    assert torch.allclose(new_policy.trunk[0].weight[:, fused_width + old_wfw:], fresh_trunk_tail)

    # The migrated modules are usable at the new size.
    logits = new_policy(torch.zeros(1, fused_width), torch.zeros(1, new_wfw))
    assert logits.shape == (1, len(new_keys))
    value = new_critic(torch.zeros(1, fused_width), torch.zeros(1, new_wfw))
    assert value.shape == (1,)


def test_neural_checkpoint_action_space_growth_requires_opt_in(tmp_path):
    old_keys = ["NULL", "JUMP"]
    new_keys = ["NULL", "JUMP", "ATTACK"]
    path = tmp_path / "agent.pt"

    old_policy, old_critic, fused_width, _ = _mlp_pair(old_keys, seed=3)
    NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=old_keys,
        policy=old_policy, critic=old_critic,
    ).save()

    new_policy, new_critic, _, _ = _mlp_pair(new_keys, fused_width=fused_width, seed=4)
    new_manager = NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=new_keys,
        policy=new_policy, critic=new_critic,
    )
    with pytest.raises(CheckpointCompatibilityError, match="action-space.*allow_action_space_growth"):
        new_manager.load()  # allow_action_space_growth defaults to False


def test_neural_checkpoint_action_space_growth_rejects_non_prefix_change(tmp_path):
    """A reordered/non-superset action space is not a growth case, even with
    the opt-in -- it still fails loudly rather than silently mis-migrating."""
    old_keys = ["NULL", "JUMP"]
    reordered_keys = ["JUMP", "NULL", "ATTACK"]
    path = tmp_path / "agent.pt"

    old_policy, old_critic, fused_width, _ = _mlp_pair(old_keys, seed=5)
    NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=old_keys,
        policy=old_policy, critic=old_critic,
    ).save()

    new_policy, new_critic, _, _ = _mlp_pair(reordered_keys, fused_width=fused_width, seed=6)
    new_manager = NeuralAgentCheckpoint(
        str(path), layout_hash=LAYOUT_HASH, action_keys=reordered_keys,
        policy=new_policy, critic=new_critic,
    )
    with pytest.raises(CheckpointCompatibilityError):
        new_manager.load(allow_action_space_growth=True)


def test_neural_checkpoint_sidecar_is_json_inspectable_without_torch_load(tmp_path):
    path = tmp_path / "agent.pt"
    _manager(path, _modules(seed=7)).save(reason="episode_end")

    sidecar = checkpoint_metadata_path(str(path))
    with open(sidecar, encoding="utf-8") as fh:
        metadata = json.load(fh)

    assert metadata["format"] == "neural-agent-checkpoint-v1"
    assert metadata["reason"] == "episode_end"
    assert metadata["layout_hash"] == LAYOUT_HASH
    assert metadata["action_space_hash"] == action_space_hash(ACTION_KEYS)
    assert metadata["replay_metadata"]["buffer_size"] == 128
    assert metadata["modules"]["encoders"]["stream_encoder.body_health"]["state_keys"] == [
        "linear.bias",
        "linear.weight",
    ]
    assert metadata["training_stats"] == {"loss": 0.25}

