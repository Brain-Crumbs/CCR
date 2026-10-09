"""Optional, bounded CPU checks; torch is imported only inside test bodies.

Synthetic constant 4x4 RGB frames are project-authored under MIT, seed 7.
No model downloads, datasets, market payloads or live capability claims.
"""
from dataclasses import replace
import copy
import hashlib
import json

import pytest

pytestmark = pytest.mark.market_extended


@pytest.mark.parametrize("name", ["adam", "adamw"])
def test_toy_step_json_state_resume_and_reject_wrong_class(name):
    torch = pytest.importorskip("torch")
    from cognitive_runtime.training.optimizer_config import build_optimizer, optimizer_manifest, restore_optimizer
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    config = {"name": name, "lr": 0.03, "weight_decay": 0.1, "betas": [0.6, 0.8]}
    optimizer = build_optimizer(torch, model.parameters(), config)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    manifest = json.loads(json.dumps(optimizer_manifest(model, optimizer)))
    target = torch.nn.Linear(2, 1)
    target.load_state_dict(model.state_dict())
    resumed = build_optimizer(torch, target.parameters(), config)
    restore_optimizer(target, resumed, copy.deepcopy(optimizer.state_dict()), manifest)
    assert optimizer_manifest(target, resumed) == manifest
    for current, opt in ((model, optimizer), (target, resumed)):
        opt.zero_grad()
        current(torch.ones(1, 2)).square().sum().backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), target.parameters()))
    wrong = build_optimizer(torch, target.parameters(), {**config, "name": "adamw" if name == "adam" else "adam"})
    with pytest.raises(ValueError, match="effective optimizer"):
        restore_optimizer(target, wrong, resumed.state_dict(), manifest)


@pytest.mark.parametrize("objective", ["windowed_rollout", "autoregressive"])
@pytest.mark.parametrize("name", ["adam", "adamw"])
def test_both_real_trainer_paths_resume_exactly(objective, name):
    torch = pytest.importorskip("torch")
    import numpy as np
    from cognitive_runtime.training.action_world_model import ActionWorldModelConfig, ActionSequenceDataset, EpisodeActionFrames, train_action_world_model
    from cognitive_runtime.training.optimizer_config import OPTIMIZER_FORMAT, resolve_optimizer
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        frames = [np.full((4, 4, 3), i * 16, dtype=np.uint8) for i in range(7)]
        assert hashlib.sha256(b"".join(frame.tobytes() for frame in frames)).hexdigest() == "2be67a06c2b553ff910234d8c14b3438e8ecbb704fd392857b2ca40687f93724"
        episode = EpisodeActionFrames(session_dir="synthetic", episode_id="one", frames=frames, actions=[0, 1] * 3, yaw=[None] * 7, ticks=list(range(7)))
        dataset = ActionSequenceDataset(episodes=[episode], action_keys=["NULL", "LEFT"], pixel_shape=(4, 4, 3), sources=["synthetic"])
        config = ActionWorldModelConfig(latent_width=4, hidden_dim=8, action_embed_dim=2, reconstruction_size=4, horizons_ticks=(1,), warmup_frames=1, rollout_frames=2, batch_size=2, epochs=2, lr=0.002, seed=7, device="cpu", collapse_gate_enabled=False, context_length_curriculum=False, training_objective=objective, optimizer_behavior=OPTIMIZER_FORMAT, optimizer=resolve_optimizer({"name": name, "lr": 0.002, "weight_decay": 0.1, "betas": [0.7, 0.8]}))
        full, full_stats = train_action_world_model(dataset, config)
        partial, stats = train_action_world_model(dataset, replace(config, epochs=1))
        resumed, resumed_stats = train_action_world_model(dataset, config, initial_model=partial, resume_state=stats["resume_state"])
        assert full_stats["effective_optimizer"]["class"].lower() == name
        assert resumed_stats["effective_optimizer"] == full_stats["effective_optimizer"]
        assert resumed_stats["resume_state"].effective_optimizer == full_stats["effective_optimizer"]
        for key, value in full.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[key]), key
        wrong = replace(config, epochs=3, optimizer={**config.optimizer, "weight_decay": 0.2})
        with pytest.raises(ValueError, match="effective optimizer"):
            train_action_world_model(dataset, wrong, initial_model=resumed, resume_state=resumed_stats["resume_state"])
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def test_resume_rejects_reordered_parameter_state_ids():
    torch = pytest.importorskip("torch")
    from cognitive_runtime.training.optimizer_config import build_optimizer, optimizer_manifest, restore_optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(torch, model.parameters(), {})
    state = optimizer.state_dict()
    state["param_groups"][0]["params"].reverse()
    with pytest.raises(ValueError, match="state groups"):
        restore_optimizer(model, optimizer, state, optimizer_manifest(model, optimizer))
