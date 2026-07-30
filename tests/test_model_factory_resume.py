"""Resumable trainer regression coverage (issue #220, MF-B3).

Exact resume equivalence requires a **declared deterministic device/
precision policy** (the acceptance criteria's own wording): even two
independent, non-resumed calls to :func:`train_action_world_model` with
identical arguments are not bit-for-bit identical on stock CPU PyTorch,
because some CPU kernels use nondeterministic multi-threaded reductions.
``torch.use_deterministic_algorithms(True)`` plus a single thread removes
that noise; every test below runs under both. This is a real precondition
of the acceptance criterion, not a workaround for a bug in the trainer.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.training.action_world_model import (  # noqa: E402
    ActionSequenceDataset,
    ActionWorldModelConfig,
    EpisodeActionFrames,
    train_action_world_model,
)
from cognitive_runtime.training.model_factory.checkpoint import (  # noqa: E402
    BestValidationTracker,
    TrainerResumeState,
    checkpoint_due,
    load_factory_checkpoint,
    save_factory_checkpoint,
)
from cognitive_runtime.training.model_factory.contracts import (  # noqa: E402
    ArchitectureContract,
    DataContract,
    TrainingContract,
)

PIXEL_SHAPE = (4, 4, 3)
ACTION_KEYS = ("NULL", "LEFT", "RIGHT")


@pytest.fixture(autouse=True)
def _deterministic_cpu_policy():
    """The declared deterministic device/precision policy every resume-
    equivalence test in this file relies on (see module docstring)."""
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.set_num_threads(previous_threads)


def _frame(value: int):
    import numpy as np

    return np.full(PIXEL_SHAPE, value, dtype="uint8")


def _episode(n_frames: int, episode_id: str, seed: int) -> EpisodeActionFrames:
    rng = random.Random(seed)
    frames = [_frame(rng.randrange(0, 256)) for _ in range(n_frames)]
    actions = [rng.randrange(len(ACTION_KEYS)) for _ in range(n_frames - 1)]
    return EpisodeActionFrames(
        session_dir="synthetic-resume", episode_id=episode_id,
        frames=frames, actions=actions, yaw=[None] * n_frames, ticks=list(range(n_frames)),
    )


def _dataset(n_episodes: int = 3, n_frames: int = 14, seed: int = 0) -> ActionSequenceDataset:
    episodes = [_episode(n_frames, f"ep-{i}", seed + i) for i in range(n_episodes)]
    return ActionSequenceDataset(
        episodes=episodes, action_keys=list(ACTION_KEYS), pixel_shape=PIXEL_SHAPE,
        sources=[f"synthetic-resume/ep-{i}" for i in range(n_episodes)],
    )


def _cfg(**overrides) -> ActionWorldModelConfig:
    base = dict(
        latent_width=8, hidden_dim=16, action_embed_dim=4, reconstruction_size=4,
        warmup_frames=2, rollout_frames=3, batch_size=4, lr=1e-2,
        scheduled_sampling_p=0.5, closed_loop_pixel_loss_weight=0.1,
        closed_loop_latent_loss_weight=0.1, seed=123,
    )
    base.update(overrides)
    return ActionWorldModelConfig(**base)


def _assert_state_dict_equal(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), f"mismatch at {key!r}"


@pytest.mark.parametrize("objective", ["windowed_rollout", "autoregressive"])
def test_resume_matches_uninterrupted_training(objective):
    """Acceptance: training 6 epochs uninterrupted, and training 3 then
    resuming 3, must produce identical weights and identical per-epoch
    losses -- for both objectives."""
    dataset = _dataset()

    model_full, stats_full = train_action_world_model(
        dataset, _cfg(epochs=6, training_objective=objective),
    )

    model_first, stats_first = train_action_world_model(
        dataset, _cfg(epochs=3, training_objective=objective),
    )
    model_resumed, stats_resumed = train_action_world_model(
        dataset, _cfg(epochs=6, training_objective=objective),
        initial_model=model_first, resume_state=stats_first["resume_state"],
    )

    _assert_state_dict_equal(model_full.state_dict(), model_resumed.state_dict())
    assert (
        stats_full["loss_curves"]["total_loss"][3:6]
        == stats_resumed["loss_curves"]["total_loss"]
    )
    assert stats_resumed["global_step"] == stats_full["global_step"]
    # resume_state=None (the fresh 6-epoch call) must not have been perturbed
    # by anything resume-specific: it still reflects a from-scratch trainer.
    assert stats_full["resume_state"].epoch == 6


def test_resume_matches_uninterrupted_training_with_transition_weights():
    """Weighted window sampling (issue #202) is a distinct RNG consumer from
    the epoch permutation; it must resume exactly too."""
    dataset = _dataset()
    transition_weights = [
        [1.0 + 0.25 * i for i in range(len(episode.actions))]
        for episode in dataset.episodes
    ]

    model_full, stats_full = train_action_world_model(
        dataset, _cfg(epochs=6, training_objective="windowed_rollout"),
        transition_weights=transition_weights,
    )

    model_first, stats_first = train_action_world_model(
        dataset, _cfg(epochs=3, training_objective="windowed_rollout"),
        transition_weights=transition_weights,
    )
    model_resumed, stats_resumed = train_action_world_model(
        dataset, _cfg(epochs=6, training_objective="windowed_rollout"),
        initial_model=model_first, resume_state=stats_first["resume_state"],
        transition_weights=transition_weights,
    )

    _assert_state_dict_equal(model_full.state_dict(), model_resumed.state_dict())
    assert (
        stats_full["loss_curves"]["total_loss"][3:6]
        == stats_resumed["loss_curves"]["total_loss"]
    )


def test_resume_state_none_is_unchanged_from_pre_220_behavior():
    """The full existing ``test_action_world_model.py`` suite already proves
    this at large; this is the narrow, explicit version: two identical
    fresh (non-resumed) calls under the declared deterministic policy must
    match exactly, and adding the new keyword-only arguments with their
    defaults must not perturb a single value."""
    dataset = _dataset()
    cfg = _cfg(epochs=3, training_objective="windowed_rollout")

    model_a, stats_a = train_action_world_model(dataset, cfg)
    model_b, stats_b = train_action_world_model(dataset, cfg, resume_state=None)

    _assert_state_dict_equal(model_a.state_dict(), model_b.state_dict())
    assert stats_a["loss_curves"] == stats_b["loss_curves"]


def test_resume_rejects_a_checkpoint_epoch_past_the_configured_budget():
    dataset = _dataset()
    _model, stats = train_action_world_model(
        dataset, _cfg(epochs=3, training_objective="windowed_rollout"),
    )
    with pytest.raises(ValueError, match="already reached or passed"):
        train_action_world_model(
            dataset, _cfg(epochs=2, training_objective="windowed_rollout"),
            resume_state=stats["resume_state"],
        )


def test_resume_rejects_an_already_complete_checkpoint_instead_of_crashing():
    """A checkpoint saved exactly at the epoch budget (e.g. the last
    periodic save before a job recorded completion) must fail clearly, not
    raise IndexError deep inside empty loss-curve bookkeeping."""
    dataset = _dataset()
    _model, stats = train_action_world_model(
        dataset, _cfg(epochs=3, training_objective="windowed_rollout"),
    )
    with pytest.raises(ValueError, match="already reached or passed"):
        train_action_world_model(
            dataset, _cfg(epochs=3, training_objective="windowed_rollout"),
            resume_state=stats["resume_state"],
        )


def test_checkpoint_selection_mode_must_be_min_or_max():
    dataset = _dataset()
    with pytest.raises(ValueError, match="min.*max"):
        train_action_world_model(
            dataset, _cfg(epochs=1, training_objective="windowed_rollout"),
            checkpoint_selection_mode="lowest",
        )


def test_checkpoint_due_cadence_hook():
    assert checkpoint_due(3, 3) is True
    assert checkpoint_due(6, 3) is True
    assert checkpoint_due(2, 3) is False
    assert checkpoint_due(0, 3) is False
    assert checkpoint_due(5, None) is False
    assert checkpoint_due(5, 0) is False


def test_best_validation_tracker_offers_only_on_improvement():
    tracker = BestValidationTracker(mode="min")
    assert tracker.offer(5.0) is True
    assert tracker.offer(6.0) is False
    assert tracker.offer(4.0) is True
    assert tracker.best == 4.0

    maximizing = BestValidationTracker(mode="max")
    assert maximizing.offer(1.0) is True
    assert maximizing.offer(0.5) is False
    assert maximizing.offer(2.0) is True
    assert maximizing.best == 2.0


def _contracts_for(model):
    architecture = ArchitectureContract(
        pixel_shape=tuple(model.pixel_shape), rgb_preprocessing_version="v1",
        action_vocabulary=tuple(model.action_keys), workspace_modalities={},
        workspace_layout_hash=None, latent_width=model.latent_width,
        hidden_dim=model.hidden_dim, action_embed_dim=model.config.action_embed_dim,
        reconstruction_shape=tuple(model.reconstruction_shape),
        visual_architecture=model.visual_architecture,
        semantic_classes=model.config.semantic_classes,
        horizons_ticks=tuple(model.horizons_ticks),
        direct_horizon_topology=tuple(model.horizons_ticks),
        backbone=model.config.backbone, context_length=model.config.context_length,
        backbone_kwargs=dict(model.config.backbone_kwargs),
    )
    data = DataContract(
        world="test", backend="test", program_config={}, scenario_names=(),
        scenario_code_version="v1", train_session_ids=(), train_session_hashes=(),
        validation_session_ids=(), validation_session_hashes=(), test_session_ids=(),
        test_session_hashes=(), seed_assignments={}, pixel_provenance="test",
        semantic_vocabulary_version="v1", preprocessing_version="v1",
        horizons_ticks=tuple(model.horizons_ticks), ticks_per_frame=1.0,
    )
    training = TrainingContract(
        objective="test", optimizer={
            "name": "Adam", "lr": 0.01, "betas": (0.9, 0.999),
            "eps": 1e-8, "weight_decay": 0.0, "grad_clip": None,
        }, batch_size=4, seed=123, rollout_frames=3, warmup_frames=2,
        scheduled_sampling_p=0.5, loss_weights={},
        checkpoint_selection_policy={"metric": "synthetic", "mode": "min"},
        device="cpu", precision="fp32", determinism_policy={"seed": 123},
    )
    return architecture, data, training


def test_checkpoint_cadence_and_best_validation_fire_independently(tmp_path):
    """Periodic checkpoints appear at the declared cadence; best-validation
    tracks a supplied validation metric, never the training loss -- and
    each saved snapshot is independently loadable and resumable."""
    dataset = _dataset()
    readings = [5.0, 6.0, 4.0, 7.0, 3.0, 8.0]  # per-epoch validation readings
    calls = []

    def validation_fn(model):
        calls.append(len(calls))
        return readings[len(calls) - 1]

    saved = {}

    def on_checkpoint(step_model, state, *, is_best):
        # The hook receives the live model explicitly (issue #253 review):
        # for a fresh call it is built inside train_action_world_model and
        # not returned until every epoch finishes, so this is the only way
        # a caller can reach the epoch-specific weights to persist.
        architecture, data, training = _contracts_for(step_model)
        path = tmp_path / f"checkpoint-epoch-{state.epoch}.pt"
        save_factory_checkpoint(
            str(path), step_model, torch.optim.Adam(step_model.parameters()), None,
            {"epoch": state.epoch, "global_step": state.global_step,
             "best_validation_metric": state.best_validation_metric},
            architecture_contract=architecture, data_contract_hash=data,
            training_contract=training,
        )
        saved[state.epoch] = (state, is_best, path, architecture, data, training)

    model, stats = train_action_world_model(
        dataset, _cfg(epochs=6, training_objective="windowed_rollout"),
        checkpoint_cadence_epochs=3, on_checkpoint=on_checkpoint,
        validation_fn=validation_fn,
    )

    # cadence=3 -> due at {3, 6}; readings improve (new min) at {1, 3, 5}.
    assert set(saved) == {1, 3, 5, 6}
    assert saved[1][1] is True and saved[1][0].best_validation_metric == 5.0
    assert saved[3][1] is True and saved[3][0].best_validation_metric == 4.0
    assert saved[5][1] is True and saved[5][0].best_validation_metric == 3.0
    assert saved[6][1] is False and saved[6][0].best_validation_metric == 3.0
    # best-validation tracks the declared metric, never the training loss.
    training_losses = set(stats["loss_curves"]["total_loss"])
    assert 3.0 not in training_losses and 5.0 not in training_losses
    assert stats["resume_state"].best_validation_metric == 3.0
    assert len(calls) == 6

    # Each snapshot is independently loadable and genuinely holds *that*
    # epoch's weights -- not the run's final weights mislabeled under every
    # earlier epoch (the regression this test guards against).
    loaded_weights = {}
    for epoch, (state, _is_best, path, architecture, data, training) in saved.items():
        loaded = load_factory_checkpoint(
            str(path), model=type(model)(model.pixel_shape, model.action_keys, model.config),
            optimizer=torch.optim.Adam(model.parameters()), resume=True,
            architecture_contract=architecture, data_contract_hash=data,
            training_contract=training,
        )
        assert loaded.resumed is True
        assert loaded.trainer_state["epoch"] == epoch
        loaded_weights[epoch] = {k: v.clone() for k, v in loaded.model.state_dict().items()}
    any_key = next(iter(loaded_weights[1]))
    assert not torch.equal(loaded_weights[1][any_key], loaded_weights[6][any_key])


def test_validation_runs_every_epoch_without_a_checkpoint_callback():
    """``validation_fn`` must run even when the caller only wants the
    tracked best metric and never configures ``on_checkpoint``."""
    dataset = _dataset()
    readings = [5.0, 3.0, 4.0]
    calls = []

    def validation_fn(model):
        calls.append(len(calls))
        return readings[len(calls) - 1]

    _model, stats = train_action_world_model(
        dataset, _cfg(epochs=3, training_objective="windowed_rollout"),
        validation_fn=validation_fn,
    )

    assert len(calls) == 3
    assert stats["resume_state"].best_validation_metric == 3.0


def test_resume_matches_uninterrupted_training_with_ema_target():
    """The EMA target encoder is a second, separately-evolving model whose
    Polyak-averaged history must be restored explicitly at resume, not
    rebuilt as a fresh copy of the just-resumed online weights."""
    dataset = _dataset()
    cfg_kwargs = dict(training_objective="windowed_rollout", ema_target_decay=0.9)

    model_full, stats_full = train_action_world_model(dataset, _cfg(epochs=6, **cfg_kwargs))

    model_first, stats_first = train_action_world_model(dataset, _cfg(epochs=3, **cfg_kwargs))
    model_resumed, stats_resumed = train_action_world_model(
        dataset, _cfg(epochs=6, **cfg_kwargs),
        initial_model=model_first, resume_state=stats_first["resume_state"],
    )

    _assert_state_dict_equal(model_full.state_dict(), model_resumed.state_dict())
    assert (
        stats_full["loss_curves"]["total_loss"][3:6]
        == stats_resumed["loss_curves"]["total_loss"]
    )
    assert stats_first["resume_state"].target_encoder_state_dict is not None
