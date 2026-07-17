"""Milestone 2 proof (issue #92, docs/v2/phases/phase-2-predictive-cortex.md):
scoring-gate correctness and the action-ablation claim that action-
conditioning is load-bearing, not decorative.

``tests/test_action_world_model.py`` already covers the cortex's forward
shape and the promotion shim (issue #91); this file covers the three
things issue #92 adds on top of it: the frozen-rollout detector tripping on
a genuinely degenerate (identity-collapsed) model, the copy-last baseline
being computed correctly (independent of the model under test), and the
action-ablation harness actually detecting a real action dependency.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from cognitive_runtime.training.action_world_model import (  # noqa: E402
    ActionSequenceDataset,
    ActionWorldModelConfig,
    EpisodeActionFrames,
    _episode_tensors,
    build_action_sequence_dataset,
    evaluate_action_world_model,
    train_action_world_model,
)
from cognitive_runtime.training.nursery import (  # noqa: E402
    NURSERY_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
    run_action_ablation_eval,
)
from cognitive_runtime.training.statistical_evaluation import MetricStats  # noqa: E402


def _small_nursery_config(**overrides) -> NurseryConfig:
    base = dict(
        train_seeds=(0, 1),
        holdout_seeds=(1000,),
        episode_ticks=26,
        world_size=16,
        horizons=(1, 3),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
    )
    base.update(overrides)
    return NurseryConfig(**base)


def _small_model_config(**overrides) -> ActionWorldModelConfig:
    base = dict(
        latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=2, batch_size=16, warmup_frames=2, rollout_frames=3,
    )
    base.update(overrides)
    return ActionWorldModelConfig(**base)


@pytest.fixture(scope="module")
def turn_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("cortex-sessions")
    cfg = _small_nursery_config()
    return _record_scenario_episode(
        str(root), "cortex-turn", 0, NURSERY_SCENARIOS["turn_in_place"], cfg
    )


# --------------------------------------------------------------- scoring gates


def test_frozen_rollout_detector_trips_on_degenerate_identity_model(turn_session):
    """A model whose decoder collapses to a constant output regardless of
    its input latent -- the "predict no change" identity attractor the
    phase doc warns is MSE's cheapest long-rollout solution -- must trip
    the frozen-rollout detector when the real scenario keeps moving."""
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())

    # Degenerate: the decoder ignores its input, so every horizon decodes to
    # the same (zero) frame -- the collapsed fixed point, not a trained
    # model that merely predicts poorly.
    with torch.no_grad():
        for p in model.decoder.parameters():
            p.zero_()

    report = evaluate_action_world_model(model, dataset, [1, 3], warmup_frames=2)
    health = report["rollout_health"]
    assert health["prediction_dispersion"] == pytest.approx(0.0, abs=1e-12)
    assert health["target_dispersion"] > 0.0
    assert health["frozen_rollout"] is True


def test_copy_last_baseline_computed_correctly(turn_session):
    """``copy_last_mse`` must equal MSE(targets[t], targets[t+h]) averaged
    over exactly the same (episode, start) samples the function scores --
    a baseline computed from the recorded targets alone, independent of
    whatever the model under test predicts."""
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    warmup_frames = 2
    horizons = [1, 3]

    report = evaluate_action_world_model(model, dataset, horizons, warmup_frames=warmup_frames)

    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    max_horizon = max(horizons)
    expected: dict = {h: [] for h in horizons}
    for _episode, _pixels, targets, _actions in episodes:
        n = targets.shape[0]
        if n <= max_horizon + warmup_frames:
            continue
        for t in range(warmup_frames, n - max_horizon):
            for h in horizons:
                expected[h].append(float(F.mse_loss(targets[t], targets[t + h])))

    for h in horizons:
        assert report["horizons"][h]["n_samples"] == len(expected[h])
        assert report["horizons"][h]["copy_last_mse"] == pytest.approx(
            sum(expected[h]) / len(expected[h]), rel=1e-6
        )


# --------------------------------------------------------------- action ablation


PIXEL_SHAPE = (4, 4, 3)
_ACAB_ACTIONS = ("NULL", "LEFT_TURN", "RIGHT_TURN")
_NEUTRAL = np.full(PIXEL_SHAPE, 128, dtype=np.uint8)
_BLACK = np.zeros(PIXEL_SHAPE, dtype=np.uint8)
_WHITE = np.full(PIXEL_SHAPE, 255, dtype=np.uint8)


def _diverging_episode(frames: int, branch_left: bool, idx: int) -> EpisodeActionFrames:
    """A scripted episode whose only branch point is which of two actions
    fires at tick 2 -- every frame up to and including tick 2 is byte-
    identical between branches, and only the *action* (never anything
    visible) determines whether the world then turns black or white. A
    model that never sees its action has, at that point, strictly less
    information than one that does: it cannot do better than guessing."""
    diverged = _BLACK if branch_left else _WHITE
    divergent_action = _ACAB_ACTIONS.index("LEFT_TURN" if branch_left else "RIGHT_TURN")
    null_action = _ACAB_ACTIONS.index("NULL")
    pixels = [_NEUTRAL] * 3 + [diverged] * (frames - 3)
    actions = [null_action, null_action, divergent_action] + [null_action] * (frames - 4)
    return EpisodeActionFrames(
        session_dir="synthetic-ablation", episode_id=f"{'train' if frames == 5 else 'eval'}-{idx}",
        frames=pixels, actions=actions, yaw=[None] * frames, ticks=list(range(frames)),
    )


def _diverging_dataset(frames: int, n_episodes: int, offset: int = 0) -> ActionSequenceDataset:
    episodes = [_diverging_episode(frames, i % 2 == 0, offset + i) for i in range(n_episodes)]
    return ActionSequenceDataset(
        episodes=episodes,
        action_keys=list(_ACAB_ACTIONS),
        pixel_shape=PIXEL_SHAPE,
        sources=[f"synthetic-ablation/{offset + i}" for i in range(n_episodes)],
    )


def test_action_ablation_measurably_worsens_prediction():
    """The Milestone 2 claim in miniature: two frame sequences are byte-
    identical up to the tick where one of two actions fires, after which
    they diverge (black vs. white) -- purely as a function of that action,
    never anything visible beforehand. Training with the action stream
    withheld (``ActionWorldModelConfig.withhold_actions``) must measurably
    worsen held-out prediction at exactly that divergence, since the
    ablated model has no way to tell the two branches apart."""
    train_dataset = _diverging_dataset(frames=5, n_episodes=40, offset=0)
    eval_dataset = _diverging_dataset(frames=4, n_episodes=20, offset=1000)

    model_cfg = ActionWorldModelConfig(
        latent_width=8, hidden_dim=16, reconstruction_size=4,
        epochs=60, batch_size=8, warmup_frames=1, rollout_frames=3, seed=0,
    )
    model_with, _ = train_action_world_model(train_dataset, replace(model_cfg, withhold_actions=False))
    model_without, _ = train_action_world_model(train_dataset, replace(model_cfg, withhold_actions=True))

    # warmup_frames=2 isolates the single (episode, start) sample straddling
    # the divergent action -- every earlier/later start is identical across
    # branches and would dilute the comparison with easy, uninformative
    # samples both models trivially get right.
    report_with = evaluate_action_world_model(model_with, eval_dataset, [1], warmup_frames=2)
    report_without = evaluate_action_world_model(model_without, eval_dataset, [1], warmup_frames=2)

    mse_with = report_with["horizons"][1]["model_mse"]
    mse_without = report_without["horizons"][1]["model_mse"]
    assert mse_with < 0.05, f"action-aware model should learn the branch cleanly, got {mse_with}"
    assert mse_without > 0.15, (
        f"action-withheld model has no way to distinguish the branches and should sit near "
        f"chance-level MSE, got {mse_without}"
    )
    assert mse_without > 3 * mse_with


def test_run_action_ablation_eval_reports_valid_structure(tmp_path):
    """The nursery-scenario-backed harness (issue #92's "runnable eval") --
    exercised end-to-end on real recorded scenarios -- returns a
    well-formed, statistically-annotated comparison. (The tiny configs
    tests can afford don't have the training budget to reproduce the full
    Milestone 2 effect size on real scenarios; that's what
    ``test_action_ablation_measurably_worsens_prediction`` proves under
    controlled conditions instead.)"""
    cfg = _small_nursery_config(episode_ticks=26, horizons=(1,))
    report = run_action_ablation_eval(
        str(tmp_path),
        train_scenarios=["walk_forward", "turn_in_place"],
        eval_scenario="turn_in_place",
        config=cfg,
        model_config=_small_model_config(),
    )
    assert report.train_scenarios == ["walk_forward", "turn_in_place"]
    assert report.eval_scenario == "turn_in_place"
    assert set(report.with_actions_metrics["horizons"]) == set(report.without_actions_metrics["horizons"])
    for metrics in (report.with_actions_metrics, report.without_actions_metrics):
        assert "per_episode_model_mse" in metrics
        for h, values in metrics["per_episode_model_mse"].items():
            assert len(values) == len(cfg.holdout_seeds)
    for stats in (report.with_actions_stats, report.without_actions_stats):
        for h, metric_stats in stats.items():
            assert isinstance(metric_stats, MetricStats)
    for h, comparison in report.comparisons.items():
        assert comparison.metric == f"horizon_{h}_model_mse"
        assert comparison.direction in {"improved", "regressed", "no_significant_difference"}
    assert isinstance(report.action_withholding_degrades, bool)


def test_run_action_ablation_eval_rejects_eval_scenario_outside_train_scenarios(tmp_path):
    with pytest.raises(ValueError, match="must be one of the trained scenarios"):
        run_action_ablation_eval(
            str(tmp_path),
            train_scenarios=["walk_forward"],
            eval_scenario="turn_in_place",
            config=_small_nursery_config(),
        )


def test_run_action_ablation_eval_warm_starts_both_runs_from_separate_checkpoints(tmp_path, monkeypatch):
    """issue #134: both runs used to be fresh, disposable models every
    call, discarding any progress -- ``cortex_checkpoint_path`` warm-starts
    the with-actions run from (and saves it back to) a persisted
    checkpoint, so repeated calls keep improving the same cortex.

    PR #155 review: warm-starting *only* the with-actions run would let it
    accumulate strictly more total training than a freshly-initialized
    control every attempt, so ``action_ablation_margin`` could turn positive
    from more training alone rather than from access to actions. The
    without-actions control therefore warm-starts too, from its own
    sibling checkpoint -- never the with-actions one -- so both keep an
    equal accumulated training budget while the control still never sees
    actions."""
    import cognitive_runtime.training.nursery as nursery_module

    checkpoint_path = str(tmp_path / "cortex.pt")
    control_checkpoint_path = checkpoint_path + ".control"
    cfg = _small_nursery_config(episode_ticks=26, horizons=(1,))

    captured = []
    original = nursery_module.train_action_world_model

    def spy(dataset, config=None, *, initial_model=None):
        captured.append((config.withhold_actions, initial_model))
        return original(dataset, config, initial_model=initial_model)

    monkeypatch.setattr(nursery_module, "train_action_world_model", spy)

    run_action_ablation_eval(
        str(tmp_path),
        train_scenarios=["walk_forward", "turn_in_place"],
        eval_scenario="turn_in_place",
        config=cfg,
        model_config=_small_model_config(),
        cortex_checkpoint_path=checkpoint_path,
    )
    assert os.path.exists(checkpoint_path), "first call must save the with-actions cortex"
    assert os.path.exists(control_checkpoint_path), "first call must save the without-actions control"
    (with_flag_1, initial_1), (without_flag_1, initial_2) = captured
    assert with_flag_1 is False and initial_1 is None
    assert without_flag_1 is True and initial_2 is None

    run_action_ablation_eval(
        str(tmp_path),
        train_scenarios=["walk_forward", "turn_in_place"],
        eval_scenario="turn_in_place",
        config=cfg,
        model_config=_small_model_config(),
        cortex_checkpoint_path=checkpoint_path,
    )
    (with_flag_2, initial_3), (without_flag_2, initial_4) = captured[2:]
    assert with_flag_2 is False
    assert initial_3 is not None, "with-actions run must warm-start from its saved checkpoint"
    assert without_flag_2 is True
    assert initial_4 is not None, "without-actions control must warm-start from its own sibling checkpoint"
    assert initial_4 is not initial_3, "the control must never warm-start from the with-actions model"
