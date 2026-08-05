"""Trial orchestration acceptance tests for ``run_trial`` (issue #225, MF-C3).

Every scenario trains a real (tiny) cortex against real (tiny) recorded
Crafter sessions -- seconds, not minutes -- rather than mocking the trainer,
so these tests exercise the actual integration the epic calls "the main
integration point," not a stand-in for it.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.training.model_factory.corpus import build_corpus  # noqa: E402
from cognitive_runtime.training.model_factory.runner import run_trial  # noqa: E402
from cognitive_runtime.training.model_factory.state import (  # noqa: E402
    IncompleteRunError,
    require_completed_for_promotion,
    state_path,
    load_state,
)

ORGANISM = "Test"


def _build_corpus(root: Path, corpus_id: str = "crafter-tiny-v1", *, world_size: int = 16):
    spec = {
        "root": str(root),
        "organism": ORGANISM,
        "corpus_id": corpus_id,
        "generator": {
            "world": "crafter",
            "backend": "crafter",
            "episode_ticks": 24,
            "world_size": world_size,
            # The quality/split-overlap gates are already covered by MF-A5's
            # own tests; disabling the quality gate here keeps this fixture
            # from being sensitive to a real scenario's behavioral
            # thresholds, which is not what run_trial's own tests are for.
            "data_quality_gate": False,
        },
        "splits": {
            "train": {"walk_forward_short": [0, 1]},
            "validation": {"walk_forward_short": [100, 101]},
            "test": {},
        },
    }
    return build_corpus(spec)


def _build_navigation_corpus(root: Path, generic_corpus_id: str):
    return build_corpus({
        "root": str(root),
        "organism": ORGANISM,
        "corpus_id": "crafter-navigation-tiny-v1",
        "generator": {
            "world": "crafter",
            "backend": "crafter",
            "episode_ticks": 24,
            "world_size": 48,
            "data_quality_gate": False,
            "navigation_random_action_fraction": 0.25,
        },
        "splits": {
            "train": {"navigate_open_goal": [0, 1]},
            "validation": {"navigate_open_goal": [1000, 1001]},
            "test": {},
        },
        "behavior_mixture": {"random_action_fraction": 0.25},
        "retention_policy": {
            "suite": "generic_action_effects_v1",
            "corpus_id": generic_corpus_id,
            "metric": "heldout_prediction_loss",
            "max_regression": 1.0,
            "replay_mixture": {
                "generic_action_effects_v1": 0.25,
                "goal_navigation_v1": 0.75,
            },
        },
        "quality_policy": {"enabled": False, "expected_pixel_source": "crafter"},
    })


def _spec_dict(corpus_id, *, mode="fresh", parent=None, training_overrides=None):
    training = {
        "objective": "windowed_rollout",
        "optimizer": {"name": "adamw", "lr": 0.02},
        "batch_size": 4,
        "seed": 0,
        "rollout_frames": 2,
        "warmup_frames": 2,
        "scheduled_sampling_p": 0.0,
        "epoch_budget": 2,
        "checkpoint_cadence_epochs": 1,
        "device": "cpu",
        "precision": "fp32",
    }
    if training_overrides:
        training.update(training_overrides)
    doc = {
        "organism": ORGANISM,
        "mode": mode,
        "data": {"corpus_id": corpus_id, "world": "crafter", "horizons_ticks": [1, 2]},
        "model": {
            "backbone": "transformer", "latent_width": 8, "hidden_dim": 16,
            "action_embed_dim": 4, "reconstruction_size": 4, "context_length": 3,
            "backbone_kwargs": {"n_heads": 2, "n_layers": 1},
        },
        "training": training,
        "evaluation": {"selection_metric": "rollout.t+1.model_over_copy_last_mse"},
    }
    if parent is not None:
        doc["parent"] = parent
    return doc


def _parent_block(result):
    # TrialResult.checkpoint_path/checkpoint_sha256 always name the
    # best-validation checkpoint (see TrialResult's docstring).
    return {"run_id": result.run_id, "checkpoint": "best-validation.pt", "sha256": result.checkpoint_sha256}


@pytest.fixture()
def corpus(tmp_path):
    return _build_corpus(tmp_path / "corpora")


def _run(spec_doc, tmp_path, *, run_id=None):
    return run_trial(
        spec_doc, root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"), run_id=run_id,
    )


def test_run_trial_reuses_corpus_with_different_experiment_horizons(tmp_path):
    corpus = _build_corpus(tmp_path / "corpora")
    spec = _spec_dict(corpus.corpus_id)
    spec["data"]["horizons_ticks"] = [1]
    spec["evaluation"]["selection_metric"] = "rollout.t+1.model_over_copy_last_mse"

    result = _run(spec, tmp_path)

    assert result.state == "completed"


def test_run_trial_rejects_horizon_beyond_recorded_temporal_coverage(tmp_path):
    corpus = _build_corpus(tmp_path / "corpora")
    spec = _spec_dict(corpus.corpus_id)
    spec["data"]["horizons_ticks"] = [50]
    spec["evaluation"]["selection_metric"] = "rollout.t+50.model_over_copy_last_mse"

    with pytest.raises(ValueError, match=r"cannot support trial horizons \[50\].*tick_span"):
        _run(spec, tmp_path)

    assert not (tmp_path / "runs").exists()


def test_fresh_trial_completes_and_writes_artifacts(tmp_path, corpus):
    spec = _spec_dict(corpus.corpus_id, training_overrides={
        "transition_balance_policy": {"stationary_cap": 0.4},
    })
    result = _run(spec, tmp_path)

    assert result.mode == "fresh"
    assert result.state == "completed"
    assert result.comparison is None
    directory = result.directory
    assert (directory / "checkpoints" / "last.pt").is_file()
    assert (directory / "checkpoints" / "best-validation.pt").is_file()
    assert (directory / "metrics" / "validation.json").is_file()
    assert (directory / "metrics" / "budget_report.json").is_file()
    assert (directory / "experiment_report.json").is_file()
    assert not (directory / "metrics" / "test.json").exists()
    assert not (directory / "metrics" / "comparison.json").exists()

    state = load_state(state_path(directory))
    assert state.state == "completed"
    require_completed_for_promotion(state)  # does not raise

    report = json.loads((directory / "experiment_report.json").read_text(encoding="utf-8"))
    assert report["checkpoint"]["sha256"] == result.checkpoint_sha256
    assert report["training_stats"]["completion_status"] == "completed"
    balance = result.training_stats["transition_balance_policy"]
    assert balance["stationary_cap"] == pytest.approx(0.4)
    assert balance["transitions"] > 0
    assert balance["stationary_transitions"] >= 0
    assert balance["weighted_stationary_fraction"] is not None


def test_factory_loss_weight_aliases_reach_the_real_trainer_config():
    from cognitive_runtime.training.model_factory.runner import _action_world_model_config
    from cognitive_runtime.training.model_factory.spec import resolve

    spec = resolve(_spec_dict("unused", training_overrides={
        "loss_weights": {"pixel": 0.2, "latent": 0.3, "semantic": 0.4},
    }))
    cfg = _action_world_model_config(spec)

    assert cfg.pixel_loss_weight == pytest.approx(0.2)
    assert cfg.latent_loss_weight == pytest.approx(0.3)
    assert cfg.semantic_loss_weight == pytest.approx(0.4)


def test_fresh_trial_writes_clinic_session_index_and_exports_validation_predictions(tmp_path, corpus):
    """The auto-export the clinic (viewer/) relies on: every train/validation
    session is indexed for the Episode tab, and the promoted checkpoint's
    predictions land next to a validation session's own stream log --
    inside the frozen corpus, not the run directory (so a sibling run
    against the same corpus can be compared against it later)."""
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)

    index_path = result.directory / "clinic_sessions.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["format"] == "clinic-session-index-v1"
    train_entries = corpus.manifest["sessions"]["train"]
    validation_entries = corpus.manifest["sessions"]["validation"]
    assert len(index["sessions"]) == len(train_entries) + len(validation_entries)
    by_split = {"train": set(), "evaluation": set()}
    for entry in index["sessions"]:
        by_split[entry["split"]].add(entry["session_dir"])
    assert by_split["train"] == {str(Path(entry["session_path"]).resolve()) for entry in train_entries}
    assert by_split["evaluation"] == {str(Path(entry["session_path"]).resolve()) for entry in validation_entries}

    exported = [
        path for entry in validation_entries
        for path in Path(entry["session_path"]).glob(f"{result.run_id}-predictions_*.json")
    ]
    assert exported, "expected at least one clinic prediction export for a validation episode"
    payload = json.loads(exported[0].read_text(encoding="utf-8"))
    assert payload["format"] == "pixel-predictions-v2"
    assert payload["experiment"]["experiment_id"] == result.run_id
    assert payload["prediction_mode"] == "rollout"


def test_export_predictions_failure_does_not_fail_the_trial(tmp_path, corpus, monkeypatch):
    """The clinic export is diagnostic-only: it must never gate or break
    training, even if every episode's export blows up."""
    import cognitive_runtime.training.prediction_export as prediction_export_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(prediction_export_module, "export_cortex_session_predictions", _boom)

    result = _run(_spec_dict(corpus.corpus_id), tmp_path)

    assert result.state == "completed"
    validation_entries = corpus.manifest["sessions"]["validation"]
    exported = [
        path for entry in validation_entries
        for path in Path(entry["session_path"]).glob(f"{result.run_id}-predictions_*.json")
    ]
    assert not exported


def test_no_export_predictions_flag_skips_the_export_but_keeps_the_session_index(tmp_path, corpus):
    result = run_trial(
        _spec_dict(corpus.corpus_id), root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
        export_predictions=False,
    )

    assert (result.directory / "clinic_sessions.json").is_file()
    validation_entries = corpus.manifest["sessions"]["validation"]
    exported = [
        path for entry in validation_entries
        for path in Path(entry["session_path"]).glob(f"{result.run_id}-predictions_*.json")
    ]
    assert not exported


def test_two_clone_siblings_share_identical_data_manifest_and_report_comparison(tmp_path, corpus):
    parent_result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    parent = _parent_block(parent_result)

    clone_a = _run(
        _spec_dict(corpus.corpus_id, mode="clone", parent=parent,
                    training_overrides={"scheduled_sampling_p": 0.1}),
        tmp_path,
    )
    clone_b = _run(
        _spec_dict(corpus.corpus_id, mode="clone", parent=parent,
                    training_overrides={"scheduled_sampling_p": 0.3}),
        tmp_path,
    )

    assert clone_a.state == "completed"
    assert clone_b.state == "completed"

    manifest_a = json.loads((clone_a.directory / "data_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((clone_b.directory / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["sessions"] == manifest_b["sessions"]
    assert manifest_a["data_contract_hash"] == manifest_b["data_contract_hash"]

    for clone in (clone_a, clone_b):
        comparison_path = clone.directory / "metrics" / "comparison.json"
        assert comparison_path.is_file()
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        assert payload["parent_run_id"] == parent_result.run_id
        assert payload["tier"] == "fast"
        assert isinstance(payload["measured_training_seconds"], float)
        assert payload["decision"] in ("candidate_improves", "hold")
        assert payload["episode_ids"], "per-episode deltas must be present"
        assert payload["per_episode"]["deltas"]
        field_names = {entry["field"] for entry in payload["field_diff"]}
        assert field_names == {"training.scheduled_sampling_p"}
        assert clone.comparison == payload


def test_fine_tune_mode_continues_weights_under_a_new_training_contract(tmp_path, corpus):
    parent_result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    parent = _parent_block(parent_result)

    child = _run(
        _spec_dict(corpus.corpus_id, mode="fine_tune", parent=parent,
                    training_overrides={"optimizer": {"name": "adamw", "lr": 0.05}}),
        tmp_path,
    )

    assert child.mode == "fine_tune"
    assert child.state == "completed"
    assert child.training_contract_hash != parent_result.training_contract_hash
    assert child.architecture_hash == parent_result.architecture_hash


def test_navigation_fine_tune_writes_navigation_and_generic_retention_evidence(tmp_path):
    corpus = _build_corpus(
        tmp_path / "corpora", "crafter-retention-tiny-v1", world_size=48,
    )
    parent_result = _run(
        _spec_dict(corpus.corpus_id, training_overrides={"epoch_budget": 1}), tmp_path,
    )
    navigation_corpus = _build_navigation_corpus(tmp_path / "corpora", corpus.corpus_id)
    child_spec = _spec_dict(
        navigation_corpus.corpus_id,
        mode="fine_tune",
        parent=_parent_block(parent_result),
        training_overrides={"epoch_budget": 1},
    )
    child_spec["evaluation"]["selection_metric"] = "goal_navigation.success_rate"

    child = _run(child_spec, tmp_path)

    validation = json.loads(
        (child.directory / "metrics" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["goal_navigation"]["format"] == "goal-navigation-metrics-v1"
    assert 0.0 <= validation["goal_navigation"]["success_rate"] <= 1.0
    retention = json.loads(
        (child.directory / "metrics" / "retention.json").read_text(encoding="utf-8")
    )
    assert retention["old_scenario"] == "generic_action_effects_v1"
    assert retention["new_scenario"] == "goal_navigation_v1"
    assert retention["corpus_id"] == corpus.corpus_id
    assert retention["episode_ids"]
    assert isinstance(retention["retained"], bool)
    from cognitive_runtime.cli import _factory_population_candidate

    candidate = _factory_population_candidate(
        child.directory,
        objective="goal_navigation.success_rate",
        declared_genome_fields=(),
    )
    assert candidate.retention_metric == pytest.approx(retention["forgetting_amount"])


def test_resume_continues_an_interrupted_run_to_completion(tmp_path, corpus):
    # Build the interrupted run by hand: allocate it exactly like a fresh
    # run_trial call would, run one real chunk, then leave the state record
    # at "running" -- the only state a genuine (uncatchable) worker death
    # leaves behind, and the only state resume is entitled to pick up
    # (state.py's own module docstring; a failed/completed record is never
    # silently reopened).
    from cognitive_runtime.training.model_factory import (
        artifacts as artifacts_module,
        checkpoint as checkpoint_module,
        contracts as contracts_module,
        spec as spec_module,
        state as state_module,
    )
    from cognitive_runtime.training import action_world_model as awm
    from cognitive_runtime.training import nursery as nursery_module
    from cognitive_runtime.training.model_factory.runner import (
        _action_world_model_config,
        _architecture_contract,
    )

    raw = _spec_dict(corpus.corpus_id, training_overrides={"epoch_budget": 3, "checkpoint_cadence_epochs": 1})
    resolved = spec_module.resolve(raw)
    vocabulary = nursery_module._action_keys_for_world(resolved.data["world"])
    train_paths = [entry["session_path"] for entry in corpus.manifest["sessions"]["train"]]
    train_dataset = awm.build_action_sequence_dataset(train_paths, action_keys=vocabulary)
    cfg = _action_world_model_config(resolved)
    expected_workspace_modalities = awm._workspace_modalities(train_dataset, cfg.workspace_enabled)
    expected_workspace_layout = train_dataset.workspace_layout_hash if expected_workspace_modalities else None
    model = awm.build_action_world_model(
        train_dataset.pixel_shape, train_dataset.action_keys, cfg,
        workspace_modalities=expected_workspace_modalities, workspace_layout_hash=expected_workspace_layout,
    )
    architecture_contract = _architecture_contract(model)
    training_contract = contracts_module.TrainingContract(**dict(resolved.training))

    runs_root = tmp_path / "runs"
    interrupted_run_id = "interrupted-run"
    artifacts = artifacts_module.allocate_run_artifacts(
        str(runs_root), resolved, architecture_contract, corpus.data_contract, training_contract,
        run_id=interrupted_run_id,
    )
    state_module.create_state(state_path(artifacts.directory), artifacts.run_id, heartbeat_timeout_seconds=300.0)
    state_module.transition(state_path(artifacts.directory), state_module.STATE_RUNNING)
    # A stale heartbeat, not merely an active state, is what actually marks
    # a worker as gone: run_trial's resume path now requires this too, so a
    # live worker's own resume attempt can't race a second one against it.
    state_module.write_heartbeat(
        state_module.heartbeat_path(artifacts.directory), run_id=artifacts.run_id,
        clock=lambda: time.time() - 10_000,
    )

    partial_cfg = dataclasses.replace(cfg, epochs=1)
    trained_model, stats = awm.train_action_world_model(train_dataset, partial_cfg, initial_model=model)
    resume_state = stats["resume_state"]
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=cfg.lr)
    optimizer.load_state_dict(resume_state.optimizer_state_dict)
    checkpoint_module.save_factory_checkpoint(
        str(artifacts.checkpoints_dir / "last.pt"), trained_model, optimizer,
        trainer_state={
            "epoch": resume_state.epoch, "global_step": resume_state.global_step,
            "best_validation_metric": resume_state.best_validation_metric,
        },
        architecture_contract=architecture_contract, data_contract_hash=corpus.data_contract,
        training_contract=training_contract, rng_state=resume_state.rng_state,
    )
    # Deliberately do NOT transition to a terminal state: this is the
    # left-running record a crashed worker would leave behind.

    resume_sha = checkpoint_module.read_factory_checkpoint_metadata(
        str(artifacts.checkpoints_dir / "last.pt")
    )["checkpoint_sha256"]
    resume_spec_doc = dict(raw)
    resume_spec_doc["mode"] = "resume"
    resume_spec_doc["parent"] = {"run_id": interrupted_run_id, "checkpoint": "last.pt", "sha256": resume_sha}

    result = _run(resume_spec_doc, tmp_path)

    assert result.mode == "resume"
    assert result.run_id == interrupted_run_id
    assert result.state == "completed"
    assert result.training_stats["completion_status"] == "completed"


def test_resume_never_regresses_a_better_pre_crash_best_checkpoint(tmp_path, corpus):
    """Codex review (PR #258): seeding a fresh BestValidationTracker on
    resume made the first post-resume validation always count as "the
    best," silently overwriting a genuinely better pre-crash checkpoint.
    An unbeatable pre-crash metric must survive resuming."""
    from cognitive_runtime.training.model_factory import (
        artifacts as artifacts_module,
        checkpoint as checkpoint_module,
        contracts as contracts_module,
        spec as spec_module,
        state as state_module,
    )
    from cognitive_runtime.training import action_world_model as awm
    from cognitive_runtime.training import nursery as nursery_module
    from cognitive_runtime.training.model_factory.runner import (
        _action_world_model_config,
        _architecture_contract,
    )

    raw = _spec_dict(corpus.corpus_id, training_overrides={"epoch_budget": 3, "checkpoint_cadence_epochs": 1})
    resolved = spec_module.resolve(raw)
    vocabulary = nursery_module._action_keys_for_world(resolved.data["world"])
    train_paths = [entry["session_path"] for entry in corpus.manifest["sessions"]["train"]]
    train_dataset = awm.build_action_sequence_dataset(train_paths, action_keys=vocabulary)
    cfg = _action_world_model_config(resolved)
    expected_workspace_modalities = awm._workspace_modalities(train_dataset, cfg.workspace_enabled)
    expected_workspace_layout = train_dataset.workspace_layout_hash if expected_workspace_modalities else None
    model = awm.build_action_world_model(
        train_dataset.pixel_shape, train_dataset.action_keys, cfg,
        workspace_modalities=expected_workspace_modalities, workspace_layout_hash=expected_workspace_layout,
    )
    architecture_contract = _architecture_contract(model)
    training_contract = contracts_module.TrainingContract(**dict(resolved.training))

    runs_root = tmp_path / "runs"
    interrupted_run_id = "interrupted-run-unbeatable-best"
    artifacts = artifacts_module.allocate_run_artifacts(
        str(runs_root), resolved, architecture_contract, corpus.data_contract, training_contract,
        run_id=interrupted_run_id,
    )
    state_module.create_state(state_path(artifacts.directory), artifacts.run_id, heartbeat_timeout_seconds=300.0)
    state_module.transition(state_path(artifacts.directory), state_module.STATE_RUNNING)
    state_module.write_heartbeat(
        state_module.heartbeat_path(artifacts.directory), run_id=artifacts.run_id,
        clock=lambda: time.time() - 10_000,
    )

    partial_cfg = dataclasses.replace(cfg, epochs=1)
    trained_model, stats = awm.train_action_world_model(train_dataset, partial_cfg, initial_model=model)
    resume_state = stats["resume_state"]
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=cfg.lr)
    optimizer.load_state_dict(resume_state.optimizer_state_dict)
    # A real per-episode MSE can never beat this: any post-resume chunk
    # must leave the pre-crash best-validation.pt exactly as it is now.
    unbeatable_metric = -1000.0
    trainer_state = {
        "epoch": resume_state.epoch, "global_step": resume_state.global_step,
        "best_validation_metric": unbeatable_metric,
    }
    checkpoint_module.save_factory_checkpoint(
        str(artifacts.checkpoints_dir / "last.pt"), trained_model, optimizer,
        trainer_state=trainer_state,
        architecture_contract=architecture_contract, data_contract_hash=corpus.data_contract,
        training_contract=training_contract, rng_state=resume_state.rng_state,
    )
    checkpoint_module.save_factory_checkpoint(
        str(artifacts.checkpoints_dir / "best-validation.pt"), trained_model, optimizer,
        trainer_state=trainer_state,
        architecture_contract=architecture_contract, data_contract_hash=corpus.data_contract,
        training_contract=training_contract, rng_state=resume_state.rng_state,
    )
    pre_resume_best_sha = checkpoint_module.read_factory_checkpoint_metadata(
        str(artifacts.checkpoints_dir / "best-validation.pt")
    )["checkpoint_sha256"]

    resume_sha = checkpoint_module.read_factory_checkpoint_metadata(
        str(artifacts.checkpoints_dir / "last.pt")
    )["checkpoint_sha256"]
    resume_spec_doc = dict(raw)
    resume_spec_doc["mode"] = "resume"
    resume_spec_doc["parent"] = {"run_id": interrupted_run_id, "checkpoint": "last.pt", "sha256": resume_sha}

    result = _run(resume_spec_doc, tmp_path)

    assert result.state == "completed"
    post_resume_best_sha = checkpoint_module.read_factory_checkpoint_metadata(
        str(artifacts.checkpoints_dir / "best-validation.pt")
    )["checkpoint_sha256"]
    assert post_resume_best_sha == pre_resume_best_sha
    assert result.checkpoint_sha256 == pre_resume_best_sha
    assert result.evaluation is not None


def test_routine_trial_never_writes_metrics_test_json(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    assert not (result.directory / "metrics" / "test.json").exists()


def test_corpus_cache_miss_fails_before_training_and_creates_no_run_directory(tmp_path, corpus):
    doc = _spec_dict("this-corpus-does-not-exist")

    with pytest.raises(FileNotFoundError):
        _run(doc, tmp_path)

    runs_root = tmp_path / "runs"
    assert not runs_root.exists() or not any((runs_root / ORGANISM).glob("*"))


def test_failure_mid_trial_leaves_state_failed_not_running(tmp_path, corpus, monkeypatch):
    import cognitive_runtime.training.model_factory.runner as runner_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(runner_module, "_evaluate", _boom)

    doc = _spec_dict(corpus.corpus_id)
    with pytest.raises(RuntimeError, match="simulated stage failure"):
        _run(doc, tmp_path, run_id="failing-trial")

    run_directory = tmp_path / "runs" / ORGANISM / "failing-trial"
    state = load_state(state_path(run_directory))
    assert state.state == "failed"
    assert "simulated stage failure" in (state.reason or "")
    with pytest.raises(IncompleteRunError):
        require_completed_for_promotion(state)
    assert not (run_directory / "checkpoints" / "best-validation.pt").exists()


def test_existing_nursery_and_evaluation_tests_are_unaffected():
    """run_trial adds a caller around the existing trainer/evaluator; it
    must not fork or alter them (epic #212 §6.1)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_nursery.py", "tests/test_action_world_model.py"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_runner_module_imports_without_torch():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockTorch(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'")
        return None

sys.meta_path.insert(0, BlockTorch())
import cognitive_runtime.training.model_factory.runner  # noqa: F401
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
