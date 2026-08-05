"""``ccr factory ...`` CLI wiring (issue #228, MF-C6, epic #212 §20).

A thin dispatcher: argument parsing plus one call into
``cognitive_runtime.training.model_factory``'s runner/registry/promotion/
corpus/confirmation modules. These tests cover the acceptance criteria
directly:

* every subcommand parses, dispatches, and has ``--help`` text;
* ``ccr factory show`` prints the fully resolved config, all three contract
  hashes, the display name, and the completion status;
* ``ccr factory lineage`` renders the ancestor chain including both
  configuration parents and the weight donor for a bred child;
* ``ccr factory clone --set`` rejects an unknown dotted path with a message
  naming the nearest valid field;
* ``ccr factory --help`` and ``ccr factory show`` succeed with torch not
  installed.

Building fixture runs directly through ``allocate_run_artifacts``/
``resolve``/``create_state`` (rather than ``run_trial``) keeps this whole
file torch-free, matching every Model Factory module except the neural
trainer itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cognitive_runtime.cli import build_parser, main
from cognitive_runtime.training.model_factory.artifacts import RunArtifacts, allocate_run_artifacts
from cognitive_runtime.training.model_factory.contracts import ArchitectureContract, DataContract, TrainingContract
from cognitive_runtime.training.model_factory.spec import ExperimentSpec, dump_canonical_json, resolve
from cognitive_runtime.training.model_factory.state import (
    STATE_COMPLETED,
    STATE_RUNNING,
    cancel_request_path,
    create_state,
    heartbeat_path,
    load_state,
    state_path,
    transition,
    write_heartbeat,
)


# --------------------------------------------------------------------- fixture helpers


def _spec(*, seed: int = 0, parent=None, evolution=None, organism: str = "crafter-baseline") -> ExperimentSpec:
    doc = {
        "organism": organism,
        "mode": "clone" if parent else "fresh",
        "data": {"corpus_id": "generic-v1", "world": "crafter", "horizons_ticks": [1, 4]},
        "model": {"backbone": "transformer", "latent_width": 8, "hidden_dim": 16, "context_length": 4},
        "training": {"objective": "windowed_rollout", "seed": seed, "rollout_frames": 4, "warmup_frames": 2},
        "evaluation": {"selection_metric": "direct.t+4.model_mse"},
    }
    if parent:
        doc["parent"] = parent
    if evolution:
        doc["evolution"] = evolution
    return resolve(doc)


def _contracts() -> tuple[ArchitectureContract, DataContract]:
    architecture = ArchitectureContract(
        pixel_shape=(3, 16, 16), rgb_preprocessing_version="v1", action_vocabulary=("noop",),
        workspace_modalities={}, workspace_layout_hash=None, latent_width=8, hidden_dim=16,
        action_embed_dim=4, reconstruction_shape=(3, 8, 8), visual_architecture="cnn",
        semantic_classes=0, horizons_ticks=(1, 4), direct_horizon_topology=(1, 4),
        backbone="transformer", context_length=4,
    )
    data = DataContract(
        world="crafter", backend="crafter", program_config={}, scenario_names=("walk_forward",),
        scenario_code_version="v1", train_session_ids=("a",), train_session_hashes=("h1",),
        validation_session_ids=("b",), validation_session_hashes=("h2",),
        test_session_ids=(), test_session_hashes=(), seed_assignments={"a": 0, "b": 1},
        pixel_provenance="crafter", semantic_vocabulary_version="none", preprocessing_version="native",
        horizons_ticks=(1, 4), ticks_per_frame=1.0,
    )
    return architecture, data


def _make_run(
    root: Path, run_id: str, spec: ExperimentSpec, *, state: str | None = STATE_COMPLETED,
) -> RunArtifacts:
    architecture, data = _contracts()
    training = TrainingContract(**dict(spec.training))
    artifacts = allocate_run_artifacts(root, spec, architecture, data, training, run_id=run_id)
    create_state(state_path(artifacts.directory), run_id)
    if state is not None:
        transition(state_path(artifacts.directory), STATE_RUNNING)
        if state != STATE_RUNNING:
            transition(state_path(artifacts.directory), state)
    return artifacts


def _write_checkpoint_metadata(artifacts: RunArtifacts, *, sha256: str = "a" * 64, name: str = "best-validation.pt") -> None:
    path = artifacts.checkpoints_dir / f"{name}.json"
    path.write_text(json.dumps({"format": "model-factory-checkpoint-header-v1", "checkpoint_sha256": sha256}))


def _write_validation(
    artifacts: RunArtifacts, episode_ids: list[str], values: list[float],
    *, ticks: int = 4, selection_metric: str = "direct.t+4.model_mse",
) -> None:
    payload = {
        "format": "model-factory-validation-metrics-v1",
        "selection_metric": selection_metric,
        "rollout": {"horizons": {}, "per_episode_model_mse": {}},
        "direct": {
            "horizons": {str(ticks): {"model_mse": sum(values) / len(values)}},
            "per_episode_model_mse": {str(ticks): values},
        },
        "rollout_health": {},
        "per_episode_model_mse": {str(ticks): values},
        "episode_ids": episode_ids,
    }
    (artifacts.metrics_dir / "validation.json").write_text(json.dumps(payload))


def _write_budget_report(artifacts: RunArtifacts, *, tier: str = "fast") -> None:
    payload = {"format": "model-factory-budget-report-v1", "tier": tier, "completion_status": "completed"}
    (artifacts.metrics_dir / "budget_report.json").write_text(json.dumps(payload))


def _write_experiment_report(
    artifacts: RunArtifacts, *, checkpoint_sha256: str = "a" * 64, beats_copy_last: bool = True, healthy: bool = True,
) -> None:
    """A minimal ``experiment_report.json`` that clears every
    ``evaluate_promotion`` gate that reads it directly (checkpoint identity,
    evaluation mode, training-time budget, rollout-beats-copy-last, rollout
    health, metric schema, event-stratified evaluability)."""
    payload = {
        "checkpoint": {"path": str(artifacts.checkpoints_dir / "best-validation.pt"), "sha256": checkpoint_sha256},
        "training_stats": {"evaluation_mode": "held_out", "completion_status": "completed"},
        "metrics": {
            "rollout": {
                "horizons": {"4": {"beats_copy_last": beats_copy_last}},
                "rollout_health": {"state": "healthy" if healthy else "frozen_rollout"},
            },
            "direct": {},
        },
        "event_stratified_metrics": {},
    }
    (artifacts.directory / "experiment_report.json").write_text(json.dumps(payload))


def _build_passing_corpus(tmp_path: Path, monkeypatch, *, organism: str, corpus_id: str) -> Path:
    """A real, quality/split-overlap-gated corpus on disk via ``build_corpus``
    with the recorder faked out -- mirrors
    ``tests/test_model_factory_corpus.py``'s own fixture pattern. World is
    ``minecraft``/``simulated`` (not ``crafter``) so building it never touches
    ``training.nursery`` (and therefore never imports torch): only a
    ``crafter``-world corpus's ``DataContract.semantic_vocabulary_version``
    reaches into that module.
    """
    import cognitive_runtime.training.model_factory.corpus as corpus_module
    from types import SimpleNamespace

    scenario = SimpleNamespace(name="synthetic")
    monkeypatch.setattr(corpus_module, "_scenarios_for_world", lambda world: {"synthetic": scenario})
    monkeypatch.setattr(corpus_module, "validate_nursery_recordings", lambda *args, **kwargs: [])

    cache = tmp_path / "episode-cache" / corpus_id

    def record_or_reuse(_record_dir, _session_id, seed, scenario_, _cfg):
        session = cache / f"{scenario_.name}-{seed}"
        session.mkdir(parents=True, exist_ok=True)
        (session / "session.json").write_text(json.dumps({"session_id": session.name}))
        (session / "episode_00000.decisions.jsonl").write_text("{}\n")
        (session / "episode_00000.streams.jsonl").write_text(
            json.dumps({"stream_id": "vision.frame.pixels", "frame_ref": f"frame-{seed}"}) + "\n"
        )
        return str(session)

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", record_or_reuse)

    corpus_root = tmp_path / "corpora"
    spec = {
        "root": str(corpus_root),
        "organism": organism,
        "corpus_id": corpus_id,
        "generator": {"world": "minecraft", "backend": "simulated", "episode_cache_dir": str(cache)},
        "splits": {"train": {"synthetic": [1]}, "validation": {"synthetic": [2]}, "test": {"synthetic": [3]}},
    }
    corpus_module.build_corpus(spec)
    return corpus_root


# --------------------------------------------------------------------- parsing + dispatch


FACTORY_DISPATCH = [
    (["factory", "baseline", "spec.yaml"], "cmd_factory_baseline"),
    (["factory", "search", "spec.yaml"], "cmd_factory_search"),
    (["factory", "clone", "some-run"], "cmd_factory_clone"),
    (["factory", "resume", "some-run"], "cmd_factory_resume"),
    (["factory", "breed", "parent-a", "parent-b"], "cmd_factory_breed"),
    (["factory", "compare", "run-a", "run-b"], "cmd_factory_compare"),
    (["factory", "promote", "some-run"], "cmd_factory_promote"),
    (["factory", "show"], "cmd_factory_show"),
    (["factory", "show", "some-run"], "cmd_factory_show"),
    (["factory", "lineage"], "cmd_factory_lineage"),
    (["factory", "lineage", "some-run"], "cmd_factory_lineage"),
    (["factory", "corpus", "build", "spec.yaml"], "cmd_factory_corpus_build"),
    (["factory", "test", "some-run"], "cmd_factory_test"),
]


@pytest.mark.parametrize("argv,expected_func", FACTORY_DISPATCH, ids=[".".join(a[1:3]) for a, _ in FACTORY_DISPATCH])
def test_factory_subcommand_parses_and_dispatches(argv, expected_func):
    args = build_parser().parse_args(argv)
    assert args.func.__name__ == expected_func


HELP_ARGVS = [
    ["factory", "--help"],
    ["factory", "baseline", "--help"],
    ["factory", "search", "--help"],
    ["factory", "clone", "--help"],
    ["factory", "resume", "--help"],
    ["factory", "breed", "--help"],
    ["factory", "compare", "--help"],
    ["factory", "promote", "--help"],
    ["factory", "show", "--help"],
    ["factory", "lineage", "--help"],
    ["factory", "corpus", "--help"],
    ["factory", "corpus", "build", "--help"],
    ["factory", "test", "--help"],
]


@pytest.mark.parametrize("argv", HELP_ARGVS, ids=[".".join(a[:-1]) for a in HELP_ARGVS])
def test_factory_help_succeeds(argv):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 0


def test_factory_command_is_a_registered_top_level_choice():
    parser = build_parser()
    # argparse raises SystemExit(2) for an unknown subcommand; "factory"
    # must instead be accepted and routed to its own nested parser.
    args = parser.parse_args(["factory", "show"])
    assert args.command == "factory"
    assert args.factory_command == "show"


def test_factory_search_dry_run_prints_stable_candidate_specs_without_training(capsys):
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--candidates", "2", "--seed", "13",
        "--budgets", "1", "2", "--run-id-prefix", "preview-13", "--dry-run", "--no-trace",
        "--set", "evaluation.selection_metric=rollout.t+2.model_over_copy_last_mse",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["campaign_id"] == "preview-13"
    assert payload["dry_run"] is True
    assert payload["budgets"] == [1, 2]
    assert [candidate["run_id"] for candidate in payload["candidates"]] == [
        "preview-13-r0-c0", "preview-13-r0-c1",
    ]
    assert all(candidate["spec"]["data"]["corpus_id"] == "crafter-generic-action-effects-v1"
               for candidate in payload["candidates"])


def test_factory_search_defaults_to_evolutionary_populations(capsys):
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--population-size", "4", "--populations", "2",
        "--seed", "13", "--run-id-prefix", "preview-13", "--dry-run", "--no-trace",
        "--set", "evaluation.selection_metric=rollout.t+2.model_over_copy_last_mse",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["workflow"] == "evolutionary"
    assert payload["schema"] == "generic_action_effects_v2"
    assert payload["population_size"] == 4
    assert payload["population_count"] == 2
    assert "budgets" not in payload
    assert [candidate["run_id"] for candidate in payload["candidates"]] == [
        "preview-13-p0-c0", "preview-13-p0-c1", "preview-13-p0-c2", "preview-13-p0-c3",
    ]
    assert all(
        {"pixel", "latent", "semantic"}
        <= set(candidate["spec"]["training"]["loss_weights"])
        for candidate in payload["candidates"]
    )


def test_factory_search_dry_run_accepts_repeatable_seed_runs(capsys):
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path),
        "--seed-run", "prior-a", "--seed-run", "prior-b",
        "--population-size", "4", "--populations", "2",
        "--run-id-prefix", "continued", "--dry-run", "--no-trace",
        "--set", "evaluation.selection_metric=rollout.t+2.model_over_copy_last_mse",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["seed_run_ids"] == ["prior-a", "prior-b"]
    assert [candidate["run_id"] for candidate in payload["candidates"]] == [
        "prior-a", "prior-b", "continued-p0-c2", "continued-p0-c3",
    ]
    assert [candidate["source"] for candidate in payload["candidates"]] == [
        "seed_run", "seed_run", "fresh_proposal", "fresh_proposal",
    ]


def test_factory_search_dry_run_resolves_reference_run_checkpoint_sha256(tmp_path, capsys):
    """--reference-run is a convenience wholly analogous to clone/resume's
    own parent-block construction: a human should never have to look up
    and type a checkpoint's sha256 by hand."""
    root = tmp_path / "runs"
    reference = _make_run(root, "champion-0001", _spec(organism="Crafter"))
    _write_checkpoint_metadata(reference, sha256="c" * 64, name="best-validation.pt")

    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--root", str(root),
        "--candidates", "2", "--seed", "13", "--budgets", "1", "2",
        "--run-id-prefix", "preview-13", "--dry-run", "--no-trace",
        "--reference-run", "champion-0001",
        "--set", "evaluation.selection_metric=rollout.t+2.model_over_copy_last_mse",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["candidates"], "dry-run must still propose candidates"
    for candidate in payload["candidates"]:
        assert candidate["spec"]["evaluation"]["reference_run"] == {
            "run_id": "champion-0001", "checkpoint": "best-validation.pt", "sha256": "c" * 64,
        }


def test_factory_search_reference_run_defaults_to_the_best_validation_checkpoint(tmp_path, capsys):
    root = tmp_path / "runs"
    reference = _make_run(root, "champion-0001", _spec(organism="Crafter"))
    _write_checkpoint_metadata(reference, sha256="d" * 64, name="best-validation.pt")

    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--root", str(root),
        "--candidates", "2", "--seed", "13", "--budgets", "1",
        "--run-id-prefix", "preview-13", "--dry-run", "--no-trace",
        "--reference-run", "champion-0001",
        "--set", "evaluation.selection_metric=rollout.t+2.model_over_copy_last_mse",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["candidates"][0]["spec"]["evaluation"]["reference_run"]["checkpoint"] == "best-validation.pt"


def test_factory_search_unknown_reference_run_is_a_concise_cli_error(tmp_path):
    root = tmp_path / "runs"
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    with pytest.raises(SystemExit, match="no run 'no-such-run'"):
        main([
            "factory", "search", str(spec_path), "--root", str(root), "--dry-run", "--no-trace",
            "--reference-run", "no-such-run",
        ])


def test_factory_search_missing_reference_checkpoint_names_the_checkpoint_not_the_spec(tmp_path):
    """A --reference-run whose checkpoint header is absent must say so.

    The run exists and its spec file reads fine, so blaming "spec file not
    found: <spec>" -- as a single broad ``except FileNotFoundError`` around
    both reads did -- names a file that was read successfully and hides the
    one that is actually missing. The clinic's Evolve tab offers a
    reference-run picker over every completed run, including runs with no
    checkpoint on disk, and shows this string to a human verbatim.
    """
    root = tmp_path / "runs"
    _make_run(root, "champion-0001", _spec(organism="Crafter"))  # no checkpoint written
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    with pytest.raises(SystemExit) as excinfo:
        main([
            "factory", "search", str(spec_path), "--root", str(root), "--dry-run", "--no-trace",
            "--reference-run", "champion-0001",
        ])
    message = str(excinfo.value)
    assert "best-validation.pt" in message
    assert "spec file not found" not in message


def test_factory_search_architecture_dry_run_previews_the_outer_population(capsys):
    """``--genome architecture`` previews architectures, not training genes."""
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--genome", "architecture",
        "--outer-population-size", "3", "--outer-populations", "2",
        "--population-size", "2", "--populations", "1",
        "--seed", "13", "--run-id-prefix", "preview-13", "--dry-run", "--no-trace",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["workflow"] == "architecture"
    assert payload["dry_run"] is True
    assert payload["campaign_id"] == "preview-13"
    assert payload["architecture_schema_version"] == "architecture_search_v1"
    assert payload["hyperparameter_schema_version"] == "generic_action_effects_v2"
    assert [candidate["architecture_id"] for candidate in payload["candidates"]] == [
        "preview-13-arch0-a0", "preview-13-arch0-a1", "preview-13-arch0-a2",
    ]
    # Each previewed candidate's genome is genuinely merged into its spec's
    # model block -- the outer axis, never the training block.
    for candidate in payload["candidates"]:
        assert candidate["spec"]["model"]["hidden_dim"] == candidate["genome"]["hidden_dim"]
        assert candidate["spec"]["mode"] == "fresh"
        assert candidate["spec"]["parent"] is None


def test_factory_search_architecture_always_prints_the_multiplicative_cost_bound(capsys):
    """The cost bound is the one thing this workflow must never launch without.

    Printed to stderr so stdout stays the single JSON document every other
    factory subcommand prints.
    """
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    main([
        "factory", "search", str(spec_path), "--genome", "architecture",
        "--outer-population-size", "4", "--outer-populations", "2",
        "--population-size", "3", "--populations", "1",
        "--seed", "1", "--dry-run", "--no-trace",
    ])
    captured = capsys.readouterr()

    assert "24 trial(s)" in captured.err
    assert "4 architectures x 2 outer generation(s) x 3 candidates" in captured.err
    assert json.loads(captured.out)["estimate"]["estimated_max_trials"] == 24


def test_factory_search_architecture_rejects_the_legacy_halving_flag(tmp_path):
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    with pytest.raises(SystemExit, match="--budgets"):
        main([
            "factory", "search", str(spec_path), "--genome", "architecture",
            "--budgets", "1", "2", "--dry-run", "--no-trace",
        ])


def test_factory_search_architecture_rejects_an_unknown_outer_schema(tmp_path):
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    with pytest.raises(SystemExit, match="unknown architecture genome schema"):
        main([
            "factory", "search", str(spec_path), "--genome", "architecture",
            "--outer-schema", "architecture_search_v99", "--dry-run", "--no-trace",
        ])


def test_factory_meta_declares_the_architecture_schemas_and_backbone_presets(capsys):
    """The Evolve tab's NAS mode sources its options from the backend's enums."""
    main(["factory", "meta", "--no-trace"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["architecture_genome_schemas"] == ["architecture_search_v1"]
    assert payload["backbone_presets"]["transformer_h4_l2"] == {
        "backbone": "transformer", "backbone_kwargs": {"n_heads": 4, "n_layers": 2},
    }
    assert payload["backbone_presets"]["gru"] == {"backbone": "gru", "backbone_kwargs": {}}
    # The two schema registries stay separate: neither can be offered where
    # the other is expected.
    assert not set(payload["architecture_genome_schemas"]) & set(payload["genome_schemas"])


def test_factory_search_missing_spec_file_still_names_the_spec(tmp_path):
    with pytest.raises(SystemExit, match="spec file not found"):
        main([
            "factory", "search", str(tmp_path / "no-such-spec.json"),
            "--root", str(tmp_path / "runs"), "--dry-run", "--no-trace",
        ])


def test_factory_search_dry_run_rejects_invalid_budget_schedule():
    spec_path = Path(__file__).resolve().parents[1] / "specs" / "crafter-baseline.yaml"
    with pytest.raises(SystemExit, match="strictly increasing"):
        main([
            "factory", "search", str(spec_path), "--budgets", "2", "2",
            "--dry-run", "--no-trace",
        ])


def test_factory_search_missing_spec_is_a_concise_cli_error(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(SystemExit) as excinfo:
        main(["factory", "search", str(missing), "--dry-run", "--no-trace"])
    message = str(excinfo.value)
    assert message.startswith("invalid factory search:")
    assert str(missing) in message


def test_factory_breed_dry_run_prints_explicit_parent_and_weight_donor_lineage(tmp_path, capsys):
    root = tmp_path / "runs"
    base_doc = _spec().to_dict()
    base_doc["training"] = dict(base_doc["training"])
    base_doc["training"].update({
        "max_training_seconds": 600.0,
        "scheduled_sampling_p": 0.1,
        "loss_weights": {
            "closed_loop_pixel_loss_weight": 0.2,
            "closed_loop_latent_loss_weight": 0.3,
            "pixel": 1.0,
            "latent": 1.0,
            "semantic": 1.0,
        },
        "transition_balance_policy": {"stationary_cap": 0.4},
    })
    parent_a = _make_run(root, "parent-a", resolve(base_doc))
    base_doc["training"]["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.00001}
    base_doc["training"]["scheduled_sampling_p"] = 0.4
    parent_b = _make_run(root, "parent-b", resolve(base_doc))
    for artifacts, sha in ((parent_a, "a" * 64), (parent_b, "b" * 64)):
        _write_checkpoint_metadata(artifacts, sha256=sha)
        _write_budget_report(artifacts, tier="fast")

    main([
        "factory", "breed", "parent-a", "parent-b", "--root", str(root),
        "--seed", "19", "--weight-donor", "parent-b", "--dry-run", "--no-trace",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["child_spec"]["mode"] == "clone"
    assert payload["child_spec"]["parent"]["run_id"] == "parent-b"
    assert payload["breeding_lineage"]["parent_a_run_id"] == "parent-a"
    assert payload["breeding_lineage"]["parent_b_run_id"] == "parent-b"
    assert payload["breeding_lineage"]["weight_donor_run_id"] == "parent-b"
    assert sorted(path.name for path in (root / "crafter-baseline").iterdir() if path.is_dir()) == [
        "parent-a", "parent-b",
    ]


def _fake_trial_result(run_id, directory):
    """A run_trial() stand-in shaped just enough for _print_trial_result --
    used to test CLI wiring (argument threading) without importing torch."""
    from types import SimpleNamespace

    return SimpleNamespace(
        run_id=run_id, display_name="fake-display", mode="fresh", state="completed", directory=directory,
        architecture_hash="a" * 8, data_contract_hash="b" * 8, training_contract_hash="c" * 8,
        checkpoint_path=str(directory / "checkpoints" / "best-validation.pt"), checkpoint_sha256="d" * 64,
        training_stats={}, evaluation={}, budget_report={}, comparison=None, experiment_report_path=None,
    )


def test_factory_baseline_passes_an_explicit_run_id_through_to_run_trial(tmp_path, capsys, monkeypatch):
    """--run-id (added alongside Phase 2's job registry, which must be able
    to precompute a launched job's run id before run_trial's own
    auto-naming would otherwise decide it) threads straight through."""
    import cognitive_runtime.training.model_factory.runner as runner_module

    captured = {}

    def fake_run_trial(spec, **kwargs):
        captured.update(kwargs)
        return _fake_trial_result(kwargs.get("run_id") or "auto-named", tmp_path)

    monkeypatch.setattr(runner_module, "run_trial", fake_run_trial)
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(dump_canonical_json(_spec()))

    main(["factory", "baseline", str(spec_path), "--factory-run-id", "explicit-run-7", "--no-trace"])

    assert captured["run_id"] == "explicit-run-7"
    assert "run_id: explicit-run-7" in capsys.readouterr().out


def test_factory_clone_passes_an_explicit_run_id_through_to_run_trial(tmp_path, monkeypatch):
    import cognitive_runtime.training.model_factory.runner as runner_module

    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_checkpoint_metadata(artifacts)
    captured = {}

    def fake_run_trial(spec, **kwargs):
        captured.update(kwargs)
        return _fake_trial_result(kwargs.get("run_id") or "auto-named", tmp_path)

    monkeypatch.setattr(runner_module, "run_trial", fake_run_trial)

    main([
        "factory", "clone", "--root", str(root), "crafter-baseline-0001",
        "--factory-run-id", "clone-child-1", "--no-trace",
    ])

    assert captured["run_id"] == "clone-child-1"


def test_factory_breed_passes_an_explicit_run_id_through_to_run_trial(tmp_path, monkeypatch):
    import cognitive_runtime.training.model_factory.runner as runner_module

    root = tmp_path / "runs"
    base_doc = _spec().to_dict()
    base_doc["training"] = dict(base_doc["training"])
    base_doc["training"].update({
        "max_training_seconds": 600.0,
        "scheduled_sampling_p": 0.1,
        "loss_weights": {
            "closed_loop_pixel_loss_weight": 0.2,
            "closed_loop_latent_loss_weight": 0.3,
            # generic_action_effects_v2 (the default schema) also varies the
            # three reconstruction weights, and breeding requires every gene
            # path the schema declares to be present on both parents.
            "pixel": 1.0,
            "latent": 1.0,
            "semantic": 1.0,
        },
        "transition_balance_policy": {"stationary_cap": 0.4},
    })
    parent_a = _make_run(root, "parent-a", resolve(base_doc))
    base_doc["training"]["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.00001}
    base_doc["training"]["scheduled_sampling_p"] = 0.4
    parent_b = _make_run(root, "parent-b", resolve(base_doc))
    for artifacts, sha in ((parent_a, "a" * 64), (parent_b, "b" * 64)):
        _write_checkpoint_metadata(artifacts, sha256=sha)
        _write_budget_report(artifacts, tier="fast")
    captured = {}

    def fake_run_trial(spec, **kwargs):
        captured.update(kwargs)
        run_id = kwargs.get("run_id") or "auto-named"
        # cmd_factory_breed enriches lineage.json right after run_trial
        # returns (record_breeding_lineage); a real run_trial call would
        # have allocated it via allocate_run_artifacts, so this stand-in
        # writes the same minimal stub.
        child_directory = tmp_path / run_id
        child_directory.mkdir(parents=True, exist_ok=True)
        (child_directory / "lineage.json").write_text(json.dumps({
            "format": "model-factory-lineage-v1", "mode": "clone", "parent": None,
            "configuration_parents": ["parent-a", "parent-b"], "weight_donor": "parent-b",
        }))
        return _fake_trial_result(run_id, child_directory)

    monkeypatch.setattr(runner_module, "run_trial", fake_run_trial)

    main([
        "factory", "breed", "parent-a", "parent-b", "--root", str(root),
        "--seed", "19", "--weight-donor", "parent-b", "--factory-run-id", "bred-child-1", "--no-trace",
    ])

    assert captured["run_id"] == "bred-child-1"


# --------------------------------------------------------------------- show


def test_factory_show_prints_config_hashes_name_and_status(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())

    main(["factory", "show", "--root", str(root), "crafter-baseline-0001", "--no-trace"])
    out = capsys.readouterr().out

    assert "run_id: crafter-baseline-0001" in out
    assert f"display_name: {artifacts.display_name}" in out
    assert "completion_status: completed" in out
    with (artifacts.directory / "contracts.json").open() as handle:
        contracts = json.load(handle)
    assert contracts["architecture_hash"] in out
    assert contracts["data_contract_hash"] in out
    assert contracts["training_contract_hash"] in out
    assert "trial_horizons_ticks: [1, 4]" in out
    assert "model_horizons_ticks: [1, 4]" in out
    assert "horizons_aligned: True" in out
    assert "corpus_temporal_coverage: legacy/unavailable" in out
    # The complete effective config: every resolved spec field must appear,
    # not just a summary (epic #212 success criterion 5).
    with (artifacts.directory / "trial_spec.json").open() as handle:
        trial_spec = json.load(handle)
    assert json.dumps(trial_spec, indent=2, sort_keys=True) in out


def test_factory_show_defaults_to_latest_run(tmp_path, capsys):
    root = tmp_path / "runs"
    _make_run(root, "crafter-baseline-0001", _spec(seed=0))
    _make_run(root, "crafter-baseline-0002", _spec(seed=1))

    main(["factory", "show", "--root", str(root), "--no-trace"])
    out = capsys.readouterr().out
    assert "run_id: crafter-baseline-0002" in out


def test_factory_show_works_without_torch(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "crafter-baseline-0001", _spec())

    script = f"""
import builtins
_import = builtins.__import__

def block_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return _import(name, *args, **kwargs)

builtins.__import__ = block_torch
from cognitive_runtime.cli import main
main(["factory", "show", "--root", {str(root)!r}, "crafter-baseline-0001", "--no-trace"])
"""
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "run_id: crafter-baseline-0001" in completed.stdout
    assert "completion_status: completed" in completed.stdout


def test_factory_help_works_without_torch():
    script = """
import builtins
_import = builtins.__import__

def block_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return _import(name, *args, **kwargs)

builtins.__import__ = block_torch
from cognitive_runtime.cli import build_parser
try:
    build_parser().parse_args(["factory", "--help"])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("--help did not exit")
"""
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


# --------------------------------------------------------------------- lineage


def test_factory_lineage_renders_configuration_parents_and_weight_donor(tmp_path, capsys):
    root = tmp_path / "runs"
    _make_run(root, "crafter-baseline-0001", _spec(seed=0))
    parent = {"run_id": "crafter-baseline-0001", "checkpoint": "best-validation.pt", "sha256": "x" * 64}
    _make_run(root, "child-a", _spec(seed=1, parent=parent))
    _make_run(root, "child-b", _spec(seed=2, parent=parent))
    _make_run(
        root, "bred-child",
        _spec(seed=3, evolution={"configuration_parents": ["child-a", "child-b"], "weight_donor": "child-b"}),
    )

    main(["factory", "lineage", "--root", str(root), "bred-child", "--no-trace"])
    out = capsys.readouterr().out

    assert "bred-child" in out
    assert "configuration_parents: child-a, child-b" in out
    assert "weight_donor: child-b" in out
    # The deeper ancestor chain (each config parent's own parent) is walked too.
    assert "child-a" in out and "child-b" in out
    assert "crafter-baseline-0001" in out


def test_factory_lineage_defaults_to_latest_run(tmp_path, capsys):
    root = tmp_path / "runs"
    _make_run(root, "crafter-baseline-0001", _spec())

    main(["factory", "lineage", "--root", str(root), "--no-trace"])
    out = capsys.readouterr().out
    assert "lineage for crafter-baseline-0001" in out


# --------------------------------------------------------------------- clone --set


def test_factory_clone_rejects_unknown_top_level_field_with_nearest_match(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_checkpoint_metadata(artifacts)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "factory", "clone", "--root", str(root), "crafter-baseline-0001",
            "--set", "trainng.seed=5", "--no-trace",
        ])
    assert "trainng" in str(excinfo.value)
    assert "training" in str(excinfo.value)


def test_factory_clone_rejects_unknown_nested_field_with_nearest_match(tmp_path):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_checkpoint_metadata(artifacts)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "factory", "clone", "--root", str(root), "crafter-baseline-0001",
            "--set", "data.corpus_i=foo", "--no-trace",
        ])
    assert "corpus_i" in str(excinfo.value)
    assert "corpus_id" in str(excinfo.value)


# --------------------------------------------------------------------- resume


def test_factory_build_resume_spec_reopens_the_same_run_from_its_last_checkpoint(tmp_path):
    from cognitive_runtime.cli import _factory_build_resume_spec

    root = tmp_path / "runs"
    spec = _spec()
    artifacts = _make_run(root, "crafter-baseline-0001", spec, state=STATE_RUNNING)
    _write_checkpoint_metadata(artifacts, sha256="b" * 64, name="last.pt")

    resume_doc = _factory_build_resume_spec(artifacts.directory, "crafter-baseline-0001")

    assert resume_doc["mode"] == "resume"
    assert resume_doc["organism"] == spec.organism
    assert resume_doc["parent"] == {
        "run_id": "crafter-baseline-0001", "checkpoint": "last.pt", "sha256": "b" * 64,
    }

    # The resume doc must itself resolve/validate cleanly -- run_trial calls
    # resolve() on exactly this shape before it ever checks state.json -- and,
    # once resolved, its data/model/training/evaluation blocks must be
    # byte-identical to the original run's: resume requires an exact
    # continuation of the same architecture/data/training contracts, never a
    # modified sibling (run_trial rebuilds the model from these blocks and
    # checks it against the checkpoint's own recorded contracts).
    resolved = resolve(resume_doc)
    assert resolved.mode == "resume"
    assert resolved.parent == {"run_id": "crafter-baseline-0001", "checkpoint": "last.pt", "sha256": "b" * 64}
    assert resolved.data == spec.data
    assert resolved.model == spec.model
    assert resolved.training == spec.training
    assert resolved.evaluation == spec.evaluation


def test_factory_resume_missing_last_checkpoint_is_a_concise_cli_error(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "crafter-baseline-0001", _spec(), state=STATE_RUNNING)
    # No checkpoints/last.pt.json written -- only a genuinely interrupted
    # run (one that reached at least one checkpoint chunk) is resumable.

    with pytest.raises(SystemExit) as excinfo:
        main(["factory", "resume", "--root", str(root), "crafter-baseline-0001", "--no-trace"])
    assert "last.pt" in str(excinfo.value)


# --------------------------------------------------------------------- cancel / reconcile


def test_factory_cancel_writes_the_cancellation_flag(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec(), state=STATE_RUNNING)

    main(["factory", "cancel", "--root", str(root), "crafter-baseline-0001", "--reason", "operator requested"])

    payload = json.loads(cancel_request_path(artifacts.directory).read_text())
    assert payload["reason"] == "operator requested"
    assert "cancellation requested" in capsys.readouterr().out


def test_factory_cancel_unknown_run_is_a_concise_cli_error(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(SystemExit, match="run 'missing' was not found"):
        main(["factory", "cancel", "--root", str(root), "missing"])


def test_factory_reconcile_reports_recovered_runs_as_json(tmp_path, capsys):
    root = tmp_path / "runs"
    run_directory = root / "Crafter" / "stale-run"
    run_directory.mkdir(parents=True)
    create_state(state_path(run_directory), "stale-run", heartbeat_timeout_seconds=300.0)
    transition(state_path(run_directory), STATE_RUNNING)
    write_heartbeat(heartbeat_path(run_directory), run_id="stale-run", clock=lambda: time.time() - 10_000)

    main(["factory", "reconcile", "--root", str(root), "Crafter"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["organism"] == "Crafter"
    assert payload["recovered"] == [{"run_id": "stale-run", "reason": "stale_worker_watchdog_timeout"}]
    assert load_state(state_path(run_directory)).state == "failed"


def test_factory_reconcile_reports_nothing_when_no_run_is_stale(tmp_path, capsys):
    root = tmp_path / "runs"
    run_directory = root / "Crafter" / "fresh-run"
    run_directory.mkdir(parents=True)
    create_state(state_path(run_directory), "fresh-run", heartbeat_timeout_seconds=300.0)
    transition(state_path(run_directory), STATE_RUNNING)
    write_heartbeat(heartbeat_path(run_directory), run_id="fresh-run")

    main(["factory", "reconcile", "--root", str(root), "Crafter"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["recovered"] == []
    assert load_state(state_path(run_directory)).state == "running"


def test_factory_reconcile_kill_transitions_a_running_run_to_cancelled(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec(), state=STATE_RUNNING)

    main(["factory", "reconcile-kill", "--root", str(root), "crafter-baseline-0001", "--state", "cancelled"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "run_id": "crafter-baseline-0001", "recovered": True,
        "state": "cancelled", "reason": "killed_by_operator",
    }
    assert load_state(state_path(artifacts.directory)).state == "cancelled"


def test_factory_reconcile_kill_is_a_noop_for_an_already_terminal_run(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec(), state=STATE_COMPLETED)

    main(["factory", "reconcile-kill", "--root", str(root), "crafter-baseline-0001", "--state", "cancelled"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"run_id": "crafter-baseline-0001", "recovered": False, "state": None, "reason": None}
    assert load_state(state_path(artifacts.directory)).state == "completed"


# --------------------------------------------------------------------- compare / promote


def test_factory_compare_reports_paired_decision(tmp_path, capsys):
    root = tmp_path / "runs"
    baseline = _make_run(root, "crafter-baseline-0001", _spec(seed=0))
    parent = {"run_id": "crafter-baseline-0001", "checkpoint": "best-validation.pt", "sha256": "x" * 64}
    child_a = _make_run(root, "child-a", _spec(seed=1, parent=parent))
    child_b = _make_run(root, "child-b", _spec(seed=2, parent=parent))

    episodes = [f"e{i}" for i in range(5)]
    _write_validation(baseline, episodes, [1.0, 1.1, 0.9, 1.05, 0.95])
    _write_validation(child_a, episodes, [0.5, 0.55, 0.45, 0.52, 0.48])
    _write_validation(child_b, episodes, [1.5, 1.4, 1.6, 1.55, 1.45])

    main(["factory", "compare", "--root", str(root), "crafter-baseline-0001", "child-a", "child-b", "--no-trace"])
    out = capsys.readouterr().out

    assert "child-a: status=evaluable" in out
    assert "decision=candidate_improves" in out
    assert "child-b: status=evaluable" in out
    assert "decision=hold" in out


def test_factory_promote_records_champion_when_every_gate_passes(tmp_path, capsys, monkeypatch):
    """Matches the epic proposal's own terse example: ``factory promote
    child-b`` with no --family/--tier/--objective -- now that promotion is
    gated, this also proves a passing candidate still promotes cleanly."""
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    corpus_root = _build_passing_corpus(tmp_path, monkeypatch, organism="crafter-baseline", corpus_id="generic-v1")
    _write_checkpoint_metadata(artifacts)
    _write_validation(artifacts, ["b"], [0.5])
    _write_budget_report(artifacts, tier="fast")
    _write_experiment_report(artifacts)

    main([
        "factory", "promote", "--root", str(root), "--corpus-root", str(corpus_root),
        "crafter-baseline-0001", "--no-trace",
    ])
    out = capsys.readouterr().out

    assert "[PASS] checkpoint_identity" in out
    assert "[PASS] rollout_beats_copy_last" in out
    assert "leading_champion: crafter-baseline-0001" in out
    with (root / "crafter-baseline" / "registry.json").open() as handle:
        registry = json.load(handle)
    slot = registry["slots"]["crafter-baseline"]["fast"]["direct.t+4.model_mse"]
    assert slot["leading_champion"] == "crafter-baseline-0001"
    assert slot["population"][0]["run_id"] == "crafter-baseline-0001"


def test_factory_promote_refuses_and_holds_when_a_gate_fails(tmp_path, capsys, monkeypatch):
    """A run that would only 'win an A/B' without clearing its gates (here:
    it does not beat copy-last) must be refused, not promoted -- and the
    refusal is itself recorded as a hold, never silently dropped."""
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    corpus_root = _build_passing_corpus(tmp_path, monkeypatch, organism="crafter-baseline", corpus_id="generic-v1")
    _write_checkpoint_metadata(artifacts)
    _write_validation(artifacts, ["b"], [0.5])
    _write_budget_report(artifacts, tier="fast")
    _write_experiment_report(artifacts, beats_copy_last=False)

    with pytest.raises(SystemExit) as excinfo:
        main([
            "factory", "promote", "--root", str(root), "--corpus-root", str(corpus_root),
            "crafter-baseline-0001", "--no-trace",
        ])
    assert "gates failed" in str(excinfo.value)
    assert "rollout" in str(excinfo.value)

    with (root / "crafter-baseline" / "registry.json").open() as handle:
        registry = json.load(handle)
    slot = registry["slots"]["crafter-baseline"]["fast"]["direct.t+4.model_mse"]
    assert slot["leading_champion"] is None
    assert slot["population"] == []
    assert slot["history"][0]["action"] == "hold"


def test_factory_promote_hold_flag_skips_gate_evaluation(tmp_path, capsys):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_budget_report(artifacts, tier="fast")

    main(["factory", "promote", "--root", str(root), "--hold", "--reason", "not enough margin",
          "crafter-baseline-0001", "--no-trace"])

    with (root / "crafter-baseline" / "registry.json").open() as handle:
        registry = json.load(handle)
    slot = registry["slots"]["crafter-baseline"]["fast"]["direct.t+4.model_mse"]
    assert slot["leading_champion"] is None
    assert slot["population"] == []
    assert slot["history"][0]["action"] == "hold"


# --------------------------------------------------------------------- test (MF-C5)


def test_factory_test_refuses_without_prior_seed_confirmation(tmp_path):
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_budget_report(artifacts, tier="fast")

    with pytest.raises(SystemExit) as excinfo:
        main(["factory", "test", "--root", str(root), "crafter-baseline-0001", "--no-trace"])
    assert "seed_confirmation" in str(excinfo.value)
