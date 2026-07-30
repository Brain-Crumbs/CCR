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
from pathlib import Path

import pytest

from cognitive_runtime.cli import build_parser, main
from cognitive_runtime.training.model_factory.artifacts import RunArtifacts, allocate_run_artifacts
from cognitive_runtime.training.model_factory.contracts import ArchitectureContract, DataContract, TrainingContract
from cognitive_runtime.training.model_factory.spec import ExperimentSpec, resolve
from cognitive_runtime.training.model_factory.state import (
    STATE_COMPLETED,
    STATE_RUNNING,
    create_state,
    state_path,
    transition,
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


# --------------------------------------------------------------------- parsing + dispatch


FACTORY_DISPATCH = [
    (["factory", "baseline", "spec.yaml"], "cmd_factory_baseline"),
    (["factory", "clone", "some-run"], "cmd_factory_clone"),
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
    ["factory", "clone", "--help"],
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


def test_factory_promote_records_champion_with_zero_extra_flags(tmp_path, capsys):
    """Matches the epic proposal's own terse example: ``factory promote
    child-b`` with no --family/--tier/--objective."""
    root = tmp_path / "runs"
    artifacts = _make_run(root, "crafter-baseline-0001", _spec())
    _write_checkpoint_metadata(artifacts)
    _write_validation(artifacts, ["b"], [0.5])
    _write_budget_report(artifacts, tier="fast")

    main(["factory", "promote", "--root", str(root), "crafter-baseline-0001", "--no-trace"])
    out = capsys.readouterr().out

    assert "leading_champion: crafter-baseline-0001" in out
    with (root / "crafter-baseline" / "registry.json").open() as handle:
        registry = json.load(handle)
    slot = registry["slots"]["crafter-baseline"]["fast"]["direct.t+4.model_mse"]
    assert slot["leading_champion"] == "crafter-baseline-0001"
    assert slot["population"][0]["run_id"] == "crafter-baseline-0001"


def test_factory_promote_hold_records_decision_without_touching_population(tmp_path, capsys):
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
