"""Torch-free regression for checked-in Factory smoke inputs and CLI wiring."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".github" / "fixtures"


@pytest.fixture
def smoke():
    spec = importlib.util.spec_from_file_location(
        "factory_smoke_contract", ROOT / ".github" / "scripts" / "run_factory_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factory_inputs_exist_with_complete_verified_provenance():
    manifest = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["fixtures"]}
    assert set(entries) == {"micro-corpus.yaml", "micro-baseline.yaml"}
    for name, entry in entries.items():
        assert entry["sha256"] == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        for key in ("source", "seed", "license", "rights", "intended_edge_case"):
            assert key in entry and entry[key] is not None and entry[key] != ""
        assert entry["validation_status"] == "synthetic_only"


def test_micro_recipes_resolve_without_neural_imports():
    from cognitive_runtime.training.model_factory.corpus import _config_from_spec, _split_assignments
    from cognitive_runtime.training.model_factory.spec import apply_overrides, load_spec, resolve
    from cognitive_runtime.training.model_factory.effective_config import validate_execution_spec

    corpus = load_spec(FIXTURES / "micro-corpus.yaml")
    baseline = load_spec(FIXTURES / "micro-baseline.yaml")
    config = _config_from_spec(corpus)
    splits = _split_assignments(corpus, config)
    assert baseline["data"]["corpus_id"] == corpus["corpus_id"]
    assert baseline["organism"] == corpus["organism"]
    seeds = [set(seed for values in splits[split].values() for seed in values)
             for split in ("train", "validation", "test")]
    assert all(len(values) == 2 for values in seeds)
    assert sum(map(len, seeds)) == len(set.union(*seeds))
    assert config.episode_ticks <= 6 and config.world_size <= 16
    assert config.data_quality_gate and corpus["quality_policy"]["enabled"]
    resolved = resolve(baseline)
    validate_execution_spec(resolved)
    assert resolved.training["device"] == "cpu"
    assert resolved.training["epoch_budget"] == 1
    assert resolved.training["max_training_seconds"] <= 15
    assert resolved.model["latent_width"] <= 8 and resolved.model["hidden_dim"] <= 16
    for value in (0.125, 0.5):
        sibling = resolve(apply_overrides(baseline, [f"training.loss_weights.closed_loop_pixel_loss_weight={value}"]))
        validate_execution_spec(sibling)
        assert sibling.training["loss_weights"]["closed_loop_pixel_loss_weight"] == value


@pytest.mark.parametrize("output", ["", "run_id: \n", "run_id: first\nrun_id: second\n"])
def test_missing_or_ambiguous_run_id_fails(smoke, output):
    with pytest.raises(ValueError, match="exactly one"):
        smoke.result_field(output, "run_id")


def test_actual_generated_run_id_is_preserved(smoke):
    assert smoke.result_field("log line\nrun_id: randomized-20261009-f39a\nstate: completed\n", "run_id") == "randomized-20261009-f39a"


def test_smoke_commands_use_the_current_cli_contract(smoke):
    from cognitive_runtime.cli import build_parser

    cases = [
        ("corpus", "build", str(FIXTURES / "micro-corpus.yaml"), "--root", "corpora"),
        ("baseline", str(FIXTURES / "micro-baseline.yaml"), "--root", "runs", "--corpus-root", "corpora", "--no-export-predictions"),
        ("clone", "actual-run-id", "--root", "runs", "--corpus-root", "corpora", "--set", "training.loss_weights.closed_loop_pixel_loss_weight=0.125", "--no-export-predictions"),
        ("compare", "actual-baseline", "actual-sibling-a", "actual-sibling-b", "--root", "runs"),
    ]
    for arguments in cases:
        command = smoke.factory_command(*arguments)
        args = build_parser().parse_args(command[3:])
        assert args.no_trace is True
        assert callable(args.func)


def test_unavailable_factory_fails_and_leaves_failure_evidence(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=2, stdout="", stderr="factory CLI unavailable",
    ))
    output = tmp_path / "failed-smoke"
    with pytest.raises(RuntimeError, match="factory CLI unavailable"):
        smoke.run_smoke(output)
    summary = json.loads((output / "smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["stages"][0]["returncode"] == 2
    assert not summary["run_ids"]


def test_subprocess_timeout_fails_instead_of_skipping(smoke, monkeypatch, tmp_path):
    def timeout(command, **kwargs):
        assert 0 < kwargs["timeout"] <= 60
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        raise smoke.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(smoke.subprocess, "run", timeout)
    output = tmp_path / "timeout-smoke"
    with pytest.raises(smoke.subprocess.TimeoutExpired):
        smoke.run_smoke(output)
    assert json.loads((output / "smoke-summary.json").read_text())["status"] == "failed"


def test_missing_fixture_is_a_hard_failure(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "FIXTURES", tmp_path)
    with pytest.raises(FileNotFoundError, match="micro-corpus.yaml"):
        smoke.run_smoke(tmp_path / "missing-inputs")


def test_smoke_will_not_reuse_a_previous_output_directory(smoke, tmp_path):
    with pytest.raises(FileExistsError):
        smoke.run_smoke(tmp_path)


def test_workflow_is_manual_bounded_and_has_no_success_shaped_skip():
    path = ROOT / ".github" / "workflows" / "nightly-factory.yml"
    # BaseLoader deliberately keeps YAML 1.1's `on` key as a string.
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert int(workflow["jobs"]["smoke"]["timeout-minutes"]) <= 5
    text = path.read_text(encoding="utf-8")
    assert "steps.detect" not in text and "skipping (issue" not in text
    assert "run_factory_smoke.py" in text
    assert "test_resume_matches_uninterrupted_training" in text
    assert "test_budget_exceeded_checkpoint_is_valid_and_loadable" in text
    assert "test_no_search_step_ever_opens_the_sealed_test_partition" in text
    assert "pytest tests -q -k" not in text


def test_smoke_chains_actual_cli_ids_and_records_success(smoke, monkeypatch, tmp_path):
    output = tmp_path / "smoke"
    corpus_directory = output / "corpora" / "organism" / "corpus"
    ids = ["generated-parent-a71c", "generated-sibling-ff02", "generated-sibling-811a"]
    observed = []

    def execute(command, **kwargs):
        arguments = command[4:]
        observed.append(arguments)
        operation = arguments[0]
        if operation == "corpus":
            corpus_directory.mkdir(parents=True, exist_ok=True)
            manifest = {"sessions": {name: [{"sha256": name}] for name in ("train", "validation", "test")}}
            (corpus_directory / "corpus_manifest.json").write_text(json.dumps(manifest))
            stdout = f"directory: {corpus_directory}\n"
        elif operation in ("baseline", "clone"):
            run_id = ids[sum(args[0] in ("baseline", "clone") for args in observed) - 1]
            if operation == "clone":
                assert arguments[1] == ids[0]
            directory = output / "runs" / "organism" / run_id
            for name in ("trial_spec.json", "contracts.json", "lineage.json", "execution.json",
                         "metrics/validation.json", "metrics/budget_report.json", "experiment_report.json",
                         "checkpoints/last.pt", "checkpoints/best-validation.pt"):
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            stdout = f"run_id: {run_id}\nstate: completed\ndirectory: {directory}\n"
        else:
            assert operation == "compare"
            assert arguments[1:4] == ids
            stdout = f"baseline: {ids[0]} (selection_metric=validation_loss)\n"
            stdout += "".join(f"{run_id}: status=evaluable mean_delta=0.0\n" for run_id in ids[1:])
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", execute)
    summary = smoke.run_smoke(output)
    assert summary["status"] == "passed"
    assert summary["run_ids"] == ids
    assert len(summary["stages"]) == 6
    assert summary["corpus_manifest_sha256"]
