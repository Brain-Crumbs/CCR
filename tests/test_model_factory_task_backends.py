"""Offline lifecycle acceptance using project-authored in-memory synthetic data.

Seed 285; MIT project fixture. No provider data. Hashes are computed over exact
feature/target records in prepare(), making changed evidence change the corpus.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from brain.cortex.model_contracts import InferenceInput, require_inference_input
from cognitive_runtime.training.model_factory import task_registry as registry
from cognitive_runtime.training.model_factory.contracts import contract_hash
from cognitive_runtime.training.model_factory.checkpoint import load_factory_checkpoint
from cognitive_runtime.training.model_factory.runner import run_trial
from cognitive_runtime.training.model_factory.spec import load_spec, resolve, TASK_DOCUMENT_FORMAT
from cognitive_runtime.training.model_factory.state import load_state, state_path, heartbeat_path
from cognitive_runtime.training.model_factory.task_contracts import (
    TaskIdentity, TaskDataContract, ModelDefinition, TrainingBatch, EvaluationBatch,
    PreparedTask, TaskCancelled,
)


class FakeBackend:
    capabilities = frozenset({"prepare", "fit", "predict", "evaluate", "save", "load", "resume", "clone"})

    def __init__(self, domain="market"):
        self.identity = TaskIdentity(domain, "synthetic-test", "fake-" + domain)
        self.calls = []
        self.fail = None
        self.cancel = False

    def validate(self, spec):
        assert set(spec.model) == {"backend"}
        if set(spec.training) - {"device", "precision", "determinism_policy", "max_training_seconds"}:
            raise ValueError("unsupported training setting")
        if spec.evaluation:
            raise ValueError("unsupported evaluation setting")

    def prepare(self, spec, *, corpus_root):
        self.calls.append("prepare")
        identity = self.identity.to_dict()
        inputs = InferenceInput("numeric-v1", ("sample-285",), ((1.0, 2.0),))
        digest = contract_hash({"seed": 285, "features": inputs.features, "targets": [[3.0]],
                                "license": "MIT", "source": "project-authored", "edge_case": "domain separation"})
        data = TaskDataContract(identity, spec.data["corpus_id"], "numeric-v1",
                                {"train": [{"id": "synthetic", "sha256": digest}], "validation": [], "test": []})
        return PreparedTask(data, ModelDefinition(identity, {"kind": "fake"}, "numeric-v1"),
                            TrainingBatch(inputs, ((3.0,),)), EvaluationBatch(inputs, ((3.0,),)))

    def build(self, definition):
        self.calls.append("build")
        return {"value": 0, "steps": 0}

    def fit(self, model, data, configuration, control, *, resume):
        self.calls.append("fit-resume" if resume else "fit")
        control.check()
        if self.fail == "fit":
            raise RuntimeError("synthetic training failure")
        model["value"] = sum(data.inputs.features[0])
        model["steps"] += 1
        if self.cancel:
            raise TaskCancelled("synthetic cancellation")
        return {"steps": model["steps"]}

    def predict(self, model, inputs):
        self.calls.append("predict")
        require_inference_input(inputs)
        if self.fail == "predict":
            raise RuntimeError("synthetic inference failure")
        return ((model["value"],),)

    def evaluate(self, predictions, data):
        self.calls.append("evaluate")
        return {self.identity.domain + "_error": abs(predictions[0][0] - data.labels[0][0])}

    def save(self, model):
        self.calls.append("save")
        return dict(model)

    def load(self, definition, state, *, resume):
        self.calls.append("load-resume" if resume else "load")
        return dict(state) if resume else {"value": state["value"], "steps": 0}


@pytest.fixture
def backends(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    result = [FakeBackend("crafter"), FakeBackend("market")]
    for backend in result:
        registry.register_backend(registry.BackendRegistration(backend.identity, lambda b=backend: b, backend.capabilities))
    return result


def spec(backend):
    return {"format": TASK_DOCUMENT_FORMAT, "organism": "ContractTest", "mode": "fresh",
            "data": {"corpus_id": "synthetic-285"}, "model": {"backend": backend.identity.backend}}


def continuation(raw, result, mode="clone"):
    return {**raw, "mode": mode, "parent": {"run_id": result.run_id,
        "checkpoint": "last.json", "sha256": result.checkpoint_sha256}}


@pytest.mark.parametrize("index", [0, 1])
def test_both_domains_use_real_factory_lifecycle(tmp_path, backends, index):
    backend = backends[index]
    result = run_trial(spec(backend), root=tmp_path, run_id="original", naming_seed=285)
    assert result.state == "completed"
    assert backend.calls == ["prepare", "build", "fit", "save", "predict", "evaluate"]
    assert result.evaluation == {backend.identity.domain + "_error": 0}
    report = json.loads(Path(result.experiment_report_path).read_text())
    assert report["task_identity"]["domain"] == backend.identity.domain
    assert {r["kind"] for r in report["artifacts"]} == {"checkpoint", "evaluation"}
    for ref in report["artifacts"]:
        assert hashlib.sha256((result.directory / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    paths = [Path(result.checkpoint_path), Path(result.checkpoint_path + ".json"), result.directory / "trial_spec.json"]
    before = [p.read_bytes() for p in paths]
    loaded = load_factory_checkpoint(result.checkpoint_path)
    assert loaded.model == {"value": 3.0, "steps": 0}  # inspection loads weights, not training progress
    assert loaded.payload["state"] == {"value": 3.0, "steps": 1}
    assert [p.read_bytes() for p in paths] == before
    clone = run_trial(continuation(spec(backend), result), root=tmp_path, run_id="clone", naming_seed=286)
    assert clone.state == "completed" and clone.training_stats["steps"] == 1


def test_cross_domain_rejected_before_prepare(tmp_path, backends):
    parent = run_trial(spec(backends[0]), root=tmp_path, run_id="parent")
    for mode in ("clone", "resume"):
        with pytest.raises(ValueError, match="cross-domain"):
            run_trial(continuation(spec(backends[1]), parent, mode), root=tmp_path)
    assert backends[1].calls == []
    assert len(list((tmp_path / "ContractTest").iterdir())) >= 1


class CursorBackend(FakeBackend):
    """Two deterministic updates with explicit cursor, momentum and RNG state."""
    stop_after = None
    stop_on_save = None

    def save(self, model):
        import os
        if model["steps"] == self.stop_on_save:
            os._exit(24)
        return dict(model)

    def build(self, definition):
        return {"value": 0.0, "steps": 0, "momentum": 0.0, "rng": 285}

    def load(self, definition, state, *, resume):
        if resume:
            return dict(state)
        model = self.build(definition)
        model["value"] = state["value"]
        return model

    def fit(self, model, data, configuration, control, *, resume):
        import os
        while model["steps"] < 2:
            control.check()
            model["rng"] = (1664525 * model["rng"] + 1013904223) % (2 ** 32)
            model["momentum"] = 0.9 * model["momentum"] + model["rng"] / (2 ** 32)
            model["value"] += model["momentum"]
            model["steps"] += 1
            control.checkpoint()
            if model["steps"] == self.stop_after:
                os._exit(23)  # actual dead worker: no finally/terminal transition
        return {"steps": model["steps"]}


@pytest.mark.parametrize("kill_in_save", [False, True])
def test_resume_restores_state_and_preserves_manifests(tmp_path, backends, monkeypatch, kill_in_save):
    backend = CursorBackend()
    monkeypatch.setitem(registry._REGISTRY, backend.identity.backend,
                        registry.BackendRegistration(backend.identity, lambda: backend, backend.capabilities))
    code = """
import sys
from tests.test_model_factory_task_backends import CursorBackend, spec
from cognitive_runtime.training.model_factory import task_registry as registry
from cognitive_runtime.training.model_factory.runner import run_trial
backend = CursorBackend()
if sys.argv[2] == 'save':
    backend.stop_on_save = 2  # die during second save, retaining first checkpoint
else:
    backend.stop_after = 1
registry.register_backend(registry.BackendRegistration(backend.identity, lambda: backend, backend.capabilities))
run_trial(spec(backend), root=sys.argv[1], run_id='interrupted')
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), "save" if kill_in_save else "fit"], timeout=20)
    assert child.returncode == (24 if kill_in_save else 23)
    directory = tmp_path / "ContractTest/interrupted"
    assert load_state(state_path(directory)).state == ("checkpointing" if kill_in_save else "running")
    assert not (directory / "experiment_report.json").exists()
    # Advance only the persisted heartbeat age to avoid a 300-second test wait.
    heartbeat = json.loads(heartbeat_path(directory).read_text())
    heartbeat["heartbeat_seconds"] = 0
    heartbeat_path(directory).write_text(json.dumps(heartbeat))
    immutable = [directory / name for name in ("trial_spec.json", "contracts.json", "data_manifest.json", "execution.json")]
    before = [p.read_bytes() for p in immutable]
    sha = json.loads((directory / "checkpoints/last.json.json").read_text())["checkpoint_sha256"]
    raw = {**spec(backend), "mode": "resume", "parent": {
        "run_id": "interrupted", "checkpoint": "last.json", "sha256": sha}}
    resumed = run_trial(raw, root=tmp_path)
    assert resumed.run_id == "interrupted" and resumed.training_stats["steps"] == 2
    uninterrupted = run_trial(spec(backend), root=tmp_path, run_id="uninterrupted")
    actual = load_factory_checkpoint(resumed.checkpoint_path).payload["state"]
    expected = load_factory_checkpoint(uninterrupted.checkpoint_path).payload["state"]
    assert actual == expected  # weights, cursor, optimizer-like momentum and RNG
    assert [p.read_bytes() for p in immutable] == before
    with pytest.raises(Exception, match="not an active"):
        run_trial(continuation(spec(backend), resumed, "resume"), root=tmp_path)


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_failure_is_persisted_not_completed(tmp_path, backends, stage):
    backend = backends[1]
    backend.fail = stage
    with pytest.raises(RuntimeError, match="synthetic"):
        run_trial(spec(backend), root=tmp_path, run_id="failure")
    directory = tmp_path / "ContractTest" / "failure"
    assert load_state(state_path(directory)).state == "failed"
    report = json.loads((directory / "experiment_report.json").read_text())
    assert report["status"] == "failed" and report["failure"]


def test_cancelled_checkpoint_is_not_promotable(tmp_path, backends):
    backend = backends[1]
    backend.cancel = True
    result = run_trial(spec(backend), root=tmp_path)
    assert result.state == "cancelled" and result.checkpoint_sha256
    assert "predict" not in backend.calls


def test_unknown_backend_and_unsupported_configuration_fail_before_artifacts(tmp_path, backends):
    raw = spec(backends[1])
    with pytest.raises(ValueError, match="unknown task/model backend"):
        run_trial({**raw, "model": {"backend": "does-not-exist"}}, root=tmp_path)
    with pytest.raises(ValueError, match="unsupported training"):
        run_trial({**raw, "training": {"invented": True}}, root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_checkpoint_tampering_and_training_mismatch_fail_before_load(tmp_path, backends):
    backend = backends[1]
    result = run_trial(spec(backend), root=tmp_path)
    backend.calls.clear()
    raw = continuation(spec(backend), result, "resume")
    with pytest.raises(ValueError, match="training contract differs"):
        run_trial({**raw, "training": {"max_training_seconds": 61}}, root=tmp_path)
    assert "load-resume" not in backend.calls
    Path(result.checkpoint_path).write_text("{}")
    with pytest.raises(ValueError, match="bytes do not match"):
        load_factory_checkpoint(result.checkpoint_path)


def test_inference_has_no_label_or_metadata_slot():
    inputs = InferenceInput("v1", ("id",), ((1.0,),))
    for batch in (TrainingBatch(inputs, ((2.0,),)), EvaluationBatch(inputs, ((2.0,),)), {"labels": [2]}):
        with pytest.raises(TypeError, match="InferenceInput"):
            require_inference_input(batch)
    with pytest.raises(TypeError):
        InferenceInput("v1", ("id",), ((1.0,),), labels=[2])
    with pytest.raises(TypeError, match="finite numbers"):
        InferenceInput("v1", ("id",), (({"terminal_settlement": 1},),))
    assert not hasattr(inputs, "__dict__")


def test_legacy_configuration_load_does_not_add_backend_or_change_hash():
    path = Path(".github/fixtures/micro-baseline.yaml")
    before = path.read_bytes()
    resolved = resolve(load_spec(path))
    assert "backend" not in resolved.model
    assert resolve(resolved.to_dict()).hash == resolved.hash
    assert path.read_bytes() == before


def test_imports_do_not_attempt_optional_frameworks():
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers', 'alpaca', 'openai'}:
            raise AssertionError('optional import attempted: ' + fullname)
sys.meta_path.insert(0, Block())
import brain.cortex.model_contracts
import cognitive_runtime.training.model_factory.runner
import cognitive_runtime.training.model_factory.task_contracts
import cognitive_runtime.training.model_factory.task_registry as registry
registry.backend_registration('crafter')
assert 'cognitive_runtime.training.model_factory.legacy_adapter' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_budget_outcome_and_nonfinite_evaluation(tmp_path, backends, monkeypatch):
    backend = backends[1]
    original_fit = backend.fit
    def exceed(model, data, configuration, control, *, resume):
        from cognitive_runtime.training.model_factory.task_runner import TaskBudgetExceeded
        original_fit(model, data, configuration, control, resume=resume)
        raise TaskBudgetExceeded("synthetic exhausted budget")
    monkeypatch.setattr(backend, "fit", exceed)
    result = run_trial(spec(backend), root=tmp_path, run_id="budget")
    assert result.state == "budget_exceeded" and "predict" not in backend.calls
    monkeypatch.setattr(backend, "fit", original_fit)
    monkeypatch.setattr(backend, "evaluate", lambda *_: {"metric": float("nan")})
    with pytest.raises(ValueError, match="non-finite"):
        run_trial(spec(backend), root=tmp_path, run_id="nonfinite")
    report = json.loads((tmp_path / "ContractTest/nonfinite/experiment_report.json").read_text())
    assert report["status"] == "failed"


def test_changed_model_and_data_reject_continuation(tmp_path, backends, monkeypatch):
    backend = backends[1]
    result = run_trial(spec(backend), root=tmp_path)
    prepare = backend.prepare
    def changed(specification, *, corpus_root):
        prepared = prepare(specification, corpus_root=corpus_root)
        return dataclasses.replace(prepared, model_definition=dataclasses.replace(prepared.model_definition, model={"kind": "different"}))
    monkeypatch.setattr(backend, "prepare", changed)
    with pytest.raises(ValueError, match="incompatible model"):
        run_trial(continuation(spec(backend), result), root=tmp_path)
    def changed_data(specification, *, corpus_root):
        prepared = prepare(specification, corpus_root=corpus_root)
        return dataclasses.replace(prepared, data_contract=dataclasses.replace(prepared.data_contract, split_artifacts={"train": []}))
    monkeypatch.setattr(backend, "prepare", changed_data)
    with pytest.raises(ValueError, match="data contract differs"):
        run_trial(continuation(spec(backend), result), root=tmp_path)


def test_public_checkpoint_loader_enforces_modes_and_capabilities(tmp_path, backends, monkeypatch):
    from cognitive_runtime.training.model_factory.task_contracts import TaskTrainingContract
    backend = backends[1]
    result = run_trial(spec(backend), root=tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        load_factory_checkpoint(result.checkpoint_path, mode="clone", resume=True)
    backend.capabilities = backend.capabilities - {"resume"}
    monkeypatch.setitem(registry._REGISTRY, backend.identity.backend,
                        registry.BackendRegistration(backend.identity, lambda: backend, backend.capabilities))
    resolved = resolve(spec(backend))
    prepared = backend.prepare(resolved, corpus_root=None)
    backend.calls.clear()
    with pytest.raises(ValueError, match="does not support 'resume'"):
        load_factory_checkpoint(result.checkpoint_path, mode="resume",
                                architecture_contract=prepared.model_definition,
                                data_contract_hash=prepared.data_contract,
                                training_contract=TaskTrainingContract(resolved.training))
    assert "load-resume" not in backend.calls
