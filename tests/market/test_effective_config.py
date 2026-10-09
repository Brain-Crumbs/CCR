"""Project-authored synthetic constructor fixtures (MIT, no external data).

The tiny named-parameter fixture is defined inline, deterministic (no RNG),
with intended edge cases: class substitution, group drift, and JSON resume.
No torch import or training occurs in this default/offline test module.
"""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cognitive_runtime.training.optimizer_config import (
    OPTIMIZER_FORMAT, LEGACY_BEHAVIOR, action_optimizer_config, build_optimizer,
    effective_hash, optimizer_manifest, resolve_optimizer, restore_optimizer,
)
from cognitive_runtime.training.model_factory.effective_config import validate_execution_spec
from cognitive_runtime.training.model_factory.spec import SpecError, resolve


class Parameter:
    shape = (2, 1)


class Toy:
    def __init__(self):
        self.weight = Parameter()

    def parameters(self):
        return [self.weight]

    def named_parameters(self):
        return [("weight", self.weight)]


class Adam:
    def __init__(self, parameters, **kwargs):
        self.param_groups = [{"params": list(parameters), **kwargs}]
        self.loaded = False

    def state_dict(self):
        return {"state": {}, "param_groups": [{**self.param_groups[0], "params": [0]}]}

    def load_state_dict(self, state):
        self.loaded = True


class AdamW(Adam):
    pass


TORCH = SimpleNamespace(optim=SimpleNamespace(Adam=Adam, AdamW=AdamW))


def spec_doc(**training):
    return {"organism": "Test", "mode": "fresh", "data": {"corpus_id": "synthetic", "world": "crafter"},
            "evaluation": {"selection_metric": "validation_loss"}, "training": training}


@pytest.mark.parametrize("name", ["adam", "adamw"])
def test_constructor_and_manifest_match_all_resolved_settings(name):
    config = resolve_optimizer({"name": name, "lr": 0.02, "weight_decay": 0.3, "betas": [0.7, 0.8], "eps": 1e-6, "amsgrad": True})
    toy = Toy()
    optimizer = build_optimizer(TORCH, toy.parameters(), config)
    manifest = optimizer_manifest(toy, optimizer)
    assert manifest["class"].lower() == name
    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
        assert manifest["parameter_groups"][0][key] == config[key]
    assert manifest["parameter_groups"][0]["parameters"] == [{"name": "weight", "shape": [2, 1]}]
    restored = json.loads(json.dumps(manifest))
    assert effective_hash(restored) == effective_hash(manifest)
    restore_optimizer(toy, optimizer, optimizer.state_dict(), restored)
    assert optimizer.loaded


def test_two_supported_configs_have_different_constructors_and_hashes():
    toy = Toy()
    manifests = [optimizer_manifest(toy, build_optimizer(TORCH, toy.parameters(), {"name": name})) for name in ("adam", "adamw")]
    assert manifests[0]["class"] != manifests[1]["class"]
    assert effective_hash(manifests[0]) != effective_hash(manifests[1])


@pytest.mark.parametrize("bad", [
    {"name": "sgd"}, {"momentum": 0.9}, {"lr": float("nan")}, {"lr": -1},
    {"betas": [0.9, 1.0]}, {"betas": [0.9]}, {"weight_decay": float("inf")},
    {"amsgrad": "false"}, {"params": []}, {"format": "invented"},
])
def test_unsupported_optimizer_fails_before_constructor(bad):
    with pytest.raises(ValueError):
        build_optimizer(SimpleNamespace(), [], bad)


def test_explicit_versioned_legacy_adam_defaults_and_migration():
    legacy = action_optimizer_config(SimpleNamespace(optimizer_behavior=LEGACY_BEHAVIOR, optimizer=None, lr=1e-3))
    assert legacy["name"] == "adam" and legacy["weight_decay"] == 0.0
    resolved = resolve(spec_doc())
    assert resolved.training["optimizer"]["format"] == OPTIMIZER_FORMAT
    assert resolved.training["optimizer"]["name"] == "adamw"
    assert effective_hash(legacy) != effective_hash(dict(resolved.training["optimizer"]))


def test_unknown_optimizer_fails_spec_resolution():
    with pytest.raises(SpecError, match="unsupported optimizer"):
        resolve(spec_doc(optimizer={"name": "sgd"}))


@pytest.mark.parametrize("training", [
    {"loss_weights": {"semantic": 0.4}},
    {"objective": "autoregressive", "batch_size": 32},
    {"scheduler": {"name": "StepLR"}}, {"precision": "bf16"}, {"step_budget": 10},
    {"early_stopping_policy": {"patience": 3}}, {"loss_weights": {"typo": 2}},
    {"loss_weights": {"lr": 2}}, {"loss_weights": {"device": "cuda"}},
    {"loss_weights": {"pixel": 1, "pixel_loss_weight": 2}},
    {"objective": "autoregressive", "loss_weights": {"closed_loop_pixel_loss_weight": 1}},
    {"loss_weights": {"autoregressive_rollout_pixel_weight": 1}},
    {"objective": "autoregressive", "transition_balance_policy": {"stationary_cap": 0.5}},
    {"transition_balance_policy": {"invented_gene": 0.5}},
    {"objective": "autoregressive", "loss_weights": {"autoregressive_rollout_weight": 1, "autoregressive_rollout_latent_weight": 1}},
])
def test_inert_or_unsupported_configuration_fails_before_model_import(training):
    with pytest.raises(ValueError):
        validate_execution_spec(resolve(spec_doc(**training)))


@pytest.mark.parametrize("change", [{"name": "adam"}, {"lr": 0.04}, {"betas": [0.5, 0.8]}, {"weight_decay": 0.2}])
def test_resume_rejects_configuration_change_before_load(change):
    toy = Toy()
    original = build_optimizer(TORCH, toy.parameters(), {})
    changed = build_optimizer(TORCH, toy.parameters(), change)
    with pytest.raises(ValueError, match="effective optimizer"):
        restore_optimizer(toy, changed, original.state_dict(), optimizer_manifest(toy, original))
    assert not changed.loaded


def test_resume_rejects_missing_evidence_and_corrupt_saved_groups():
    toy = Toy()
    optimizer = build_optimizer(TORCH, toy.parameters(), {})
    with pytest.raises(ValueError, match="without effective"):
        restore_optimizer(toy, optimizer, optimizer.state_dict(), None)
    state = optimizer.state_dict()
    state["param_groups"][0]["lr"] = 99
    with pytest.raises(ValueError, match="state groups"):
        restore_optimizer(toy, optimizer, state, optimizer_manifest(toy, optimizer))
    assert not optimizer.loaded


def test_module_imports_are_torch_free():
    subprocess.run([sys.executable, "-c", "import sys; import cognitive_runtime.training.optimizer_config; import cognitive_runtime.training.model_factory.runner; assert 'torch' not in sys.modules"], check=True)


def test_runner_rejects_unsupported_config_before_corpus_or_training(monkeypatch):
    from cognitive_runtime.training.model_factory import runner
    def forbidden(*args, **kwargs):
        pytest.fail("invalid config reached corpus/model work")
    monkeypatch.setattr(runner, "resolve_corpus", forbidden)
    monkeypatch.setattr(runner, "_action_world_model_module", forbidden)
    with pytest.raises(ValueError, match="scheduler"):
        runner.run_trial(spec_doc(scheduler={"name": "StepLR"}))


def test_inactive_semantic_search_gene_fails_closed():
    from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V2
    semantic_gene = GENERIC_ACTION_EFFECTS_V2.genes["loss_weights.semantic"]
    with pytest.raises(ValueError, match="semantic loss is inactive"):
        validate_execution_spec(resolve(spec_doc(loss_weights={"semantic": semantic_gene.default})))


def test_autoregressive_and_seed_defaults_materialize_actual_settings():
    spec = resolve(spec_doc(objective="autoregressive", seed=123))
    assert spec.training["batch_size"] == 1
    assert spec.training["determinism_policy"]["seed"] == 123
    validate_execution_spec(spec)


def test_inert_checkpoint_policy_cannot_be_searched():
    with pytest.raises(ValueError, match="checkpoint_selection_policy"):
        validate_execution_spec(resolve(spec_doc(checkpoint_selection_policy={"metric": "ignored", "mode": "max"})))


@pytest.mark.parametrize("objective", ["windowed_rollout", "autoregressive"])
def test_new_default_search_schema_proposals_pass_execution_preflight(objective):
    from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V1, GENERIC_ACTION_EFFECTS_V2, GENERIC_ACTION_EFFECTS_V3
    from cognitive_runtime.training.model_factory.search import propose
    assert GENERIC_ACTION_EFFECTS_V1.content_hash == "e247442c3fac00d30f90ca411cce0a7c4b7428247f963e3034dc22b0e408135d"
    assert GENERIC_ACTION_EFFECTS_V2.content_hash == "f2ccc7d2dfcd490d57c5b30822e83603f66bea1b77634c36af63ccea1aeb6925"
    assert GENERIC_ACTION_EFFECTS_V3.content_hash == "8b4659dd6cf5097469530d466dbb610bebbeccdd0d5330f34f998a38809c49c3"
    base = resolve(spec_doc(objective=objective))
    proposals = propose(base, GENERIC_ACTION_EFFECTS_V3, n=3, seed=7)
    for proposal in proposals:
        validate_execution_spec(proposal)
        assert "semantic" not in proposal.training["loss_weights"]
        if objective == "autoregressive":
            assert "closed_loop_pixel_loss_weight" not in proposal.training["loss_weights"]
            assert not proposal.training["transition_balance_policy"]


def test_autoregressive_v3_proposal_parent_and_breeding_roundtrip():
    from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V3 as schema
    from cognitive_runtime.training.model_factory.search import propose, _parent_record_from_result
    from cognitive_runtime.training.model_factory.breeding import _extract_genome, breed
    proposals = propose(resolve(spec_doc(objective="autoregressive")), schema, n=2, seed=7)
    parents = []
    for index, proposal in enumerate(proposals):
        result = SimpleNamespace(checkpoint_path=f"parent-{index}/best.pt", checkpoint_sha256=str(index) * 64, run_id=f"parent-{index}", architecture_hash="a" * 64, data_contract_hash="d" * 64)
        parent = _parent_record_from_result(result, proposal, schema)
        assert parent.genome == _extract_genome(schema, proposal.training)
        assert set(parent.genome) == set(schema.genes)
        parents.append(parent)
    result = breed(parents[0], parents[1], schema, objective="autoregressive", generation=1, seed=7)
    child = result.child_spec
    validate_execution_spec(child)
    assert "closed_loop_pixel_loss_weight" not in child.training["loss_weights"]
    assert not child.training["transition_balance_policy"]
