"""Model Factory contract + canonical hashing tests (issue #213).

Deliberately torch-free: these run in the ``factory-contracts`` CI tier
under two different ``PYTHONHASHSEED`` values (see .github/workflows/ci.yml),
because non-deterministic ordering leaking into the canonical JSON is the
most likely way "reproducible identity" would silently stop being
reproducible.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    ExecutionProvenance,
    TrainingContract,
    canonical_json,
    contract_hash,
)


def _make_architecture(**overrides) -> ArchitectureContract:
    fields = dict(
        pixel_shape=(3, 64, 64),
        rgb_preprocessing_version="v1",
        action_vocabulary=("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "NULL"),
        workspace_modalities={"proprioception": 4, "vision": 128},
        workspace_layout_hash="abc123",
        latent_width=32,
        hidden_dim=64,
        action_embed_dim=8,
        reconstruction_shape=(3, 16, 16),
        visual_architecture="spatial_residual_v1",
        semantic_classes=12,
        horizons_ticks=(1, 4, 8),
        direct_horizon_topology=(4, 8),
        backbone="transformer",
        context_length=8,
        backbone_kwargs={"n_heads": 2, "n_layers": 1},
    )
    fields.update(overrides)
    return ArchitectureContract(**fields)


def _make_data(**overrides) -> DataContract:
    fields = dict(
        world="crafter",
        backend="crafter",
        program_config={"scenario": "approach_entity"},
        scenario_names=("motor_babbling_open", "motor_babbling_walls"),
        scenario_code_version="v1",
        train_session_ids=("session-1", "session-2"),
        train_session_hashes=("hash-1", "hash-2"),
        validation_session_ids=("session-3",),
        validation_session_hashes=("hash-3",),
        test_session_ids=("session-4",),
        test_session_hashes=("hash-4",),
        seed_assignments={"train": 0, "validation": 1, "test": 2},
        pixel_provenance="crafter-renderer-v1",
        semantic_vocabulary_version="v1",
        preprocessing_version="v1",
        horizons_ticks=(1, 4, 8),
        ticks_per_frame=1.0,
        quality_policy={"min_episode_length": 32},
        split_overlap_policy={"max_overlap": 0.0},
    )
    fields.update(overrides)
    return DataContract(**fields)


def _make_training(**overrides) -> TrainingContract:
    fields = dict(
        objective="windowed_rollout",
        optimizer={"name": "adamw", "lr": 0.0003, "weight_decay": 1e-5},
        batch_size=32,
        seed=0,
        rollout_frames=8,
        warmup_frames=3,
        scheduled_sampling_p=0.25,
        loss_weights={"pixel": 1.0, "latent": 1.0, "semantic": 1.0},
        checkpoint_selection_policy={"metric": "rollout.t+4.model_over_copy_last_mse"},
        device="cpu",
        precision="fp32",
        determinism_policy={"deterministic": True},
    )
    fields.update(overrides)
    return TrainingContract(**fields)


def _make_execution(**overrides) -> ExecutionProvenance:
    fields = dict(
        source_commit="deadbeef",
        dirty=False,
        environment_hash="env-hash",
        python_version="3.12.0",
        device="cpu",
        precision="fp32",
        determinism_policy={"deterministic": True},
        pytorch_version=None,
        cuda_version=None,
    )
    fields.update(overrides)
    return ExecutionProvenance(**fields)


# --------------------------------------------------------------------- canonical_json


def test_canonical_json_sorts_keys_and_strips_whitespace():
    payload = {"b": 1, "a": {"z": 2, "y": 3}}
    encoded = canonical_json(payload)
    assert encoded == b'{"a":{"y":3,"z":2},"b":1}'


def test_canonical_json_normalizes_tuples_and_lists_identically():
    assert canonical_json({"x": (1, 2, 3)}) == canonical_json({"x": [1, 2, 3]})


def test_canonical_json_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_contract_hash_is_sha256_hex_of_canonical_json():
    payload = {"a": 1}
    assert contract_hash(payload) == __import__("hashlib").sha256(
        canonical_json(payload)
    ).hexdigest()


# --------------------------------------------------------------------- dataclass basics


def test_to_dict_and_hash_are_exposed_on_every_contract():
    for contract in (
        _make_architecture(),
        _make_data(),
        _make_training(),
        _make_execution(),
    ):
        assert isinstance(contract.to_dict(), dict)
        assert isinstance(contract.hash, str)
        assert len(contract.hash) == 64  # sha256 hex digest


def test_unknown_field_is_a_construction_error():
    with pytest.raises(TypeError):
        _make_architecture(unexpected_field=123)


def test_identical_contracts_built_in_different_field_orders_hash_identically():
    ordered = ArchitectureContract(
        pixel_shape=(3, 64, 64),
        rgb_preprocessing_version="v1",
        action_vocabulary=("a", "b"),
        workspace_modalities={"vision": 1, "proprioception": 2},
        workspace_layout_hash="h",
        latent_width=32,
        hidden_dim=64,
        action_embed_dim=8,
        reconstruction_shape=(3, 16, 16),
        visual_architecture="spatial_residual_v1",
        semantic_classes=1,
        horizons_ticks=(1, 4),
        direct_horizon_topology=(4,),
        backbone="gru",
        context_length=8,
        backbone_kwargs={"n_heads": 2, "n_layers": 1},
    )
    reordered = ArchitectureContract(
        backbone_kwargs={"n_layers": 1, "n_heads": 2},
        context_length=8,
        backbone="gru",
        direct_horizon_topology=(4,),
        horizons_ticks=(1, 4),
        semantic_classes=1,
        visual_architecture="spatial_residual_v1",
        reconstruction_shape=(3, 16, 16),
        action_embed_dim=8,
        hidden_dim=64,
        latent_width=32,
        workspace_layout_hash="h",
        workspace_modalities={"proprioception": 2, "vision": 1},
        action_vocabulary=("a", "b"),
        rgb_preprocessing_version="v1",
        pixel_shape=(3, 64, 64),
    )
    assert ordered.hash == reordered.hash
    assert ordered.to_dict() == reordered.to_dict()


# --------------------------------------------------------------------- acceptance criteria


def test_changing_a_5_1_field_changes_architecture_hash():
    baseline = _make_architecture()
    changed = _make_architecture(latent_width=baseline.latent_width + 1)
    assert baseline.hash != changed.hash


def test_changing_only_a_loss_weight_changes_training_hash_but_not_architecture_hash():
    architecture = _make_architecture()
    baseline_training = _make_training()
    changed_weights = dict(baseline_training.loss_weights)
    changed_weights["pixel"] += 0.5
    changed_training = _make_training(loss_weights=changed_weights)

    assert baseline_training.hash != changed_training.hash
    # The architecture contract never even sees the training contract's
    # fields, so it trivially can't change -- assert this explicitly so a
    # future refactor that merges the two payloads gets caught here.
    assert architecture.hash == _make_architecture().hash


def test_changing_a_validation_session_hash_changes_data_contract_hash():
    baseline = _make_data()
    changed = _make_data(validation_session_hashes=("different-hash",))
    assert baseline.hash != changed.hash


_HASH_STABILITY_SCRIPT = """
import sys
sys.path.insert(0, {path!r})
from cognitive_runtime.training.model_factory.contracts import ArchitectureContract

contract = ArchitectureContract(
    pixel_shape=(3, 64, 64),
    rgb_preprocessing_version="v1",
    action_vocabulary=("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "NULL"),
    workspace_modalities={{"proprioception": 4, "vision": 128}},
    workspace_layout_hash="abc123",
    latent_width=32,
    hidden_dim=64,
    action_embed_dim=8,
    reconstruction_shape=(3, 16, 16),
    visual_architecture="spatial_residual_v1",
    semantic_classes=12,
    horizons_ticks=(1, 4, 8),
    direct_horizon_topology=(4, 8),
    backbone="transformer",
    context_length=8,
    backbone_kwargs={{"n_heads": 2, "n_layers": 1}},
)
print(contract.hash)
"""


def _hash_under_pythonhashseed(seed: str) -> str:
    import os

    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [sys.executable, "-c", _HASH_STABILITY_SCRIPT.format(path=repo_root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_hashes_are_identical_across_different_pythonhashseed_subprocesses():
    hash_seed_0 = _hash_under_pythonhashseed("0")
    hash_seed_other = _hash_under_pythonhashseed("12345")
    assert hash_seed_0 == hash_seed_other
    assert len(hash_seed_0) == 64


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.contracts  # noqa: F401
print("OK")
"""


def test_contracts_module_imports_cleanly_without_torch():
    import os

    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=repo_root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_contract_dataclasses_are_frozen():
    architecture = _make_architecture()
    with pytest.raises(Exception):
        architecture.latent_width = 999  # type: ignore[misc]


