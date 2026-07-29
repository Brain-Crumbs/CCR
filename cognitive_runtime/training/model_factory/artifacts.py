"""Immutable, crash-safe Model Factory run artifacts.

This module owns the initial on-disk state for a factory trial.  It is kept
separate from the runner so the full provenance record is durable before a
runner imports a model or reserves a GPU (epic #212, Phase A).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Union

from cognitive_runtime.observability.trace import (
    _device_info,
    _environment,
    _git_info,
    _package_versions,
    new_run_id,
)
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.spec import DOCUMENT_FORMAT, ExperimentSpec

if TYPE_CHECKING:
    from cognitive_runtime.training.prediction_export import ExperimentIdentity


EXPERIMENT_IDENTITY_FORMAT = "experiment-identity-v1"
CONTRACTS_FORMAT = "model-factory-contracts-v1"
LINEAGE_FORMAT = "model-factory-lineage-v1"
EXECUTION_FORMAT = "model-factory-execution-v1"
DATA_MANIFEST_FORMAT = "model-factory-data-manifest-v1"


def _jsonable(value: Any) -> Any:
    """Convert immutable contract/spec containers to ordinary JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(
    path: Union[str, Path],
    payload: Mapping[str, Any],
    *,
    after_temp_write: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Write JSON atomically, leaving the old manifest intact on failure.

    ``after_temp_write`` is intentionally injectable for fault-injection
    tests.  It runs after the complete temporary file has been fsynced and
    before its atomic replacement of ``path``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path = Path(temp_name)
        if after_temp_write is not None:
            after_temp_write(temp_path)
        os.replace(temp_path, target)
        return target
    except BaseException:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return value


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a new immutable manifest, refusing to overwrite any file."""
    if path.exists():
        raise FileExistsError(f"immutable manifest already exists: {path}")
    atomic_write_json(path, payload)


def _verify_or_write_resume(path: Path, payload: Mapping[str, Any]) -> None:
    """Resume may complete a failed setup, but may never alter an artifact."""
    expected = _jsonable(payload)
    if path.exists():
        if _load_json(path) != expected:
            raise ValueError(f"resume manifest does not match requested immutable state: {path}")
        return
    atomic_write_json(path, expected)


def execution_manifest(
    *,
    device: str,
    precision: str,
    determinism_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Capture source and runtime facts needed to attribute a crashed trial."""
    # ``_device_info`` intentionally avoids importing torch.  Importing it
    # here is still before model/GPU work, and ensures the manifest contains
    # the requested PyTorch/CUDA facts when the neural extra is installed.
    try:
        __import__("torch")
    except ImportError:
        pass
    git = _git_info()
    device_info = _device_info()
    torch_module = sys.modules.get("torch")
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return {
        "format": EXECUTION_FORMAT,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git.get("commit"),
        "dirty_tree": bool(git.get("dirty", False)),
        "source": git,
        "package_versions": _package_versions(),
        "python_version": platform.python_version(),
        "torch_version": device_info.get("torch_version"),
        "cuda_version": cuda_version,
        "device": device,
        "device_info": device_info,
        "precision": precision,
        "determinism_policy": _jsonable(determinism_policy),
        "environment": _environment(),
    }


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Paths and identities allocated for one immutable factory run."""

    run_id: str
    experiment: ExperimentIdentity
    directory: Path
    experiment_path: Path
    trial_spec_path: Path
    contracts_path: Path
    lineage_path: Path
    execution_path: Path
    data_manifest_path: Path
    checkpoints_dir: Path
    metrics_dir: Path


def allocate_run_artifacts(
    root: Union[str, Path],
    spec: ExperimentSpec,
    architecture_contract: ArchitectureContract,
    data_contract: DataContract,
    training_contract: TrainingContract,
    *,
    run_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    sibling_group: Optional[str] = None,
    data_manifest: Optional[Mapping[str, Any]] = None,
    resume: Optional[bool] = None,
) -> RunArtifacts:
    """Allocate a run directory and durably write its initial manifests.

    A caller may omit ``run_id`` to use the standard unique ID generator.
    Passing an existing ID is rejected for every mode except ``resume``.
    ``resume=True`` is useful when the persisted trial was originally a
    ``fresh``/``clone``/``fine_tune`` trial: it preserves that immutable mode
    while allowing its interrupted execution to continue.  Otherwise the
    value is inferred from ``spec.mode == 'resume'``.
    """
    if spec.training_contract_hash != training_contract.hash:
        raise ValueError("ExperimentSpec training contract does not match training_contract")

    # prediction_export imports torch for its model-export functions, so keep
    # this import at the allocation boundary.  Importing contracts/specs (or
    # this module for its atomic helper) remains safe in a core-only install.
    from cognitive_runtime.training.prediction_export import ExperimentIdentity, experiment_directory

    assigned_run_id = run_id or new_run_id(spec.organism)
    experiment = ExperimentIdentity.create(assigned_run_id, spec.organism, trace_id=trace_id)
    resuming = spec.mode == "resume" if resume is None else resume
    directory = Path(experiment_directory(str(root), experiment, resume=resuming))
    checkpoints_dir = directory / "checkpoints"
    metrics_dir = directory / "metrics"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    spec_payload = {"format": DOCUMENT_FORMAT, **_jsonable(spec.to_dict())}
    contracts_payload = {
        "format": CONTRACTS_FORMAT,
        "architecture_hash": architecture_contract.hash,
        "data_contract_hash": data_contract.hash,
        "training_contract_hash": training_contract.hash,
        "architecture_contract": _jsonable(architecture_contract.to_dict()),
        "data_contract": _jsonable(data_contract.to_dict()),
        "training_contract": _jsonable(training_contract.to_dict()),
    }
    lineage_payload = {
        "format": LINEAGE_FORMAT,
        "mode": spec.mode,
        "parent": _jsonable(spec.parent) if spec.parent else None,
        "parent_checkpoint_sha": (spec.parent or {}).get("sha256"),
        "sibling_group": sibling_group,
        "configuration_parents": _jsonable((spec.evolution or {}).get("configuration_parents", [])),
        "weight_donor": (spec.evolution or {}).get("weight_donor"),
    }
    data_payload = {
        "format": DATA_MANIFEST_FORMAT,
        "corpus_id": spec.data.get("corpus_id"),
        "resolved_data": _jsonable(data_manifest if data_manifest is not None else spec.data),
    }
    execution_payload = execution_manifest(
        device=str(spec.training["device"]),
        precision=str(spec.training["precision"]),
        determinism_policy=spec.training["determinism_policy"],
    )

    experiment_path = directory / "experiment.json"
    if resuming:
        # ``experiment_directory`` already checked these authoritative fields.
        # Preserve the original timestamp and source identity rather than
        # replacing them with facts from the resume process.
        saved_experiment = _load_json(experiment_path)
        if saved_experiment.get("format") not in (None, EXPERIMENT_IDENTITY_FORMAT):
            raise ValueError(f"not an experiment identity manifest: {experiment_path}")
        if saved_experiment.get("format") is None:
            # A process can die after experiment_directory() has installed
            # its identity guard but before the factory adds its schema tag.
            # This is an upgrade of that bootstrap record, not a change to
            # its identity or timestamp.
            atomic_write_json(experiment_path, {"format": EXPERIMENT_IDENTITY_FORMAT, **saved_experiment})
    else:
        # ``experiment_directory`` atomically installed this versioned
        # identity manifest as part of reserving the fresh run ID.
        if _load_json(experiment_path).get("format") != EXPERIMENT_IDENTITY_FORMAT:
            raise ValueError(f"new experiment identity manifest has an unexpected format: {experiment_path}")

    immutable_paths_and_payloads = (
        (directory / "trial_spec.json", spec_payload),
        (directory / "contracts.json", contracts_payload),
        (directory / "lineage.json", lineage_payload),
        (directory / "data_manifest.json", data_payload),
    )
    for path, payload in immutable_paths_and_payloads:
        if resuming:
            _verify_or_write_resume(path, payload)
        else:
            _write_once(path, payload)

    # The execution record attributes the initial launch.  Resuming a run
    # must not erase that provenance merely because its wall-clock time or
    # current dirty-tree state differs.
    execution_path = directory / "execution.json"
    if resuming and execution_path.exists():
        saved_execution = _load_json(execution_path)
        if saved_execution.get("format") != EXECUTION_FORMAT:
            raise ValueError(f"not an execution manifest: {execution_path}")
    elif resuming:
        atomic_write_json(execution_path, execution_payload)
    else:
        _write_once(execution_path, execution_payload)

    return RunArtifacts(
        run_id=assigned_run_id,
        experiment=experiment,
        directory=directory,
        experiment_path=experiment_path,
        trial_spec_path=directory / "trial_spec.json",
        contracts_path=directory / "contracts.json",
        lineage_path=directory / "lineage.json",
        execution_path=execution_path,
        data_manifest_path=directory / "data_manifest.json",
        checkpoints_dir=checkpoints_dir,
        metrics_dir=metrics_dir,
    )


__all__ = [
    "EXPERIMENT_IDENTITY_FORMAT",
    "CONTRACTS_FORMAT",
    "LINEAGE_FORMAT",
    "EXECUTION_FORMAT",
    "DATA_MANIFEST_FORMAT",
    "RunArtifacts",
    "atomic_write_json",
    "execution_manifest",
    "allocate_run_artifacts",
]
