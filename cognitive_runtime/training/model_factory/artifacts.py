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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Union

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
from cognitive_runtime.training.model_factory.naming import (
    DisplayNameAssignment,
    DisplayNameParent,
    assign_display_name,
    load_display_name,
    new_naming_seed,
)

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


@contextmanager
def _display_name_lock(organism_directory: Path) -> Iterator[None]:
    """Serialize sibling-name reservation across factory processes.

    A run directory is reserved atomically by ``experiment_directory``, but
    the sibling scan that detects a cosmetic-name collision spans directories.
    This small persistent lock serializes that scan and the corresponding
    ``display_name.json`` write.  It is deliberately separate from run
    identity and is never consulted for lineage or registry lookup.
    """

    lock_path = organism_directory / ".display_name.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lineage_parent_run_ids(spec: ExperimentSpec) -> tuple[str, ...]:
    """Return the authoritative naming-parent IDs encoded by the trial spec."""

    configuration_parents = tuple((spec.evolution or {}).get("configuration_parents", ()))
    if configuration_parents:
        return tuple(str(run_id) for run_id in configuration_parents)
    if spec.parent:
        return (str(spec.parent["run_id"]),)
    return ()


def _resolve_naming_parents(
    organism_directory: Path,
    spec: ExperimentSpec,
    supplied_parents: Sequence[DisplayNameParent],
) -> tuple[DisplayNameParent, ...]:
    """Load cosmetic parent records using the lineage's real run IDs."""

    parent_run_ids = _lineage_parent_run_ids(spec)
    if supplied_parents:
        supplied_ids = tuple(parent.run_id for parent in supplied_parents)
        if supplied_ids != parent_run_ids:
            raise ValueError(
                "naming_parents do not match the parent run IDs declared by ExperimentSpec"
            )
        return tuple(supplied_parents)

    resolved = []
    for run_id in parent_run_ids:
        display_name_path = organism_directory / run_id / "display_name.json"
        try:
            assignment = load_display_name(display_name_path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"lineage parent {run_id!r} has no persisted display-name manifest: "
                f"{display_name_path}"
            ) from exc
        resolved.append(DisplayNameParent(run_id=run_id, display_name=assignment.display_name))
    return tuple(resolved)


def _session_artifacts(session_ids: Any, session_hashes: Any, *, split: str) -> list[Dict[str, str]]:
    """Pair every frozen session ID with its immutable content hash."""
    if len(session_ids) != len(session_hashes):
        raise ValueError(
            f"data contract {split} session IDs and hashes have different lengths "
            f"({len(session_ids)} != {len(session_hashes)})"
        )
    return [
        {"session_id": str(session_id), "sha256": str(session_hash)}
        for session_id, session_hash in zip(session_ids, session_hashes)
    ]


def _contract_session_manifest(data_contract: DataContract) -> Dict[str, Any]:
    """Exact evidence identities required even before MF-A5 adds metadata."""
    return {
        "train": _session_artifacts(
            data_contract.train_session_ids, data_contract.train_session_hashes, split="train"
        ),
        "validation": _session_artifacts(
            data_contract.validation_session_ids, data_contract.validation_session_hashes,
            split="validation",
        ),
        "test": _session_artifacts(
            data_contract.test_session_ids, data_contract.test_session_hashes, split="test"
        ),
    }


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
    display_name_path: Path
    display_name: str
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
    naming_seed: Optional[Union[str, int]] = None,
    naming_parents: Sequence[DisplayNameParent] = (),
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
        "data_contract_hash": data_contract.hash,
        "sessions": _contract_session_manifest(data_contract),
        "resolved_data": _jsonable(data_manifest if data_manifest is not None else spec.data),
    }
    execution_payload = execution_manifest(
        device=str(spec.training["device"]),
        precision=str(spec.training["precision"]),
        determinism_policy=spec.training["determinism_policy"],
    )

    experiment_path = directory / "experiment.json"
    display_name_path = directory / "display_name.json"
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
        try:
            experiment = ExperimentIdentity(
                experiment_id=saved_experiment["experiment_id"],
                organism=saved_experiment["organism"],
                trace_id=saved_experiment.get("trace_id"),
                created_at=saved_experiment["created_at"],
                git_commit=saved_experiment.get("git_commit"),
                format=saved_experiment.get("format", EXPERIMENT_IDENTITY_FORMAT),
            )
        except KeyError as exc:
            raise ValueError(f"invalid experiment identity manifest: {experiment_path}") from exc
    else:
        # ``experiment_directory`` atomically installed this versioned
        # identity manifest as part of reserving the fresh run ID.
        if _load_json(experiment_path).get("format") != EXPERIMENT_IDENTITY_FORMAT:
            raise ValueError(f"new experiment identity manifest has an unexpected format: {experiment_path}")

    if resuming and display_name_path.exists():
        display_name_assignment = load_display_name(display_name_path)
    else:
        resolved_naming_parents = _resolve_naming_parents(
            directory.parent, spec, naming_parents,
        )
        # Each run directory is an identity boundary.  The lock only guards
        # collision reservation; display names still never locate, merge, or
        # otherwise identify a run.
        with _display_name_lock(directory.parent):
            existing_names = set()
            for sibling in directory.parent.iterdir():
                if sibling == directory or not sibling.is_dir():
                    continue
                sibling_display_name = sibling / "display_name.json"
                if sibling_display_name.exists():
                    existing_names.add(load_display_name(sibling_display_name).display_name)
            display_name_assignment = assign_display_name(
                run_id=assigned_run_id,
                naming_seed=new_naming_seed() if naming_seed is None else naming_seed,
                parents=resolved_naming_parents,
                existing_names=existing_names,
            )
            if resuming:
                _verify_or_write_resume(display_name_path, display_name_assignment.to_dict())
            else:
                _write_once(display_name_path, display_name_assignment.to_dict())

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
        display_name_path=display_name_path,
        display_name=display_name_assignment.display_name,
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
