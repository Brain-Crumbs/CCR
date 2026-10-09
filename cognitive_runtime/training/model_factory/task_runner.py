"""Neutral lifecycle at Factory's existing run/artifact/state boundaries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from brain.cortex.model_contracts import require_inference_input
from .artifacts import allocate_run_artifacts, atomic_write_json, _jsonable
from .checkpoint import load_task_checkpoint, read_factory_checkpoint_metadata, save_task_checkpoint
from .contracts import contract_hash
from .task_contracts import TaskCancelled, TaskControl, TaskTrainingContract
from .state import (cancellation_requested, claim_stale_worker, create_state, heartbeat_path,
                    state_path, transition, write_heartbeat)


class TaskBudgetExceeded(Exception):
    pass


def run_backend_trial(spec, backend, *, root, corpus_root, run_id, naming_seed,
                      heartbeat_timeout_seconds):
    from .runner import TrialResult, _load_existing_run_artifacts, _parent_checkpoint_path
    required = {"prepare", "fit", "predict", "evaluate", "save", "load"}
    if not required <= backend.capabilities:
        raise ValueError(f"backend lacks lifecycle capabilities: {sorted(required - backend.capabilities)}")
    backend.validate(spec)
    # Identity rejection must precede data/model construction and optional imports.
    parent_path = None
    if spec.parent:
        parent_path = _parent_checkpoint_path(root, spec)
        header = read_factory_checkpoint_metadata(str(parent_path))
        if header.get("task_identity") != _jsonable(backend.identity.to_dict()):
            raise ValueError("cross-domain clone/resume is forbidden")
        if header.get("checkpoint_sha256") != spec.parent["sha256"]:
            raise ValueError("parent checkpoint sha256 differs")
    from .corpus import prepare_task_corpus
    prepared = prepare_task_corpus(backend, spec, corpus_root=corpus_root)
    definition, data = prepared.model_definition, prepared.data_contract
    identity = backend.identity.to_dict()
    training = TaskTrainingContract(spec.training)
    if parent_path:
        model = load_task_checkpoint(str(parent_path), mode=spec.mode,
                                    architecture_contract=definition, data_contract_hash=data,
                                    training_contract=training).model
    else:
        model = backend.build(definition)
    if spec.mode == "resume":
        parent_id = str(spec.parent["run_id"])
        if run_id is not None and run_id != parent_id:
            raise ValueError("resume must append to its own parent run")
        artifacts = _load_existing_run_artifacts(root, spec.organism, parent_id)
        if (artifacts.experiment.experiment_id, artifacts.experiment.organism) != (parent_id, spec.organism):
            raise ValueError("resume identity does not match existing run")
        saved = json.loads(artifacts.contracts_path.read_text())
        if (saved["architecture_hash"], saved["data_contract_hash"], saved["training_contract_hash"]) != (definition.hash, data.hash, training.hash):
            raise ValueError("resume run contracts differ from checkpoint/request")
        claim_stale_worker(state_path(artifacts.directory), heartbeat_path(artifacts.directory), run_id=parent_id)
    else:
        artifacts = allocate_run_artifacts(root, spec, definition, data, training,
                                          run_id=run_id, naming_seed=naming_seed)
        create_state(state_path(artifacts.directory), artifacts.run_id,
                     heartbeat_timeout_seconds=heartbeat_timeout_seconds or max(300, 2 * spec.training["max_training_seconds"]))
        transition(state_path(artifacts.directory), "running")
    started = time.monotonic()
    limit = spec.training["max_training_seconds"]
    checkpoint = artifacts.checkpoints_dir / "last.json"
    stats, evaluation, refs = {}, {}, []
    digest = None
    status, failure = "completed", None

    def cancelled():
        write_heartbeat(heartbeat_path(artifacts.directory), run_id=artifacts.run_id)
        if time.monotonic() - started >= limit:
            raise TaskBudgetExceeded("task cooperative wall-clock budget exceeded")
        return cancellation_requested(artifacts.directory)

    control = TaskControl(cancelled)
    try:
        control.check()
        fitted = backend.fit(model, prepared.training, spec.training, control, resume=spec.mode == "resume")
        contract_hash(fitted)
        stats = fitted
        control.check()
        digest = save_task_checkpoint(checkpoint, backend, model, definition, data, training)
        # This is the only input object crossing the inference boundary.
        predictions = backend.predict(model, require_inference_input(prepared.validation.inputs))
        control.check()
        evaluated = backend.evaluate(predictions, prepared.validation)
        contract_hash(evaluated)  # Reject nonfinite/nonserializable outcomes.
        evaluation = evaluated
        control.check()
        atomic_write_json(artifacts.metrics_dir / "validation.json", evaluation)
    except TaskCancelled as exc:
        status, failure = "cancelled", str(exc)
        try:
            digest = save_task_checkpoint(checkpoint, backend, model, definition, data, training)
        except BaseException as save_error:
            status, failure = "failed", f"checkpoint failure: {save_error}"
            raise
    except TaskBudgetExceeded as exc:
        status, failure = "budget_exceeded", str(exc)
        try:
            digest = save_task_checkpoint(checkpoint, backend, model, definition, data, training)
        except BaseException as save_error:
            status, failure = "failed", f"checkpoint failure: {save_error}"
            raise
    except BaseException as exc:
        status, failure = "failed", f"{type(exc).__name__}: {exc}"
        # Persist a failure result before propagating the original exception.
        raise
    finally:
        elapsed = time.monotonic() - started
        for kind, path in (("checkpoint", checkpoint), ("evaluation", artifacts.metrics_dir / "validation.json")):
            if path.is_file():
                refs.append({"kind": kind, "path": str(path.relative_to(artifacts.directory)),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        budget = {"format": "model-factory-task-budget-v1", "elapsed_seconds": elapsed,
                  "max_training_seconds": limit, "enforcement": "cooperative", "status": status}
        atomic_write_json(artifacts.metrics_dir / "budget_report.json", budget)
        report_path = artifacts.directory / "experiment_report.json"
        atomic_write_json(report_path, {"format": "model-factory-task-result-v1", "run_id": artifacts.run_id,
            "task_identity": identity, "status": status, "failure": failure,
            "architecture_hash": definition.hash, "data_contract_hash": data.hash,
            "training_contract_hash": training.hash, "artifacts": refs,
            "training_stats": stats, "evaluation": evaluation, "budget": budget})
        transition(state_path(artifacts.directory), status, reason=failure or status)
    return TrialResult(run_id=artifacts.run_id, display_name=artifacts.display_name, mode=spec.mode,
                       state=status, directory=artifacts.directory, architecture_hash=definition.hash,
                       data_contract_hash=data.hash, training_contract_hash=training.hash,
                       checkpoint_path=str(checkpoint), checkpoint_sha256=digest,
                       training_stats=stats, evaluation=evaluation, budget_report=budget,
                       comparison=None, experiment_report_path=str(report_path))
