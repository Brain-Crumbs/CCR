"""Versioned task boundaries; training/evaluation data never enter predict."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from brain.cortex.model_contracts import InferenceInput
from .contracts import _ContractMixin


@dataclass(frozen=True)
class TaskIdentity(_ContractMixin):
    domain: str
    task: str
    backend: str
    version: str = "1"

    def __post_init__(self):
        super().__post_init__()
        if any(not isinstance(x, str) or not x for x in (self.domain, self.task, self.backend, self.version)):
            raise ValueError("task identity fields must be nonempty strings")


@dataclass(frozen=True)
class TaskDataContract(_ContractMixin):
    task_identity: Mapping[str, Any]
    corpus_id: str
    feature_schema: str
    split_artifacts: Mapping[str, Any]
    format: str = "model-factory-task-data-v1"


@dataclass(frozen=True)
class ModelDefinition(_ContractMixin):
    task_identity: Mapping[str, Any]
    model: Mapping[str, Any]
    feature_schema: str
    format: str = "model-factory-model-definition-v1"


@dataclass(frozen=True)
class TaskTrainingContract(_ContractMixin):
    configuration: Mapping[str, Any]
    format: str = "model-factory-task-training-v1"


@dataclass(frozen=True)
class TrainingBatch:
    inputs: InferenceInput
    targets: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        from brain.cortex.model_contracts import require_inference_input
        require_inference_input(self.inputs)
        values = tuple(tuple(row) for row in self.targets)
        if len(values) != len(self.inputs.sample_ids):
            raise ValueError("training targets must align with inputs")
        object.__setattr__(self, "targets", values)


@dataclass(frozen=True)
class EvaluationBatch:
    inputs: InferenceInput
    labels: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        from brain.cortex.model_contracts import require_inference_input
        require_inference_input(self.inputs)
        values = tuple(tuple(row) for row in self.labels)
        if len(values) != len(self.inputs.sample_ids):
            raise ValueError("evaluation labels must align with inputs")
        object.__setattr__(self, "labels", values)


@dataclass(frozen=True)
class PreparedTask:
    data_contract: TaskDataContract
    model_definition: ModelDefinition
    training: TrainingBatch
    validation: EvaluationBatch


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    path: str
    sha256: str


class TaskCancelled(Exception):
    """Cooperative cancellation; never a completed/promotable run."""


@dataclass(frozen=True)
class TaskControl:
    is_cancelled: Callable[[], bool]
    save_checkpoint: Callable[[], ArtifactReference]

    def check(self) -> None:
        if self.is_cancelled():
            raise TaskCancelled("task cancellation requested")

    def checkpoint(self) -> ArtifactReference:
        """Persist a resumable boundary chosen by the backend during fit."""
        self.check()
        return self.save_checkpoint()


class TaskBackend(Protocol):
    identity: TaskIdentity
    capabilities: frozenset[str]

    def validate(self, spec: Any) -> None: ...
    def prepare(self, spec: Any, *, corpus_root: Path | None) -> PreparedTask: ...
    def build(self, definition: ModelDefinition) -> Any: ...
    def fit(self, model: Any, data: TrainingBatch, configuration: Mapping[str, Any],
            control: TaskControl, *, resume: bool) -> Mapping[str, Any]: ...
    def predict(self, model: Any, inputs: InferenceInput) -> Any: ...
    def evaluate(self, predictions: Any, data: EvaluationBatch) -> Mapping[str, Any]: ...
    def save(self, model: Any) -> Mapping[str, Any]: ...
    def load(self, definition: ModelDefinition, state: Mapping[str, Any], *, resume: bool) -> Any: ...


class FeatureBuilder(Protocol):
    """Fit preprocessing on training only; transform inference features separately."""
    def fit(self, training: TrainingBatch) -> None: ...
    def transform(self, inputs: InferenceInput) -> InferenceInput: ...
