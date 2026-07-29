"""Model Factory: reproducible checkpoint lineage and budgeted experiments.

Phase A (issue #213) adds the contract dataclasses and canonical hashing
primitive that every later phase's ``run_id``, ``architecture_hash``,
``data_contract_hash``, ``training_contract_hash``, and
``parent_checkpoint_sha`` identities are derived from. Phase A, step 2
(issue #214) adds ``ExperimentSpec``: the immutable, fully-resolved
description of one trial. See epic #212.
"""

from __future__ import annotations

from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    ExecutionProvenance,
    TrainingContract,
    canonical_json,
    contract_hash,
)
from cognitive_runtime.training.model_factory.spec import (
    DEFAULT_SPEC,
    DOCUMENT_FORMAT,
    VALID_MODES,
    ExperimentSpec,
    SpecError,
    apply_overrides,
    dump_canonical_json,
    load_spec,
    resolve,
    validate,
)

__all__ = [
    "ArchitectureContract",
    "DataContract",
    "ExecutionProvenance",
    "TrainingContract",
    "canonical_json",
    "contract_hash",
    "DEFAULT_SPEC",
    "DOCUMENT_FORMAT",
    "VALID_MODES",
    "ExperimentSpec",
    "SpecError",
    "apply_overrides",
    "dump_canonical_json",
    "load_spec",
    "resolve",
    "validate",
]
