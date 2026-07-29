"""Model Factory: reproducible checkpoint lineage and budgeted experiments.

Phase A (issue #213) adds the contract dataclasses and canonical hashing
primitive that every later phase's ``run_id``, ``architecture_hash``,
``data_contract_hash``, ``training_contract_hash``, and
``parent_checkpoint_sha`` identities are derived from. See epic #212.
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

__all__ = [
    "ArchitectureContract",
    "DataContract",
    "ExecutionProvenance",
    "TrainingContract",
    "canonical_json",
    "contract_hash",
]
