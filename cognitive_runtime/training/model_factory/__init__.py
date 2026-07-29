"""Model Factory: reproducible checkpoint lineage and budgeted experiments.

Phase A (issue #213) adds the contract dataclasses and canonical hashing
primitive that every later phase's ``run_id``, ``architecture_hash``,
``data_contract_hash``, ``training_contract_hash``, and
``parent_checkpoint_sha`` identities are derived from. Phase A, step 2
(issue #214) adds ``ExperimentSpec``: the immutable, fully-resolved
description of one trial. See epic #212.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from cognitive_runtime.training.model_factory.artifacts import (
    CONTRACTS_FORMAT,
    DATA_MANIFEST_FORMAT,
    EXECUTION_FORMAT,
    EXPERIMENT_IDENTITY_FORMAT,
    LINEAGE_FORMAT,
    RunArtifacts,
    allocate_run_artifacts,
    atomic_write_json,
    execution_manifest,
)
from cognitive_runtime.training.model_factory.naming import (
    DISPLAY_NAME_FORMAT,
    DisplayNameAssignment,
    DisplayNameParent,
    assign_display_name,
    load_display_name,
    new_naming_seed,
    surname_sequence,
)
from cognitive_runtime.training.model_factory.checkpoint import (
    FORMAT as FACTORY_CHECKPOINT_FORMAT,
    FactoryCheckpoint,
    capture_rng_state,
    load_factory_checkpoint,
    restore_rng_state,
    save_factory_checkpoint,
)
if TYPE_CHECKING:
    from cognitive_runtime.training.model_factory.corpus import (
        CORPUS_MANIFEST_FORMAT,
        CORPUS_SPEC_FORMAT,
        QUALITY_REPORT_FORMAT,
        SPLIT_OVERLAP_REPORT_FORMAT,
        ResolvedCorpus,
        build_corpus,
        resolve_corpus,
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
    "EXPERIMENT_IDENTITY_FORMAT",
    "CONTRACTS_FORMAT",
    "LINEAGE_FORMAT",
    "EXECUTION_FORMAT",
    "DATA_MANIFEST_FORMAT",
    "RunArtifacts",
    "atomic_write_json",
    "execution_manifest",
    "allocate_run_artifacts",
    "DISPLAY_NAME_FORMAT",
    "DisplayNameAssignment",
    "DisplayNameParent",
    "assign_display_name",
    "load_display_name",
    "new_naming_seed",
    "surname_sequence",
    "FACTORY_CHECKPOINT_FORMAT",
    "FactoryCheckpoint",
    "capture_rng_state",
    "restore_rng_state",
    "save_factory_checkpoint",
    "load_factory_checkpoint",
    "CORPUS_SPEC_FORMAT",
    "CORPUS_MANIFEST_FORMAT",
    "QUALITY_REPORT_FORMAT",
    "SPLIT_OVERLAP_REPORT_FORMAT",
    "ResolvedCorpus",
    "build_corpus",
    "resolve_corpus",
]


_LAZY_CORPUS_EXPORTS = {
    "CORPUS_SPEC_FORMAT",
    "CORPUS_MANIFEST_FORMAT",
    "QUALITY_REPORT_FORMAT",
    "SPLIT_OVERLAP_REPORT_FORMAT",
    "ResolvedCorpus",
    "build_corpus",
    "resolve_corpus",
}


def __getattr__(name: str):
    """Keep the existing core-only Model Factory import boundary torch-free."""
    if name in _LAZY_CORPUS_EXPORTS:
        from cognitive_runtime.training.model_factory import corpus

        return getattr(corpus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
