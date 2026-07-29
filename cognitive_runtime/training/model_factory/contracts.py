"""Model Factory contract dataclasses and canonical SHA-256 hashing.

Phase A, steps 1-2 of the Model Factory epic (#212, issue #213). These four
frozen dataclasses are the epic's three-plus-one identities: architecture
compatibility (5.1), data evidence (5.2), training procedure (5.3), and
execution environment (5.4). A run's ``architecture_hash``,
``data_contract_hash``, ``training_contract_hash``, and ``environment_hash``
are all derived from the payloads defined here.

Pure Python -- no torch import -- so this module is constructible and
hashable in the core-only install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from functools import cached_property
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


def _require_str_keys(mapping: Mapping[Any, Any]) -> None:
    """Reject non-``str`` mapping keys instead of coercing them.

    ``str(key)`` coercion is lossy: ``{1: "a", "1": "b"}`` and ``{"1": "b"}``
    would otherwise canonicalize identically, letting two distinct contracts
    collide on the same SHA-256 identity.
    """
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(
                "mapping keys must be str for a stable contract identity; "
                f"got {key!r} ({type(key).__name__})"
            )


def _canonicalize(value: Any) -> Any:
    """Recursively normalize ``value`` into JSON-safe, order-stable data.

    Tuples and sets become lists so a tuple-vs-list field choice never
    changes the hash; NaN/Inf floats are rejected rather than silently
    emitted as the non-standard, non-round-trippable ``NaN``/``Infinity``
    tokens ``json.dumps`` would otherwise produce.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float is not canonicalizable: {value!r}")
        return value
    if isinstance(value, Mapping):
        _require_str_keys(value)
        return {key: _canonicalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # Set iteration order depends on PYTHONHASHSEED; sort the
        # canonicalized elements' own JSON encoding for a stable order.
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonicalize value of type {type(value).__name__}: {value!r}")


def _deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers into immutable ones.

    Contract fields often arrive as caller-owned dicts/lists (an optimizer
    block, a policy dict, ``backbone_kwargs``). Without this, editing that
    dict after construction would change ``to_dict()`` while a previously
    cached ``.hash`` silently kept identifying the old contract -- exactly
    the kind of mismatch this module exists to prevent. Applied once in
    ``__post_init__``, so a contract's fields are immutable for its entire
    lifetime, not just at hashing time.
    """
    if isinstance(value, Mapping):
        _require_str_keys(value)
        return MappingProxyType({key: _deep_freeze(val) for key, val in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Serialize ``payload`` to a deterministic JSON byte string.

    Keys are sorted recursively and there is no incidental whitespace, so
    two semantically identical payloads -- built in different field or key
    orders, or in different processes under different ``PYTHONHASHSEED``
    values -- always produce identical bytes. Floats use Python's
    shortest-round-trip ``repr`` encoding (``json``'s default), which is
    itself independent of hash randomization.
    """
    canonical = _canonicalize(dict(payload))
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def contract_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of ``payload``'s canonical JSON encoding."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class _ContractMixin:
    """Shared ``to_dict``/``hash``/freezing behavior for the contracts.

    Not itself a dataclass -- each subclass supplies its own frozen
    ``@dataclass`` fields. ``cached_property`` writes directly into the
    instance ``__dict__`` rather than through ``__setattr__``, so caching
    works even though the dataclasses are frozen. dataclass-generated
    ``__init__`` calls ``__post_init__`` (found via inheritance) after
    setting every field, so this runs on every construction.
    """

    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, f.name, _deep_freeze(getattr(self, f.name)))

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

    @cached_property
    def hash(self) -> str:
        return contract_hash(self.to_dict())


@dataclass(frozen=True)
class ArchitectureContract(_ContractMixin):
    """Checkpoint compatibility identity (epic #212 §5.1).

    Two checkpoints may resume or clone into one another only when every
    field here matches; any difference starts a fresh architecture branch.
    """

    pixel_shape: Tuple[int, int, int]
    rgb_preprocessing_version: str
    action_vocabulary: Tuple[str, ...]
    workspace_modalities: Mapping[str, int]
    workspace_layout_hash: Optional[str]
    latent_width: int
    hidden_dim: int
    action_embed_dim: int
    reconstruction_shape: Tuple[int, int, int]
    visual_architecture: str
    semantic_classes: int
    horizons_ticks: Tuple[int, ...]
    direct_horizon_topology: Tuple[int, ...]
    backbone: str
    context_length: int
    backbone_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataContract(_ContractMixin):
    """Meaning of the training/evaluation evidence (epic #212 §5.2)."""

    world: str
    backend: str
    program_config: Mapping[str, Any]
    scenario_names: Tuple[str, ...]
    scenario_code_version: str
    train_session_ids: Tuple[str, ...]
    train_session_hashes: Tuple[str, ...]
    validation_session_ids: Tuple[str, ...]
    validation_session_hashes: Tuple[str, ...]
    test_session_ids: Tuple[str, ...]
    test_session_hashes: Tuple[str, ...]
    seed_assignments: Mapping[str, int]
    pixel_provenance: str
    semantic_vocabulary_version: str
    preprocessing_version: str
    horizons_ticks: Tuple[int, ...]
    ticks_per_frame: float
    quality_policy: Mapping[str, Any] = field(default_factory=dict)
    split_overlap_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingContract(_ContractMixin):
    """Optimization procedure identity (epic #212 §5.3)."""

    objective: str
    optimizer: Mapping[str, Any]
    batch_size: int
    seed: int
    rollout_frames: int
    warmup_frames: int
    scheduled_sampling_p: float
    loss_weights: Mapping[str, float]
    checkpoint_selection_policy: Mapping[str, Any]
    device: str
    precision: str
    determinism_policy: Mapping[str, Any]
    epoch_budget: Optional[int] = None
    step_budget: Optional[int] = None
    scheduler: Optional[Mapping[str, Any]] = None
    max_training_seconds: Optional[float] = None
    checkpoint_cadence_epochs: Optional[int] = None
    ema_target_decay: Optional[float] = None
    transition_balance_policy: Mapping[str, Any] = field(default_factory=dict)
    early_stopping_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionProvenance(_ContractMixin):
    """Execution environment identity (epic #212 §5.4)."""

    source_commit: str
    dirty: bool
    environment_hash: str
    python_version: str
    device: str
    precision: str
    determinism_policy: Mapping[str, Any]
    pytorch_version: Optional[str] = None
    cuda_version: Optional[str] = None
