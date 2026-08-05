"""Model Factory experiment specifications (epic #212 §8, issue #214).

Phase A, step 2 of the Model Factory epic. ``ExperimentSpec`` is the
immutable, fully-resolved description of one trial: what organism, what
mode (fresh/clone/resume/fine_tune), what data, model, training, evaluation
and (optionally) evolution configuration it uses.

The on-disk JSON is canonical; YAML is optional input convenience only
(epic §8). A resolved spec must be fully explicit -- every default
materialised -- because "a trial must never be reproducible only from a
notebook cell and its current defaults."

Pure Python -- no torch import -- so this module is usable in the core-only
install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import yaml

from cognitive_runtime.training.model_factory.contracts import (
    TrainingContract,
    _ContractMixin,
    canonical_json,
    contract_hash,
)

DOCUMENT_FORMAT = "model-factory-spec-v1"

VALID_MODES: Tuple[str, ...] = ("fresh", "clone", "resume", "fine_tune")

# Keys that the factory assigns at launch time (MF-A3/MF-A4). A template
# that sets them is trying to forge an identity that isn't its to choose.
LAUNCH_ASSIGNED_KEYS: Tuple[str, ...] = ("run_id", "display_name")

# Top-level keys an input template may declare.
TOP_LEVEL_KEYS: Tuple[str, ...] = (
    "format",
    "organism",
    "mode",
    "parent",
    "data",
    "model",
    "training",
    "evaluation",
    "evolution",
)

PARENT_KEYS: Tuple[str, ...] = ("run_id", "checkpoint", "sha256")

DATA_KEYS: Tuple[str, ...] = (
    "corpus_id",
    "world",
    "train_scenarios",
    "train_sessions",
    "validation_sessions",
    "test_sessions",
    "expected_pixel_source",
    "horizons_ticks",
    "quality_policy",
)

MODEL_KEYS: Tuple[str, ...] = (
    "backbone",
    "latent_width",
    "hidden_dim",
    "action_embed_dim",
    "reconstruction_size",
    "context_length",
    "backbone_kwargs",
)

# Mirrors cognitive_runtime.training.model_factory.contracts.TrainingContract
# field-for-field: resolve() builds a TrainingContract straight out of this
# block, so a spec's training_contract_hash is defined, not derived.
TRAINING_KEYS: Tuple[str, ...] = (
    "objective",
    "optimizer",
    "batch_size",
    "seed",
    "rollout_frames",
    "warmup_frames",
    "scheduled_sampling_p",
    "loss_weights",
    "checkpoint_selection_policy",
    "device",
    "precision",
    "determinism_policy",
    "epoch_budget",
    "step_budget",
    "scheduler",
    "max_training_seconds",
    "checkpoint_cadence_epochs",
    "ema_target_decay",
    "transition_balance_policy",
    "early_stopping_policy",
)

EVALUATION_KEYS: Tuple[str, ...] = ("selection_metric", "gates", "confidence", "reference_run")

#: Same shape as ``PARENT_KEYS`` -- a checkpoint reference with a declared
#: hash -- but ``reference_run`` is loaded read-only for evaluation-time
#: comparison (epic clinic redesign, "beat this model, not just
#: copy-last"), never as a weight donor: it is unrelated to ``mode``/
#: ``parent`` and coexists with ``mode: fresh``.
REFERENCE_RUN_KEYS: Tuple[str, ...] = ("run_id", "checkpoint", "sha256")

EVOLUTION_KEYS: Tuple[str, ...] = (
    "generation",
    "configuration_parents",
    "weight_donor",
    "genome_operator",
    "mutations",
)

# A resolvable metric path: "rollout.t+4.model_over_copy_last_mse",
# "validation_loss", "goal_navigation.success_rate", ...
_METRIC_PATH_RE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*(\.(t\+\d+|[a-zA-Z_][a-zA-Z0-9_]*))*$"
)

DEFAULT_DATA: Mapping[str, Any] = {
    "train_scenarios": [],
    "train_sessions": [],
    "validation_sessions": [],
    "test_sessions": [],
    "expected_pixel_source": "crafter",
    "horizons_ticks": [1, 2, 3, 4],
    "quality_policy": {},
}

DEFAULT_MODEL: Mapping[str, Any] = {
    "backbone": "transformer",
    "latent_width": 128,
    "hidden_dim": 256,
    "action_embed_dim": 8,
    "reconstruction_size": 64,
    "context_length": 8,
    "backbone_kwargs": {},
}

DEFAULT_TRAINING: Mapping[str, Any] = {
    "objective": "windowed_rollout",
    "optimizer": {"name": "adamw", "lr": 0.0003, "weight_decay": 1e-5},
    "batch_size": 32,
    "seed": 0,
    "rollout_frames": 8,
    "warmup_frames": 4,
    "scheduled_sampling_p": 0.0,
    "loss_weights": {},
    "checkpoint_selection_policy": {"metric": "validation_loss", "mode": "min"},
    "device": "cpu",
    "precision": "fp32",
    "determinism_policy": {"deterministic": True, "seed": 0},
    "epoch_budget": None,
    "step_budget": None,
    "scheduler": None,
    "max_training_seconds": None,
    "checkpoint_cadence_epochs": None,
    "ema_target_decay": None,
    "transition_balance_policy": {},
    "early_stopping_policy": {},
}

DEFAULT_EVALUATION: Mapping[str, Any] = {
    "gates": [],
    "confidence": 0.95,
}

# The full default template that resolve() merges caller data onto. Required
# identifiers (data.corpus_id, data.world, evaluation.selection_metric, ...)
# are deliberately absent -- there is no safe default for them, so their
# absence after resolution is a validation error, not a silently-filled gap.
DEFAULT_SPEC: Mapping[str, Any] = {
    "data": DEFAULT_DATA,
    "model": DEFAULT_MODEL,
    "training": DEFAULT_TRAINING,
    "evaluation": DEFAULT_EVALUATION,
}


_MISSING = object()


class SpecError(ValueError):
    """A malformed, ambiguous, or unresolvable ExperimentSpec."""


def _nearest_match(key: str, valid_keys: Sequence[str]) -> Optional[str]:
    matches = difflib.get_close_matches(key, valid_keys, n=1)
    return matches[0] if matches else None


def _check_unknown_keys(location: str, mapping: Mapping[str, Any], valid_keys: Sequence[str]) -> None:
    for key in mapping:
        if key not in valid_keys:
            suggestion = _nearest_match(key, valid_keys)
            hint = f"; did you mean {suggestion!r}?" if suggestion else ""
            raise SpecError(f"unknown key {key!r} in {location}{hint}")


def _thaw(value: Any) -> Any:
    """Recursively convert (possibly frozen) mappings/sequences to plain
    ``dict``/``list``.

    ``copy.deepcopy`` cannot copy ``types.MappingProxyType`` (contracts.py
    freezes every nested mapping into one via ``_deep_freeze``), so a
    resolved ``ExperimentSpec``'s blocks need this instead of ``deepcopy``
    wherever they flow back into ``apply_overrides``/``resolve``.
    """
    if isinstance(value, Mapping):
        return {key: _thaw(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings."""
    merged: Dict[str, Any] = _thaw(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _thaw(value)
    return merged


@dataclass(frozen=True)
class ExperimentSpec(_ContractMixin):
    """A resolved (or in-progress) Model Factory experiment specification.

    ``data``, ``model``, ``training``, ``evaluation``, ``parent`` and
    ``evolution`` stay plain mappings rather than nested dataclasses: their
    leaves (an optimizer block, loss weights, backbone kwargs) are
    intentionally open-ended, and ``training`` maps field-for-field onto
    :class:`TrainingContract` so its hash is defined directly rather than
    re-derived through a parallel schema.
    """

    format: str
    organism: str
    mode: str
    data: Mapping[str, Any]
    model: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    parent: Optional[Mapping[str, Any]] = None
    evolution: Optional[Mapping[str, Any]] = None

    @property
    def training_contract_hash(self) -> str:
        return TrainingContract(**dict(self.training)).hash


def load_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a raw (unresolved) spec document from ``.yaml``/``.yml``/``.json``.

    Returns the plain input mapping -- not yet defaulted, validated, or
    frozen. JSON is canonical on disk; YAML is accepted as input convenience
    only (epic §8).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise SpecError(f"unsupported spec file extension {suffix!r} for {path}")
    if not isinstance(raw, Mapping):
        raise SpecError(f"spec file {path} must contain a mapping at the top level")
    return dict(raw)


def _coerce(raw_value: str, reference: Any) -> Any:
    """Coerce a ``--set`` string value using ``reference``'s declared type.

    Never uses ``eval``. ``bool`` is checked before ``int``/``float``
    because ``bool`` is a subclass of ``int`` in Python. When there is no
    reference value to infer a type from (a brand-new leaf key), fall back
    to the narrowest type the string actually parses as: int, then float,
    then bool, then str.
    """
    if isinstance(reference, bool):
        lowered = raw_value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise SpecError(f"cannot coerce {raw_value!r} to bool")
    if isinstance(reference, int):
        return int(raw_value)
    if isinstance(reference, float):
        return float(raw_value)
    if isinstance(reference, (list, tuple)):
        return [_infer_scalar(item.strip()) for item in raw_value.split(",")]
    if isinstance(reference, str):
        return raw_value
    if reference is None:
        return _infer_scalar(raw_value)
    raise SpecError(
        f"cannot coerce override value for a field of type {type(reference).__name__}"
    )


def _infer_scalar(raw_value: str) -> Any:
    lowered = raw_value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "null" or lowered == "none":
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def apply_overrides(spec: Mapping[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``"dotted.path=value"`` overrides onto a copy of ``spec``.

    Type coercion is driven by the current value at that path (or, for a
    brand-new leaf, by the narrowest type the string parses as) -- never by
    ``eval``.
    """
    result: Dict[str, Any] = _thaw(spec)
    for override in overrides:
        if "=" not in override:
            raise SpecError(f"malformed override {override!r}; expected dotted.path=value")
        dotted_path, raw_value = override.split("=", 1)
        parts = dotted_path.split(".")
        if not parts or not all(parts):
            raise SpecError(f"malformed override path {dotted_path!r}")

        cursor = result
        for part in parts[:-1]:
            existing = cursor.get(part, _MISSING)
            if existing is _MISSING:
                existing = {}
            elif not isinstance(existing, Mapping):
                raise SpecError(
                    f"override path {dotted_path!r} traverses through "
                    f"{part!r}, an existing non-mapping value ({existing!r})"
                )
            cursor[part] = dict(existing)
            cursor = cursor[part]

        leaf = parts[-1]
        reference = cursor.get(leaf)
        cursor[leaf] = _coerce(raw_value, reference)
    return result


def resolve(
    spec: Mapping[str, Any], defaults: Mapping[str, Any] = DEFAULT_SPEC
) -> ExperimentSpec:
    """Expand ``spec`` into a fully-explicit, validated ``ExperimentSpec``.

    Every default is materialised into the result; nothing is left implicit
    (epic §8: "a trial must never be reproducible only from a notebook cell
    and its current defaults").
    """
    raw = dict(spec)

    for key in LAUNCH_ASSIGNED_KEYS:
        if key in raw:
            raise SpecError(
                f"{key!r} is assigned by the factory at launch and must not "
                "be set in an input template"
            )
    _check_unknown_keys("spec", raw, TOP_LEVEL_KEYS)

    merged = _deep_merge(defaults, raw)
    merged.setdefault("format", DOCUMENT_FORMAT)

    resolved = ExperimentSpec(
        format=merged.get("format", DOCUMENT_FORMAT),
        organism=merged.get("organism"),
        mode=merged.get("mode"),
        data=merged.get("data", {}),
        model=merged.get("model", {}),
        training=merged.get("training", {}),
        evaluation=merged.get("evaluation", {}),
        parent=merged.get("parent"),
        evolution=merged.get("evolution"),
    )
    validate(resolved)
    return resolved


def validate(spec: ExperimentSpec) -> None:
    """Structural and cross-field validation of a resolved ``ExperimentSpec``.

    Raises :class:`SpecError` with an actionable message on the first
    violation found.
    """
    if spec.format != DOCUMENT_FORMAT:
        raise SpecError(
            f"unsupported spec format {spec.format!r}; expected {DOCUMENT_FORMAT!r}"
        )
    if not spec.organism:
        raise SpecError("'organism' is required")
    if spec.mode not in VALID_MODES:
        raise SpecError(f"invalid mode {spec.mode!r}; expected one of {VALID_MODES}")

    _validate_parent(spec)
    _validate_data(spec.data)
    _validate_model(spec.model)
    _validate_training(spec.training)
    _validate_warmup_rollout_fits_corpus(spec)
    _validate_evaluation(spec.evaluation)
    _validate_evaluation_horizons(spec)
    _validate_evolution(spec.evolution)


def _validate_parent(spec: ExperimentSpec) -> None:
    if spec.mode == "fresh":
        if spec.parent:
            raise SpecError("mode 'fresh' must not declare a 'parent'")
        return

    if not spec.parent:
        raise SpecError(f"mode {spec.mode!r} requires a 'parent' block")

    _check_unknown_keys("parent", spec.parent, PARENT_KEYS)
    for required in ("run_id", "sha256"):
        if not spec.parent.get(required):
            raise SpecError(
                f"mode {spec.mode!r} requires 'parent.{required}'; missing from spec"
            )


def _validate_data(data: Mapping[str, Any]) -> None:
    _check_unknown_keys("data", data, DATA_KEYS)
    for required in ("corpus_id", "world"):
        if not data.get(required):
            raise SpecError(f"'data.{required}' is required")

    horizons = data.get("horizons_ticks") or ()
    if any(not isinstance(tick, int) or tick <= 0 for tick in horizons):
        raise SpecError("'data.horizons_ticks' must be a sequence of positive integers")


def _validate_model(model: Mapping[str, Any]) -> None:
    _check_unknown_keys("model", model, MODEL_KEYS)
    for required in ("backbone", "latent_width", "hidden_dim", "context_length"):
        if model.get(required) in (None, ""):
            raise SpecError(f"'model.{required}' is required")


def _validate_training(training: Mapping[str, Any]) -> None:
    _check_unknown_keys("training", training, TRAINING_KEYS)
    if not training.get("objective"):
        raise SpecError("'training.objective' is required")

    warmup = training.get("warmup_frames", 0) or 0
    rollout = training.get("rollout_frames", 0) or 0
    if rollout <= 0:
        raise SpecError("'training.rollout_frames' must be a positive integer")
    if warmup < 1:
        # train_action_world_model() rejects warmup_frames < 1 unconditionally
        # (cognitive_runtime/training/action_world_model.py), for both the
        # windowed_rollout and autoregressive objectives -- a spec that
        # resolves with warmup_frames=0 would pass validation here and then
        # fail immediately at training.
        raise SpecError("'training.warmup_frames' must be a positive integer (>= 1)")

    # This is exactly the required-field subset of TrainingContract's
    # non-defaulted fields; constructing one below is both the validation
    # and the mechanism that produces `training_contract_hash`.
    try:
        TrainingContract(**dict(training))
    except TypeError as exc:
        raise SpecError(f"'training' block does not resolve to a TrainingContract: {exc}") from exc


def _validate_warmup_rollout_fits_corpus(spec: ExperimentSpec) -> None:
    """``warmup_frames + rollout_frames`` must fit the corpus's shortest episode.

    Declared under ``data.quality_policy.min_episode_length``; absent that
    declaration there is nothing to check against, so this is a no-op.
    """
    min_episode_length = spec.data.get("quality_policy", {}).get("min_episode_length")
    if min_episode_length is None:
        return
    warmup = spec.training.get("warmup_frames", 0) or 0
    rollout = spec.training.get("rollout_frames", 0) or 0
    if warmup + rollout >= min_episode_length:
        raise SpecError(
            "'training.warmup_frames' + 'training.rollout_frames' "
            f"({warmup + rollout}) leaves no target frame within 'data.quality_policy."
            f"min_episode_length' ({min_episode_length})"
        )


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    _check_unknown_keys("evaluation", evaluation, EVALUATION_KEYS)
    selection_metric = evaluation.get("selection_metric")
    if not selection_metric:
        raise SpecError("'evaluation.selection_metric' is required")
    if not _METRIC_PATH_RE.match(selection_metric):
        raise SpecError(
            f"'evaluation.selection_metric' {selection_metric!r} does not "
            "parse into a resolvable metric path (e.g. "
            "'rollout.t+4.model_over_copy_last_mse')"
        )

    reference_run = evaluation.get("reference_run")
    if reference_run is not None:
        _check_unknown_keys("evaluation.reference_run", reference_run, REFERENCE_RUN_KEYS)
        # "checkpoint" is required alongside run_id/sha256, not merely
        # optional metadata: runner._reference_checkpoint_path() indexes
        # reference_run["checkpoint"] unconditionally, so omitting it here
        # would pass resolve()/validate() cleanly and then crash with a raw
        # KeyError at trial startup instead of this clear SpecError.
        for required in ("run_id", "checkpoint", "sha256"):
            if not reference_run.get(required):
                raise SpecError(f"'evaluation.reference_run.{required}' is required when 'reference_run' is set")


def _validate_evaluation_horizons(spec: ExperimentSpec) -> None:
    """Every ``t+N`` metric must name a horizon the trial actually builds."""
    selection_metric = str(spec.evaluation["selection_metric"])
    requested = tuple(int(value) for value in re.findall(r"t\+(\d+)", selection_metric))
    declared = tuple(int(value) for value in spec.data["horizons_ticks"])
    missing = tuple(value for value in requested if value not in declared)
    if missing:
        raise SpecError(
            f"'evaluation.selection_metric' {selection_metric!r} requires horizon tick(s) "
            f"{list(missing)!r}, but 'data.horizons_ticks' is {list(declared)!r}"
        )


def _validate_evolution(evolution: Optional[Mapping[str, Any]]) -> None:
    if evolution is None:
        return
    _check_unknown_keys("evolution", evolution, EVOLUTION_KEYS)
    parents = evolution.get("configuration_parents")
    if parents is not None:
        # A `str` is itself a `Sequence`, so it must be excluded explicitly --
        # otherwise a two-character string like "ab" would pass a bare
        # `len(parents) == 2` check without naming two actual parent run IDs.
        if isinstance(parents, str) or not isinstance(parents, (list, tuple)):
            raise SpecError(
                "'evolution.configuration_parents' must be a list/tuple of "
                f"parent run IDs; got {type(parents).__name__}"
            )
        if len(parents) not in (0, 2) or any(
            not isinstance(parent, str) or not parent for parent in parents
        ):
            raise SpecError(
                "'evolution.configuration_parents' must name exactly two "
                f"non-empty parent run IDs when present; got {parents!r}"
            )
    if parents and not evolution.get("weight_donor"):
        raise SpecError("'evolution.weight_donor' is required when 'configuration_parents' is set")


def dump_canonical_json(spec: ExperimentSpec) -> bytes:
    """Canonical JSON bytes for ``spec`` -- byte-identical across formatting."""
    return canonical_json(spec.to_dict())


__all__ = [
    "DOCUMENT_FORMAT",
    "VALID_MODES",
    "DEFAULT_SPEC",
    "ExperimentSpec",
    "SpecError",
    "load_spec",
    "apply_overrides",
    "resolve",
    "validate",
    "dump_canonical_json",
]
