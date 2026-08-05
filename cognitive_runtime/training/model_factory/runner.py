"""Trial orchestration: ``run_trial(spec) -> TrialResult`` (epic #212, Phase C
step 1-2, issue #225).

This is the epic's main integration point: it wires every earlier phase
(spec resolution, run provenance, the frozen corpus, checkpoint lineage,
the resumable trainer, the budget policy, the state machine, and paired
comparison) around the *existing* ``train_action_world_model`` and
``evaluate_action_world_model_milestone`` functions. Per epic #212 §6.1 it
must not duplicate model training or metrics -- ``run_nursery_joint`` is not
reused here because it records/reuses its own nursery-keyed episodes
internally and therefore cannot consume a Model Factory corpus's already
frozen, differently-named session directories without recording a
replacement episode; the lower-level trainer/evaluator it itself calls are
reused directly instead.

Two design decisions worth naming up front:

* ``train_action_world_model`` has no mid-run interrupt hook -- the only
  primitive for a graceful stop is the documented stop-and-continue
  contract (train N epochs, get ``stats["resume_state"]``, train more).
  This runner therefore trains in chunks sized by the declared
  ``training.checkpoint_cadence_epochs`` (or the full epoch budget in one
  chunk when no cadence is declared -- there is then no boundary to
  gracefully stop at other than the hard deadline after that one chunk),
  checking :class:`~.budget.TrainingBudget` after every chunk.
* Corpus resolution, dataset construction, and checkpoint-continuation
  validation all happen *before* the run directory is allocated, because
  :func:`~.artifacts.allocate_run_artifacts` itself requires an already
  -resolved ``DataContract``/``ArchitectureContract`` to write
  ``contracts.json``. A cache-miss or a rejected continuation therefore
  never creates a run directory at all -- a strictly stronger guarantee
  than "no artifact eligible for promotion". Only training, evaluation,
  and report-writing run under the persisted state machine.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from cognitive_runtime.observability.trace import RunTrace
from cognitive_runtime.training.model_factory.artifacts import (
    RunArtifacts,
    allocate_run_artifacts,
    atomic_write_json,
)
from cognitive_runtime.training.model_factory.budget import (
    BUDGET_TIERS,
    STATUS_BUDGET_EXCEEDED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    EpochTimer,
    TrainingBudget,
    budget_seconds_for_tier,
    build_budget_report,
    write_budget_report,
)
from cognitive_runtime.training.model_factory.checkpoint import (
    BestValidationTracker,
    TrainerResumeState,
    _field_differences,
    load_factory_checkpoint,
    read_factory_checkpoint_metadata,
    save_factory_checkpoint,
)
from cognitive_runtime.training.model_factory.comparison import compare_paired_episodes
from cognitive_runtime.training.model_factory.contracts import ArchitectureContract, TrainingContract
from cognitive_runtime.training.model_factory.corpus import resolve_corpus
from cognitive_runtime.training.model_factory.navigation_metrics import evaluate_navigation_sessions
from cognitive_runtime.training.model_factory.spec import ExperimentSpec
from cognitive_runtime.training.model_factory.spec import resolve as resolve_spec
from cognitive_runtime.training.model_factory.spec import validate as validate_spec
from cognitive_runtime.training.model_factory.state import (
    STATE_BUDGET_EXCEEDED,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    StateError,
    cancellation_requested,
    claim_stale_worker,
    create_state,
    heartbeat_path,
    release_devices,
    reserve_devices,
    state_path,
    transition,
    write_heartbeat,
)

#: The factory's default epoch count when a trial spec declares no
#: ``training.epoch_budget``. Purely a runner-internal fallback -- it never
#: changes ``training_contract_hash``, which faithfully records the
#: declared (possibly ``None``) budget.
DEFAULT_EPOCH_BUDGET = 10

#: Versioned identity for how ``run_trial`` derives ``rgb_preprocessing_version``
#: in the architecture contracts it builds. Bump only if the pixel
#: preprocessing this runner assumes actually changes.
RGB_PREPROCESSING_VERSION = "model-factory-runner-rgb-v1"

RUNS_ROOT_DEFAULT = "runs"

_SELECTION_METRIC_RE = re.compile(
    r"^(?P<report>rollout|direct)\.t\+(?P<ticks>\d+)\.(?P<stat>[a-zA-Z_][a-zA-Z0-9_]*)$"
)
_GOAL_NAVIGATION_METRICS = frozenset(
    ("success_rate", "geodesic_efficiency", "collision_rate", "replan_recovery")
)


def _torch():
    import torch

    return torch


def _action_world_model_module():
    """Import the torch-dependent trainer/evaluator only when a trial runs.

    Keeps ``import cognitive_runtime.training.model_factory.runner`` itself
    usable in the core-only install, matching every other Model Factory
    module's lazy-import discipline.
    """
    from cognitive_runtime.training import action_world_model

    return action_world_model


def _nursery_module():
    from cognitive_runtime.training import nursery

    return nursery


def _statistical_evaluation_module():
    from cognitive_runtime.training import statistical_evaluation

    return statistical_evaluation


@dataclass(frozen=True)
class TrialResult:
    """Everything a caller needs after one immutable factory trial.

    ``evaluation`` is the best-validation checkpoint's full
    ``evaluate_action_world_model_milestone`` report; ``comparison`` is
    ``None`` for a ``fresh`` or ``resume`` trial (there is no sibling
    baseline to pair against) and the merged paired-comparison payload
    (also persisted to ``metrics/comparison.json``) for ``clone``/
    ``fine_tune``.
    """

    run_id: str
    display_name: str
    mode: str
    state: str
    directory: Path
    architecture_hash: str
    data_contract_hash: str
    training_contract_hash: str
    checkpoint_path: str
    checkpoint_sha256: Optional[str]
    training_stats: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    budget_report: Mapping[str, Any]
    comparison: Optional[Mapping[str, Any]]
    experiment_report_path: str


def _tier_for_seconds(seconds: float) -> str:
    """Name ``seconds`` against the declared §15.1 tier portfolio.

    ``ExperimentSpec`` has no separate ``tier`` field -- a trial declares
    ``training.max_training_seconds`` directly -- so the comparison report's
    "tier" is derived by matching that value against the known tiers rather
    than resolved from a dedicated spec field.
    """
    for tier, tier_seconds in BUDGET_TIERS.items():
        if seconds == tier_seconds:
            return tier
    return "custom"


def _final_loss(stats: Mapping[str, Any]) -> Optional[float]:
    curve = (stats.get("loss_curves") or {}).get("total_loss")
    return float(curve[-1]) if curve else None


def _architecture_contract(model: Any) -> ArchitectureContract:
    """Derive the checkpoint-compatibility identity from a built cortex.

    Mirrors ``checkpoint._model_definition`` field-for-field so a run's
    ``architecture_hash`` is defined by the same facts a checkpoint header
    would embed.
    """
    cfg = model.config
    return ArchitectureContract(
        pixel_shape=tuple(model.pixel_shape),
        rgb_preprocessing_version=RGB_PREPROCESSING_VERSION,
        action_vocabulary=tuple(model.action_keys),
        workspace_modalities=dict(model.workspace_modalities),
        workspace_layout_hash=model.workspace_layout_hash,
        latent_width=int(model.latent_width),
        hidden_dim=int(model.hidden_dim),
        action_embed_dim=int(cfg.action_embed_dim),
        reconstruction_shape=tuple(model.reconstruction_shape),
        visual_architecture=str(model.visual_architecture),
        semantic_classes=int(cfg.semantic_classes),
        horizons_ticks=tuple(model.horizons_ticks),
        direct_horizon_topology=tuple(model.horizons_ticks),
        backbone=str(cfg.backbone),
        context_length=int(cfg.context_length),
        backbone_kwargs=dict(cfg.backbone_kwargs),
    )


def _action_world_model_config(spec: ExperimentSpec) -> Any:
    """Map a resolved spec's ``model``/``training``/``data`` blocks onto
    ``ActionWorldModelConfig``.

    Not every declared training-contract field is threaded into the real
    trainer: ``train_action_world_model`` always builds
    ``torch.optim.Adam(model.parameters(), lr=cfg.lr)``, so
    ``training.optimizer.name``/``weight_decay`` are recorded for contract
    identity but do not change the optimizer actually used -- an existing
    property of the reused trainer, not something this runner introduces.
    """
    awm = _action_world_model_module()
    model_block = spec.model
    training_block = spec.training
    valid_fields = {field.name for field in dataclasses.fields(awm.ActionWorldModelConfig)}
    kwargs: Dict[str, Any] = dict(
        latent_width=int(model_block["latent_width"]),
        hidden_dim=int(model_block["hidden_dim"]),
        action_embed_dim=int(model_block["action_embed_dim"]),
        reconstruction_size=int(model_block["reconstruction_size"]),
        backbone=str(model_block["backbone"]),
        context_length=int(model_block["context_length"]),
        backbone_kwargs=dict(model_block.get("backbone_kwargs") or {}),
        horizons_ticks=tuple(int(tick) for tick in spec.data["horizons_ticks"]),
        training_objective=str(training_block["objective"]),
        lr=float(training_block["optimizer"]["lr"]),
        batch_size=int(training_block["batch_size"]),
        seed=int(training_block["seed"]),
        rollout_frames=int(training_block["rollout_frames"]),
        warmup_frames=int(training_block["warmup_frames"]),
        scheduled_sampling_p=float(training_block["scheduled_sampling_p"]),
        ema_target_decay=training_block.get("ema_target_decay"),
        device=str(training_block["device"]),
        epochs=int(training_block.get("epoch_budget") or DEFAULT_EPOCH_BUDGET),
    )
    # Factory specs use concise public names while ActionWorldModelConfig
    # retains the older explicit field names. Preserve support for callers
    # already using the explicit names, but make the documented/spec-shipped
    # aliases actually reach the loss calculation.
    loss_weight_aliases = {
        "pixel": "pixel_loss_weight",
        "latent": "latent_loss_weight",
        "semantic": "semantic_loss_weight",
    }
    loss_weights = dict(training_block.get("loss_weights") or {})
    for alias, field_name in loss_weight_aliases.items():
        if alias in loss_weights and field_name in loss_weights:
            raise ValueError(
                f"training.loss_weights declares both {alias!r} and {field_name!r}; "
                "use only one name for the same loss weight"
            )
    for key, value in loss_weights.items():
        field_name = loss_weight_aliases.get(key, key)
        if field_name in valid_fields:
            kwargs[field_name] = value
    return awm.ActionWorldModelConfig(**kwargs)


def _transition_balance_weights(
    dataset: Any,
    train_entries: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Tuple[Optional[List[List[float]]], Optional[Dict[str, Any]]]:
    """Build trainer-aligned transition weights for a Factory corpus.

    Labels are computed from ``dataset.episodes`` itself, then sliced back
    into that exact episode order. This is the same alignment discipline as
    the nursery trainer and prevents manifest order or multi-episode sessions
    from assigning one episode's weights to another.
    """
    if "stationary_cap" not in policy:
        return None, None
    stationary_cap = float(policy["stationary_cap"])
    if not math.isfinite(stationary_cap) or not 0.0 <= stationary_cap <= 1.0:
        raise ValueError(
            "training.transition_balance_policy.stationary_cap must be finite and in [0, 1], "
            f"got {policy['stationary_cap']!r}"
        )

    def session_key(value: Any) -> str:
        return str(Path(str(value)).resolve()).casefold()

    session_to_scenario: Dict[str, str] = {}
    for entry in train_entries:
        key = session_key(entry["session_path"])
        scenario = str(entry["scenario"])
        prior = session_to_scenario.setdefault(key, scenario)
        if prior != scenario:
            raise ValueError(
                f"training session {entry['session_path']!r} is assigned to both "
                f"{prior!r} and {scenario!r}"
            )

    nursery = _nursery_module()
    labels_by_episode: List[List[Any]] = []
    for episode in dataset.episodes:
        key = session_key(episode.session_dir)
        if key not in session_to_scenario:
            raise ValueError(
                f"training dataset episode {episode.session_dir!r}/{episode.episode_id} "
                "does not correspond to a training corpus manifest entry"
            )
        labels = nursery.compute_scenario_transition_labels(
            episode.session_dir, episode.episode_id, session_to_scenario[key]
        )
        if len(labels) != len(episode.actions):
            raise ValueError(
                f"transition labels for {episode.session_dir!r}/{episode.episode_id} have "
                f"{len(labels)} transitions, but the training dataset has {len(episode.actions)}"
            )
        labels_by_episode.append(labels)

    flat_labels = [label for episode_labels in labels_by_episode for label in episode_labels]
    flat_weights = nursery.balanced_transition_weights(
        flat_labels, max_stationary_fraction=stationary_cap,
    )
    transition_weights: List[List[float]] = []
    cursor = 0
    for episode_labels in labels_by_episode:
        count = len(episode_labels)
        transition_weights.append(flat_weights[cursor : cursor + count])
        cursor += count

    total_weight = sum(flat_weights)
    stationary_weight = sum(
        weight for weight, label in zip(flat_weights, flat_labels)
        if label.movement_state == "blocked"
    )
    report = {
        "stationary_cap": stationary_cap,
        "transitions": len(flat_labels),
        "stationary_transitions": sum(
            label.movement_state == "blocked" for label in flat_labels
        ),
        "weighted_stationary_fraction": (
            stationary_weight / total_weight if total_weight > 0 else None
        ),
    }
    return transition_weights, report


def _parent_checkpoint_path(root: Union[str, Path], spec: ExperimentSpec) -> Path:
    parent = spec.parent or {}
    return Path(root) / spec.organism / str(parent["run_id"]) / "checkpoints" / str(parent["checkpoint"])


def _reference_checkpoint_path(root: Union[str, Path], spec: ExperimentSpec) -> Path:
    reference_run = spec.evaluation.get("reference_run") or {}
    return Path(root) / spec.organism / str(reference_run["run_id"]) / "checkpoints" / str(reference_run["checkpoint"])


def _load_reference_model(root: Union[str, Path], spec: ExperimentSpec) -> Optional[Any]:
    """Load ``evaluation.reference_run``'s checkpoint for read-only
    comparison, or ``None`` if the spec declares none.

    Unlike a ``parent`` weight donor, a reference model is never loaded
    against the candidate's own architecture/data/training contracts --
    ``load_factory_checkpoint`` with no ``mode``/``model`` given just
    reconstructs the model the checkpoint itself declares (the same
    "plain inspect" call other read-only reloads in this module use, e.g.
    ``best_eval``'s reconstruction fallback below), matching the docstring
    at spec.py's ``REFERENCE_RUN_KEYS``: read-only comparison, never a
    weight donor.
    """
    reference_run = spec.evaluation.get("reference_run")
    if reference_run is None:
        return None
    reference_path = str(_reference_checkpoint_path(root, spec))
    declared_sha = str(reference_run["sha256"])
    actual_sha = read_factory_checkpoint_metadata(reference_path).get("checkpoint_sha256")
    if actual_sha != declared_sha:
        raise ValueError(
            f"reference_run checkpoint sha256 mismatch for {reference_run.get('run_id')!r}: "
            f"spec declares {declared_sha!r}, file is {actual_sha!r}"
        )
    return load_factory_checkpoint(reference_path).model


def _load_existing_run_artifacts(root: Union[str, Path], organism: str, run_id: str) -> RunArtifacts:
    """Reopen an already-allocated run directory for a resume trial.

    A resume trial's declared ``mode``/``parent`` necessarily differ from
    whatever the interrupted run was originally launched with, so its
    ``trial_spec.json`` can never satisfy ``allocate_run_artifacts``'s
    resume path (an exact byte-for-byte match of the *entire* immutable
    payload -- that mechanism recovers a crash *during setup* of the
    original launch, not a later continuation of training under a new
    spec). "Only resume may append to an interrupted run" (epic #212 §11)
    means exactly that: none of the run's immutable manifests are rewritten
    for a resume, they are only read back.
    """
    from cognitive_runtime.training.prediction_export import ExperimentIdentity

    directory = Path(root) / organism / run_id
    experiment_path = directory / "experiment.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(f"cannot resume {run_id!r}: no existing run directory at {directory}")
    with experiment_path.open(encoding="utf-8") as handle:
        experiment_doc = json.load(handle)
    experiment = ExperimentIdentity(
        experiment_id=experiment_doc["experiment_id"], organism=experiment_doc["organism"],
        trace_id=experiment_doc.get("trace_id"), created_at=experiment_doc["created_at"],
        git_commit=experiment_doc.get("git_commit"),
        format=experiment_doc.get("format", "experiment-identity-v1"),
    )
    with (directory / "display_name.json").open(encoding="utf-8") as handle:
        display_name = json.load(handle)["display_name"]
    return RunArtifacts(
        run_id=run_id, experiment=experiment, directory=directory,
        experiment_path=experiment_path, trial_spec_path=directory / "trial_spec.json",
        contracts_path=directory / "contracts.json", lineage_path=directory / "lineage.json",
        display_name_path=directory / "display_name.json", display_name=display_name,
        execution_path=directory / "execution.json", data_manifest_path=directory / "data_manifest.json",
        checkpoints_dir=directory / "checkpoints", metrics_dir=directory / "metrics",
    )


def _evaluate(
    model: Any,
    dataset: Any,
    spec: ExperimentSpec,
    cfg: Any,
    *,
    navigation_metrics: Optional[Mapping[str, Any]] = None,
    reference_model: Optional[Any] = None,
) -> Dict[str, Any]:
    awm = _action_world_model_module()
    result = awm.evaluate_action_world_model_milestone(
        model, dataset, tuple(int(tick) for tick in spec.data["horizons_ticks"]),
        warmup_frames=cfg.warmup_frames, reference_model=reference_model,
    )
    if navigation_metrics is not None:
        result["goal_navigation"] = dict(navigation_metrics)
    return result


def _write_clinic_session_index(
    directory: Path,
    train_entries: Sequence[Mapping[str, Any]],
    validation_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Bind a factory run to the frozen corpus sessions it trained/validated
    against, mirroring ``nursery._write_clinic_session_index``'s
    ``clinic-session-index-v1`` format.

    Written as soon as the run directory exists -- well before training
    finishes -- so the clinic can show a still-running trial's complete
    train/validation session set (including EEG/attention/action data,
    which lives in each session's own streams/decisions files independent
    of any prediction export) immediately, not only once a checkpoint has
    been promoted.
    """
    entries = [
        {"scenario": str(entry["scenario"]), "split": split, "session_dir": str(Path(str(entry["session_path"])).resolve())}
        for split, group in (("train", train_entries), ("evaluation", validation_entries))
        for entry in group
    ]
    atomic_write_json(directory / "clinic_sessions.json", {"format": "clinic-session-index-v1", "sessions": entries})


def _validate_temporal_capability(
    dataset: Any,
    horizons_ticks: Sequence[int],
    *,
    warmup_frames: int,
    rollout_frames: int = 0,
    purpose: str,
) -> None:
    """Reject recorded sequences that cannot support a trial's time offsets.

    The corpus owns frames, recorded tick values, and episode boundaries. The
    experiment owns horizons. This check joins those facts after dataset load,
    without requiring the corpus to advertise the same horizon topology.
    """
    horizons = tuple(int(value) for value in horizons_ticks)
    if not dataset.episodes:
        raise ValueError(f"{purpose} corpus contains no recorded episodes")
    horizon_frames = _action_world_model_module().horizons_ticks_to_frames(
        horizons, dataset.ticks_per_frame,
    )
    max_horizon_ticks = max(horizons)
    max_horizon_frames = max(horizon_frames)
    too_short: list[str] = []
    invalid_ticks: list[str] = []
    for episode in dataset.episodes:
        source = f"{episode.session_dir}/{episode.episode_id}"
        frame_count = len(episode.frames)
        if len(episode.ticks) != frame_count or any(
            right < left for left, right in zip(episode.ticks, episode.ticks[1:])
        ):
            invalid_ticks.append(source)
            continue
        tick_span = episode.ticks[-1] - episode.ticks[0] if len(episode.ticks) >= 2 else 0
        required_frames = warmup_frames + max_horizon_frames
        if purpose == "training":
            required_frames = max(required_frames, warmup_frames + rollout_frames)
        # Evaluators/trainers use a strict `n > offset + warmup` window so
        # equality still supplies no valid starting index.
        if frame_count <= required_frames or tick_span < max_horizon_ticks:
            too_short.append(
                f"{source} (frames={frame_count}, tick_span={tick_span})"
            )
    if invalid_ticks:
        raise ValueError(
            f"{purpose} corpus has missing or non-monotonic recorded tick indexes: "
            + ", ".join(invalid_ticks)
        )
    if too_short:
        raise ValueError(
            f"{purpose} corpus cannot support trial horizons {list(horizons)!r} ticks "
            f"with warmup={warmup_frames} and rollout={rollout_frames}: "
            + "; ".join(too_short)
        )


def _clinic_prediction_mode(selection_metric: str) -> str:
    """Derive the clinic export's rollout/direct mode from a selection
    metric, mirroring ``_resolve_selection_metric``'s own prefix parsing.
    A goal-navigation metric carries no rollout/direct prefix -- rollout is
    the only mode goal-navigation trials actually train with.
    """
    match = _SELECTION_METRIC_RE.match(selection_metric)
    return str(match.group("report")) if match else "rollout"


def _export_clinic_predictions(
    resolved_spec: ExperimentSpec,
    checkpoint_path: Path,
    validation_dataset: Any,
    experiment: Any,
    training_stats: Mapping[str, Any],
    *,
    max_episodes: int,
) -> None:
    """Best-effort clinic (viewer/) prediction export for the promoted
    checkpoint's validation episodes (issue: auto-export Model Factory
    predictions).

    Reloads the checkpoint that was actually selected/promoted rather than
    reusing the last in-memory training chunk, so the export reflects the
    weights the run actually reports. Diagnostic-only, matching the clinic's
    existing discipline that it never gates or affects training: a bad
    export -- per-episode or for the whole call -- is swallowed rather than
    raised, and must never fail the trial.
    """
    try:
        prediction_mode = _clinic_prediction_mode(resolved_spec.evaluation["selection_metric"])
        ticks = tuple(int(tick) for tick in resolved_spec.data["horizons_ticks"])
        awm = _action_world_model_module()
        horizon_frames = (
            awm.horizons_ticks_to_frames(ticks, validation_dataset.ticks_per_frame)
            if prediction_mode == "rollout" else list(ticks)
        )
        loaded = load_factory_checkpoint(str(checkpoint_path))
        from cognitive_runtime.training.prediction_export import export_cortex_session_predictions

        for episode in validation_dataset.episodes[:max_episodes]:
            try:
                export_cortex_session_predictions(
                    loaded.model, validation_dataset, episode.session_dir, episode.episode_id,
                    horizon_frames=horizon_frames, prediction_mode=prediction_mode,
                    checkpoint_path=str(checkpoint_path), experiment=experiment,
                    training_stats=training_stats,
                )
            except Exception:
                continue
    except Exception:
        pass


def _selection_metric_mode(selection_metric: str) -> str:
    """Best-checkpoint direction for one supported selection path."""
    if selection_metric in (
        "goal_navigation.success_rate",
        "goal_navigation.geodesic_efficiency",
        "goal_navigation.replan_recovery",
    ):
        return "max"
    return "min"


def _resolve_selection_metric(
    eval_result: Mapping[str, Any], selection_metric: str, ticks_per_frame: float,
) -> Tuple[float, List[float]]:
    """Resolve a spec's ``evaluation.selection_metric`` against one
    ``evaluate_action_world_model_milestone`` report.

    The second return value is always a lower-is-better per-episode error for
    paired comparison.  Prediction metrics already are errors; higher-is-
    better navigation scores are converted to ``1 - score`` while the scalar
    remains the advertised raw score used for checkpoint selection/reporting.
    """
    navigation_prefix = "goal_navigation."
    if selection_metric.startswith(navigation_prefix):
        metric = selection_metric[len(navigation_prefix):]
        if metric not in _GOAL_NAVIGATION_METRICS:
            raise ValueError(
                f"unknown goal-navigation selection metric {metric!r}; expected one of "
                f"{sorted(_GOAL_NAVIGATION_METRICS)!r}"
            )
        report = eval_result.get("goal_navigation")
        if not isinstance(report, Mapping):
            raise ValueError(
                f"selection metric {selection_metric!r} requires goal-navigation evaluation evidence"
            )
        value = report.get(metric)
        per_episode = (report.get("per_episode") or {}).get(metric)
        if value is None or not isinstance(per_episode, Sequence):
            raise ValueError(
                f"selection metric {selection_metric!r} was not evaluable on the held-out corpus"
            )
        if metric == "collision_rate":
            errors = [float(item) for item in per_episode]
        elif metric == "replan_recovery":
            errors = [0.0 if item is None else 1.0 - float(item) for item in per_episode]
        else:
            errors = [1.0 - float(item) for item in per_episode]
        return float(value), errors

    match = _SELECTION_METRIC_RE.match(selection_metric)
    if not match:
        raise ValueError(
            "run_trial resolves goal_navigation.<metric> or selection metrics of the form "
            f"'<rollout|direct>.t+<ticks>.<stat>'; got {selection_metric!r}"
        )
    report_name = match.group("report")
    ticks = int(match.group("ticks"))
    stat = match.group("stat")
    report = eval_result[report_name]
    if report_name == "direct":
        horizon_key: Union[int, float] = ticks
    else:
        horizon_key = _action_world_model_module().horizons_ticks_to_frames((ticks,), ticks_per_frame)[0]
    horizons = report.get("horizons", {})
    if horizon_key not in horizons:
        raise ValueError(
            f"selection metric {selection_metric!r} horizon t+{ticks} was not evaluated "
            f"(available: {sorted(horizons)!r})"
        )
    value = horizons[horizon_key][stat]
    per_episode = report["per_episode_model_mse"][horizon_key]
    return float(value), [float(item) for item in per_episode]


def _per_episode_retention_losses(eval_result: Mapping[str, Any]) -> List[float]:
    """Reduce held-out multi-horizon prediction errors to one loss per episode."""
    by_horizon = eval_result.get("per_episode_model_mse") or {}
    series = [list(values) for _, values in sorted(by_horizon.items(), key=lambda item: str(item[0]))]
    if not series or not series[0]:
        raise ValueError("generic retention evaluation produced no per-episode prediction losses")
    episode_count = len(series[0])
    if any(len(values) != episode_count for values in series):
        raise ValueError("generic retention horizons disagree on held-out episode count")
    return [statistics.fmean(float(values[index]) for values in series) for index in range(episode_count)]


def _optimizer_for_state(model: Any, lr: float, optimizer_state_dict: Optional[Mapping[str, Any]]) -> Any:
    """A throwaway optimizer whose ``state_dict()`` matches a trainer chunk's
    ``resume_state`` -- ``save_factory_checkpoint`` needs a live optimizer
    object, but the trainer's own internal optimizer is not returned."""
    torch = _torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return optimizer


def _build_comparison(
    spec: ExperimentSpec,
    root: Union[str, Path],
    child_trial_spec_path: Union[str, Path],
    candidate_eval: Mapping[str, Any],
    baseline_eval: Mapping[str, Any],
    episode_ids: Sequence[str],
    ticks_per_frame: float,
    *,
    tier: str,
    measured_training_seconds: float,
) -> Dict[str, Any]:
    """Paired per-episode comparison plus everything issue #225 asks the
    comparison report to name beyond ``comparison.py``'s own scope: the
    parent, the single changed declared field, tier, and measured training
    time.

    ``minimum_episode_count=1`` deliberately overrides ``compare_paired_episodes``'s
    default of 5: whether a handful of paired episodes is *statistically*
    enough to trust is a promotion-policy question (epic #212's MF-C4,
    issue #226, not yet built), not a reason for the trial runner to omit
    the comparison outright.
    """
    selection_metric = spec.evaluation["selection_metric"]
    _, candidate_per_episode = _resolve_selection_metric(candidate_eval, selection_metric, ticks_per_frame)
    _, baseline_per_episode = _resolve_selection_metric(baseline_eval, selection_metric, ticks_per_frame)
    if len(candidate_per_episode) != len(episode_ids) or len(baseline_per_episode) != len(episode_ids):
        raise ValueError(
            "per-episode validation counts do not match the frozen validation session "
            "count; an episode was likely too short for the declared selection horizon"
        )
    candidate_by_episode = dict(zip(episode_ids, candidate_per_episode))
    baseline_by_episode = dict(zip(episode_ids, baseline_per_episode))
    comparison = compare_paired_episodes(
        candidate_by_episode, baseline_by_episode, primary_metric=selection_metric,
        confidence=float(spec.evaluation.get("confidence", 0.95)), minimum_episode_count=1,
    )

    parent = spec.parent or {}
    parent_run_id = str(parent["run_id"])
    parent_spec_path = Path(root) / spec.organism / parent_run_id / "trial_spec.json"
    with parent_spec_path.open(encoding="utf-8") as handle:
        parent_spec_doc = json.load(handle)
    # Diff against the child's own persisted trial_spec.json rather than
    # spec.to_dict() directly: both are written through the same
    # atomic_write_json/_jsonable pipeline, so tuples and MappingProxyType
    # blocks are normalised identically on both sides -- comparing the raw
    # in-memory spec against the parent's disk-loaded JSON would otherwise
    # flag every tuple-valued field as a spurious mismatch.
    with Path(child_trial_spec_path).open(encoding="utf-8") as handle:
        child_doc = json.load(handle)
    field_diff = []
    for section in ("data", "model", "training", "evaluation"):
        differences = _field_differences(
            dict(parent_spec_doc.get(section) or {}), dict(child_doc.get(section) or {}), section,
        )
        field_diff.extend(
            {"field": mismatch.field, "parent": mismatch.parent, "child": mismatch.requested}
            for mismatch in differences
        )
    decision = (
        "candidate_improves"
        if comparison.status == "evaluable" and comparison.mean_delta is not None and comparison.mean_delta < 0
        else "hold"
    )

    payload = comparison.to_dict()
    payload.update({
        "parent_run_id": parent_run_id,
        "field_diff": field_diff,
        "tier": tier,
        "measured_training_seconds": measured_training_seconds,
        "decision": decision,
    })
    return payload


def run_trial(
    spec: Union[Mapping[str, Any], ExperimentSpec],
    *,
    root: Union[str, Path] = RUNS_ROOT_DEFAULT,
    corpus_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    naming_seed: Optional[Union[str, int]] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    heartbeat_timeout_seconds: Optional[float] = None,
    export_predictions: bool = True,
    export_predictions_max_episodes: int = 3,
) -> TrialResult:
    """Execute one fresh/clone/resume/fine_tune Model Factory trial.

    Wires the frozen corpus (MF-A5), checkpoint lineage and continuation
    validation (MF-B2), the resumable/budgeted trainer (MF-B3/B4), and the
    trial state machine (MF-B5) around the existing
    ``train_action_world_model``/``evaluate_action_world_model_milestone``.
    Writes ``checkpoints/last.pt``, ``checkpoints/best-validation.pt``,
    ``metrics/validation.json`` and ``metrics/budget_report.json``; a
    navigation continuation also writes ``metrics/retention.json``, while
    ``clone``/``fine_tune`` writes ``metrics/comparison.json``. Every run
    writes ``experiment_report.json``. Never writes ``metrics/test.json`` -- that
    is reserved for an explicit final-test action (MF-C5, issue #227).

    Also writes ``clinic_sessions.json`` (a fresh/clone/fine_tune launch
    only) and, unless ``export_predictions`` is false, best-effort exports
    up to ``export_predictions_max_episodes`` validation episodes'
    predictions from the promoted checkpoint -- presentation tooling for
    the clinic's Episode tab (viewer/), not part of the experiment's
    identity/reproducibility contract, so neither is spec-declared or
    contract-hashed.
    """
    resolved_spec = spec if isinstance(spec, ExperimentSpec) else resolve_spec(spec)
    validate_spec(resolved_spec)
    # A resume trial appends to its own interrupted run directory (epic
    # #212 §11: "only resume may append to an interrupted run") rather than
    # creating a sibling -- its declared parent *is* that same run, so the
    # run_id defaults to the parent's unless a caller overrides it.
    if resolved_spec.mode == "resume" and run_id is None:
        run_id = str((resolved_spec.parent or {})["run_id"])

    corpus = resolve_corpus(
        resolved_spec.data["corpus_id"], allow_record=False,
        root=corpus_root, organism=resolved_spec.organism,
    )
    trial_horizons = tuple(int(value) for value in resolved_spec.data["horizons_ticks"])
    train_entries = corpus.manifest["sessions"]["train"]
    validation_entries = corpus.manifest["sessions"]["validation"]
    if not train_entries:
        raise ValueError(f"corpus {corpus.corpus_id!r} has no training sessions")
    if not validation_entries:
        raise ValueError(f"corpus {corpus.corpus_id!r} has no validation sessions")
    validation_episode_ids = [str(entry["session_id"]) for entry in validation_entries]
    navigation_evaluation = (
        evaluate_navigation_sessions(validation_entries)
        if corpus.data_contract.behavior_mixture_policy else None
    )

    awm = _action_world_model_module()
    vocabulary = _nursery_module()._action_keys_for_world(resolved_spec.data["world"])
    train_dataset = awm.build_action_sequence_dataset(
        [entry["session_path"] for entry in train_entries], action_keys=vocabulary,
    )
    transition_weights, transition_balance_report = _transition_balance_weights(
        train_dataset,
        train_entries,
        dict(resolved_spec.training.get("transition_balance_policy") or {}),
    )
    validation_dataset = awm.build_action_sequence_dataset(
        [entry["session_path"] for entry in validation_entries], action_keys=vocabulary,
    )
    _validate_temporal_capability(
        train_dataset, trial_horizons,
        warmup_frames=int(resolved_spec.training["warmup_frames"]),
        rollout_frames=int(resolved_spec.training["rollout_frames"]),
        purpose="training",
    )
    _validate_temporal_capability(
        validation_dataset, trial_horizons,
        warmup_frames=int(resolved_spec.training["warmup_frames"]),
        purpose="validation",
    )

    retention_policy = dict(corpus.data_contract.retention_policy)
    retention_dataset = None
    retention_episode_ids: List[str] = []
    if retention_policy:
        if resolved_spec.mode == "fresh":
            raise ValueError(
                "a navigation retention contract requires a parent checkpoint; use fine_tune "
                "from a generic_action_effects_v1 candidate so before/after retention is measurable"
            )
        retention_corpus = resolve_corpus(
            str(retention_policy["corpus_id"]), allow_record=False,
            root=corpus_root, organism=resolved_spec.organism,
        )
        if retention_corpus.data_contract.behavior_mixture_policy:
            raise ValueError(
                "retention_policy.corpus_id must resolve to the generic action-effects corpus, "
                "not another navigation corpus"
            )
        retention_entries = retention_corpus.manifest["sessions"]["validation"]
        if not retention_entries:
            raise ValueError("generic retention corpus has no validation sessions")
        retention_episode_ids = [str(entry["session_id"]) for entry in retention_entries]
        retention_dataset = awm.build_action_sequence_dataset(
            [entry["session_path"] for entry in retention_entries], action_keys=vocabulary,
        )
        _validate_temporal_capability(
            retention_dataset, trial_horizons,
            warmup_frames=int(resolved_spec.training["warmup_frames"]),
            purpose="retention validation",
        )

    cfg = _action_world_model_config(resolved_spec)
    torch = _torch()
    expected_workspace_modalities = awm._workspace_modalities(train_dataset, cfg.workspace_enabled)
    expected_workspace_layout = train_dataset.workspace_layout_hash if expected_workspace_modalities else None
    candidate_model = awm.build_action_world_model(
        train_dataset.pixel_shape, train_dataset.action_keys, cfg,
        workspace_modalities=expected_workspace_modalities,
        workspace_layout_hash=expected_workspace_layout,
    )
    architecture_contract = _architecture_contract(candidate_model)
    training_contract = TrainingContract(**dict(resolved_spec.training))

    model = candidate_model
    resume_state: Optional[TrainerResumeState] = None
    parent_checkpoint_sha: Optional[str] = None
    baseline_eval: Optional[Dict[str, Any]] = None
    retention_before_eval: Optional[Dict[str, Any]] = None

    if resolved_spec.mode != "fresh":
        parent_path = str(_parent_checkpoint_path(root, resolved_spec))
        declared_sha = str((resolved_spec.parent or {})["sha256"])
        actual_sha = read_factory_checkpoint_metadata(parent_path).get("checkpoint_sha256")
        if actual_sha != declared_sha:
            raise ValueError(
                f"parent checkpoint sha256 mismatch for {(resolved_spec.parent or {}).get('run_id')!r}: "
                f"spec declares {declared_sha!r}, file is {actual_sha!r}"
            )
        # A live optimizer object is required only to satisfy
        # load_factory_checkpoint's resume-path API contract; the state
        # dict it restores is read back out of the loaded payload below and
        # handed to the trainer's own stop-and-continue mechanism instead
        # (train_action_world_model constructs its own internal optimizer).
        optimizer = torch.optim.Adam(candidate_model.parameters(), lr=cfg.lr)
        loaded = load_factory_checkpoint(
            parent_path, model=candidate_model, optimizer=optimizer,
            mode=resolved_spec.mode, architecture_contract=architecture_contract,
            data_contract_hash=corpus.data_contract, training_contract=training_contract,
        )
        model = loaded.model
        parent_checkpoint_sha = declared_sha
        if resolved_spec.mode == "resume":
            trainer_state = loaded.trainer_state
            resume_state = TrainerResumeState(
                epoch=int(trainer_state["epoch"]), global_step=int(trainer_state["global_step"]),
                optimizer_state_dict=loaded.payload["optimizer_state_dict"],
                rng_state=loaded.payload["rng_state"],
                best_validation_metric=trainer_state.get("best_validation_metric"),
                # See the matching comment where this is saved: it lives
                # inside trainer_state, not as a top-level payload field.
                target_encoder_state_dict=trainer_state.get("target_encoder_state_dict"),
            )

    reference_model = _load_reference_model(root, resolved_spec)

    if retention_dataset is not None:
        retention_before_eval = _evaluate(
            model, retention_dataset, resolved_spec, cfg, reference_model=reference_model,
        )

    if resolved_spec.mode == "resume":
        artifacts = _load_existing_run_artifacts(root, resolved_spec.organism, run_id)
    else:
        artifacts = allocate_run_artifacts(
            root, resolved_spec, architecture_contract, corpus.data_contract, training_contract,
            run_id=run_id, naming_seed=naming_seed,
        )
        _write_clinic_session_index(artifacts.directory, train_entries, validation_entries)

    device = str(resolved_spec.training["device"])
    devices: Tuple[str, ...] = () if device in ("cpu", "auto") else (device,)
    trial_state_path = state_path(artifacts.directory)
    trial_heartbeat_path = heartbeat_path(artifacts.directory)
    declared_max_seconds = resolved_spec.training.get("max_training_seconds")
    max_training_seconds = float(declared_max_seconds or budget_seconds_for_tier("fast"))
    resolved_heartbeat_timeout = float(heartbeat_timeout_seconds or max(300.0, 2.0 * max_training_seconds))

    if trial_state_path.exists():
        # Only a genuinely interrupted worker -- one that died without ever
        # reaching a terminal transition -- leaves an active record behind
        # for a resume to pick back up (epic #212 §16; state.py's own
        # module docstring: a stale/failed record is never silently
        # reopened, only a still-active one is a legitimate resume target).
        # claim_stale_worker checks staleness and claims the run (by
        # writing a fresh heartbeat) as one locked critical section, so two
        # concurrent resume attempts can never both pass the check and both
        # proceed to train against the same run directory (Codex review,
        # PR #277).
        try:
            claim_stale_worker(trial_state_path, trial_heartbeat_path, run_id=artifacts.run_id)
        except StateError as exc:
            raise ValueError(str(exc)) from exc
    else:
        create_state(trial_state_path, artifacts.run_id, heartbeat_timeout_seconds=resolved_heartbeat_timeout)
        transition(trial_state_path, STATE_RUNNING)
        write_heartbeat(trial_heartbeat_path, run_id=artifacts.run_id)

    try:
        if devices:
            reserve_devices(root, artifacts.run_id, devices)

        with RunTrace(
            "model-factory-trial", run_id=artifacts.run_id,
            trace_dir=str(trace_dir) if trace_dir is not None else str(artifacts.directory),
            config=resolved_spec.to_dict(),
        ):
            if resolved_spec.mode in ("clone", "fine_tune"):
                baseline_eval = _evaluate(
                    model, validation_dataset, resolved_spec, cfg,
                    navigation_metrics=navigation_evaluation, reference_model=reference_model,
                )

            total_epochs = int(resolved_spec.training.get("epoch_budget") or DEFAULT_EPOCH_BUDGET)
            epoch = resume_state.epoch if resume_state is not None else 0
            if epoch >= total_epochs:
                raise ValueError(
                    f"resume checkpoint epoch ({epoch}) has already reached training.epoch_budget "
                    f"({total_epochs}); raise it to continue training"
                )
            cadence = resolved_spec.training.get("checkpoint_cadence_epochs")
            chunk_size = int(cadence) if cadence else total_epochs
            budget = TrainingBudget(
                max_training_seconds=max_training_seconds, total_epochs=total_epochs,
                checkpoint_cadence_epochs=cadence,
            )
            # Seed the tracker with the pre-crash best (if any): otherwise a
            # resumed run's first post-resume validation always counts as
            # "the best" regardless of the actual metric, silently
            # overwriting a genuinely better pre-crash checkpoint.
            best_tracker = BestValidationTracker(
                mode=_selection_metric_mode(resolved_spec.evaluation["selection_metric"]),
                best=resume_state.best_validation_metric if resume_state is not None else None,
            )
            trial_started = time.monotonic()
            completion_status = STATUS_COMPLETED
            best_eval: Optional[Dict[str, Any]] = None
            last_stats: Dict[str, Any] = {}

            while epoch < total_epochs:
                chunk_start_epoch = epoch
                next_epoch = min(epoch + chunk_size, total_epochs)
                chunk_cfg = replace(cfg, epochs=next_epoch)
                with EpochTimer() as timer:
                    model, stats = awm.train_action_world_model(
                        train_dataset, chunk_cfg, initial_model=model,
                        transition_weights=transition_weights, resume_state=resume_state,
                    )
                if transition_balance_report is not None:
                    stats["transition_balance_policy"] = dict(transition_balance_report)
                epoch = next_epoch
                resume_state = stats["resume_state"]
                last_stats = stats
                # One chunk can cover several epochs (chunk_size ==
                # checkpoint_cadence_epochs); record_epoch expects one
                # sample per epoch, or its median-based ETA projection
                # multiplies a whole chunk's duration by the *individual*
                # remaining-epoch count, wildly overestimating time left and
                # triggering a premature budget_exceeded stop. Split the
                # chunk's measured duration evenly across the epochs it
                # covered instead -- the finest granularity actually
                # available, since the trainer reports no per-epoch timing
                # within one call.
                epochs_in_chunk = epoch - chunk_start_epoch
                per_epoch_seconds = (timer.duration_seconds or 0.0) / epochs_in_chunk
                for covered_epoch in range(chunk_start_epoch + 1, epoch + 1):
                    decision = budget.record_epoch(covered_epoch, per_epoch_seconds)

                trainer_state_payload = {
                    "epoch": resume_state.epoch, "global_step": resume_state.global_step,
                    "best_validation_metric": resume_state.best_validation_metric,
                    # Not a top-level save_factory_checkpoint field -- folded
                    # into trainer_state (a free-form mapping) so a later
                    # resume can restore the EMA target's Polyak-averaged
                    # history instead of silently rebuilding it as a fresh
                    # copy of the resumed online weights (checkpoint.py's
                    # TrainerResumeState docstring).
                    "target_encoder_state_dict": resume_state.target_encoder_state_dict,
                }
                chunk_optimizer = _optimizer_for_state(model, cfg.lr, resume_state.optimizer_state_dict)
                save_factory_checkpoint(
                    str(artifacts.checkpoints_dir / "last.pt"), model, chunk_optimizer,
                    trainer_state=trainer_state_payload,
                    architecture_contract=architecture_contract, data_contract_hash=corpus.data_contract,
                    training_contract=training_contract, parent_checkpoint_sha=parent_checkpoint_sha,
                    training_stats={"epoch": resume_state.epoch, "final_total_loss": _final_loss(stats)},
                    rng_state=resume_state.rng_state,
                )
                write_heartbeat(trial_heartbeat_path, run_id=artifacts.run_id)

                eval_result = _evaluate(
                    model, validation_dataset, resolved_spec, cfg,
                    navigation_metrics=navigation_evaluation, reference_model=reference_model,
                )
                metric_value, _ = _resolve_selection_metric(
                    eval_result, resolved_spec.evaluation["selection_metric"], train_dataset.ticks_per_frame,
                )
                if best_tracker.offer(metric_value):
                    save_factory_checkpoint(
                        str(artifacts.checkpoints_dir / "best-validation.pt"), model, chunk_optimizer,
                        trainer_state=trainer_state_payload,
                        architecture_contract=architecture_contract, data_contract_hash=corpus.data_contract,
                        training_contract=training_contract, parent_checkpoint_sha=parent_checkpoint_sha,
                        training_stats={
                            "epoch": resume_state.epoch,
                            "selection_metric": resolved_spec.evaluation["selection_metric"],
                            "selection_metric_value": metric_value,
                        },
                        rng_state=resume_state.rng_state,
                    )
                    best_eval = eval_result

                # A cancellation request is honored at this same checkpoint
                # boundary as a graceful budget-exceeded stop: the chunk
                # just completed is still evaluated and checkpointed above,
                # only the next chunk never starts. Cancellation takes
                # priority in the unlikely event both fire on the same
                # chunk -- it is the more specific, deliberate request.
                cancelled = cancellation_requested(artifacts.directory)
                if decision.should_stop_now or cancelled:
                    completion_status = STATUS_CANCELLED if cancelled else STATUS_BUDGET_EXCEEDED
                    break

            if best_eval is None:
                # A resumed run whose seeded best (the pre-crash checkpoint)
                # was never beaten leaves checkpoints/best-validation.pt
                # untouched -- correctly still the pre-crash weights -- but
                # this call's own loop never captured its evaluation report.
                # Reconstruct it by re-evaluating that same checkpoint
                # rather than reporting the (worse) last-chunk model here.
                existing_best = load_factory_checkpoint(str(artifacts.checkpoints_dir / "best-validation.pt"))
                best_eval = _evaluate(
                    existing_best.model, validation_dataset, resolved_spec, cfg,
                    navigation_metrics=navigation_evaluation, reference_model=reference_model,
                )
            total_trial_seconds = time.monotonic() - trial_started
            budget_report = build_budget_report(
                budget, completion_status=completion_status, total_trial_seconds=total_trial_seconds,
                precision=str(resolved_spec.training["precision"]), tier=_tier_for_seconds(max_training_seconds),
            )
            write_budget_report(str(artifacts.metrics_dir / "budget_report.json"), budget_report)

            training_stats = dict(last_stats)
            training_stats.pop("resume_state", None)
            training_stats["completion_status"] = completion_status
            training_stats["evaluation_mode"] = "held_out"

            checkpoint_path = artifacts.checkpoints_dir / "best-validation.pt"
            checkpoint_meta = read_factory_checkpoint_metadata(str(checkpoint_path))
            validation_payload = {
                "format": "model-factory-validation-metrics-v1",
                "selection_metric": resolved_spec.evaluation["selection_metric"],
                "rollout": best_eval["rollout"],
                "direct": best_eval["direct"],
                "rollout_health": best_eval["rollout_health"],
                "per_episode_model_mse": best_eval["per_episode_model_mse"],
                "episode_ids": validation_episode_ids,
            }
            if "goal_navigation" in best_eval:
                validation_payload["goal_navigation"] = best_eval["goal_navigation"]
            atomic_write_json(artifacts.metrics_dir / "validation.json", validation_payload)

            if export_predictions:
                _export_clinic_predictions(
                    resolved_spec, checkpoint_path, validation_dataset, artifacts.experiment,
                    training_stats, max_episodes=export_predictions_max_episodes,
                )

            if retention_dataset is not None and retention_before_eval is not None:
                from sleep.forgetting import compute_forgetting_metric

                best_checkpoint = load_factory_checkpoint(str(checkpoint_path))
                retention_after_eval = _evaluate(
                    best_checkpoint.model, retention_dataset, resolved_spec, cfg,
                    reference_model=reference_model,
                )
                forgetting = compute_forgetting_metric(
                    _per_episode_retention_losses(retention_before_eval),
                    _per_episode_retention_losses(retention_after_eval),
                    old_scenario=str(retention_policy["suite"]),
                    new_scenario="goal_navigation_v1",
                    tolerance=float(retention_policy["max_regression"]),
                )
                retention_payload = forgetting.to_dict()
                retention_payload.update({
                    "metric": str(retention_policy["metric"]),
                    "corpus_id": str(retention_policy["corpus_id"]),
                    "episode_ids": retention_episode_ids,
                    "checkpoint_path": str(checkpoint_path),
                })
                atomic_write_json(artifacts.metrics_dir / "retention.json", retention_payload)

            comparison_payload: Optional[Dict[str, Any]] = None
            if resolved_spec.mode in ("clone", "fine_tune") and baseline_eval is not None:
                comparison_payload = _build_comparison(
                    resolved_spec, root, artifacts.trial_spec_path, best_eval, baseline_eval,
                    validation_episode_ids, train_dataset.ticks_per_frame,
                    tier=_tier_for_seconds(max_training_seconds),
                    measured_training_seconds=budget.elapsed_seconds,
                )
                atomic_write_json(artifacts.metrics_dir / "comparison.json", comparison_payload)

            experiment_report = _statistical_evaluation_module().build_experiment_report(
                experiment=dataclasses.asdict(artifacts.experiment),
                training_stats=training_stats,
                direct_metrics=best_eval["direct"], rollout_metrics=best_eval["rollout"],
                checkpoint={"path": str(checkpoint_path), "sha256": checkpoint_meta.get("checkpoint_sha256")},
                action_world_model_config=cfg, data_config=dict(resolved_spec.data),
            )
            report_path = str(artifacts.directory / "experiment_report.json")
            _statistical_evaluation_module().write_experiment_report(report_path, experiment_report)

        final_state = {
            STATUS_COMPLETED: STATE_COMPLETED,
            STATUS_BUDGET_EXCEEDED: STATE_BUDGET_EXCEEDED,
            STATUS_CANCELLED: STATE_CANCELLED,
        }[completion_status]
        transition(trial_state_path, final_state, reason=completion_status)
    except Exception as exc:
        transition(trial_state_path, STATE_FAILED, reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if devices:
            release_devices(root, artifacts.run_id)

    return TrialResult(
        run_id=artifacts.run_id, display_name=artifacts.display_name, mode=resolved_spec.mode,
        state=final_state, directory=artifacts.directory,
        architecture_hash=architecture_contract.hash, data_contract_hash=corpus.data_contract.hash,
        training_contract_hash=training_contract.hash,
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_meta.get("checkpoint_sha256"),
        training_stats=training_stats, evaluation=best_eval, budget_report=budget_report,
        comparison=comparison_payload, experiment_report_path=report_path,
    )


__all__ = ["TrialResult", "run_trial", "RUNS_ROOT_DEFAULT", "DEFAULT_EPOCH_BUDGET"]
