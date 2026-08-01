"""Immutable, quality-gated nursery corpora for Model Factory trials.

The nursery episode cache answers *which episode was requested*.  A corpus
answers *which exact bytes a trial is allowed to consume*.  That distinction
is intentional: a cache miss can record while building a corpus, but it is a
hard error while resolving one for a clone or search trial.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import yaml

from cognitive_runtime.record.quality import SplitOverlapReport, audit_split_overlap
from cognitive_runtime.training.action_effect_taxonomy import ACTION_EFFECT_LABEL_VERSION
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.contracts import DataContract, contract_hash
CORPUS_SPEC_FORMAT = "model-factory-corpus-spec-v1"
CORPUS_MANIFEST_FORMAT = "model-factory-corpus-manifest-v1"
QUALITY_REPORT_FORMAT = "model-factory-corpus-quality-v1"
SPLIT_OVERLAP_REPORT_FORMAT = "model-factory-corpus-split-overlap-v1"

#: Epic #212 Sec 12.6's "role of each split" field, applied when a spec
#: doesn't declare its own ``split_roles``.
DEFAULT_SPLIT_ROLES: Dict[str, str] = {
    "train": "generic_training",
    "validation": "validation",
    "test": "sealed_test",
}

# In-process convenience only.  The on-disk fallback in ``_corpus_directory``
# keeps resolution usable by a later process when the conventional root is
# used, while this permits callers to choose an explicit temporary root.
# Multiple roots may deliberately hold the same corpus ID, so an unqualified
# ID is usable only when it identifies exactly one known directory.
_KNOWN_CORPORA: Dict[str, set[Path]] = {}


@dataclass
class CorpusNurseryConfig:
    """Recording-relevant NurseryConfig shape that does not import torch.

    It intentionally mirrors :class:`training.nursery.NurseryConfig`.  The
    nursery recorder accesses this configuration structurally, so this keeps
    corpus manifest inspection and synthetic corpus tests available in the
    core-only install while preserving the recording configuration passed to
    a real nursery build.
    """

    train_seeds: Sequence[int] = (0, 1, 2, 3)
    holdout_seeds: Sequence[int] = (1000, 1001)
    episode_ticks: int = 400
    world_size: int = 48
    # Model Factory's default evidence source is the deterministic Crafter
    # nursery.  Minecraft remains available when a corpus spec requests it,
    # but must never be selected merely by omitting generator fields.
    world: str = "crafter"
    backend: str = "crafter"
    realtime: bool = False
    horizons: Sequence[int] = (1, 10, 100)
    latent_width: int = 32
    hidden_dim: int = 64
    reconstruction_size: int = 16
    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    max_train_samples: Optional[int] = None
    ssim_window: int = 3
    consistency_epochs: int = 15
    consistency_lr: float = 1e-3
    entity_persistence_epochs: int = 30
    data_quality_gate: bool = True
    split_overlap_gate: bool = False
    overfit_evaluation: bool = False
    max_corresponding_frame_fraction: float = 0.35
    export_predictions: bool = True
    expected_pixel_source: Optional[str] = None
    name: Optional[str] = None
    episode_cache_dir: Optional[str] = None
    navigation_random_action_fraction: float = 0.25


def _nursery_module():
    """Import the recording implementation only when a corpus is built.

    ``nursery`` owns the optional neural training path and imports torch.
    Corpus manifests themselves remain inspectable in Model Factory's
    core-only environment, where torch is deliberately unavailable.
    """
    from cognitive_runtime.training import nursery

    return nursery


def _record_or_reuse_scenario_episode(*args: Any, **kwargs: Any) -> str:
    return _nursery_module()._record_or_reuse_scenario_episode(*args, **kwargs)


def _scenarios_for_world(world: str) -> Mapping[str, Any]:
    return _nursery_module()._scenarios_for_world(world)


def validate_nursery_recordings(*args: Any, **kwargs: Any) -> list[str]:
    return _nursery_module().validate_nursery_recordings(*args, **kwargs)


@dataclass(frozen=True)
class ResolvedCorpus:
    """A verified, immutable corpus and the data contract it supplies."""

    corpus_id: str
    directory: Path
    manifest: Mapping[str, Any]
    data_contract: DataContract

    @property
    def data_contract_hash(self) -> str:
        return self.data_contract.hash


def _ordinary(value: Any) -> Any:
    """Convert dataclass/frozen containers to JSON-compatible ordinary data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _ordinary(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _ordinary(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_ordinary(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _config_from_spec(spec: Mapping[str, Any]) -> CorpusNurseryConfig:
    supplied = spec.get("generator", spec.get("nursery_config", {}))
    if isinstance(supplied, CorpusNurseryConfig):
        return supplied
    # Accept a caller-provided NurseryConfig without importing its
    # torch-dependent defining module here.  Both config classes have the
    # same public field shape and the recorder treats them structurally.
    if dataclasses.is_dataclass(supplied) and not isinstance(supplied, type):
        return CorpusNurseryConfig(**_ordinary(supplied))
    if not isinstance(supplied, Mapping):
        raise TypeError("corpus spec generator must be a mapping or NurseryConfig")
    allowed = {field.name for field in dataclasses.fields(CorpusNurseryConfig)}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"unknown NurseryConfig fields in corpus generator: {sorted(unknown)!r}")
    return CorpusNurseryConfig(**dict(supplied))


def _split_assignments(spec: Mapping[str, Any], cfg: CorpusNurseryConfig) -> Dict[str, Dict[str, tuple[int, ...]]]:
    """Normalize the explicit ``split -> scenario -> seeds`` corpus plan.

    ``splits`` is deliberately explicit so a future change cannot silently
    reassign an episode.  For a compact single-scenario caller, top-level
    ``scenario_names`` plus the nursery seed pools is accepted as a useful
    shorthand; that form gives validation its usual holdout split and leaves
    test empty.
    """
    raw = spec.get("splits")
    if raw is None:
        names = spec.get("scenario_names")
        if names is None:
            raise ValueError("corpus spec requires explicit splits or scenario_names")
        raw = {
            "train": {name: list(cfg.train_seeds) for name in names},
            "validation": {name: list(cfg.holdout_seeds) for name in names},
            "test": {},
        }
    if not isinstance(raw, Mapping):
        raise TypeError("corpus spec splits must be a mapping")
    result: Dict[str, Dict[str, tuple[int, ...]]] = {}
    for split in ("train", "validation", "test"):
        values = raw.get(split, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"corpus split {split!r} must map scenarios to seed lists")
        normalized: Dict[str, tuple[int, ...]] = {}
        for scenario, seeds in values.items():
            if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
                raise TypeError(f"corpus split {split!r} scenario {scenario!r} seeds must be a sequence")
            normalized[str(scenario)] = tuple(int(seed) for seed in seeds)
        result[split] = normalized
    unexpected = set(raw) - set(result)
    if unexpected:
        raise ValueError(f"unknown corpus split names: {sorted(unexpected)!r}")
    if not any(result[split] for split in result):
        raise ValueError("corpus has no sessions")
    return result


def _scenario_recording_config(
    cfg: CorpusNurseryConfig,
    splits: Mapping[str, Mapping[str, Sequence[int]]],
    scenario_name: str,
) -> CorpusNurseryConfig:
    """Bind a scenario's declared corpus splits to its nursery seed pools.

    Several scripted scenarios choose their train/held-out layout parameters
    through ``cfg.train_seeds`` and ``cfg.holdout_seeds``.  A corpus has an
    additional sealed-test split, which is held out for generation purposes
    just like validation.  Passing the spec defaults here made a valid test
    seed such as 2000 look unassigned and stopped a corpus build late in its
    recording pass.
    """
    train_seeds = tuple(int(seed) for seed in splits["train"].get(scenario_name, ()))
    held_out_seeds = tuple(
        int(seed)
        for split in ("validation", "test")
        for seed in splits[split].get(scenario_name, ())
    )
    return dataclasses.replace(
        cfg,
        train_seeds=train_seeds,
        holdout_seeds=held_out_seeds,
    )


def _scenario_mix_policy(spec: Mapping[str, Any], splits: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize an optional declared per-scenario data-collection mix
    (epic #212 Sec 12.2). Returns ``None`` when the spec declares none --
    the mix gate is opt-in, since not every corpus (e.g. a single-scenario
    canary-only corpus) has a percentage policy to check.
    """
    raw = spec.get("scenario_mix")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("corpus spec scenario_mix must be a mapping")
    split = str(raw.get("split", "train"))
    if split not in splits:
        raise ValueError(f"scenario_mix split {split!r} is not one of train/validation/test")
    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, Mapping) or not weights_raw:
        raise ValueError("corpus spec scenario_mix requires a non-empty weights mapping")
    weights = {str(scenario): float(fraction) for scenario, fraction in weights_raw.items()}
    unknown = set(weights) - set(splits[split])
    if unknown:
        raise ValueError(
            f"scenario_mix weights name scenarios not in the {split!r} split: {sorted(unknown)!r}"
        )
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError(f"scenario_mix weights must sum to 1.0, got {total_weight!r}")
    tolerance = float(raw.get("tolerance", 0.1))
    if tolerance < 0:
        raise ValueError(f"scenario_mix tolerance must be non-negative, got {tolerance!r}")
    return {"split": split, "weights": weights, "tolerance": tolerance}


def _scenario_mix_report(
    sessions: Mapping[str, Sequence[Mapping[str, Any]]],
    policy: Optional[Mapping[str, Any]],
) -> tuple[Optional[Dict[str, Any]], list[str]]:
    """Realised-vs-declared scenario mix.

    The declared percentages are "a starting data-collection policy, not a
    fixed scientific constant" (epic #212 Sec 12.2), but a build that misses
    them beyond the declared ``tolerance`` fails its quality gate rather
    than being silently accepted (issue #237's acceptance criterion) -- a
    generator bug or a rejected-episode retry loop that skews the realised
    mix must be visible, not just the requested one.
    """
    if not policy:
        return None, []
    split = str(policy["split"])
    weights: Mapping[str, float] = policy["weights"]
    tolerance = float(policy["tolerance"])
    entries = sessions.get(split, [])
    total = len(entries)
    counts: Dict[str, int] = {}
    for entry in entries:
        scenario = str(entry["scenario"])
        counts[scenario] = counts.get(scenario, 0) + 1
    realised = {
        scenario: (counts.get(scenario, 0) / total if total else 0.0)
        for scenario in weights
    }
    issues: list[str] = []
    for scenario, declared_fraction in weights.items():
        actual = realised[scenario]
        if abs(actual - declared_fraction) > tolerance:
            issues.append(
                f"scenario mix: {split!r} split's realised {scenario!r} fraction "
                f"{actual:.1%} is outside the declared {declared_fraction:.1%} "
                f"+/- {tolerance:.1%} tolerance"
            )
    report = {
        "format": "model-factory-corpus-scenario-mix-v1",
        "split": split,
        "declared": dict(weights),
        "tolerance": tolerance,
        "realised": realised,
        "realised_counts": counts,
        "total_sessions": total,
        "within_tolerance": not issues,
    }
    return report, issues


def _behavior_mixture_policy(
    spec: Mapping[str, Any], cfg: CorpusNurseryConfig, splits: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze the navigation expert/random schedule into the DataContract."""
    scenario_names = {
        str(name) for per_scenario in splits.values() for name in per_scenario
    }
    if not any(name.startswith("navigate_") or name == "replan_after_block" for name in scenario_names):
        return {}
    fraction = float(cfg.navigation_random_action_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"navigation_random_action_fraction must be between 0 and 1, got {fraction!r}"
        )
    raw = spec.get("behavior_mixture") or {}
    if not isinstance(raw, Mapping):
        raise TypeError("corpus spec behavior_mixture must be a mapping")
    declared_fraction = raw.get("random_action_fraction", fraction)
    if not math.isclose(float(declared_fraction), fraction, abs_tol=1e-12):
        raise ValueError(
            "behavior_mixture.random_action_fraction must match generator."
            "navigation_random_action_fraction"
        )
    implemented = {
        "expert_policy": "astar",
        "expert_action_fraction": 1.0 - fraction,
        "random_action_fraction": fraction,
        "random_action_subset": [
            "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
        ],
        "schedule": "seeded_stratified_per_active_step",
        "seed_rule": "episode_seed",
    }
    for key, value in raw.items():
        if key in implemented and value != implemented[key]:
            raise ValueError(
                f"behavior_mixture.{key}={value!r} contradicts the recorded "
                f"navigation policy {implemented[key]!r}"
            )
    return implemented


def _retention_policy(spec: Mapping[str, Any], splits: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_names = {
        str(name) for per_scenario in splits.values() for name in per_scenario
    }
    is_navigation = any(
        name.startswith("navigate_") or name == "replan_after_block"
        for name in scenario_names
    )
    raw = spec.get("retention_policy")
    if not is_navigation and raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("navigation corpus spec requires a retention_policy mapping")
    suite = str(raw.get("suite", ""))
    if suite != "generic_action_effects_v1":
        raise ValueError(
            "navigation retention_policy.suite must be 'generic_action_effects_v1'"
        )
    max_regression = float(raw.get("max_regression", -1.0))
    if max_regression < 0:
        raise ValueError("retention_policy.max_regression must be non-negative")
    replay = raw.get("replay_mixture")
    if not isinstance(replay, Mapping) or not replay:
        raise ValueError("retention_policy requires a non-empty replay_mixture")
    replay_mixture = {str(key): float(value) for key, value in replay.items()}
    if any(value < 0 for value in replay_mixture.values()) or not math.isclose(
        sum(replay_mixture.values()), 1.0, abs_tol=1e-6
    ):
        raise ValueError("retention_policy.replay_mixture values must be non-negative and sum to 1.0")
    metric = str(raw.get("metric", "heldout_prediction_loss"))
    if metric != "heldout_prediction_loss":
        raise ValueError(
            "navigation retention_policy.metric must be 'heldout_prediction_loss'"
        )
    corpus_id = str(raw.get("corpus_id", "crafter-generic-action-effects-v1"))
    if not corpus_id:
        raise ValueError("navigation retention_policy.corpus_id must not be empty")
    return {
        "suite": suite,
        "corpus_id": corpus_id,
        "metric": metric,
        "max_regression": max_regression,
        "replay_mixture": replay_mixture,
        "forgetting_metric": "sleep.forgetting.compute_forgetting_metric",
    }


def _corpus_spec_payload(spec: Mapping[str, Any], cfg: CorpusNurseryConfig, splits: Mapping[str, Any]) -> Dict[str, Any]:
    """The immutable declaration, excluding where the corpus happens to live."""
    generator = _ordinary(cfg)
    # The episode cache is an implementation location, not a content input.
    generator.pop("episode_cache_dir", None)
    payload = {
        "format": CORPUS_SPEC_FORMAT,
        "corpus_id": str(spec["corpus_id"]),
        "organism": str(spec["organism"]),
        "generator": generator,
        "splits": _ordinary(splits),
        "scenario_code_version": str(spec.get("scenario_code_version", "nursery-scenarios-v1")),
        "preprocessing_version": str(spec.get("preprocessing_version", "native")),
        "quality_policy": _ordinary(spec.get("quality_policy", {
            "enabled": bool(cfg.data_quality_gate),
            "expected_pixel_source": cfg.expected_pixel_source,
        })),
        "split_overlap_policy": _ordinary(spec.get("split_overlap_policy", {
            "max_corresponding_frame_fraction": cfg.max_corresponding_frame_fraction,
        })),
        # Epic #212 Sec 12.6's "role of each split" field.
        "split_roles": _ordinary(spec.get("split_roles", DEFAULT_SPLIT_ROLES)),
        # Epic #212 Sec 12.2's declared data-collection mix, when present.
        "scenario_mix_policy": _scenario_mix_policy(spec, splits),
        "behavior_mixture_policy": _behavior_mixture_policy(spec, cfg, splits),
        "retention_policy": _retention_policy(spec, splits),
    }
    return payload


def _corpus_directory(corpus_id: str, *, root: str | Path | None = None, organism: str | None = None) -> Path:
    direct = Path(corpus_id)
    if direct.is_dir():
        return direct.resolve()
    base = Path(root) if root is not None else Path("corpora")
    if organism is not None:
        return (base / organism / corpus_id).resolve()
    if len(direct.parts) > 1:
        return (base / direct).resolve()
    candidates = set(_KNOWN_CORPORA.get(corpus_id, set()))
    candidates.update(path.resolve() for path in base.glob(f"*/{corpus_id}") if path.is_dir())
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise FileNotFoundError(f"corpus {corpus_id!r} was not found under {base}")
    raise ValueError(f"corpus id {corpus_id!r} is ambiguous under {base}; use organism/corpus_id")


def _session_hash(session_dir: Path) -> str:
    """SHA-256 over every file that constitutes a recorded session.

    Path names are included as well as bytes, so replacing a stream or frame
    with a differently named file is detectable.  ``nursery_cache.json`` is
    identity metadata, not recorded content, and is intentionally excluded.
    """
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session directory is missing: {session_dir}")
    digest = hashlib.sha256()
    files = sorted(
        path for path in session_dir.rglob("*")
        if path.is_file() and path.name != "nursery_cache.json"
    )
    if not files:
        raise ValueError(f"session directory contains no record files: {session_dir}")
    for path in files:
        relative = path.relative_to(session_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _json_report(report: SplitOverlapReport) -> Dict[str, Any]:
    return _ordinary(report)


def _write_split_lists(directory: Path, sessions: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for split, entries in sessions.items():
        atomic_write_json(directory / f"{split}_sessions.json", {
            "format": "model-factory-corpus-session-list-v1",
            "split": split,
            "sessions": list(entries),
        })


def _session_quality_evidence(
    sessions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Read recorder-produced quality evidence without importing nursery.

    Session metadata is part of each frozen session hash.  Copying its
    declared generator parameters and realised action-effect mix into the
    corpus report and DataContract makes a corpus self-describing without
    making Model Factory's manifest-only import depend on torch.
    """
    evidence: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        for entry in sessions[split]:
            path = Path(str(entry["session_path"])) / "session.json"
            with path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            quality = metadata.get("quality_report")
            program_config = metadata.get("program_config") or {}
            generator = program_config.get("motor_babbling") or program_config.get("navigation")
            if quality is not None or generator is not None:
                evidence[str(entry["session_id"])] = {
                    "scenario": str(entry["scenario"]),
                    "split": split,
                    "quality_report": quality,
                    "generator": generator,
                }
    return evidence


def _scenario_generator_summary(
    generator_evidence: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per-scenario generator identity (epic #212 Sec 12.6), deduplicated
    from per-session evidence.

    Strips the fields that vary *per episode* by construction --
    ``generator_seed``, and ``layout_distribution``'s own
    ``layout_seed``/``realised_layout`` -- leaving only the generator's
    constant declared shape: name, version, action subset, burst
    distribution, and (for ``motor_babbling_walls``) the wall/layout
    distribution. Every episode of one scenario shares this shape by
    construction (it comes from that scenario's one generator function), so
    the first episode's cleaned evidence stands for the whole scenario.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for entry in generator_evidence.values():
        generator = entry.get("generator")
        if not generator:
            continue
        scenario = str(entry["scenario"])
        if scenario in summary:
            continue
        cleaned = {key: value for key, value in generator.items() if key != "generator_seed"}
        layout = cleaned.get("layout_distribution")
        if isinstance(layout, Mapping):
            cleaned["layout_distribution"] = {
                key: value for key, value in layout.items()
                if key not in ("layout_seed", "terrain_seed", "realised_layout")
            }
        summary[scenario] = cleaned
    # Navigation layouts deliberately vary by split and seed.  Keep the
    # per-session realised cells in ``program_config`` evidence above, while
    # making this per-scenario summary describe the *distribution* rather
    # than accidentally presenting the first episode's wall coordinates as
    # a constant generator parameter.
    for scenario in list(summary):
        navigation_layouts = [
            entry["generator"].get("layout_distribution") or {}
            for entry in generator_evidence.values()
            if entry.get("scenario") == scenario
            and entry.get("generator")
            and entry["generator"].get("behavior_mixture")
        ]
        if not navigation_layouts:
            continue
        families_by_split: Dict[str, set[str]] = {}
        for layout in navigation_layouts:
            split = str(layout.get("split", "unspecified"))
            family = layout.get("layout_family")
            if family:
                families_by_split.setdefault(split, set()).add(str(family))
        summary[scenario]["layout_distribution"] = {
            "layout_families_by_split": {
                split: sorted(families) for split, families in sorted(families_by_split.items())
            },
            "solvability_check": "astar_before_recording",
            "per_episode_realisations": "program_config.scenario_generator_evidence",
            "dynamic_block": next(
                (layout.get("dynamic_block") for layout in navigation_layouts if layout.get("dynamic_block")),
                None,
            ),
        }
    return summary


def _navigation_contract_policy(
    generator_evidence: Mapping[str, Mapping[str, Any]], key: str,
) -> Dict[str, Any]:
    """One navigation policy declaration, requiring cross-session identity."""
    declarations = [
        dict(entry["generator"][key])
        for entry in generator_evidence.values()
        if entry.get("generator") and entry["generator"].get(key)
    ]
    if not declarations:
        return {}
    first = declarations[0]
    if any(declaration != first for declaration in declarations[1:]):
        raise ValueError(f"navigation sessions disagree on declared {key}")
    return first


def _data_contract(spec: Mapping[str, Any], sessions: Mapping[str, Sequence[Mapping[str, Any]]]) -> DataContract:
    generator = spec["generator"]
    quality_policy = spec["quality_policy"]
    all_entries = [entry for split in ("train", "validation", "test") for entry in sessions[split]]
    program_config = dict(generator)
    generator_evidence = _session_quality_evidence(sessions)
    if generator_evidence:
        # The session id keys make the seed-specific layout realisation
        # explicit.  This includes the declared layout distribution and mix
        # bounds, not just post-hoc aggregate counts.
        program_config["scenario_generator_evidence"] = generator_evidence
    is_crafter = generator["world"] == "crafter"
    behavior_mixture_policy = dict(spec.get("behavior_mixture_policy") or {})
    recorded_behavior_mixture = _navigation_contract_policy(
        generator_evidence, "behavior_mixture",
    )
    if behavior_mixture_policy and recorded_behavior_mixture != behavior_mixture_policy:
        raise ValueError(
            "navigation session behavior_mixture evidence does not match the corpus declaration: "
            f"recorded={recorded_behavior_mixture!r}, declared={behavior_mixture_policy!r}"
        )
    return DataContract(
        world=str(generator["world"]),
        backend=str(generator["backend"]),
        program_config=program_config,
        scenario_names=tuple(sorted({str(entry["scenario"]) for entry in all_entries})),
        scenario_code_version=str(spec["scenario_code_version"]),
        train_session_ids=tuple(str(entry["session_id"]) for entry in sessions["train"]),
        train_session_hashes=tuple(str(entry["sha256"]) for entry in sessions["train"]),
        validation_session_ids=tuple(str(entry["session_id"]) for entry in sessions["validation"]),
        validation_session_hashes=tuple(str(entry["sha256"]) for entry in sessions["validation"]),
        test_session_ids=tuple(str(entry["session_id"]) for entry in sessions["test"]),
        test_session_hashes=tuple(str(entry["sha256"]) for entry in sessions["test"]),
        seed_assignments={str(entry["session_id"]): int(entry["seed"]) for entry in all_entries},
        pixel_provenance=str(
            quality_policy.get("expected_pixel_source", generator.get("expected_pixel_source"))
            or "unspecified"
        ),
        semantic_vocabulary_version=(
            _nursery_module().SEMANTIC_VOCABULARY_VERSION if is_crafter else "none"
        ),
        preprocessing_version=str(spec["preprocessing_version"]),
        horizons_ticks=tuple(int(value) for value in generator["horizons"]),
        ticks_per_frame=1.0,
        quality_policy=quality_policy,
        split_overlap_policy=spec["split_overlap_policy"],
        scenario_generators=_scenario_generator_summary(generator_evidence),
        action_effect_label_schema_version=(ACTION_EFFECT_LABEL_VERSION if is_crafter else ""),
        split_roles=dict(spec.get("split_roles", DEFAULT_SPLIT_ROLES)),
        scenario_mix_policy=dict(spec.get("scenario_mix_policy") or {}),
        goal_distribution_policy=_navigation_contract_policy(
            generator_evidence, "goal_distribution_policy",
        ),
        oracle_planner_policy=_navigation_contract_policy(
            generator_evidence, "oracle_planner_policy",
        ),
        goal_reward_policy=_navigation_contract_policy(
            generator_evidence, "goal_reward_policy",
        ),
        behavior_mixture_policy=recorded_behavior_mixture,
        retention_policy=dict(spec.get("retention_policy") or {}),
    )


def _check_split_sets(sessions: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    ids = {split: {str(entry["session_id"]) for entry in entries} for split, entries in sessions.items()}
    paths = {
        split: {str(Path(entry["session_path"]).resolve()) for entry in entries}
        for split, entries in sessions.items()
    }
    issues = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(ids[left] & ids[right])
        if overlap:
            issues.append(f"{left} and {right} splits share session IDs: {overlap}")
        path_overlap = sorted(paths[left] & paths[right])
        if path_overlap:
            issues.append(f"{left} and {right} splits share session paths: {path_overlap}")
    return issues


def _verify_manifest_sessions(
    corpus_id: str, sessions: Any, data_contract: DataContract,
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    """Ensure the consumable session list is exactly the contracted evidence."""
    if not isinstance(sessions, Mapping) or set(sessions) != {"train", "validation", "test"}:
        raise ValueError(f"corpus {corpus_id!r} manifest must contain exactly train, validation, and test sessions")
    contract_splits = {
        "train": (data_contract.train_session_ids, data_contract.train_session_hashes),
        "validation": (data_contract.validation_session_ids, data_contract.validation_session_hashes),
        "test": (data_contract.test_session_ids, data_contract.test_session_hashes),
    }
    for split, (contract_ids, contract_hashes) in contract_splits.items():
        entries = sessions[split]
        if not isinstance(entries, list) or not all(isinstance(entry, Mapping) for entry in entries):
            raise ValueError(f"corpus {corpus_id!r} manifest {split!r} sessions are invalid")
        manifest_pairs = tuple(
            (str(entry.get("session_id")), str(entry.get("sha256"))) for entry in entries
        )
        contract_pairs = tuple(zip(map(str, contract_ids), map(str, contract_hashes)))
        if manifest_pairs != contract_pairs:
            raise ValueError(
                f"corpus {corpus_id!r} manifest {split!r} sessions do not match its data contract"
            )
    return sessions


def load_corpus_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a raw corpus spec document from ``.yaml``/``.yml``/``.json``.

    Mirrors ``spec.load_spec``'s input-convenience contract: YAML/JSON is
    accepted input, the plain mapping is returned unresolved and
    unvalidated -- :func:`build_corpus` is what freezes and gates it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"unsupported corpus spec file extension {suffix!r} for {path}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"corpus spec file {path} must contain a mapping at the top level")
    return dict(raw)


def build_corpus(spec: Mapping[str, Any]) -> ResolvedCorpus:
    """Record/reuse all requested episodes, gate them, then freeze their hashes.

    Required keys are ``corpus_id`` and ``organism``.  ``root`` defaults to
    ``corpora`` and ``generator`` accepts ``NurseryConfig`` fields.  Splits
    are ``{"train": {scenario: [seed, ...]}, ...}``.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("corpus spec must be a mapping")
    for key in ("corpus_id", "organism"):
        if not spec.get(key):
            raise ValueError(f"corpus spec requires {key!r}")
    cfg = _config_from_spec(spec)
    splits = _split_assignments(spec, cfg)
    frozen_spec = _corpus_spec_payload(spec, cfg, splits)
    directory = (Path(spec.get("root", "corpora")) / str(spec["organism"]) / str(spec["corpus_id"])).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _KNOWN_CORPORA.setdefault(str(spec["corpus_id"]), set()).add(directory)
    spec_path = directory / "corpus_spec.json"
    if spec_path.exists():
        with spec_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != frozen_spec:
            raise ValueError(
                f"corpus id {spec['corpus_id']!r} already has a different generator declaration; "
                "choose a new corpus_id"
            )
        if (directory / "corpus_manifest.json").is_file():
            # The content manifest is the corpus identity.  A second build
            # of the same declaration must never re-record a remote or
            # non-deterministic episode and replace those frozen bytes.
            return resolve_corpus(
                str(spec["corpus_id"]), root=spec.get("root"), organism=str(spec["organism"]),
            )
    else:
        atomic_write_json(spec_path, frozen_spec)

    scenarios = _scenarios_for_world(cfg.world)
    sessions: Dict[str, list[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    quality_issues: list[str] = []
    for split, per_scenario in splits.items():
        for scenario_name, seeds in per_scenario.items():
            try:
                scenario = scenarios[scenario_name]
            except KeyError as exc:
                raise ValueError(f"unknown {cfg.world} nursery scenario {scenario_name!r}") from exc
            paths: list[str] = []
            scenario_cfg = _scenario_recording_config(cfg, splits, scenario_name)
            for seed in seeds:
                session_id = f"{scenario_name}-{split}-{seed}"
                session_dir = Path(_record_or_reuse_scenario_episode(
                    str(directory / "recordings"), session_id, seed, scenario, scenario_cfg,
                )).resolve()
                paths.append(str(session_dir))
                sessions[split].append({
                    "session_id": session_id,
                    "session_path": str(session_dir),
                    "scenario": scenario_name,
                    "seed": seed,
                    "sha256": _session_hash(session_dir),
                })
            if frozen_spec["quality_policy"].get("enabled", True):
                quality_issues.extend(validate_nursery_recordings(
                    paths,
                    scenario,
                    expected_pixel_source=frozen_spec["quality_policy"].get(
                        "expected_pixel_source", cfg.expected_pixel_source,
                    ),
                ))

    mix_report, mix_issues = _scenario_mix_report(sessions, frozen_spec.get("scenario_mix_policy"))
    quality_issues.extend(mix_issues)

    set_issues = _check_split_sets(sessions)
    pair_reports: Dict[str, Dict[str, Any]] = {}
    overlap_issues = list(set_issues)
    threshold = float(frozen_spec["split_overlap_policy"].get(
        "max_corresponding_frame_fraction", cfg.max_corresponding_frame_fraction,
    ))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        report = audit_split_overlap(
            [entry["session_path"] for entry in sessions[left]],
            [entry["session_path"] for entry in sessions[right]],
            max_corresponding_frame_fraction=threshold,
        )
        pair_reports[f"{left}_vs_{right}"] = _json_report(report)
        overlap_issues.extend(report.issues)

    quality_payload = {
        "format": QUALITY_REPORT_FORMAT,
        "corpus_id": str(spec["corpus_id"]),
        "passed": not quality_issues,
        "issues": quality_issues,
        "action_effect_evidence": _session_quality_evidence(sessions),
        "scenario_mix_report": mix_report,
    }
    split_payload = {
        "format": SPLIT_OVERLAP_REPORT_FORMAT,
        "corpus_id": str(spec["corpus_id"]),
        "session_ids": {split: [entry["session_id"] for entry in entries] for split, entries in sessions.items()},
        "disjoint": not overlap_issues,
        "issues": overlap_issues,
        "pair_reports": pair_reports,
    }
    atomic_write_json(directory / "quality_report.json", quality_payload)
    atomic_write_json(directory / "split_overlap_report.json", split_payload)
    if quality_issues or overlap_issues:
        problems = quality_issues + overlap_issues
        raise ValueError(f"corpus {spec['corpus_id']!r} did not pass its gates: {'; '.join(problems)}")

    data_contract = _data_contract(frozen_spec, sessions)
    manifest = {
        "format": CORPUS_MANIFEST_FORMAT,
        "corpus_id": str(spec["corpus_id"]),
        "spec_hash": contract_hash(frozen_spec),
        "label_versions": {"semantic_vocabulary": data_contract.semantic_vocabulary_version},
        "sessions": sessions,
        "data_contract": _ordinary(data_contract.to_dict()),
        "data_contract_hash": data_contract.hash,
    }
    atomic_write_json(directory / "corpus_manifest.json", manifest)
    _write_split_lists(directory, sessions)
    return resolve_corpus(str(spec["corpus_id"]), root=spec.get("root"), organism=str(spec["organism"]))


def resolve_corpus(
    corpus_id: str,
    *,
    allow_record: bool = False,
    root: str | Path | None = None,
    organism: str | None = None,
) -> ResolvedCorpus:
    """Load a corpus only if every frozen session still has its exact hash.

    ``allow_record`` exists to make the policy explicit.  Recording a missing
    session is deliberately unsupported here: callers must invoke
    :func:`build_corpus` to create a fresh, quality-gated corpus instead.
    """
    if allow_record:
        raise ValueError("resolve_corpus cannot record replacement episodes; call build_corpus for a new corpus")
    directory = _corpus_directory(corpus_id, root=root, organism=organism)
    manifest_path = directory / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"corpus {corpus_id!r} has no frozen manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != CORPUS_MANIFEST_FORMAT:
        raise ValueError(f"not a {CORPUS_MANIFEST_FORMAT} manifest: {manifest_path}")
    try:
        data_contract = DataContract(**manifest["data_contract"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"corpus {corpus_id!r} has an invalid data contract") from exc
    if data_contract.hash != manifest.get("data_contract_hash"):
        raise ValueError(f"corpus {corpus_id!r} data contract hash does not match its frozen manifest")
    sessions = _verify_manifest_sessions(corpus_id, manifest.get("sessions"), data_contract)
    for split, entries in sessions.items():
        for entry in entries:
            session_id = entry.get("session_id", "?")
            session_path = Path(entry.get("session_path", ""))
            if not session_path.is_dir():
                raise FileNotFoundError(
                    f"corpus {corpus_id!r} session {session_id!r} is missing: {session_path}"
                )
            actual = _session_hash(session_path)
            if actual != entry.get("sha256"):
                raise ValueError(
                    f"corpus {corpus_id!r} session {session_id!r} content hash changed "
                    f"(expected {entry.get('sha256')}, got {actual})"
                )
    return ResolvedCorpus(str(manifest.get("corpus_id", corpus_id)), directory, manifest, data_contract)


__all__ = [
    "CORPUS_SPEC_FORMAT",
    "CORPUS_MANIFEST_FORMAT",
    "QUALITY_REPORT_FORMAT",
    "SPLIT_OVERLAP_REPORT_FORMAT",
    "DEFAULT_SPLIT_ROLES",
    "ResolvedCorpus",
    "build_corpus",
    "load_corpus_spec",
    "resolve_corpus",
]
