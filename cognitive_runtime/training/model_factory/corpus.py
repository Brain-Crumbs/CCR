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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from cognitive_runtime.record.quality import SplitOverlapReport, audit_split_overlap
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.contracts import DataContract, contract_hash
CORPUS_SPEC_FORMAT = "model-factory-corpus-spec-v1"
CORPUS_MANIFEST_FORMAT = "model-factory-corpus-manifest-v1"
QUALITY_REPORT_FORMAT = "model-factory-corpus-quality-v1"
SPLIT_OVERLAP_REPORT_FORMAT = "model-factory-corpus-split-overlap-v1"

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


def _data_contract(spec: Mapping[str, Any], sessions: Mapping[str, Sequence[Mapping[str, Any]]]) -> DataContract:
    generator = spec["generator"]
    quality_policy = spec["quality_policy"]
    all_entries = [entry for split in ("train", "validation", "test") for entry in sessions[split]]
    return DataContract(
        world=str(generator["world"]),
        backend=str(generator["backend"]),
        program_config=generator,
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
            _nursery_module().SEMANTIC_VOCABULARY_VERSION
            if generator["world"] == "crafter" else "none"
        ),
        preprocessing_version=str(spec["preprocessing_version"]),
        horizons_ticks=tuple(int(value) for value in generator["horizons"]),
        ticks_per_frame=1.0,
        quality_policy=quality_policy,
        split_overlap_policy=spec["split_overlap_policy"],
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
            for seed in seeds:
                session_id = f"{scenario_name}-{split}-{seed}"
                session_dir = Path(_record_or_reuse_scenario_episode(
                    str(directory / "recordings"), session_id, seed, scenario, cfg,
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
    "ResolvedCorpus",
    "build_corpus",
    "resolve_corpus",
]
