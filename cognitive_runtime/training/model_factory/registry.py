"""JSON-backed champion registry: tiered portfolio, atomic store, lineage
graph (epic #212 Phase C step 4, §7, §15.1, issue #224).

A registry holds references only -- paths, hashes, and metrics -- and never
copied model weights. It is keyed by ``(family, tier, objective)``: a
**portfolio**, not a single global champion. A ``scale`` result never
silently replaces a ``fast`` champion for the same family and objective --
different tiers are always different registry slots (§15.1):

```text
generic_action_effects_v1 / fast  / rollout.t+4
generic_action_effects_v1 / scale / rollout.t+4
goal_navigation_v1        / fast  / success_rate
```

Each slot records a designated leading champion plus the retained champion
population (populated by MF-D5's genetic search). Writes are atomic and
locked, reusing MF-B5's (:mod:`cognitive_runtime.training.model_factory.state`)
single-process file lock and MF-A3's
(:func:`cognitive_runtime.training.model_factory.artifacts.atomic_write_json`)
temp-write/rename primitive, so two concurrent promotions can never interleave
into a corrupt document.

``lineage_graph`` is a separate concern from the registry document: it walks
each run's own persisted ``lineage.json`` (written by ``allocate_run_artifacts``)
to recover the full ancestor chain -- including, for a bred child, both
configuration parents and the weight donor -- and terminates on an injected
cycle rather than recursing forever.

Pure Python + filesystem -- no torch -- so this module is usable in the
core-only install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from cognitive_runtime.training.model_factory.artifacts import LINEAGE_FORMAT, atomic_write_json
from cognitive_runtime.training.model_factory.budget import BUDGET_TIERS
from cognitive_runtime.training.model_factory.population import (
    PopulationCandidate,
    PopulationDecision,
    PopulationPolicy,
    replace_population,
)
from cognitive_runtime.training.model_factory.state import (
    _lock_path_for,
    _locked,
    load_state,
    require_completed_for_promotion,
    state_path,
)

REGISTRY_FORMAT = "model-factory-registry-v1"
LINEAGE_MANIFEST_NAME = "lineage.json"

DECISION_PROMOTE = "promote"
DECISION_HOLD = "hold"
DECISION_TEST_USE = "record_test_use"

_JSON_SCALAR_TYPES = (str, int, bool, type(None))


class RegistryError(ValueError):
    """A malformed registry document, lineage manifest, or invalid operation."""


class LineageCycleError(RegistryError):
    """Raised when :func:`lineage_graph` detects a cycle instead of
    recursing forever."""


class TestBudgetExhaustedError(RegistryError):
    """Raised by :func:`record_test_use` when a caller-supplied ``max_uses``
    cap is already spent for ``run_id`` (epic §10.3's sealed-test-use
    budget)."""

    def __init__(self, run_id: str, max_uses: int, current_uses: int) -> None:
        self.run_id = run_id
        self.max_uses = max_uses
        self.current_uses = current_uses
        super().__init__(
            f"sealed-test-use budget exhausted for run {run_id!r}: "
            f"{current_uses} of {max_uses} action(s) already recorded; refusing this test action"
        )


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def registry_path(root: Union[str, Path]) -> Path:
    """The conventional registry document path for one organism/factory root."""
    return Path(root) / "registry.json"


def _validate_slot_key(family: str, tier: str, objective: str) -> None:
    if not isinstance(family, str) or not family:
        raise RegistryError("family must be a non-empty string")
    if not isinstance(objective, str) or not objective:
        raise RegistryError("objective must be a non-empty string")
    if tier not in BUDGET_TIERS:
        raise RegistryError(
            f"unknown runtime budget tier {tier!r}; expected one of {sorted(BUDGET_TIERS)}"
        )


def _validate_registry_safe(value: Any, *, path: str) -> None:
    """Reject anything that is not a plain JSON scalar/container.

    The registry stores only paths, hashes and metrics (epic §7: "references
    only, never copied model weights"). A tensor, a torch module, or any
    other non-JSON-safe payload is rejected here with an actionable message
    rather than failing deep inside ``json.dump`` -- or, worse, silently
    coerced into something that looks like a metric.
    """
    if isinstance(value, _JSON_SCALAR_TYPES):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegistryError(f"{path} must be finite, got {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RegistryError(f"{path} has a non-string key {key!r}")
            _validate_registry_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_registry_safe(item, path=f"{path}[{index}]")
        return
    raise RegistryError(
        f"{path} has an unsupported type {type(value).__name__!r}; the registry "
        "stores only JSON-safe paths, hashes and metrics, never tensors or "
        "model weights"
    )


@dataclasses.dataclass(frozen=True)
class ChampionEntry:
    """One population member: a reference to a checkpoint, never its weights."""

    run_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    metrics: Mapping[str, Any]
    promoted_at: str
    test_uses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metrics": dict(self.metrics),
            "promoted_at": self.promoted_at,
            "test_uses": self.test_uses,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChampionEntry":
        return cls(
            run_id=str(payload["run_id"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            metrics=dict(payload.get("metrics") or {}),
            promoted_at=str(payload["promoted_at"]),
            test_uses=int(payload.get("test_uses", 0)),
        )


@dataclasses.dataclass(frozen=True)
class RegistrySlot:
    """One ``(family, tier, objective)`` portfolio entry."""

    family: str
    tier: str
    objective: str
    leading_champion: Optional[str]
    population: Tuple[ChampionEntry, ...]
    history: Tuple[Mapping[str, Any], ...]

    def leading_entry(self) -> Optional[ChampionEntry]:
        if self.leading_champion is None:
            return None
        for entry in self.population:
            if entry.run_id == self.leading_champion:
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "tier": self.tier,
            "objective": self.objective,
            "leading_champion": self.leading_champion,
            "population": [entry.to_dict() for entry in self.population],
            "history": [dict(entry) for entry in self.history],
        }

    @classmethod
    def from_dict(cls, family: str, tier: str, objective: str, payload: Mapping[str, Any]) -> "RegistrySlot":
        return cls(
            family=family,
            tier=tier,
            objective=objective,
            leading_champion=payload.get("leading_champion"),
            population=tuple(
                ChampionEntry.from_dict(item) for item in payload.get("population") or ()
            ),
            history=tuple(dict(item) for item in payload.get("history") or ()),
        )


def _empty_document() -> Dict[str, Any]:
    return {"format": REGISTRY_FORMAT, "slots": {}}


def _read_document(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_document()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != REGISTRY_FORMAT:
        raise RegistryError(f"not a Model Factory registry document: {path}")
    if not isinstance(payload.get("slots"), dict):
        raise RegistryError(f"invalid registry document (missing 'slots'): {path}")
    return payload


def _get_slot_payload(
    document: Mapping[str, Any], family: str, tier: str, objective: str,
) -> Optional[Dict[str, Any]]:
    return document.get("slots", {}).get(family, {}).get(tier, {}).get(objective)


def _set_slot_payload(
    document: Dict[str, Any], family: str, tier: str, objective: str, slot_payload: Dict[str, Any],
) -> None:
    slots = document.setdefault("slots", {})
    slots.setdefault(family, {}).setdefault(tier, {})[objective] = slot_payload


def _new_slot_payload(family: str, tier: str, objective: str) -> Dict[str, Any]:
    return {
        "family": family,
        "tier": tier,
        "objective": objective,
        "leading_champion": None,
        "population": [],
        "history": [],
    }


def promote(
    root: Union[str, Path],
    *,
    family: str,
    tier: str,
    objective: str,
    run_id: str,
    checkpoint_path: Union[str, Path],
    checkpoint_sha256: str,
    metrics: Mapping[str, Any],
    as_leading: bool = True,
    reason: Optional[str] = None,
    now: Callable[[], str] = _now_iso,
    population_policy: Optional[PopulationPolicy] = None,
    population_candidates: Optional[Tuple[PopulationCandidate, ...]] = None,
) -> RegistrySlot:
    """Record ``run_id`` as a champion-population member of one exact
    ``(family, tier, objective)`` slot -- by default also its new leading
    champion.

    Different tiers are always different slots (epic §15.1): promoting a
    ``scale`` result never touches the ``fast`` slot for the same family and
    objective. Re-promoting an existing population member's ``run_id``
    replaces that member's checkpoint/metrics but preserves its accumulated
    ``test_uses`` ledger rather than resetting it to zero.

    When ``population_policy`` and ``population_candidates`` are supplied,
    the candidate plus the current slot are replaced atomically through
    :func:`population.replace_population`.  This is the production boundary
    that enforces the bounded, deterministic D5 policy rather than merely
    exposing it as a standalone helper.  Supplying only one of those two
    arguments is rejected so a caller cannot accidentally bypass policy
    enforcement.

    ``<root>/<run_id>/state.json`` must record MF-B5's ``completed`` trial
    state (epic §16: "never lets an incomplete artifact enter promotion").
    A running, failed, budget-exceeded, cancelled, or nonexistent run raises
    rather than entering the champion population.
    """
    _validate_slot_key(family, tier, objective)
    if not run_id:
        raise RegistryError("run_id must not be empty")
    if not checkpoint_sha256:
        raise RegistryError("checkpoint_sha256 must not be empty")
    if (population_policy is None) != (population_candidates is None):
        raise RegistryError(
            "population_policy and population_candidates must be supplied together"
        )
    _validate_registry_safe(dict(metrics), path="metrics")
    require_completed_for_promotion(load_state(state_path(Path(root) / run_id)))

    path = registry_path(root)
    with _locked(_lock_path_for(path)):
        document = _read_document(path)
        slot_payload = _get_slot_payload(document, family, tier, objective) or _new_slot_payload(
            family, tier, objective,
        )
        previous_test_uses = 0
        for item in slot_payload.get("population", []):
            if item.get("run_id") == run_id:
                previous_test_uses = int(item.get("test_uses", 0))
                break
        entry = ChampionEntry(
            run_id=run_id,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            metrics=dict(metrics),
            promoted_at=now(),
            test_uses=previous_test_uses,
        )
        population_entries = [
            ChampionEntry.from_dict(item) for item in slot_payload.get("population", [])
            if item.get("run_id") != run_id
        ]
        population_entries.append(entry)
        population_decision: Optional[PopulationDecision] = None
        if population_policy is not None and population_candidates is not None:
            candidate_ids = {candidate.run_id for candidate in population_candidates}
            missing = sorted(entry.run_id for entry in population_entries if entry.run_id not in candidate_ids)
            if missing:
                raise RegistryError(
                    "population policy requires a selection candidate for every existing "
                    f"registry member; missing {missing!r}"
                )
            if run_id not in candidate_ids:
                raise RegistryError(
                    f"population policy candidates do not include the promoted run {run_id!r}"
                )
            try:
                population_decision = replace_population(population_candidates, population_policy)
            except ValueError as exc:
                raise RegistryError(f"invalid population replacement request: {exc}") from exc
            entries_by_run_id = {member.run_id: member for member in population_entries}
            population_entries = [
                entries_by_run_id[member.run_id] for member in population_decision.members
            ]
            slot_payload["leading_champion"] = population_decision.leading_champion
        else:
            if as_leading:
                slot_payload["leading_champion"] = run_id
        slot_payload["population"] = [member.to_dict() for member in population_entries]
        slot_payload["history"] = list(slot_payload.get("history", [])) + [{
            "action": DECISION_PROMOTE,
            "run_id": run_id,
            "at": entry.promoted_at,
            "as_leading": (
                population_decision.leading_champion == run_id
                if population_decision is not None else as_leading
            ),
            "retained": run_id in {member.run_id for member in population_entries},
            "reason": reason,
            "population_replacement": (
                population_decision.to_dict() if population_decision is not None else None
            ),
        }]
        _set_slot_payload(document, family, tier, objective, slot_payload)
        atomic_write_json(path, document)
        return RegistrySlot.from_dict(family, tier, objective, slot_payload)


def hold(
    root: Union[str, Path],
    *,
    family: str,
    tier: str,
    objective: str,
    run_id: str,
    reason: Optional[str] = None,
    now: Callable[[], str] = _now_iso,
) -> RegistrySlot:
    """Record that ``run_id`` was evaluated for this slot and *not* promoted.

    Never touches ``leading_champion`` or ``population`` -- this is an
    auditable decision entry only, so a "why wasn't this candidate promoted"
    question stays answerable from the registry itself.
    """
    _validate_slot_key(family, tier, objective)
    if not run_id:
        raise RegistryError("run_id must not be empty")

    path = registry_path(root)
    with _locked(_lock_path_for(path)):
        document = _read_document(path)
        slot_payload = _get_slot_payload(document, family, tier, objective) or _new_slot_payload(
            family, tier, objective,
        )
        slot_payload["history"] = list(slot_payload.get("history", [])) + [{
            "action": DECISION_HOLD,
            "run_id": run_id,
            "at": now(),
            "reason": reason,
        }]
        _set_slot_payload(document, family, tier, objective, slot_payload)
        atomic_write_json(path, document)
        return RegistrySlot.from_dict(family, tier, objective, slot_payload)


def leading_champion(
    root: Union[str, Path], *, family: str, tier: str, objective: str,
) -> Optional[ChampionEntry]:
    """The current leading champion for one slot, or ``None`` if the slot
    has never been promoted into."""
    document = _read_document(registry_path(root))
    slot_payload = _get_slot_payload(document, family, tier, objective)
    if slot_payload is None:
        return None
    return RegistrySlot.from_dict(family, tier, objective, slot_payload).leading_entry()


def population(
    root: Union[str, Path], *, family: str, tier: str, objective: str,
) -> Tuple[ChampionEntry, ...]:
    """The retained champion population for one slot, in insertion order."""
    document = _read_document(registry_path(root))
    slot_payload = _get_slot_payload(document, family, tier, objective)
    if slot_payload is None:
        return ()
    return tuple(ChampionEntry.from_dict(item) for item in slot_payload.get("population", []))


def record_test_use(
    root: Union[str, Path],
    *,
    family: str,
    tier: str,
    objective: str,
    run_id: str,
    reason: Optional[str] = None,
    max_uses: Optional[int] = None,
    now: Callable[[], str] = _now_iso,
) -> ChampionEntry:
    """Charge one sealed-test-set use against ``run_id``'s recorded ledger
    (epic §10.3: durable promotion "test actions are recorded against a
    small explicit test-use budget").

    ``run_id`` must already be a population member of the named slot --
    test uses are attributed to an evaluated candidate, never an unrelated
    identifier.

    ``max_uses``, when supplied, caps this run_id's ledger. The cap check
    and the increment happen inside the same locked critical section as the
    read, so two callers racing to charge the final remaining use can never
    both observe the pre-increment count and both succeed -- exactly the
    check-then-act race a budget this small (epic §10.3) cannot tolerate.
    Raises :class:`TestBudgetExhaustedError` (without writing) once the cap
    is already spent.
    """
    _validate_slot_key(family, tier, objective)
    if max_uses is not None and max_uses <= 0:
        raise RegistryError("max_uses must be positive when supplied")

    path = registry_path(root)
    with _locked(_lock_path_for(path)):
        document = _read_document(path)
        slot_payload = _get_slot_payload(document, family, tier, objective)
        if slot_payload is None:
            raise RegistryError(
                f"no registry slot for family={family!r} tier={tier!r} objective={objective!r}"
            )
        rebuilt: List[ChampionEntry] = []
        updated: Optional[ChampionEntry] = None
        found = False
        for member in (ChampionEntry.from_dict(item) for item in slot_payload.get("population", [])):
            if member.run_id == run_id:
                found = True
                if max_uses is not None and member.test_uses >= max_uses:
                    raise TestBudgetExhaustedError(run_id, max_uses, member.test_uses)
                updated = dataclasses.replace(member, test_uses=member.test_uses + 1)
                rebuilt.append(updated)
            else:
                rebuilt.append(member)
        if not found:
            raise RegistryError(
                f"run_id {run_id!r} is not a population member of "
                f"family={family!r} tier={tier!r} objective={objective!r}"
            )
        slot_payload["population"] = [member.to_dict() for member in rebuilt]
        slot_payload["history"] = list(slot_payload.get("history", [])) + [{
            "action": DECISION_TEST_USE,
            "run_id": run_id,
            "at": now(),
            "reason": reason,
        }]
        _set_slot_payload(document, family, tier, objective, slot_payload)
        atomic_write_json(path, document)
        assert updated is not None
        return updated


# --------------------------------------------------------------------- lineage graph


@dataclasses.dataclass(frozen=True)
class LineageNode:
    """One visited run's lineage edges, as recorded in its own ``lineage.json``."""

    run_id: str
    mode: Optional[str]
    parent_run_id: Optional[str]
    configuration_parents: Tuple[str, ...]
    weight_donor: Optional[str]
    #: The deduplicated union of every parent edge actually walked from this
    #: node -- ``parent_run_id``, ``configuration_parents``, and
    #: ``weight_donor`` collapse to one edge whenever they name the same run.
    parents: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "parent_run_id": self.parent_run_id,
            "configuration_parents": list(self.configuration_parents),
            "weight_donor": self.weight_donor,
            "parents": list(self.parents),
        }


@dataclasses.dataclass(frozen=True)
class LineageGraph:
    """The full ancestor graph walked from one run."""

    root_run_id: str
    nodes: Mapping[str, LineageNode]
    edges: Tuple[Tuple[str, str], ...]

    @property
    def ancestor_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(run_id for run_id in self.nodes if run_id != self.root_run_id))


def _read_lineage_manifest(organism_directory: Path, run_id: str) -> Optional[Dict[str, Any]]:
    manifest_path = organism_directory / run_id / LINEAGE_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != LINEAGE_FORMAT:
        raise RegistryError(f"invalid lineage manifest for run {run_id!r}: {manifest_path}")
    return payload


def _lineage_node(organism_directory: Path, run_id: str) -> LineageNode:
    manifest = _read_lineage_manifest(organism_directory, run_id)
    if manifest is None:
        # No materialized run directory for this ancestor (for example an
        # externally seeded parent checkpoint) -- a leaf, not an error: only
        # the originally requested run_id must have a manifest to walk from.
        return LineageNode(run_id, None, None, (), None, ())

    parent = manifest.get("parent")
    parent_run_id = str(parent["run_id"]) if isinstance(parent, Mapping) and parent.get("run_id") else None
    configuration_parents = tuple(str(item) for item in manifest.get("configuration_parents") or ())
    weight_donor = manifest.get("weight_donor")
    weight_donor = str(weight_donor) if weight_donor else None

    ordered_parents = list(dict.fromkeys(
        ([parent_run_id] if parent_run_id else [])
        + list(configuration_parents)
        + ([weight_donor] if weight_donor else [])
    ))
    return LineageNode(
        run_id=run_id,
        mode=manifest.get("mode"),
        parent_run_id=parent_run_id,
        configuration_parents=configuration_parents,
        weight_donor=weight_donor,
        parents=tuple(ordered_parents),
    )


def lineage_graph(root: Union[str, Path], run_id: str) -> LineageGraph:
    """Walk ``parent_checkpoint_sha``/``lineage.json`` edges from ``run_id``.

    ``root`` is the organism directory every run lives beneath
    (``<root>/<run_id>/lineage.json``), matching
    :func:`cognitive_runtime.training.model_factory.artifacts.allocate_run_artifacts`'s
    layout. For a bred child this returns the full ancestor chain, including
    both configuration parents and the explicit weight donor.

    Terminates on an injected cycle by raising :class:`LineageCycleError`
    instead of recursing forever: a configuration-parent or weight-donor
    edge that loops back to a run already on the current walk is reported,
    never walked into infinitely. A diamond (two children sharing a common
    ancestor reached by different paths) is not a cycle and is visited once.
    """
    organism_directory = Path(root)
    if _read_lineage_manifest(organism_directory, run_id) is None:
        raise RegistryError(f"no lineage manifest for run {run_id!r} under {organism_directory}")

    nodes: Dict[str, LineageNode] = {}
    edges: List[Tuple[str, str]] = []
    path_stack: List[str] = []

    def visit(current: str) -> None:
        if current in path_stack:
            cycle = path_stack[path_stack.index(current):] + [current]
            raise LineageCycleError(
                f"lineage cycle detected while walking {run_id!r}: " + " -> ".join(cycle)
            )
        if current in nodes:
            return
        path_stack.append(current)
        node = _lineage_node(organism_directory, current)
        nodes[current] = node
        for parent_id in node.parents:
            edges.append((current, parent_id))
            visit(parent_id)
        path_stack.pop()

    visit(run_id)
    return LineageGraph(root_run_id=run_id, nodes=dict(nodes), edges=tuple(edges))


__all__ = [
    "REGISTRY_FORMAT",
    "LINEAGE_MANIFEST_NAME",
    "DECISION_PROMOTE",
    "DECISION_HOLD",
    "DECISION_TEST_USE",
    "RegistryError",
    "LineageCycleError",
    "TestBudgetExhaustedError",
    "ChampionEntry",
    "RegistrySlot",
    "LineageNode",
    "LineageGraph",
    "registry_path",
    "promote",
    "hold",
    "leading_champion",
    "population",
    "record_test_use",
    "lineage_graph",
]
