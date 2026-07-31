"""Deterministic, validation-only champion-population selection (issue #233).

The population is deliberately a policy layer rather than another promotion
path.  A candidate must already have a complete validation evaluation, and
this module never accepts (or reads) sealed-test metrics.  Its input is a
small, JSON-shaped summary of those evaluations plus the declared genome
fields used to measure diversity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


POPULATION_POLICY_FORMAT = "model-factory-champion-population-v1"
_MISSING = object()


class PopulationError(ValueError):
    """A malformed candidate, policy, or duplicate-spec request."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Canonical JSON for a fully resolved specification or declared genome."""
    def jsonable(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("mapping keys must be strings")
            return {key: jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item

    try:
        return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PopulationError("resolved specifications and genomes must be JSON-safe") from exc


def resolved_spec_fingerprint(resolved_spec: Mapping[str, Any]) -> str:
    """Stable identity for a *resolved* specification, not a source document."""
    return hashlib.sha256(_canonical_json(resolved_spec).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComputeLedger:
    """Compute attributed to one trial and to its complete weight lineage.

    ``ancestral_compute`` is inclusive: a fresh trial has equal trial and
    ancestral compute, while a child includes the accumulated cost of the
    checkpoint from which its weights descend.  This makes a cheap child of a
    very expensive donor visibly expensive in stage reports.
    """

    trial_compute: float
    ancestral_compute: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.trial_compute) and self.trial_compute >= 0):
            raise PopulationError("trial_compute must be a finite non-negative number")
        if not (math.isfinite(self.ancestral_compute) and self.ancestral_compute >= self.trial_compute):
            raise PopulationError("ancestral_compute must be finite and at least trial_compute")

    @classmethod
    def fresh(cls, trial_compute: float) -> "ComputeLedger":
        return cls(float(trial_compute), float(trial_compute))

    @classmethod
    def child(cls, trial_compute: float, weight_donor: "ComputeLedger") -> "ComputeLedger":
        return cls(float(trial_compute), float(trial_compute) + weight_donor.ancestral_compute)

    @property
    def inherited_compute(self) -> float:
        return self.ancestral_compute - self.trial_compute

    def to_dict(self) -> Dict[str, float]:
        return {"trial_compute": self.trial_compute, "ancestral_compute": self.ancestral_compute}


@dataclass(frozen=True)
class PopulationCandidate:
    """A fully evaluated candidate, carrying validation-only measurements.

    ``gates`` names the applicable quality/rollout/representation gates.  A
    missing gate is considered not configured; an explicitly false gate (or a
    non-completed status) always excludes the candidate from both champion and
    breeding pools.
    """

    run_id: str
    resolved_spec: Mapping[str, Any]
    genome: Mapping[str, Any]
    primary_metric: float
    runtime: float
    retention_metric: float
    ledger: ComputeLedger
    gates: Mapping[str, bool] = field(default_factory=dict)
    completion_status: str = "completed"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise PopulationError("run_id must not be empty")
        for name, value in (("primary_metric", self.primary_metric), ("runtime", self.runtime),
                            ("retention_metric", self.retention_metric)):
            if not math.isfinite(value):
                raise PopulationError(f"{name} must be finite")
        if self.runtime < 0:
            raise PopulationError("runtime must be non-negative")
        _canonical_json(self.resolved_spec)
        _canonical_json(self.genome)

    @property
    def spec_fingerprint(self) -> str:
        return resolved_spec_fingerprint(self.resolved_spec)

    @property
    def gate_passing(self) -> bool:
        return self.completion_status == "completed" and all(bool(value) for value in self.gates.values())

    @property
    def rejection_reason(self) -> Optional[str]:
        if self.completion_status != "completed":
            return self.completion_status
        failed = sorted(name for name, passed in self.gates.items() if not passed)
        return None if not failed else "failed gates: " + ", ".join(failed)


@dataclass(frozen=True)
class PopulationPolicy:
    """Recorded, deterministic replacement and tournament policy.

    The default directions match the factory's error-like validation metrics:
    lower primary and retained-generic-dynamics values are better.  A stage
    with a reward-like metric can set either flag to ``True`` explicitly.
    """

    declared_genome_fields: Tuple[str, ...]
    capacity: int = 6
    near_duplicate_distance: float = 0.25
    comparable_primary_margin: float = 0.0
    latency_significance: float = 0.10
    tournament_size: int = 3
    primary_higher_is_better: bool = False
    retention_higher_is_better: bool = False

    def __post_init__(self) -> None:
        if not self.declared_genome_fields or len(set(self.declared_genome_fields)) != len(self.declared_genome_fields):
            raise PopulationError("declared_genome_fields must be a non-empty tuple of unique fields")
        if not 4 <= self.capacity <= 8:
            raise PopulationError("capacity must be between 4 and 8")
        if not 0.0 <= self.near_duplicate_distance <= 1.0:
            raise PopulationError("near_duplicate_distance must be in [0, 1]")
        if self.comparable_primary_margin < 0:
            raise PopulationError("comparable_primary_margin must be non-negative")
        if not 0.0 <= self.latency_significance < 1.0:
            raise PopulationError("latency_significance must be in [0, 1)")
        if self.tournament_size < 2:
            raise PopulationError("tournament_size must be at least 2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": POPULATION_POLICY_FORMAT,
            "declared_genome_fields": list(self.declared_genome_fields),
            "capacity": self.capacity,
            "near_duplicate_distance": self.near_duplicate_distance,
            "comparable_primary_margin": self.comparable_primary_margin,
            "latency_significance": self.latency_significance,
            "tournament_size": self.tournament_size,
            "primary_higher_is_better": self.primary_higher_is_better,
            "retention_higher_is_better": self.retention_higher_is_better,
        }


def configuration_distance(left: PopulationCandidate, right: PopulationCandidate, policy: PopulationPolicy) -> float:
    """Normalized Hamming distance over *only* declared genome fields.

    Requiring every declared field avoids an accidental comparison of partial
    genomes.  Nothing from a checkpoint, runtime object, or hidden state is
    inspected, which keeps diversity reproducible and auditable.
    """
    fields = policy.declared_genome_fields
    for candidate in (left, right):
        missing = [name for name in fields if name not in candidate.genome]
        if missing:
            raise PopulationError(f"candidate {candidate.run_id!r} is missing declared genome field(s) {missing!r}")
    return sum(left.genome[name] != right.genome[name] for name in fields) / len(fields)


def _better(value: float, other: float, higher_is_better: bool) -> bool:
    return value > other if higher_is_better else value < other


def _not_worse(value: float, other: float, higher_is_better: bool) -> bool:
    return value >= other if higher_is_better else value <= other


def dominates(left: PopulationCandidate, right: PopulationCandidate, policy: PopulationPolicy) -> bool:
    """Pareto dominance on validation quality, latency, and retention.

    Configuration diversity is applied as a guard: distant configurations are
    intentionally not allowed to erase one another merely because one wins on
    all scalar metrics.  For a near duplicate, exact metric ties are resolved
    by run ID so replacement remains deterministic.
    """
    if configuration_distance(left, right, policy) > policy.near_duplicate_distance:
        return False
    axes = (
        (left.primary_metric, right.primary_metric, policy.primary_higher_is_better),
        (left.runtime, right.runtime, False),
        (left.retention_metric, right.retention_metric, policy.retention_higher_is_better),
    )
    no_worse = all(_not_worse(a, b, higher) for a, b, higher in axes)
    strictly_better = any(_better(a, b, higher) for a, b, higher in axes)
    return no_worse and (strictly_better or (left.run_id < right.run_id and left != right))


def _primary_key(candidate: PopulationCandidate, policy: PopulationPolicy) -> Tuple[float, str]:
    value = -candidate.primary_metric if policy.primary_higher_is_better else candidate.primary_metric
    return (value, candidate.run_id)


def _retention_key(candidate: PopulationCandidate, policy: PopulationPolicy) -> Tuple[float, str]:
    value = -candidate.retention_metric if policy.retention_higher_is_better else candidate.retention_metric
    return (value, candidate.run_id)


def _comparable_primary(left: PopulationCandidate, right: PopulationCandidate, policy: PopulationPolicy) -> bool:
    return abs(left.primary_metric - right.primary_metric) <= policy.comparable_primary_margin


@dataclass(frozen=True)
class PopulationDecision:
    """An auditable replacement result; ``members`` are the breeding pool."""

    members: Tuple[PopulationCandidate, ...]
    leading_champion: Optional[str]
    roles: Mapping[str, Tuple[str, ...]]
    rejected: Mapping[str, str]
    policy: PopulationPolicy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": POPULATION_POLICY_FORMAT,
            "members": [member.run_id for member in self.members],
            "leading_champion": self.leading_champion,
            "roles": {name: list(run_ids) for name, run_ids in self.roles.items()},
            "rejected": dict(self.rejected),
            "policy": self.policy.to_dict(),
        }


def replace_population(candidates: Iterable[PopulationCandidate], policy: PopulationPolicy) -> PopulationDecision:
    """Select a bounded, deterministic population from completed candidates.

    Gate failures are excluded first.  Near-duplicate dominated candidates are
    excluded next; distant configurations remain eligible.  The named primary,
    latency, and retention leaders are then filled before diversity slots use
    max-min declared-genome distance, with run ID as every final tie-breaker.
    """
    by_run_id: Dict[str, PopulationCandidate] = {}
    for candidate in candidates:
        if candidate.run_id in by_run_id:
            raise PopulationError(f"duplicate candidate run_id {candidate.run_id!r}")
        by_run_id[candidate.run_id] = candidate

    rejected: Dict[str, str] = {}
    eligible = []
    for candidate in sorted(by_run_id.values(), key=lambda item: item.run_id):
        if candidate.gate_passing:
            eligible.append(candidate)
        else:
            rejected[candidate.run_id] = candidate.rejection_reason or "failed gates"

    survivors = []
    for candidate in eligible:
        dominator = next((other for other in eligible if other != candidate and dominates(other, candidate, policy)), None)
        if dominator is None:
            survivors.append(candidate)
        else:
            rejected[candidate.run_id] = f"dominated near-duplicate of {dominator.run_id}"

    if not survivors:
        return PopulationDecision((), None, {}, rejected, policy)

    selected: list[PopulationCandidate] = []
    roles: Dict[str, list[str]] = {"primary": [], "latency": [], "retention": [], "diversity": []}

    def add(candidate: PopulationCandidate, role: str) -> None:
        if candidate not in selected and len(selected) < policy.capacity:
            selected.append(candidate)
            roles[role].append(candidate.run_id)

    # Up to two primary leaders; the second must be comparable, preserving a
    # close alternative rather than always consuming capacity with a loser.
    primary = sorted(survivors, key=lambda item: _primary_key(item, policy))
    add(primary[0], "primary")
    if len(primary) > 1 and _comparable_primary(primary[0], primary[1], policy):
        add(primary[1], "primary")

    # A latency leader earns a named slot only if timing differs materially.
    fastest = min(survivors, key=lambda item: (item.runtime, item.run_id))
    if fastest.runtime < primary[0].runtime * (1.0 - policy.latency_significance):
        add(fastest, "latency")

    for candidate in sorted(survivors, key=lambda item: _retention_key(item, policy))[:2]:
        add(candidate, "retention")

    while len(selected) < policy.capacity:
        remaining = [candidate for candidate in survivors if candidate not in selected]
        if not remaining:
            break
        candidate = sorted(
            remaining,
            key=lambda item: (
                -(min(configuration_distance(item, member, policy) for member in selected) if selected else 1.0),
                _primary_key(item, policy),
            ),
        )[0]
        add(candidate, "diversity")

    selected.sort(key=lambda item: item.run_id)
    frozen_roles = {name: tuple(run_ids) for name, run_ids in roles.items() if run_ids}
    return PopulationDecision(tuple(selected), primary[0].run_id, frozen_roles, rejected, policy)


class DuplicateSpecCache:
    """In-memory resolved-spec cache used to avoid a duplicate training run."""

    def __init__(self, results: Optional[Mapping[str, Any]] = None) -> None:
        self._results: Dict[str, Any] = dict(results or {})

    def find(self, resolved_spec: Mapping[str, Any]) -> Optional[Any]:
        return self._results.get(resolved_spec_fingerprint(resolved_spec))

    def record(self, resolved_spec: Mapping[str, Any], result: Any) -> None:
        self._results[resolved_spec_fingerprint(resolved_spec)] = result

    def reuse_or_run(self, resolved_spec: Mapping[str, Any], run: Callable[[], Any]) -> Tuple[Any, bool]:
        """Return ``(result, reused)``; ``run`` is never called for a duplicate."""
        existing = self._results.get(resolved_spec_fingerprint(resolved_spec), _MISSING)
        if existing is not _MISSING:
            return existing, True
        result = run()
        self.record(resolved_spec, result)
        return result, False


def select_parents(
    members: Sequence[PopulationCandidate], policy: PopulationPolicy, *, seed: int,
) -> Tuple[PopulationCandidate, PopulationCandidate]:
    """Seeded quality-and-diversity tournament over gate-passing members.

    The function accepts only population candidates and reads only their
    validation-derived fields; no sealed-test split is accepted or queried.
    """
    pool = sorted((member for member in members if member.gate_passing), key=lambda item: item.run_id)
    if len(pool) < 2:
        raise PopulationError("parent selection requires at least two gate-passing population members")
    rng = random.Random(seed)

    def tournament(excluded: Sequence[PopulationCandidate]) -> PopulationCandidate:
        available = [candidate for candidate in pool if candidate not in excluded]
        count = min(policy.tournament_size, len(available))
        contenders = rng.sample(available, count)
        if not excluded:
            return min(contenders, key=lambda item: _primary_key(item, policy))
        return sorted(
            contenders,
            key=lambda item: (
                -min(configuration_distance(item, chosen, policy) for chosen in excluded),
                _primary_key(item, policy),
            ),
        )[0]

    first = tournament(())
    second = tournament((first,))
    return first, second


__all__ = [
    "POPULATION_POLICY_FORMAT", "PopulationError", "ComputeLedger", "PopulationCandidate",
    "PopulationPolicy", "PopulationDecision", "configuration_distance", "dominates",
    "replace_population", "resolved_spec_fingerprint", "DuplicateSpecCache", "select_parents",
]
