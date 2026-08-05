"""Nested architecture (NAS-style) evolutionary search for the Model Factory.

:func:`.search.run_evolutionary_search` evolves a *training procedure*
against one fixed architecture -- structurally, because every gene it can
recombine is rooted at a :class:`~.contracts.TrainingContract` field
(:func:`.genome._assert_genes_are_training_contract_paths`). This module
adds the other axis, without weakening that guarantee anywhere: an **outer**
evolutionary loop over :mod:`.architecture_genome`'s architecture genomes,
where scoring one outer candidate means running a complete **inner**
:func:`.search.run_evolutionary_search` campaign over training genes, with
that architecture held fixed.

Why nested rather than one joint population
-------------------------------------------
:mod:`.search`'s own module docstring states epic §13.1's discipline: "Do
not jointly vary architecture, data split, objective, and loss weights." A
flat genome mixing both axes would violate it directly, and would also make
the resulting comparison meaningless in the ordinary way: an architecture
evaluated at one arbitrary hyperparameter setting is confounded with that
setting, so a good architecture that happens to need a smaller learning rate
loses to a mediocre one that got lucky. Giving *every* architecture
candidate the same inner search budget before it is scored is what makes
outer fitness a property of the architecture. The inner engine is used
completely unchanged, as a tested black box -- this module calls it, it does
not modify or reimplement it.

Weight inheritance is deliberately absent
-----------------------------------------
An architecture-changing child trains **from scratch** (``mode="fresh"``,
no ``parent`` block). This is not an oversight: :mod:`.breeding`'s weight
donor is always one parent's exact checkpoint (:func:`.breeding.
select_weight_donor` never averages, and :func:`.breeding.check_compatible`
requires an identical ``architecture_hash` before donation is even
considered), so there is no sound way to hand a differently-shaped
``state_dict`` to a structural child. Training structural children from
scratch is also the standard choice for evolutionary NAS (regularized
evolution / AmoebaNet-style). Cross-architecture weight sharing is an
explicit non-goal here.

An outer child's provenance is still fully recorded:
``evolution.configuration_parents`` names both architecture parents and
``evolution.weight_donor`` records the fitter one **as a label only**. That
is legal by construction rather than by luck: :func:`.spec._validate_evolution`
requires ``weight_donor`` to be *present* whenever ``configuration_parents``
is set, and never requires it to name a loadable checkpoint -- checkpoint
loading is gated entirely by the ``parent`` block, which a ``mode="fresh"``
child does not have (:func:`.spec._validate_parent` requires exactly that).

Cost
----
Outer and inner budgets multiply. A modest ``4 x 2`` outer over ``3 x 1``
inner is already 24 full trainings, so :func:`estimate_cost` is computed and
returned *before* anything launches, and both the CLI and any web caller are
expected to surface it prominently rather than bury it.

Pure Python at declaration time -- torch is only reached through
:func:`.search.run_evolutionary_search`'s own ``run_trial`` calls, so
:func:`preview_architecture_search` and :func:`estimate_cost` work in the
core-only install.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from cognitive_runtime.training.model_factory.architecture_genome import (
    ArchitectureGenomeSchema,
    apply_architecture_genome,
    canonicalize_architecture_genome,
)
from cognitive_runtime.training.model_factory.breeding import (
    _crossover_with_provenance,
    _mutate_with_provenance,
)
from cognitive_runtime.training.model_factory.genome import (
    GenomeRepairError,
    GenomeSchema,
    repair,
)
from cognitive_runtime.training.model_factory.runner import (
    RUNS_ROOT_DEFAULT,
    _selection_metric_mode,
)
from cognitive_runtime.training.model_factory.search import (
    STATE_NOT_ATTEMPTED,
    SearchError,
    _champion_comparison,
    _latin_hypercube_genomes,
    _random_genomes,
    run_evolutionary_search,
)
from cognitive_runtime.training.model_factory.spec import ExperimentSpec, _thaw
from cognitive_runtime.training.model_factory.spec import validate as validate_spec
from cognitive_runtime.training.model_factory.state import STATE_FAILED

ARCHITECTURE_SEARCH_FORMAT = "model-factory-architecture-search-v1"

#: The outer loop's genetic operator, recorded verbatim into each structural
#: child's ``evolution.genome_operator``. Distinct from
#: :data:`.breeding.GENOME_OPERATOR` so a run's own lineage says which axis
#: it was bred on, without anyone having to infer it from the gene names.
ARCHITECTURE_GENOME_OPERATOR = "architecture_uniform_crossover_v1"

#: Versioned identity for the architecture-breeding provenance record.
ARCHITECTURE_LINEAGE_FORMAT = "model-factory-architecture-breeding-lineage-v1"

#: An outer candidate whose inner campaign finished but produced no
#: surviving candidate at all (every inner trial failed or failed a quality
#: gate). Distinct from :data:`~.search.STATE_FAILED` (the inner campaign
#: itself raised) and :data:`~.search.STATE_NOT_ATTEMPTED` (never launched).
STATE_NO_INNER_SURVIVOR = "no_inner_survivor"

#: Deterministic seed spacing, mirroring :mod:`.search`'s own use of large
#: primes so two nearby (generation, candidate) coordinates never share a
#: derived seed.
_GENERATION_SEED_STRIDE = 1_000_003
_CANDIDATE_SEED_STRIDE = 10_007
_RESAMPLE_SEED_STRIDE = 104_729


class ArchitectureSearchError(SearchError):
    """A malformed architecture-search request.

    Subclasses :class:`~.search.SearchError` (itself a ``ValueError``) so
    every existing caller that already funnels campaign misconfiguration
    into one ``except (OSError, SpecError, ValueError)`` branch -- the CLI's
    ``cmd_factory_search`` in particular -- handles this identically without
    growing a second except clause.
    """


def architecture_run_id_prefix(campaign_prefix: str, generation: int, candidate_index: int) -> str:
    """The inner campaign's ``run_id_prefix`` for one outer candidate.

    Extends :func:`.search.run_evolutionary_search`'s own
    ``{prefix}-p{pop}-c{cand}`` convention with the outer coordinates, so a
    full inner run ID reads ``{prefix}-arch{gen}-a{cand}-p{pop}-c{cand}``.
    Every ID in a campaign is therefore precomputable from the campaign's
    configuration alone, before anything has run -- which is what lets a
    polling UI lay out the whole nested grid and start watching each
    ``state.json`` before the subprocess reaches that candidate.
    """
    return f"{campaign_prefix}-arch{generation}-a{candidate_index}"


@dataclass(frozen=True)
class ArchitectureSearchEstimate:
    """The campaign's worst-case cost, computed before anything launches.

    ``estimated_max_trials`` is the honest upper bound
    (``outer_size * outer_count * inner_size * inner_count``): every outer
    candidate that is ever evaluated runs a complete inner campaign, and
    every inner population can be a full one. Real campaigns come in under
    it -- both loops carry survivors forward without retraining, so only the
    bred children of a later generation cost anything -- but a bound that
    can be exceeded is not a guardrail, so this deliberately does not
    discount for that.

    ``estimated_max_seconds`` is ``estimated_max_trials`` times the base
    spec's own ``training.max_training_seconds``, and is ``None`` when the
    spec declares no per-trial wall-clock cap -- in which case the campaign
    genuinely has no bound to report, which is itself the thing a caller
    needs to see.
    """

    outer_population_size: int
    outer_population_count: int
    inner_population_size: int
    inner_population_count: int
    estimated_max_trials: int
    max_training_seconds_per_trial: Optional[float]
    estimated_max_seconds: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outer_population_size": self.outer_population_size,
            "outer_population_count": self.outer_population_count,
            "inner_population_size": self.inner_population_size,
            "inner_population_count": self.inner_population_count,
            "estimated_max_trials": self.estimated_max_trials,
            "max_training_seconds_per_trial": self.max_training_seconds_per_trial,
            "estimated_max_seconds": self.estimated_max_seconds,
        }


def estimate_cost(
    base_spec: ExperimentSpec,
    *,
    outer_population_size: int,
    outer_population_count: int,
    inner_population_size: int,
    inner_population_count: int,
) -> ArchitectureSearchEstimate:
    """The multiplicative trial/wall-clock bound for one nested campaign."""
    trials = (
        int(outer_population_size)
        * int(outer_population_count)
        * int(inner_population_size)
        * int(inner_population_count)
    )
    per_trial = base_spec.training.get("max_training_seconds")
    per_trial = float(per_trial) if per_trial is not None else None
    return ArchitectureSearchEstimate(
        outer_population_size=int(outer_population_size),
        outer_population_count=int(outer_population_count),
        inner_population_size=int(inner_population_size),
        inner_population_count=int(inner_population_count),
        estimated_max_trials=trials,
        max_training_seconds_per_trial=per_trial,
        estimated_max_seconds=None if per_trial is None else per_trial * trials,
    )


@dataclass(frozen=True)
class ArchitectureParent:
    """One scored outer candidate, as a breeding parent.

    ``run_id`` is the inner campaign's own best run -- the concrete trained
    model this architecture is represented by, and the only run ID that
    means anything to the rest of the factory. It is what a child records as
    its ``evolution.weight_donor`` *label*; no checkpoint is ever loaded
    from it (see this module's docstring).
    """

    architecture_id: str
    run_id: str
    genome: Mapping[str, Any]
    metric_value: Optional[float]


@dataclass(frozen=True)
class ArchitectureBreedingLineage:
    """The full genetic-operator record for one structural child."""

    format: str
    generation: int
    parent_a_architecture_id: str
    parent_b_architecture_id: str
    parent_a_run_id: str
    parent_b_run_id: str
    weight_donor_run_id: str
    genome_operator: str
    mutation_seed: int
    parent_a_genome: Mapping[str, Any]
    parent_b_genome: Mapping[str, Any]
    crossover_choices: Mapping[str, str]
    mutations: Mapping[str, Mapping[str, Any]]
    repairs: Mapping[str, Mapping[str, Any]]
    child_genome: Mapping[str, Any]
    child_model: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "generation": self.generation,
            "parent_a_architecture_id": self.parent_a_architecture_id,
            "parent_b_architecture_id": self.parent_b_architecture_id,
            "parent_a_run_id": self.parent_a_run_id,
            "parent_b_run_id": self.parent_b_run_id,
            "weight_donor_run_id": self.weight_donor_run_id,
            "genome_operator": self.genome_operator,
            "mutation_seed": self.mutation_seed,
            "parent_a_genome": _thaw(self.parent_a_genome),
            "parent_b_genome": _thaw(self.parent_b_genome),
            "crossover_choices": dict(self.crossover_choices),
            "mutations": {name: dict(detail) for name, detail in self.mutations.items()},
            "repairs": {name: dict(detail) for name, detail in self.repairs.items()},
            "child_genome": _thaw(self.child_genome),
            "child_model": _thaw(self.child_model),
        }


@dataclass(frozen=True)
class ArchitectureBreedingResult:
    """A ready-to-launch structural child plus its provenance record."""

    child_spec: ExperimentSpec
    child_genome: Mapping[str, Any]
    lineage: ArchitectureBreedingLineage


def _fitter_parent(
    parent_a: ArchitectureParent, parent_b: ArchitectureParent, *, selection_mode: str,
) -> ArchitectureParent:
    """The better-scoring parent, ties broken by ``run_id``.

    Deterministic rather than a seeded coin flip (:func:`.breeding.
    select_weight_donor`'s rule for real weight donation): nothing is
    inherited here, so this choice is purely a provenance label and the
    useful thing for it to record is which architecture actually scored
    better.
    """
    scored = [parent for parent in (parent_a, parent_b) if parent.metric_value is not None]
    if not scored:
        return min((parent_a, parent_b), key=lambda parent: parent.run_id)
    if len(scored) == 1:
        return scored[0]
    sign = 1.0 if selection_mode == "min" else -1.0
    return min(scored, key=lambda parent: (sign * float(parent.metric_value), parent.run_id))


def breed_architecture(
    parent_a: ArchitectureParent,
    parent_b: ArchitectureParent,
    architecture_schema: ArchitectureGenomeSchema,
    base_spec: ExperimentSpec,
    *,
    objective: str,
    generation: int,
    seed: int,
    selection_mode: str = "min",
    mutation_rate: float = 0.2,
) -> ArchitectureBreedingResult:
    """Cross and mutate two architecture genomes into one launchable child.

    Deliberately *not* a change to :func:`.breeding.breed`, which stays
    exactly as it is for same-architecture hyperparameter breeding: that
    function's whole contract is a checkpoint-inheriting ``mode="clone"``
    child of two compatible parents, and every one of those words is wrong
    for a structural child. The genetic operators themselves are shared --
    :func:`.breeding._crossover_with_provenance` and
    :func:`.breeding._mutate_with_provenance` are generic over any
    :class:`~.genome.GenomeSchema`, and recording which parent each gene
    came from matters at least as much here.

    The child's ``training``/``data``/``evaluation`` blocks come from
    ``base_spec``, not from either parent: the outer loop varies the
    architecture and nothing else, and each child's own training genes are
    about to be searched from scratch by its inner campaign anyway.

    Each :class:`ArchitectureParent`'s ``run_id`` must name a *materialised*
    run under the campaign's organism, not just an architecture label:
    :func:`.artifacts.allocate_run_artifacts` resolves every
    ``configuration_parents`` entry to that run's persisted
    ``display_name.json`` when it names the child, and refuses a parent it
    cannot find. The outer loop satisfies this naturally by recording each
    architecture's inner-campaign best run.
    """
    rng = random.Random(seed)
    donor = _fitter_parent(parent_a, parent_b, selection_mode=selection_mode)

    crossed_genome, crossover_choices = _crossover_with_provenance(
        architecture_schema, parent_a.genome, parent_b.genome, objective=objective, rng=rng,
    )
    mutated_genome, mutations = _mutate_with_provenance(
        architecture_schema, crossed_genome, objective=objective, rng=rng,
        mutation_rate=mutation_rate,
    )
    # Canonicalized for the same reason the initial population is: a child
    # that crosses a windowed parent's context_length onto a GRU backbone
    # would otherwise record a value its own model ignores, and could
    # "differ" from a sibling it is in fact identical to.
    child_genome = canonicalize_architecture_genome(
        architecture_schema, repair(architecture_schema, mutated_genome, objective=objective),
    )
    repairs = {
        name: {"from": mutated_genome[name], "to": child_genome[name]}
        for name in architecture_schema.genes
        if mutated_genome[name] != child_genome[name]
    }

    child_model = apply_architecture_genome(base_spec.model, child_genome)
    child_spec = replace(
        base_spec,
        mode="fresh",
        parent=None,
        model=child_model,
        evolution={
            "generation": generation,
            "configuration_parents": [parent_a.run_id, parent_b.run_id],
            # A provenance label, never a checkpoint reference -- see this
            # module's docstring on why that is legal by construction.
            "weight_donor": donor.run_id,
            "genome_operator": ARCHITECTURE_GENOME_OPERATOR,
            "mutations": sorted(mutations),
        },
    )
    validate_spec(child_spec)

    lineage = ArchitectureBreedingLineage(
        format=ARCHITECTURE_LINEAGE_FORMAT,
        generation=generation,
        parent_a_architecture_id=parent_a.architecture_id,
        parent_b_architecture_id=parent_b.architecture_id,
        parent_a_run_id=parent_a.run_id,
        parent_b_run_id=parent_b.run_id,
        weight_donor_run_id=donor.run_id,
        genome_operator=ARCHITECTURE_GENOME_OPERATOR,
        mutation_seed=seed,
        parent_a_genome=dict(parent_a.genome),
        parent_b_genome=dict(parent_b.genome),
        crossover_choices=crossover_choices,
        mutations=mutations,
        repairs=repairs,
        child_genome=dict(child_genome),
        child_model=dict(child_model),
    )
    return ArchitectureBreedingResult(
        child_spec=child_spec, child_genome=child_genome, lineage=lineage,
    )


@dataclass(frozen=True)
class ArchitectureCandidateResult:
    """One architecture's outcome: its own inner campaign, summarised."""

    candidate_index: int
    architecture_id: str
    origin_generation: int
    genome: Mapping[str, Any]
    model: Mapping[str, Any]
    state: str
    best_run_id: Optional[str]
    metric_value: Optional[float]
    inner_trials_started: int
    inner_epochs_executed: int
    quality_passed: bool
    retained: bool
    carried: bool
    configuration_parents: Tuple[str, ...]
    reason: str
    #: The full crossover/mutation/repair record for a structural child;
    #: ``None`` for a sampled generation-zero architecture, which has no
    #: parents. Carried on the report rather than written into the child's
    #: ``lineage.json`` (as :func:`.breeding.record_breeding_lineage` does
    #: for an inner child) because an outer candidate is not one run: it is
    #: a whole inner campaign, so there is no single run directory this
    #: record belongs to.
    breeding_lineage: Optional[ArchitectureBreedingLineage] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "architecture_id": self.architecture_id,
            "origin_generation": self.origin_generation,
            "genome": _thaw(self.genome),
            "model": _thaw(self.model),
            "state": self.state,
            "best_run_id": self.best_run_id,
            "metric_value": self.metric_value,
            "inner_trials_started": self.inner_trials_started,
            "inner_epochs_executed": self.inner_epochs_executed,
            "quality_passed": self.quality_passed,
            "retained": self.retained,
            "carried": self.carried,
            "configuration_parents": list(self.configuration_parents),
            "reason": self.reason,
            "breeding_lineage": (
                self.breeding_lineage.to_dict() if self.breeding_lineage is not None else None
            ),
        }


@dataclass(frozen=True)
class ArchitectureGenerationReport:
    """Filtering, ranking, and breeding decisions for one outer generation."""

    generation_index: int
    results: Tuple[ArchitectureCandidateResult, ...]
    survivor_architecture_ids: Tuple[str, ...]
    offspring_architecture_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "results": [result.to_dict() for result in self.results],
            "survivor_architecture_ids": list(self.survivor_architecture_ids),
            "offspring_architecture_ids": list(self.offspring_architecture_ids),
        }


@dataclass(frozen=True)
class ArchitectureSearchReport:
    """The complete record of one nested architecture campaign."""

    format: str
    seed: int
    method: str
    architecture_schema_version: str
    hyperparameter_schema_version: str
    estimate: ArchitectureSearchEstimate
    outer_mutation_rate: float
    inner_mutation_rate: float
    generations: Tuple[ArchitectureGenerationReport, ...]
    inner_campaigns_run: int
    total_trials_started: int
    total_epochs_executed: int
    stage_budget_exceeded: bool
    final_survivor_architecture_ids: Tuple[str, ...]
    best_architecture_id: Optional[str]
    best_run_id: Optional[str]
    best_metric_value: Optional[float]
    best_genome: Optional[Mapping[str, Any]]
    best_model: Optional[Mapping[str, Any]]
    champion_comparison: Optional[Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "method": self.method,
            "architecture_schema_version": self.architecture_schema_version,
            "hyperparameter_schema_version": self.hyperparameter_schema_version,
            "estimate": self.estimate.to_dict(),
            "outer_mutation_rate": self.outer_mutation_rate,
            "inner_mutation_rate": self.inner_mutation_rate,
            "generations": [generation.to_dict() for generation in self.generations],
            "inner_campaigns_run": self.inner_campaigns_run,
            "total_trials_started": self.total_trials_started,
            "total_epochs_executed": self.total_epochs_executed,
            "stage_budget_exceeded": self.stage_budget_exceeded,
            "final_survivor_architecture_ids": list(self.final_survivor_architecture_ids),
            "best_architecture_id": self.best_architecture_id,
            "best_run_id": self.best_run_id,
            "best_metric_value": self.best_metric_value,
            "best_genome": _thaw(self.best_genome) if self.best_genome is not None else None,
            "best_model": _thaw(self.best_model) if self.best_model is not None else None,
            "champion_comparison": (
                dict(self.champion_comparison) if self.champion_comparison is not None else None
            ),
        }


@dataclass(frozen=True)
class ArchitecturePreviewCandidate:
    """One generation-zero architecture, fully resolved but never launched."""

    candidate_index: int
    architecture_id: str
    genome: Mapping[str, Any]
    spec: ExperimentSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "architecture_id": self.architecture_id,
            "genome": _thaw(self.genome),
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class ArchitectureSearchPreview:
    """A dry run: the cost bound plus every generation-zero candidate."""

    format: str
    campaign_id: str
    seed: int
    method: str
    architecture_schema_version: str
    hyperparameter_schema_version: str
    estimate: ArchitectureSearchEstimate
    candidates: Tuple[ArchitecturePreviewCandidate, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "campaign_id": self.campaign_id,
            "dry_run": True,
            "seed": self.seed,
            "method": self.method,
            "architecture_schema_version": self.architecture_schema_version,
            "hyperparameter_schema_version": self.hyperparameter_schema_version,
            "estimate": self.estimate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class _ArchitectureMember:
    """Internal outer-candidate state carried between generations."""

    genome: Mapping[str, Any]
    spec: ExperimentSpec
    origin_generation: int
    configuration_parents: Tuple[str, ...] = ()
    breeding_lineage: Optional[ArchitectureBreedingLineage] = None
    evaluation: Optional[ArchitectureCandidateResult] = None


def _validate_campaign_shape(
    *,
    outer_population_size: int,
    outer_population_count: int,
    inner_population_size: int,
    inner_population_count: int,
    outer_mutation_rate: float,
    inner_mutation_rate: float,
) -> None:
    if outer_population_size < 2:
        raise ArchitectureSearchError(
            f"outer_population_size must be at least 2; got {outer_population_size!r}"
        )
    if outer_population_count < 1:
        raise ArchitectureSearchError(
            f"outer_population_count must be at least 1; got {outer_population_count!r}"
        )
    # Deliberately the same floor `run_evolutionary_search` itself enforces:
    # failing here names the parameter the caller actually passed, instead of
    # surfacing a bare inner SearchError several minutes into a campaign.
    if inner_population_size < 2:
        raise ArchitectureSearchError(
            f"inner_population_size must be at least 2; got {inner_population_size!r}"
        )
    if inner_population_count < 1:
        raise ArchitectureSearchError(
            f"inner_population_count must be at least 1; got {inner_population_count!r}"
        )
    for label, rate in (
        ("outer_mutation_rate", outer_mutation_rate), ("inner_mutation_rate", inner_mutation_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ArchitectureSearchError(f"{label} must be in [0, 1]; got {rate!r}")


def _architecture_spec(base_spec: ExperimentSpec, genome: Mapping[str, Any]) -> ExperimentSpec:
    """One outer candidate's base spec: ``base_spec`` at this architecture.

    Always ``mode="fresh"`` with no ``parent``: a differently-shaped model
    has no compatible checkpoint to continue from, and the inner campaign
    starts every one of its own candidates fresh from this spec anyway.
    """
    candidate = replace(
        base_spec,
        mode="fresh",
        parent=None,
        model=apply_architecture_genome(base_spec.model, genome),
    )
    validate_spec(candidate)
    return candidate


def _distinct_architecture_genomes(
    architecture_schema: ArchitectureGenomeSchema,
    base_spec: ExperimentSpec,
    outer_population_size: int,
    seed: int,
    *,
    method: str,
) -> List[Tuple[Dict[str, Any], ExperimentSpec]]:
    """Sample ``outer_population_size`` *distinct* architectures deterministically.

    Duplicates are worth rejecting harder here than in the inner loop: two
    identical architectures would each spend a complete inner campaign
    proving the same thing. Redraws are deterministic (a fixed prime seed
    stride), so a given ``seed`` always yields the same initial population.

    "Distinct" means *builds a different model*, not merely "has a different
    genome": genomes are canonicalized
    (:func:`.architecture_genome.canonicalize_architecture_genome`) before
    the comparison, so sampled values of a gene the chosen backbone ignores
    -- ``context_length`` under the GRU -- collapse into one candidate
    instead of spending a full inner budget each and then being ranked
    against each other on training noise.
    """
    if method not in ("random", "lhs"):
        raise ArchitectureSearchError(f"unknown method {method!r}; expected one of ('random', 'lhs')")

    accepted: List[Tuple[Dict[str, Any], ExperimentSpec]] = []
    seen: set = set()
    for attempt in range(64):
        rng = random.Random(seed + attempt * _RESAMPLE_SEED_STRIDE)
        batch = (
            _latin_hypercube_genomes(architecture_schema, outer_population_size, rng)
            if method == "lhs"
            else _random_genomes(architecture_schema, outer_population_size, rng)
        )
        for genome in batch:
            canonical = canonicalize_architecture_genome(
                architecture_schema,
                repair(architecture_schema, genome, objective=base_spec.training["objective"]),
            )
            identity = tuple(sorted((name, str(value)) for name, value in canonical.items()))
            if identity in seen:
                continue
            seen.add(identity)
            accepted.append((canonical, _architecture_spec(base_spec, canonical)))
            if len(accepted) == outer_population_size:
                return accepted
    raise ArchitectureSearchError(
        f"could not sample {outer_population_size} distinct architectures from schema "
        f"{architecture_schema.version!r} after 64 deterministic attempts; the declared "
        "search space is smaller than the requested outer population"
    )


def preview_architecture_search(
    base_spec: ExperimentSpec,
    architecture_schema: ArchitectureGenomeSchema,
    hyperparameter_schema: GenomeSchema,
    *,
    outer_population_size: int,
    outer_population_count: int,
    inner_population_size: int,
    inner_population_count: int,
    seed: int,
    method: str = "lhs",
    outer_mutation_rate: float = 0.2,
    inner_mutation_rate: float = 0.2,
    run_id_prefix: Optional[str] = None,
) -> ArchitectureSearchPreview:
    """Resolve generation zero and the cost bound without launching anything.

    Torch-free and corpus-free by construction: it samples architecture
    genomes, merges them into specs, and validates those specs -- exactly
    what ``ccr factory search --dry-run`` already does for the
    hyperparameter workflow, plus the cost estimate this workflow must never
    be launched without seeing.
    """
    _validate_campaign_shape(
        outer_population_size=outer_population_size,
        outer_population_count=outer_population_count,
        inner_population_size=inner_population_size,
        inner_population_count=inner_population_count,
        outer_mutation_rate=outer_mutation_rate,
        inner_mutation_rate=inner_mutation_rate,
    )
    campaign_id = run_id_prefix or f"arch{seed}"
    sampled = _distinct_architecture_genomes(
        architecture_schema, base_spec, outer_population_size, seed, method=method,
    )
    return ArchitectureSearchPreview(
        format=ARCHITECTURE_SEARCH_FORMAT,
        campaign_id=campaign_id,
        seed=seed,
        method=method,
        architecture_schema_version=architecture_schema.version,
        hyperparameter_schema_version=hyperparameter_schema.version,
        estimate=estimate_cost(
            base_spec,
            outer_population_size=outer_population_size,
            outer_population_count=outer_population_count,
            inner_population_size=inner_population_size,
            inner_population_count=inner_population_count,
        ),
        candidates=tuple(
            ArchitecturePreviewCandidate(
                candidate_index=index,
                architecture_id=architecture_run_id_prefix(campaign_id, 0, index),
                genome=genome,
                spec=spec,
            )
            for index, (genome, spec) in enumerate(sampled)
        ),
    )


def run_architecture_search(
    base_spec: ExperimentSpec,
    architecture_schema: ArchitectureGenomeSchema,
    hyperparameter_schema: GenomeSchema,
    *,
    outer_population_size: int,
    outer_population_count: int,
    inner_population_size: int,
    inner_population_count: int,
    seed: int,
    method: str = "lhs",
    outer_mutation_rate: float = 0.2,
    inner_mutation_rate: float = 0.2,
    root: Union[str, Path] = RUNS_ROOT_DEFAULT,
    corpus_root: Optional[Union[str, Path]] = None,
    run_id_prefix: Optional[str] = None,
    naming_seed: Optional[Union[str, int]] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    heartbeat_timeout_seconds: Optional[float] = None,
    champion_run_id: Optional[str] = None,
    min_episode_length: Optional[int] = None,
    stage_budget_seconds: Optional[float] = None,
    cost_model: Optional[Callable[[Mapping[str, Any]], float]] = None,
) -> ArchitectureSearchReport:
    """Run one nested architecture campaign.

    Generation zero samples ``outer_population_size`` distinct architectures
    from ``architecture_schema``. Each is scored by running a complete
    :func:`.search.run_evolutionary_search` campaign over
    ``hyperparameter_schema``, at ``inner_population_size`` /
    ``inner_population_count``, with that architecture fixed; the inner
    report's ``best_metric_value`` *is* the outer candidate's fitness.
    Architectures whose inner campaign failed or produced no surviving
    candidate are eliminated; the rest are ranked by the base spec's own
    ``evaluation.selection_metric`` in its declared min/max direction and the
    best half is retained. Retained architectures carry their evidence
    forward without re-running their inner campaign; structural children
    (:func:`breed_architecture`) fill the next generation back up.

    ``stage_budget_seconds`` is the campaign-wide wall-clock cap and is
    enforced at both levels: whatever remains of it is handed to each inner
    campaign as *its* stage budget, and an outer candidate that would start
    after the budget is spent is recorded ``not_attempted`` rather than
    launched. The multiplicative worst case (:func:`estimate_cost`) is on
    the returned report either way.
    """
    _validate_campaign_shape(
        outer_population_size=outer_population_size,
        outer_population_count=outer_population_count,
        inner_population_size=inner_population_size,
        inner_population_count=inner_population_count,
        outer_mutation_rate=outer_mutation_rate,
        inner_mutation_rate=inner_mutation_rate,
    )

    campaign_id = run_id_prefix or f"arch{seed}"
    objective = str(base_spec.training["objective"])
    selection_metric = base_spec.evaluation["selection_metric"]
    selection_mode = _selection_metric_mode(selection_metric)
    estimate = estimate_cost(
        base_spec,
        outer_population_size=outer_population_size,
        outer_population_count=outer_population_count,
        inner_population_size=inner_population_size,
        inner_population_count=inner_population_count,
    )
    inner_kwargs: Dict[str, Any] = {
        "method": method,
        "mutation_rate": inner_mutation_rate,
        "root": root,
        "corpus_root": corpus_root,
        "naming_seed": naming_seed,
        "trace_dir": trace_dir,
        "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        "min_episode_length": min_episode_length,
    }
    if cost_model is not None:
        inner_kwargs["cost_model"] = cost_model

    members = [
        _ArchitectureMember(genome=genome, spec=spec, origin_generation=0)
        for genome, spec in _distinct_architecture_genomes(
            architecture_schema, base_spec, outer_population_size, seed, method=method,
        )
    ]

    generations: List[ArchitectureGenerationReport] = []
    final_survivors: List[_ArchitectureMember] = []
    stage_started = time.monotonic()
    stage_budget_exceeded = False
    inner_campaigns_run = 0
    total_trials_started = 0
    total_epochs_executed = 0

    for generation_index in range(outer_population_count):
        evaluated: List[_ArchitectureMember] = []
        displayed: List[ArchitectureCandidateResult] = []

        for candidate_index, member in enumerate(members):
            architecture_id = architecture_run_id_prefix(
                campaign_id, generation_index, candidate_index,
            )
            if member.evaluation is not None:
                carried = replace(
                    member.evaluation,
                    candidate_index=candidate_index,
                    retained=False,
                    carried=True,
                    inner_trials_started=0,
                    inner_epochs_executed=0,
                    reason=(
                        "carried forward from the preceding generation without re-running "
                        "its inner hyperparameter campaign"
                    ),
                )
                member.evaluation = carried
                evaluated.append(member)
                displayed.append(carried)
                continue

            remaining: Optional[float] = None
            if stage_budget_seconds is not None:
                remaining = stage_budget_seconds - (time.monotonic() - stage_started)
                if remaining <= 0:
                    stage_budget_exceeded = True
                    skipped = ArchitectureCandidateResult(
                        candidate_index=candidate_index,
                        architecture_id=architecture_id,
                        origin_generation=member.origin_generation,
                        genome=dict(member.genome),
                        model=dict(member.spec.model),
                        state=STATE_NOT_ATTEMPTED,
                        best_run_id=None,
                        metric_value=None,
                        inner_trials_started=0,
                        inner_epochs_executed=0,
                        quality_passed=False,
                        retained=False,
                        carried=False,
                        configuration_parents=member.configuration_parents,
                        breeding_lineage=member.breeding_lineage,
                        reason=(
                            f"skipped: the campaign's stage_budget_seconds "
                            f"({stage_budget_seconds!r}) was already exhausted before this "
                            "architecture's inner campaign could start"
                        ),
                    )
                    member.evaluation = skipped
                    evaluated.append(member)
                    displayed.append(skipped)
                    continue

            inner_seed = (
                seed
                + (generation_index + 1) * _GENERATION_SEED_STRIDE
                + candidate_index * _CANDIDATE_SEED_STRIDE
            )
            try:
                inner_report = run_evolutionary_search(
                    member.spec,
                    hyperparameter_schema,
                    inner_population_size,
                    inner_population_count,
                    inner_seed,
                    run_id_prefix=architecture_id,
                    stage_budget_seconds=remaining,
                    **inner_kwargs,
                )
            except Exception as exc:
                failed = ArchitectureCandidateResult(
                    candidate_index=candidate_index,
                    architecture_id=architecture_id,
                    origin_generation=member.origin_generation,
                    genome=dict(member.genome),
                    model=dict(member.spec.model),
                    state=STATE_FAILED,
                    best_run_id=None,
                    metric_value=None,
                    inner_trials_started=0,
                    inner_epochs_executed=0,
                    quality_passed=False,
                    retained=False,
                    carried=False,
                    configuration_parents=member.configuration_parents,
                    breeding_lineage=member.breeding_lineage,
                    reason=(
                        f"inner hyperparameter campaign failed with {type(exc).__name__}: {exc}"
                    ),
                )
                member.evaluation = failed
                evaluated.append(member)
                displayed.append(failed)
                continue

            inner_campaigns_run += 1
            total_trials_started += inner_report.total_trials_started
            total_epochs_executed += inner_report.total_epochs_executed
            if inner_report.stage_budget_exceeded:
                stage_budget_exceeded = True

            quality_passed = (
                inner_report.best_run_id is not None and inner_report.best_metric_value is not None
            )
            result = ArchitectureCandidateResult(
                candidate_index=candidate_index,
                architecture_id=architecture_id,
                origin_generation=member.origin_generation,
                genome=dict(member.genome),
                model=dict(member.spec.model),
                state="completed" if quality_passed else STATE_NO_INNER_SURVIVOR,
                best_run_id=inner_report.best_run_id,
                metric_value=inner_report.best_metric_value,
                inner_trials_started=inner_report.total_trials_started,
                inner_epochs_executed=inner_report.total_epochs_executed,
                quality_passed=quality_passed,
                retained=False,
                carried=False,
                configuration_parents=member.configuration_parents,
                breeding_lineage=member.breeding_lineage,
                reason=(
                    f"scored by its inner campaign's best {selection_metric!r}"
                    if quality_passed else
                    "eliminated: the inner hyperparameter campaign produced no surviving "
                    "candidate for this architecture"
                ),
            )
            member.evaluation = result
            evaluated.append(member)
            displayed.append(result)

        eligible = sorted(
            (member for member in evaluated if member.evaluation and member.evaluation.quality_passed),
            key=lambda member: (
                member.evaluation.metric_value
                if selection_mode == "min" else -member.evaluation.metric_value,
                member.evaluation.architecture_id,
            ),
        )
        survivor_limit = (
            min(max(1, len(eligible) // 2), max(1, outer_population_size // 2)) if eligible else 0
        )
        survivors = eligible[:survivor_limit]
        survivor_member_ids = {id(member) for member in survivors}

        finalized: List[ArchitectureCandidateResult] = []
        for member, result in zip(evaluated, displayed):
            retained = id(member) in survivor_member_ids
            reason = result.reason
            if result.quality_passed:
                reason = (
                    f"retained among the best {survivor_limit} architecture(s) by "
                    f"{selection_metric!r}"
                    if retained else
                    f"ranked below the best {survivor_limit} architecture(s) by "
                    f"{selection_metric!r}"
                )
            updated = replace(result, retained=retained, reason=reason)
            member.evaluation = updated
            finalized.append(updated)

        offspring: List[_ArchitectureMember] = []
        offspring_ids: List[str] = []
        if generation_index + 1 < outer_population_count and survivors and not stage_budget_exceeded:
            parents = [
                ArchitectureParent(
                    architecture_id=member.evaluation.architecture_id,
                    run_id=member.evaluation.best_run_id,
                    genome=member.genome,
                    metric_value=member.evaluation.metric_value,
                )
                for member in survivors
            ]
            child_count = outer_population_size - len(survivors)
            for child_offset in range(child_count):
                child_slot = len(survivors) + child_offset
                base_breeding_seed = (
                    seed + (generation_index + 1) * _GENERATION_SEED_STRIDE + child_offset
                )
                bred = None
                for attempt in range(64):
                    breeding_seed = base_breeding_seed + attempt * _RESAMPLE_SEED_STRIDE
                    rng = random.Random(breeding_seed)
                    if len(parents) == 1:
                        parent_a = parent_b = parents[0]
                    else:
                        parent_a, parent_b = rng.sample(parents, 2)
                    try:
                        bred = breed_architecture(
                            parent_a, parent_b, architecture_schema, base_spec,
                            objective=objective, generation=generation_index + 1,
                            seed=breeding_seed, selection_mode=selection_mode,
                            mutation_rate=outer_mutation_rate,
                        )
                    except GenomeRepairError:
                        continue
                    break
                if bred is None:
                    raise ArchitectureSearchError(
                        f"could not breed a valid architecture child for generation "
                        f"{generation_index + 1} slot {child_slot} after 64 deterministic attempts"
                    )
                offspring.append(_ArchitectureMember(
                    genome=bred.child_genome,
                    spec=bred.child_spec,
                    origin_generation=generation_index + 1,
                    configuration_parents=(parent_a.run_id, parent_b.run_id),
                    breeding_lineage=bred.lineage,
                ))
                offspring_ids.append(
                    architecture_run_id_prefix(campaign_id, generation_index + 1, child_slot)
                )

        generations.append(ArchitectureGenerationReport(
            generation_index=generation_index,
            results=tuple(finalized),
            survivor_architecture_ids=tuple(
                member.evaluation.architecture_id for member in survivors if member.evaluation
            ),
            offspring_architecture_ids=tuple(offspring_ids),
        ))
        final_survivors = survivors

        if generation_index + 1 >= outer_population_count or not survivors or stage_budget_exceeded:
            break
        members = survivors + offspring

    best = final_survivors[0] if final_survivors else None
    best_run_id = best.evaluation.best_run_id if best else None
    return ArchitectureSearchReport(
        format=ARCHITECTURE_SEARCH_FORMAT,
        seed=seed,
        method=method,
        architecture_schema_version=architecture_schema.version,
        hyperparameter_schema_version=hyperparameter_schema.version,
        estimate=estimate,
        outer_mutation_rate=outer_mutation_rate,
        inner_mutation_rate=inner_mutation_rate,
        generations=tuple(generations),
        inner_campaigns_run=inner_campaigns_run,
        total_trials_started=total_trials_started,
        total_epochs_executed=total_epochs_executed,
        stage_budget_exceeded=stage_budget_exceeded,
        final_survivor_architecture_ids=tuple(
            member.evaluation.architecture_id for member in final_survivors if member.evaluation
        ),
        best_architecture_id=best.evaluation.architecture_id if best else None,
        best_run_id=best_run_id,
        best_metric_value=best.evaluation.metric_value if best else None,
        best_genome=dict(best.genome) if best else None,
        best_model=dict(best.spec.model) if best else None,
        champion_comparison=_champion_comparison(
            root=root,
            organism=base_spec.organism,
            champion_run_id=champion_run_id,
            candidate_run_id=best_run_id,
            selection_metric=selection_metric,
            ticks_per_frame=_ticks_per_frame(base_spec, corpus_root),
        ) if champion_run_id is not None and best_run_id is not None else None,
    )


def _ticks_per_frame(base_spec: ExperimentSpec, corpus_root: Optional[Union[str, Path]]) -> float:
    """The corpus's tick/frame ratio, needed only for a champion comparison.

    Resolved lazily (and only when a champion was actually named) so a
    campaign that declares no champion never pays for a second corpus
    resolution beyond the ones its inner campaigns already do.
    """
    from cognitive_runtime.training.model_factory.corpus import resolve_corpus

    corpus = resolve_corpus(
        base_spec.data["corpus_id"], allow_record=False, root=corpus_root,
        organism=base_spec.organism,
    )
    return float(corpus.data_contract.ticks_per_frame)


__all__ = [
    "ARCHITECTURE_SEARCH_FORMAT",
    "ARCHITECTURE_GENOME_OPERATOR",
    "ARCHITECTURE_LINEAGE_FORMAT",
    "STATE_NO_INNER_SURVIVOR",
    "ArchitectureSearchError",
    "architecture_run_id_prefix",
    "ArchitectureSearchEstimate",
    "estimate_cost",
    "ArchitectureParent",
    "ArchitectureBreedingLineage",
    "ArchitectureBreedingResult",
    "breed_architecture",
    "ArchitectureCandidateResult",
    "ArchitectureGenerationReport",
    "ArchitectureSearchReport",
    "ArchitecturePreviewCandidate",
    "ArchitectureSearchPreview",
    "preview_architecture_search",
    "run_architecture_search",
]
