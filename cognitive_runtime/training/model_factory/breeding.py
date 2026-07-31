"""Genetic operators: crossover, mutation, weight donor, lineage recording
(epic #212, Model Factory, Phase D step 3, §14.1-14.2, issue #232).

§14.1's non-negotiable distinction is that an offspring has **two
configuration parents** but **exactly one explicit weight donor**::

    parent A config --\\
                       -> validated offspring configuration -> train from donor weights
    parent B config --/                                    -> new child checkpoint

The genome (:mod:`.genome`) recombines **configuration only** -- declared,
bounded, training-only fields. This module never imports ``torch`` and never
reads a model's ``state_dict``; it reads only the JSON-shaped ``trial_spec.json``,
``contracts.json`` and checkpoint compatibility header (sidecar) of two
already-completed parent runs, so there is no code path through which a
weight tensor could be averaged or spliced across parents -- the one
:func:`select_weight_donor`/explicit ``weight_donor`` choice always names a
single parent's checkpoint, never a blend of both.

:func:`load_parent` reads one already-completed run's compatibility identity
(architecture hash, corpus/data-contract hash, budget tier, evaluation
contract) and its resolved genome (extracted from ``trial_spec.json``'s
``training`` block via the same dotted-path gene names :mod:`.genome`
declares) into a :class:`ParentRecord`. "Stage" (epic §14.1) is which
:class:`~.genome.GenomeSchema` a breeding call declares -- both parents'
genomes are always extracted under the one schema a caller passes to
:func:`load_parent`, so two parents bred together are always read as the
same stage's genes by construction; :class:`ParentRecord` additionally
records the schema's own ``version`` so :func:`check_compatible` can still
catch a caller mistake (loading the two parents under two different declared
schema versions).

:func:`check_compatible` is the initial implementation's compatibility gate
(§14.1): both parents must share one architecture hash, one corpus (data
contract hash), one budget tier, and one evaluation contract, or breeding is
refused outright. Because a :class:`~.genome.GenomeSchema`'s genes can only
ever be rooted at a :class:`~.contracts.TrainingContract` field
(:func:`.genome._assert_genes_are_training_contract_paths`), no gene reaches
an architecture field -- an offspring that would change architecture is not
merely discouraged, it is unrepresentable through this module's crossover
and mutation path.

:func:`breed` runs the deterministic pipeline from one ``seed``: draw the
weight donor (unless the caller names one explicitly), uniform-crossover the
two parent genomes (:func:`.genome.crossover`'s exact semantics, replicated
here so every gene's chosen parent is recorded, not just its final value),
mutate (:func:`.genome.Gene.mutate` per active gene -- log space for
``optimizer.lr``/``optimizer.weight_decay``, since those genes declare
``log_scale=True`` and :mod:`.genome` enforces the matching mutation
distribution at schema-build time), then :func:`.genome.repair`. Every stage
records its own provenance, so **every gene in the resolved child genome
traces to exactly one named parent, a recorded mutation, or a recorded
repair** -- never an unexplained value. The result is a
:class:`BreedingResult`: a ready-to-launch child ``ExperimentSpec`` (mode
``"clone"`` -- same architecture and corpus as both parents, a new training
procedure, per :func:`~.checkpoint.validate_continuation`) plus a
:class:`BreedingLineage` with the complete parent genomes, the crossover
choices, the mutation seed, every repair, the generation, and the resolved
child genome.

``ExperimentSpec.evolution`` (:mod:`.spec`) is a closed schema
(``EVOLUTION_KEYS``): it holds only ``generation``, ``configuration_parents``,
``weight_donor``, ``genome_operator`` and ``mutations`` (a list of mutated
gene names), and that much is already persisted into ``trial_spec.json`` (and
lineage.json's own ``configuration_parents``/``weight_donor`` fields) the
moment the child is launched via the existing
:func:`~.artifacts.allocate_run_artifacts`. The complete parent genomes,
crossover choices, mutation seed and per-gene repairs are *not* representable
in that closed schema, so :func:`record_breeding_lineage` writes them
separately, nested under lineage.json's own ``"breeding"`` key, once the
child's run directory has been allocated.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.budget import budget_seconds_for_tier
from cognitive_runtime.training.model_factory.checkpoint import (
    read_factory_checkpoint_metadata,
)
from cognitive_runtime.training.model_factory.genome import (
    GenomeSchema,
    project_cost_seconds,
    repair,
)
from cognitive_runtime.training.model_factory.spec import (
    ExperimentSpec,
    _thaw,
    resolve as resolve_spec,
    validate as validate_spec,
)

#: The one genetic operator this module implements (§14.2's "uniform
#: crossover"). Recorded verbatim into a bred child's
#: ``evolution.genome_operator`` and its :class:`BreedingLineage`.
GENOME_OPERATOR = "uniform_crossover_v1"

#: Versioned identity for the breeding-specific record nested under a bred
#: child's ``lineage.json["breeding"]`` (see :func:`record_breeding_lineage`).
LINEAGE_DETAIL_FORMAT = "model-factory-breeding-lineage-v1"

#: §13.1/§15's fixed-budget-tier discipline, mirrored from
#: ``search._FIXED_BUDGET_FIELDS``: a caller-supplied schema can declare a
#: gene rooted at any of these three ``TrainingContract`` fields, and letting
#: crossover/mutation vary them would silently move a bred child out of the
#: budget tier both parents were required (by :func:`check_compatible`) to
#: share. Restored from the weight donor's own spec after every genome merge.
_FIXED_BUDGET_FIELDS: Tuple[str, ...] = ("epoch_budget", "step_budget", "max_training_seconds")

_MISSING = object()


class BreedingError(ValueError):
    """Incompatible breeding parents, or a malformed breeding request."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_run_spec(run_directory: Path) -> ExperimentSpec:
    path = run_directory / "trial_spec.json"
    if not path.is_file():
        raise BreedingError(f"no trial_spec.json for run at {run_directory}")
    return resolve_spec(_load_json(path))


def _get_nested(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    """Read ``mapping[a][b]...`` for a dotted gene path (the inverse of the
    dotted-path merge :func:`_apply_genome` performs when building a child).
    """
    cursor: Any = mapping
    consumed: list = []
    for part in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise BreedingError(
                f"training block is missing gene path {dotted_path!r} "
                f"(failed at {'.'.join(consumed + [part])!r})"
            )
        cursor = cursor[part]
        consumed.append(part)
    return cursor


def _extract_genome(schema: GenomeSchema, training: Mapping[str, Any]) -> Dict[str, Any]:
    """Read every one of ``schema``'s declared gene values out of a resolved
    ``ExperimentSpec.training`` block -- the parent's actual historical
    configuration, never a resampled or default value."""
    return {name: _get_nested(training, name) for name in schema.genes}


def _set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        existing = cursor.get(part, _MISSING)
        if existing is _MISSING:
            existing = {}
        elif not isinstance(existing, Mapping):
            raise BreedingError(
                f"gene {dotted_path!r} traverses through {part!r}, an existing "
                f"non-mapping training value ({existing!r})"
            )
        cursor[part] = dict(existing)
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_genome(training: Mapping[str, Any], genome: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _thaw(training)
    for dotted_path, value in genome.items():
        _set_nested(merged, dotted_path, value)
    return merged


@dataclass(frozen=True)
class ParentRecord:
    """One breeding parent's compatibility identity and resolved genome.

    Loaded once, from an already-completed run's own persisted artifacts
    (never resampled/guessed): ``genome`` is exactly what that run actually
    trained with, read out of its ``trial_spec.json``.
    """

    run_id: str
    genome_schema_version: str
    architecture_hash: str
    data_contract_hash: str
    corpus_id: Optional[str]
    tier: str
    evaluation_contract: Mapping[str, Any]
    genome: Mapping[str, Any]
    checkpoint_name: str
    checkpoint_sha256: str
    spec: ExperimentSpec


def load_parent(
    root: Union[str, Path],
    organism: str,
    run_id: str,
    genome_schema: GenomeSchema,
    *,
    tier: str,
    checkpoint_name: str = "best-validation.pt",
) -> ParentRecord:
    """Load ``run_id``'s compatibility identity and resolved genome.

    ``tier`` is the caller's declared §15.1 budget-tier label for this run
    (as with the champion registry's ``(family, tier, objective)`` slots,
    :mod:`.registry`, a tier is a declared portfolio label, not something
    inferred from a single numeric field); it is checked against the run's
    own persisted ``training.max_training_seconds`` so a mislabeled tier is
    rejected immediately rather than silently corrupting a later
    :func:`check_compatible` comparison.
    """
    directory = Path(root) / organism / run_id
    spec = _load_run_spec(directory)

    expected_seconds = budget_seconds_for_tier(tier)
    actual_seconds = spec.training.get("max_training_seconds")
    if actual_seconds != expected_seconds:
        raise BreedingError(
            f"run {run_id!r} declares training.max_training_seconds={actual_seconds!r}, "
            f"which does not match tier {tier!r}'s budget ({expected_seconds!r})"
        )

    contracts = _load_json(directory / "contracts.json")
    checkpoint_path = directory / "checkpoints" / checkpoint_name
    checkpoint_sha256 = read_factory_checkpoint_metadata(str(checkpoint_path))["checkpoint_sha256"]

    return ParentRecord(
        run_id=run_id,
        genome_schema_version=genome_schema.version,
        architecture_hash=contracts["architecture_hash"],
        data_contract_hash=contracts["data_contract_hash"],
        corpus_id=spec.data.get("corpus_id"),
        tier=tier,
        evaluation_contract=dict(spec.evaluation),
        genome=_extract_genome(genome_schema, spec.training),
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha256,
        spec=spec,
    )


def check_compatible(parent_a: ParentRecord, parent_b: ParentRecord) -> None:
    """§14.1's compatibility gate: refuse breeding outright unless both
    parents share one architecture hash, one corpus, one stage (genome
    schema version), one budget tier, and one evaluation contract.

    Raises :class:`BreedingError` naming every mismatched dimension at once
    (not just the first) so a caller sees the full incompatibility in one
    message.
    """
    mismatches = []
    if parent_a.architecture_hash != parent_b.architecture_hash:
        mismatches.append(
            f"architecture_hash ({parent_a.architecture_hash!r} != {parent_b.architecture_hash!r})"
        )
    if parent_a.data_contract_hash != parent_b.data_contract_hash:
        mismatches.append(
            f"corpus/data_contract_hash ({parent_a.data_contract_hash!r} != {parent_b.data_contract_hash!r})"
        )
    if parent_a.genome_schema_version != parent_b.genome_schema_version:
        mismatches.append(
            f"stage/genome schema version ({parent_a.genome_schema_version!r} != "
            f"{parent_b.genome_schema_version!r})"
        )
    if parent_a.tier != parent_b.tier:
        mismatches.append(f"budget tier ({parent_a.tier!r} != {parent_b.tier!r})")
    if parent_a.evaluation_contract != parent_b.evaluation_contract:
        mismatches.append(
            f"evaluation contract ({parent_a.evaluation_contract!r} != {parent_b.evaluation_contract!r})"
        )
    if mismatches:
        raise BreedingError(
            f"parents {parent_a.run_id!r} and {parent_b.run_id!r} are incompatible for breeding "
            "(epic #212 §14.1 requires an identical architecture hash, corpus, stage, budget tier "
            "and evaluation contract): " + "; ".join(mismatches)
        )


def select_weight_donor(
    parent_a: ParentRecord, parent_b: ParentRecord, rng: random.Random,
) -> ParentRecord:
    """The weight-donor operator: a single seeded, uniform choice between the
    two parents. Never blends or averages -- the result always names exactly
    one parent's checkpoint."""
    return parent_a if rng.random() < 0.5 else parent_b


def _crossover_with_provenance(
    schema: GenomeSchema,
    parent_a_genome: Mapping[str, Any],
    parent_b_genome: Mapping[str, Any],
    *,
    objective: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Uniform crossover (§14.2), identical rng consumption and semantics to
    :func:`.genome.crossover`, but also recording which named parent each
    gene was chosen from (an inactive gene is always attributed to
    ``"parent_a"``, exactly like :func:`.genome.crossover` always carries it
    over from ``parent_a`` without a coin flip)."""
    child: Dict[str, Any] = {}
    choices: Dict[str, str] = {}
    for name, gene in schema.genes.items():
        if gene.is_active(objective):
            chosen = "parent_a" if rng.random() < 0.5 else "parent_b"
        else:
            chosen = "parent_a"
        child[name] = parent_a_genome[name] if chosen == "parent_a" else parent_b_genome[name]
        choices[name] = chosen
    return child, choices


def _mutate_with_provenance(
    schema: GenomeSchema,
    genome: Mapping[str, Any],
    *,
    objective: str,
    rng: random.Random,
    mutation_rate: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Per-gene bounded mutation (§14.2), identical rng consumption and
    semantics to :func:`.genome.mutate`, but also recording each mutated
    gene's before/after value."""
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate must be in [0, 1]; got {mutation_rate!r}")
    child: Dict[str, Any] = dict(genome)
    mutations: Dict[str, Dict[str, Any]] = {}
    for name, gene in schema.genes.items():
        if not gene.is_active(objective):
            continue
        if rng.random() < mutation_rate:
            before = genome[name]
            after = gene.mutate(before, rng)
            child[name] = after
            mutations[name] = {"from": before, "to": after}
    return child, mutations


@dataclass(frozen=True)
class BreedingLineage:
    """The complete, JSON-safe genetic-operator record for one bred child
    (§14.2: "the complete parent genomes, crossover choices, mutation seed,
    repairs, and resolved child configuration")."""

    format: str
    generation: int
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "generation": self.generation,
            "parent_a_run_id": self.parent_a_run_id,
            "parent_b_run_id": self.parent_b_run_id,
            "weight_donor_run_id": self.weight_donor_run_id,
            "genome_operator": self.genome_operator,
            "mutation_seed": self.mutation_seed,
            "parent_a_genome": dict(self.parent_a_genome),
            "parent_b_genome": dict(self.parent_b_genome),
            "crossover_choices": dict(self.crossover_choices),
            "mutations": {name: dict(detail) for name, detail in self.mutations.items()},
            "repairs": {name: dict(detail) for name, detail in self.repairs.items()},
            "child_genome": dict(self.child_genome),
        }


@dataclass(frozen=True)
class BreedingResult:
    """A ready-to-launch bred child plus its complete genetic-operator record."""

    child_spec: ExperimentSpec
    lineage: BreedingLineage


def breed(
    parent_a: ParentRecord,
    parent_b: ParentRecord,
    genome_schema: GenomeSchema,
    *,
    objective: str,
    generation: int,
    seed: int,
    mutation_rate: float = 0.2,
    weight_donor: Optional[str] = None,
    min_episode_length: Optional[int] = None,
    stage_budget_seconds: Optional[float] = None,
    cost_model=project_cost_seconds,
) -> BreedingResult:
    """Breed ``parent_a``/``parent_b`` into one child under ``genome_schema``.

    Refuses incompatible parents (:func:`check_compatible`) before drawing
    anything from ``seed``. The deterministic order is: draw the weight donor
    (skipped when ``weight_donor`` names one of the two parents explicitly),
    uniform-crossover every gene, mutate each active gene independently at
    ``mutation_rate``, then :func:`.genome.repair`. The same ``seed`` (and the
    same explicit ``weight_donor``, if given) always reproduces the same
    child exactly, because every random draw goes through
    ``random.Random(seed)`` in this fixed order and a
    :class:`~.genome.GenomeSchema`'s genes preserve declaration order.

    The child is always mode ``"clone"``: both parents already share one
    architecture hash and one corpus (:func:`check_compatible`), and cloning
    is exactly "same architecture and data, new training procedure"
    (:func:`~.checkpoint.validate_continuation`) -- never ``"fine_tune"``,
    which epic §14.1 reserves for a *stage transition* onto a different
    corpus. The child's ``data``/``model`` blocks are copied unchanged from
    the weight donor's own spec (compatibility already guarantees the other
    parent's are equivalent), and ``epoch_budget``/``step_budget``/
    ``max_training_seconds`` are restored from the donor's spec after the
    genome merge so a caller-supplied schema can never let crossover or
    mutation drift the child out of the budget tier both parents share.
    """
    check_compatible(parent_a, parent_b)
    rng = random.Random(seed)

    if weight_donor is None:
        donor = select_weight_donor(parent_a, parent_b, rng)
    elif weight_donor == parent_a.run_id:
        donor = parent_a
    elif weight_donor == parent_b.run_id:
        donor = parent_b
    else:
        raise BreedingError(
            f"weight_donor {weight_donor!r} must name one of the two breeding parents "
            f"({parent_a.run_id!r}, {parent_b.run_id!r})"
        )

    crossed_genome, crossover_choices = _crossover_with_provenance(
        genome_schema, parent_a.genome, parent_b.genome, objective=objective, rng=rng,
    )
    mutated_genome, mutations = _mutate_with_provenance(
        genome_schema, crossed_genome, objective=objective, rng=rng, mutation_rate=mutation_rate,
    )
    child_genome = repair(
        genome_schema, mutated_genome, objective=objective,
        min_episode_length=min_episode_length, stage_budget_seconds=stage_budget_seconds,
        cost_model=cost_model,
    )
    repairs = {
        name: {"from": mutated_genome[name], "to": child_genome[name]}
        for name in genome_schema.genes
        if mutated_genome[name] != child_genome[name]
    }

    child_training = _apply_genome(donor.spec.training, child_genome)
    for field in _FIXED_BUDGET_FIELDS:
        child_training[field] = donor.spec.training.get(field)

    child_evolution = {
        "generation": generation,
        "configuration_parents": [parent_a.run_id, parent_b.run_id],
        "weight_donor": donor.run_id,
        "genome_operator": GENOME_OPERATOR,
        "mutations": sorted(mutations),
    }
    child_spec = replace(
        donor.spec,
        mode="clone",
        parent={
            "run_id": donor.run_id,
            "checkpoint": donor.checkpoint_name,
            "sha256": donor.checkpoint_sha256,
        },
        training=child_training,
        evolution=child_evolution,
    )
    validate_spec(child_spec)

    lineage = BreedingLineage(
        format=LINEAGE_DETAIL_FORMAT,
        generation=generation,
        parent_a_run_id=parent_a.run_id,
        parent_b_run_id=parent_b.run_id,
        weight_donor_run_id=donor.run_id,
        genome_operator=GENOME_OPERATOR,
        mutation_seed=seed,
        parent_a_genome=dict(parent_a.genome),
        parent_b_genome=dict(parent_b.genome),
        crossover_choices=crossover_choices,
        mutations=mutations,
        repairs=repairs,
        child_genome=dict(child_genome),
    )
    return BreedingResult(child_spec=child_spec, lineage=lineage)


def record_breeding_lineage(run_directory: Union[str, Path], lineage: BreedingLineage) -> Path:
    """Enrich an already-allocated child run's ``lineage.json`` with the full
    breeding record.

    ``allocate_run_artifacts`` (MF-A3) already wrote lineage.json's generic,
    every-mode fields (``mode``, ``parent``, ``configuration_parents``,
    ``weight_donor``) from the child spec's own ``parent``/``evolution``
    blocks by the time this runs. This adds the operator-level detail issue
    #232 requires -- both complete parent genomes, the crossover choices, the
    mutation seed, and every repair -- nested under a ``"breeding"`` key, so
    lineage.json stays the single on-disk source for a bred child's full
    provenance without rewriting (or risking a mismatch with) the fields it
    shares with every other run mode.
    """
    path = Path(run_directory) / "lineage.json"
    if not path.is_file():
        raise BreedingError(f"no lineage.json to enrich at {run_directory}; allocate the run first")
    existing = _load_json(path)
    if existing.get("weight_donor") != lineage.weight_donor_run_id:
        raise BreedingError(
            f"lineage.json weight_donor ({existing.get('weight_donor')!r}) does not match "
            f"this breeding record's weight donor ({lineage.weight_donor_run_id!r}); refusing to "
            "attach a breeding record to a run it does not describe"
        )
    enriched = dict(existing)
    enriched["breeding"] = lineage.to_dict()
    return atomic_write_json(path, enriched)


__all__ = [
    "GENOME_OPERATOR",
    "LINEAGE_DETAIL_FORMAT",
    "BreedingError",
    "ParentRecord",
    "load_parent",
    "check_compatible",
    "select_weight_donor",
    "BreedingLineage",
    "BreedingResult",
    "breed",
    "record_breeding_lineage",
]
