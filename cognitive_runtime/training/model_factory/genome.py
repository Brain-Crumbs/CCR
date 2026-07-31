"""Versioned genome schema for Model Factory's budgeted/genetic search.

Epic #212 (Model Factory), Phase D step 1, §14.2. A genome schema declares
the bounded set of tunable *training-only* configuration genes a search
policy may recombine and mutate for one stage. It never declares an
architecture field, a scenario-generator field, or a reward-semantics field
(§14.5) -- those are simply not representable as genes here, not merely
discouraged by convention (see :func:`_assert_no_forbidden_genes`).

Each :class:`Gene` declares its type, a bounded range or a discrete set of
choices, a default, and a mutation distribution; a declaration missing any
of these fails to load with :class:`GenomeSchemaError`. A gene may also
declare which training objectives it is active under -- a windowed-only loss
gene has no effect under the ``autoregressive`` objective (the two
objectives use different loss terms; see ``ActionWorldModelConfig`` at
``action_world_model.py:413``) and is excluded from crossover and mutation
for that objective.

:func:`repair` canonicalises a genome (clipping/snapping every gene value
into its declared bounds or choices) and rejects invalid combinations before
launch: ``warmup_frames + rollout_frames`` exceeding the corpus's minimum
episode length, or a projected cost exceeding the stage budget.

Pure Python -- no torch import -- so this module is constructible and usable
in the core-only install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple

from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    contract_hash,
)

#: The two ``ActionWorldModelConfig.training_objective`` values a gene's
#: conditional validity can be expressed in terms of.
OBJECTIVES: Tuple[str, ...] = ("windowed_rollout", "autoregressive")

GENE_TYPES: Tuple[str, ...] = ("float", "int", "categorical")

MUTATION_DISTRIBUTIONS: Tuple[str, ...] = (
    "log_normal_perturb",
    "normal_perturb",
    "categorical_resample",
)

#: DataContract/generator fields (epic §5.2) that name the scenario/program
#: identity, plus a few reward-semantics names used elsewhere in the
#: codebase (``cognitive_runtime/programs/minecraft/rewards.py``). §14.5
#: forbids a gene from reaching either category. Unlike architecture fields,
#: there is no single frozen dataclass enumerating every scenario-generator
#: or reward field in the codebase, so this list is a best-effort
#: complement to the structural ``ArchitectureContract`` check below.
_SCENARIO_AND_REWARD_FIELDS: FrozenSet[str] = frozenset({
    "world", "backend", "program_config", "scenario_names",
    "scenario_code_version", "goal", "reward",
    "reward_weights", "reward_shaping", "reward_config", "terminal_condition",
})


class GenomeSchemaError(ValueError):
    """A malformed, unsafe, or unversioned genome schema."""


class GenomeRepairError(ValueError):
    """A genome (or genome combination) invalid before launch (§14.2)."""


def _require_known_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}; expected one of {OBJECTIVES}")


@dataclass(frozen=True)
class MutationSpec:
    """A gene's declared mutation distribution (§14.2's four operators).

    ``log_normal_perturb``/``normal_perturb`` perturb a bounded numeric
    value in log or linear space respectively -- log space for learning
    rate and weight decay, as §14.2 specifies. ``categorical_resample``
    picks a different allowed value from a gene's choices.
    """

    distribution: str
    sigma: Optional[float] = None

    def __post_init__(self) -> None:
        if self.distribution not in MUTATION_DISTRIBUTIONS:
            raise GenomeSchemaError(
                f"unknown mutation distribution {self.distribution!r}; "
                f"expected one of {MUTATION_DISTRIBUTIONS}"
            )
        needs_sigma = self.distribution in ("log_normal_perturb", "normal_perturb")
        if needs_sigma and not (self.sigma is not None and self.sigma > 0):
            raise GenomeSchemaError(
                f"mutation distribution {self.distribution!r} requires a positive sigma"
            )
        if not needs_sigma and self.sigma is not None:
            raise GenomeSchemaError(
                f"mutation distribution {self.distribution!r} does not use sigma; got {self.sigma!r}"
            )


@dataclass(frozen=True)
class Gene:
    """One tunable configuration gene: type, range/choices, default, mutation.

    Exactly one of ``bounds`` (a ``(low, high)`` range, for ``float``/``int``
    genes) or ``choices`` (a discrete allowed set, for any type) must be
    declared. ``active_objectives``, when declared, restricts the gene to
    the named :data:`OBJECTIVES`; ``None`` means active under every
    objective (§14.2's conditional validity).
    """

    name: str
    type: str
    default: Any
    mutation: MutationSpec
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[Tuple[Any, ...]] = None
    log_scale: bool = False
    active_objectives: Optional[FrozenSet[str]] = None

    def __post_init__(self) -> None:
        if self.type not in GENE_TYPES:
            raise GenomeSchemaError(
                f"gene {self.name!r} has unknown type {self.type!r}; expected one of {GENE_TYPES}"
            )
        if (self.bounds is None) == (self.choices is None):
            raise GenomeSchemaError(
                f"gene {self.name!r} must declare exactly one of 'bounds' or 'choices'"
            )
        if self.choices is not None:
            if not self.choices:
                raise GenomeSchemaError(f"gene {self.name!r} 'choices' must not be empty")
            if self.default not in self.choices:
                raise GenomeSchemaError(
                    f"gene {self.name!r} default {self.default!r} is not among its choices {self.choices!r}"
                )
            if self.mutation.distribution != "categorical_resample":
                raise GenomeSchemaError(
                    f"gene {self.name!r} declares 'choices' and must use the "
                    f"'categorical_resample' mutation, not {self.mutation.distribution!r}"
                )
        if self.bounds is not None:
            if self.type == "categorical":
                raise GenomeSchemaError(
                    f"gene {self.name!r} is 'categorical' and must use 'choices', not 'bounds'"
                )
            low, high = self.bounds
            if not low < high:
                raise GenomeSchemaError(f"gene {self.name!r} bounds must satisfy low < high; got {self.bounds!r}")
            if self.log_scale and low <= 0:
                raise GenomeSchemaError(f"gene {self.name!r} is log-scaled and requires a positive lower bound")
            if not low <= self.default <= high:
                raise GenomeSchemaError(
                    f"gene {self.name!r} default {self.default!r} is outside its bounds {self.bounds!r}"
                )
            expected_distribution = "log_normal_perturb" if self.log_scale else "normal_perturb"
            if self.mutation.distribution != expected_distribution:
                raise GenomeSchemaError(
                    f"gene {self.name!r} (log_scale={self.log_scale}) must use the "
                    f"{expected_distribution!r} mutation, not {self.mutation.distribution!r}"
                )
        if self.active_objectives is not None:
            if not self.active_objectives:
                raise GenomeSchemaError(f"gene {self.name!r} 'active_objectives' must not be empty when declared")
            unknown = self.active_objectives - set(OBJECTIVES)
            if unknown:
                raise GenomeSchemaError(
                    f"gene {self.name!r} declares unknown active_objectives {sorted(unknown)!r}; "
                    f"expected a subset of {OBJECTIVES}"
                )

    def is_active(self, objective: str) -> bool:
        """Whether this gene has any effect under ``objective`` (§14.2)."""
        _require_known_objective(objective)
        return self.active_objectives is None or objective in self.active_objectives

    def sample(self, rng: random.Random) -> Any:
        """Draw a fresh, in-range value -- used to seed an initial population."""
        if self.choices is not None:
            return rng.choice(self.choices)
        low, high = self.bounds  # type: ignore[misc]
        if self.log_scale:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        return int(round(value)) if self.type == "int" else float(value)

    def mutate(self, value: Any, rng: random.Random) -> Any:
        """Perturb (numeric) or resample (categorical) ``value`` per §14.2."""
        if self.choices is not None:
            remaining = [choice for choice in self.choices if choice != value]
            return rng.choice(remaining) if remaining else value
        low, high = self.bounds  # type: ignore[misc]
        if self.mutation.distribution == "log_normal_perturb":
            anchor = math.log(min(max(float(value), low), high))
            perturbed = math.exp(rng.gauss(anchor, self.mutation.sigma))
        else:
            perturbed = rng.gauss(float(value), self.mutation.sigma)
        clipped = min(max(perturbed, low), high)
        return int(round(clipped)) if self.type == "int" else float(clipped)

    def clip(self, value: Any) -> Any:
        """Canonicalize ``value`` into this gene's declared range/choices.

        Used by :func:`repair`. A choices gene rejects a value outside its
        allowed set rather than silently snapping to the nearest one --
        there is no well-defined "nearest" for an arbitrary choice type.
        """
        if self.choices is not None:
            if value not in self.choices:
                raise GenomeRepairError(
                    f"gene {self.name!r} value {value!r} is not among its allowed choices {self.choices!r}"
                )
            return value
        low, high = self.bounds  # type: ignore[misc]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise GenomeRepairError(f"gene {self.name!r} value {value!r} is not numeric") from exc
        clipped = min(max(numeric, low), high)
        return int(round(clipped)) if self.type == "int" else float(clipped)


def _build_gene(name: str, spec: Mapping[str, Any]) -> Gene:
    """Build one :class:`Gene` from a declarative mapping, naming what is missing.

    This is the "schema missing any of these fails to load" acceptance
    check: constructing ``Gene`` directly would only raise a generic
    ``TypeError`` from the dataclass machinery, not a message that names
    the offending gene and the specific missing field(s).
    """
    if not isinstance(spec, Mapping):
        raise GenomeSchemaError(f"gene {name!r} declaration must be a mapping, got {type(spec).__name__}")
    missing = [key for key in ("type", "default", "mutation") if key not in spec]
    if spec.get("bounds") is None and spec.get("choices") is None:
        missing.append("bounds or choices")
    if missing:
        raise GenomeSchemaError(f"gene {name!r} is missing required field(s): {', '.join(missing)}")

    mutation = spec["mutation"]
    if not isinstance(mutation, MutationSpec):
        if not isinstance(mutation, Mapping):
            raise GenomeSchemaError(f"gene {name!r} 'mutation' must be a mapping or MutationSpec")
        mutation = MutationSpec(**mutation)

    active_objectives = spec.get("active_objectives")
    return Gene(
        name=name,
        type=spec["type"],
        default=spec["default"],
        mutation=mutation,
        bounds=tuple(spec["bounds"]) if spec.get("bounds") is not None else None,
        choices=tuple(spec["choices"]) if spec.get("choices") is not None else None,
        log_scale=bool(spec.get("log_scale", False)),
        active_objectives=frozenset(active_objectives) if active_objectives is not None else None,
    )


_ARCHITECTURE_FIELDS: FrozenSet[str] = frozenset(
    field.name for field in dataclass_fields(ArchitectureContract)
)


def _assert_no_forbidden_genes(version: str, gene_names: Tuple[str, ...]) -> None:
    """§14.5's guardrail, enforced structurally rather than by convention.

    A dotted gene name such as ``"optimizer.lr"`` is checked by its root
    (``"optimizer"``) as well as its full name, so a gene nested under a
    forbidden top-level field is caught too.
    """
    roots = {name.split(".", 1)[0] for name in gene_names}
    forbidden = (
        (set(gene_names) | roots) & (_ARCHITECTURE_FIELDS | _SCENARIO_AND_REWARD_FIELDS)
    )
    if forbidden:
        raise GenomeSchemaError(
            f"genome schema {version!r} declares gene(s) reaching an architecture, "
            f"scenario-generator, or reward-semantics field, which §14.5 forbids: {sorted(forbidden)!r}"
        )


@dataclass(frozen=True)
class GenomeSchema:
    """A versioned, per-stage set of genes (§14.2).

    ``version`` names the stage schema (for example
    ``"generic_action_effects_v1"``). Bounds, choices, defaults, and
    mutation distributions are frozen for a given version: change any gene
    and the version must change too (see :data:`_EXPECTED_CONTENT_HASHES`
    and :func:`_verify_schema_integrity`).
    """

    version: str
    genes: Mapping[str, Gene]

    def __post_init__(self) -> None:
        object.__setattr__(self, "genes", MappingProxyType(dict(self.genes)))
        _assert_no_forbidden_genes(self.version, tuple(self.genes))

    @property
    def gene_names(self) -> Tuple[str, ...]:
        return tuple(self.genes)

    def active_gene_names(self, objective: str) -> Tuple[str, ...]:
        """Genes with any effect under ``objective`` -- see :meth:`Gene.is_active`."""
        return tuple(name for name, gene in self.genes.items() if gene.is_active(objective))

    @property
    def content_hash(self) -> str:
        """SHA-256 of every gene's declared type/bounds/default/mutation.

        Two schemas of the same version must always hash identically;
        anything that changes this hash for a shipped version is exactly
        the "gene bounds changed without a version bump" mistake §14.2
        guards against.
        """
        payload = {
            name: {
                "type": gene.type,
                "default": gene.default,
                "bounds": list(gene.bounds) if gene.bounds is not None else None,
                "choices": list(gene.choices) if gene.choices is not None else None,
                "log_scale": gene.log_scale,
                "mutation": {"distribution": gene.mutation.distribution, "sigma": gene.mutation.sigma},
                "active_objectives": (
                    sorted(gene.active_objectives) if gene.active_objectives is not None else None
                ),
            }
            for name, gene in self.genes.items()
        }
        return contract_hash(payload)


def build_schema(version: str, gene_specs: Mapping[str, Mapping[str, Any]]) -> GenomeSchema:
    """Build a :class:`GenomeSchema` from declarative ``{name: gene_spec}`` mappings."""
    genes = {name: _build_gene(name, spec) for name, spec in gene_specs.items()}
    return GenomeSchema(version=version, genes=genes)


def _verify_schema_integrity(schema: GenomeSchema, expected_content_hash: str) -> GenomeSchema:
    """Fail to load a shipped schema whose genes changed without a version bump.

    A version string alone is only a promise; this makes the promise
    enforced. Bumping bounds, a default, or a mutation distribution on
    ``schema.version`` in place changes ``content_hash`` and this raises
    immediately at import time -- the fix is a new schema version
    (``generic_action_effects_v2``), not editing v1's pinned hash.
    """
    if schema.content_hash != expected_content_hash:
        raise GenomeSchemaError(
            f"genome schema {schema.version!r} content changed (gene bounds, choices, "
            "default, or mutation) without a version bump: expected content hash "
            f"{expected_content_hash!r}, got {schema.content_hash!r}. Ship a new schema "
            "version instead of editing this one in place."
        )
    return schema


# §14.2's default training-only schema for the generic_action_effects stage.
# `transition_balance_policy.*` mirrors TrainingContract's field name
# (contracts.py) so a resolved gene value can be merged straight into an
# ExperimentSpec's `training` block via a dotted-path override.
_GENERIC_ACTION_EFFECTS_V1_GENES: Dict[str, Dict[str, Any]] = {
    "optimizer.lr": {
        "type": "float",
        "bounds": (1e-5, 1e-2),
        "log_scale": True,
        "default": 3e-4,
        "mutation": {"distribution": "log_normal_perturb", "sigma": 0.3},
    },
    "optimizer.weight_decay": {
        "type": "float",
        "bounds": (1e-7, 1e-1),
        "log_scale": True,
        "default": 1e-5,
        "mutation": {"distribution": "log_normal_perturb", "sigma": 0.5},
    },
    "scheduled_sampling_p": {
        "type": "float",
        "bounds": (0.0, 0.5),
        "default": 0.25,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.05},
    },
    # Windowed-only losses: `action_world_model.py`'s autoregressive path
    # (`_train_autoregressive_objective`) uses `autoregressive_rollout_*_weight`
    # instead, so these have no effect under that objective (issue #229).
    "closed_loop_pixel_loss_weight": {
        "type": "float",
        "bounds": (0.0, 1.0),
        "default": 0.25,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.1},
        "active_objectives": frozenset({"windowed_rollout"}),
    },
    "closed_loop_latent_loss_weight": {
        "type": "float",
        "bounds": (0.0, 1.0),
        "default": 0.25,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.1},
        "active_objectives": frozenset({"windowed_rollout"}),
    },
    "rollout_frames": {
        "type": "int",
        "choices": (4, 6, 8, 10, 12, 16),
        "default": 8,
        "mutation": {"distribution": "categorical_resample"},
    },
    "warmup_frames": {
        "type": "int",
        "choices": (1, 2, 3, 4, 6, 8),
        "default": 3,
        "mutation": {"distribution": "categorical_resample"},
    },
    "transition_balance_policy.stationary_cap": {
        "type": "float",
        "bounds": (0.0, 1.0),
        "default": 0.5,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.1},
    },
}

#: Pinned content hash for ``generic_action_effects_v1`` (see
#: :func:`_verify_schema_integrity`). Recomputed and asserted at import
#: time; a change to any gene above without also renaming the schema
#: version will fail to import.
_EXPECTED_CONTENT_HASHES: Dict[str, str] = {
    "generic_action_effects_v1": (
        "d1b981a3e03826f09cc2b255bb9a178eb0f03c7609a1bb66273dfe3662e9055c"
    ),
}

GENERIC_ACTION_EFFECTS_V1: GenomeSchema = _verify_schema_integrity(
    build_schema("generic_action_effects_v1", _GENERIC_ACTION_EFFECTS_V1_GENES),
    _EXPECTED_CONTENT_HASHES["generic_action_effects_v1"],
)

#: Registry of every declared per-stage schema (§14.2: "The factory declares
#: a versioned genome schema per stage.").
GENOME_SCHEMAS: Dict[str, GenomeSchema] = {
    GENERIC_ACTION_EFFECTS_V1.version: GENERIC_ACTION_EFFECTS_V1,
}


def get_schema(version: str) -> GenomeSchema:
    """Look up a declared schema by version; unknown versions are a hard error."""
    try:
        return GENOME_SCHEMAS[version]
    except KeyError:
        raise GenomeSchemaError(
            f"unknown genome schema version {version!r}; expected one of {sorted(GENOME_SCHEMAS)}"
        ) from None


def default_genome(schema: GenomeSchema) -> Dict[str, Any]:
    """Every gene at its declared default -- the schema's canonical baseline."""
    return {name: gene.default for name, gene in schema.genes.items()}


def sample_genome(schema: GenomeSchema, rng: random.Random) -> Dict[str, Any]:
    """Draw a fresh genome (every gene, active or not) -- seeds an initial population."""
    return {name: gene.sample(rng) for name, gene in schema.genes.items()}


def crossover(
    schema: GenomeSchema,
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    *,
    objective: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """Uniform crossover (§14.2): each *active* child gene comes from either parent.

    A gene inactive under ``objective`` is excluded from the coin flip and
    simply carried over from ``parent_a`` -- recombining a value neither
    parent's training run actually used would not be a meaningful choice.
    """
    _require_known_objective(objective)
    child: Dict[str, Any] = {}
    for name, gene in schema.genes.items():
        for parent, label in ((parent_a, "parent_a"), (parent_b, "parent_b")):
            if name not in parent:
                raise GenomeRepairError(f"{label} genome is missing gene {name!r}")
        if gene.is_active(objective):
            child[name] = parent_a[name] if rng.random() < 0.5 else parent_b[name]
        else:
            child[name] = parent_a[name]
    return child


def mutate(
    schema: GenomeSchema,
    genome: Mapping[str, Any],
    *,
    objective: str,
    rng: random.Random,
    mutation_rate: float = 0.2,
) -> Dict[str, Any]:
    """Per-gene bounded mutation (§14.2), skipping genes inactive under ``objective``.

    Each active gene independently mutates with probability
    ``mutation_rate``; an inactive gene is left untouched regardless.
    """
    _require_known_objective(objective)
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate must be in [0, 1]; got {mutation_rate!r}")
    missing = set(schema.genes) - set(genome)
    if missing:
        raise GenomeRepairError(f"genome is missing gene(s): {sorted(missing)}")
    child: Dict[str, Any] = dict(genome)
    for name, gene in schema.genes.items():
        if not gene.is_active(objective):
            continue
        if rng.random() < mutation_rate:
            child[name] = gene.mutate(genome[name], rng)
    return child


def project_cost_seconds(genome: Mapping[str, Any], *, seconds_per_frame: float = 1.0) -> float:
    """A deliberately naive pre-launch cost proxy for :func:`repair`.

    Real per-stage timing lives in :mod:`.budget` (measured, not projected)
    once a trial has actually run an epoch. Before launch there is no
    measured signal yet, so this proxies projected cost as work
    proportional to frames processed per training step
    (``warmup_frames + rollout_frames``). A caller with better prior timing
    data (for example a per-stage seconds/frame estimate from previous
    trials) should pass its own ``cost_model`` to :func:`repair` instead.
    """
    warmup = genome.get("warmup_frames", 0)
    rollout = genome.get("rollout_frames", 0)
    return float(seconds_per_frame) * float(warmup + rollout)


def repair(
    schema: GenomeSchema,
    genome: Mapping[str, Any],
    *,
    objective: str,
    min_episode_length: Optional[int] = None,
    stage_budget_seconds: Optional[float] = None,
    cost_model: Callable[[Mapping[str, Any]], float] = project_cost_seconds,
) -> Dict[str, Any]:
    """Canonicalize ``genome`` and reject invalid combinations before launch (§14.2).

    Every gene value is clipped/snapped into its declared bounds or choices
    first. Then, when declared:

    - ``warmup_frames + rollout_frames`` must fit ``min_episode_length``,
      naming both genes in the error if it does not;
    - the genome's projected cost (via ``cost_model``) must not exceed
      ``stage_budget_seconds``.

    Raises :class:`GenomeRepairError` on the first violation found.
    """
    _require_known_objective(objective)
    missing = set(schema.genes) - set(genome)
    if missing:
        raise GenomeRepairError(f"genome is missing gene(s): {sorted(missing)}")
    unknown = set(genome) - set(schema.genes)
    if unknown:
        raise GenomeRepairError(
            f"genome declares gene(s) unknown to schema {schema.version!r}: {sorted(unknown)}"
        )

    canonical: Dict[str, Any] = {
        name: gene.clip(genome[name]) for name, gene in schema.genes.items()
    }

    warmup = canonical.get("warmup_frames")
    rollout = canonical.get("rollout_frames")
    if (
        min_episode_length is not None
        and warmup is not None
        and rollout is not None
        and warmup + rollout > min_episode_length
    ):
        raise GenomeRepairError(
            "'warmup_frames' + 'rollout_frames' "
            f"({warmup} + {rollout} = {warmup + rollout}) exceeds the corpus's "
            f"minimum episode length ({min_episode_length})"
        )

    if stage_budget_seconds is not None:
        projected = cost_model(canonical)
        if projected > stage_budget_seconds:
            raise GenomeRepairError(
                f"genome's projected cost ({projected:.1f}s) exceeds the stage budget "
                f"({stage_budget_seconds:.1f}s)"
            )

    return canonical


__all__ = [
    "OBJECTIVES",
    "GENE_TYPES",
    "MUTATION_DISTRIBUTIONS",
    "GenomeSchemaError",
    "GenomeRepairError",
    "MutationSpec",
    "Gene",
    "GenomeSchema",
    "build_schema",
    "GENERIC_ACTION_EFFECTS_V1",
    "GENOME_SCHEMAS",
    "get_schema",
    "default_genome",
    "sample_genome",
    "crossover",
    "mutate",
    "project_cost_seconds",
    "repair",
]
