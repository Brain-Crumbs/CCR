"""Bounded random and Latin-hypercube trial proposals (epic #212, Phase D
step 1, §13.1, issue #230).

``propose()`` turns one fixed-parent ``ExperimentSpec`` plus a declared
``GenomeSchema`` into ``n`` sibling ``ExperimentSpec``s that vary only the
schema's training-only genes -- §13.1's discipline: "Do not jointly vary
architecture, data split, objective, and loss weights." Every other block
(``organism``, ``mode``, ``parent``, ``data``, ``model``, ``evaluation``,
``evolution``) is copied from the base spec via ``dataclasses.replace``, so
a proposal batch can never accidentally drift the corpus or architecture
declaration a search is supposed to hold fixed. The budget tier
(``training.epoch_budget``/``step_budget``/``max_training_seconds``) is
additionally restored from the base spec unconditionally after every
genome merge, because those three fields are otherwise ordinary
``TrainingContract`` leaves a caller-supplied schema is free to declare a
gene against -- letting a batch's compute budget itself vary would make
its successive-halving comparisons meaningless.

Two sampling strategies:

* ``"random"`` draws each gene independently and uniformly (in log space
  for a log-scale gene, via ``genome.Gene.sample``).
* ``"lhs"`` (Latin hypercube) partitions each gene's own range into ``n``
  equal strata and places exactly one sample per stratum, independently
  permuted per gene. With the small trial counts this epic budgets (4-8
  trials -- epic §13.1), uniform random sampling frequently leaves whole
  regions of a range unexplored; LHS stratifies each dimension so a small
  batch still covers it. A log-scale gene is stratified in *log* space
  (``math.log``/``math.exp``), not linear space -- linear-space strata over
  a wide range such as ``optimizer.lr``'s ``[1e-5, 1e-2]`` would place
  almost every stratum's mass in the top decade.

Every sampled genome is repaired (``genome.repair``) and every resulting
spec is validated (``spec.validate``) before it is returned -- an invalid
spec is never emitted. Neither error is caught and retried: a repair or
validation failure means the *declared* search space itself needs
narrowing, not that this module should silently substitute a different
sample.

``architecture_hash``/``data_contract_hash`` (epic §5.1/§5.2) are computed
from a *built model* and a *resolved corpus* respectively (see
``runner._architecture_contract`` and ``corpus.resolve_corpus``) -- both
require torch and an on-disk corpus, neither of which this module touches.
Because only ``training`` is ever overridden here, a proposal batch's
``data`` and ``model`` blocks are identical to ``base_spec``'s by
construction, which is what actually keeps those two downstream hashes
identical across a batch; there is nothing for this module to compute
beyond preserving that invariant.

Pure Python -- no torch import -- so ``propose()`` is usable in the
core-only install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from cognitive_runtime.training.model_factory.genome import (
    Gene,
    GenomeSchema,
    project_cost_seconds,
    repair,
)
from cognitive_runtime.training.model_factory.spec import (
    ExperimentSpec,
    _thaw,
)
from cognitive_runtime.training.model_factory.spec import validate as validate_spec

METHODS: Tuple[str, ...] = ("random", "lhs")

#: §13.1/§15's fixed-budget-tier discipline is a spec-level, not merely
#: schema-level, invariant: `genome._assert_genes_are_training_contract_paths`
#: allows any `TrainingContract` field as a gene root, including these three
#: budget-defining fields, so a caller-supplied schema *can* declare a gene
#: rooted at one of them. Letting that vary the budget across siblings would
#: make successive-halving comparisons across a batch meaningless (unequal
#: compute) and could silently exceed the batch's intended budget tier, so
#: `propose()` restores these three fields from `base_spec` unconditionally
#: after merging any genome, regardless of what the schema declares.
_FIXED_BUDGET_FIELDS: Tuple[str, ...] = ("epoch_budget", "step_budget", "max_training_seconds")


class SearchError(ValueError):
    """A malformed proposal request (bad ``n``, ``method``, or objective)."""


_MISSING = object()


def _set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``mapping[a][b]... = value`` for a dotted gene path.

    ``mapping`` must already be a fully mutable (fresh, ``_thaw``-ed) tree;
    every intermediate node visited is mutated in place. Mirrors how genome
    gene roots are declared (``genome._assert_genes_are_training_contract_paths``):
    every gene name is rooted at a real ``TrainingContract`` field, so this
    always lands inside a spec's ``training`` block -- but
    ``_assert_genes_are_training_contract_paths`` only checks that the
    *root* segment names a real field, not that every intermediate segment
    is itself mapping-valued (``genome.build_schema({"batch_size.foo": ...})``
    passes that check even though ``TrainingContract.batch_size`` is a
    scalar ``int``). Traversing through an existing non-mapping value would
    otherwise silently replace it with a fresh ``{}``, and the corrupted
    leaf (a ``dict`` where a scalar belongs) would not be caught by
    ``spec.validate`` -- dataclass field annotations are not enforced at
    runtime -- surfacing only much later as a ``TypeError`` deep inside the
    trainer. Raise here instead, at the moment the schema/spec mismatch is
    actually knowable.
    """
    parts = dotted_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        existing = cursor.get(part, _MISSING)
        if existing is _MISSING:
            existing = {}
        elif not isinstance(existing, Mapping):
            raise SearchError(
                f"gene {dotted_path!r} traverses through {part!r}, an existing "
                f"non-mapping training value ({existing!r}); every intermediate "
                "segment of a dotted gene path must name a mapping-valued "
                "TrainingContract field"
            )
        cursor[part] = dict(existing)
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_genome(training: Mapping[str, Any], genome: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge a repaired genome's dotted-path values onto a copy of ``training``."""
    merged = _thaw(training)
    for dotted_path, value in genome.items():
        _set_nested(merged, dotted_path, value)
    return merged


def _lhs_strata(n: int, rng: random.Random) -> List[float]:
    """``n`` independent stratified uniform samples in ``[0, 1)``.

    Classic Latin-hypercube jitter: stratum ``i`` contributes exactly one
    sample drawn uniformly from ``[i/n, (i+1)/n)``, and the strata are then
    visited in a permuted (not stratum-ordered) sequence so pairing across
    genes is randomized rather than correlated by stratum index.
    """
    order = list(range(n))
    rng.shuffle(order)
    return [(stratum + rng.random()) / n for stratum in order]


def _lhs_int_stratified_values(gene: Gene, n: int, rng: random.Random) -> List[int]:
    """Exact, collision-free stratification for a bounded, non-log-scale int gene.

    Continuous jitter-then-round (as used for float genes below) can push a
    rounded value into a *neighboring* stratum whenever the domain size
    isn't an exact multiple of ``n`` -- e.g. bounds ``(0, 8)`` (9 possible
    values) with ``n=8`` can round two different strata to the same
    integer while skipping another integer entirely, breaking the
    exactly-one-sample-per-stratum guarantee. Instead, partition the
    gene's ``domain_size = high - low + 1`` integers into ``n`` contiguous,
    non-overlapping blocks (as evenly sized as possible) and draw one
    uniformly random integer from *within* each block. Blocks never
    overlap by construction, so collisions are structurally impossible
    whenever ``n <= domain_size``; that inequality is checked explicitly
    because no assignment can satisfy the guarantee once it fails
    (pigeonhole).
    """
    low, high = int(gene.bounds[0]), int(gene.bounds[1])  # type: ignore[index]
    domain_size = high - low + 1
    if n > domain_size:
        raise SearchError(
            f"cannot draw {n} Latin-hypercube-stratified samples for int gene "
            f"{gene.name!r}, which has only {domain_size} distinct value(s) in "
            f"bounds {gene.bounds!r}; reduce n or widen the gene's bounds"
        )
    block_sizes = [domain_size // n + (1 if i < domain_size % n else 0) for i in range(n)]
    values: List[int] = []
    start = low
    for size in block_sizes:
        values.append(rng.randrange(start, start + size))
        start += size
    rng.shuffle(values)
    return values


def _lhs_bounded_values(gene: Gene, n: int, rng: random.Random) -> List[Any]:
    if gene.type == "int" and not gene.log_scale:
        return _lhs_int_stratified_values(gene, n, rng)
    # Float genes (and the rare log-scale int gene, where an exact discrete
    # partition of the log-space domain has no well-defined "one integer
    # per stratum" analogue) use continuous jitter; no shipped schema
    # declares a log-scale int gene today.
    low, high = gene.bounds  # type: ignore[misc]
    if gene.log_scale:
        low, high = math.log(low), math.log(high)
    values: List[Any] = []
    for unit in _lhs_strata(n, rng):
        raw = low + unit * (high - low)
        value = math.exp(raw) if gene.log_scale else raw
        values.append(int(round(value)) if gene.type == "int" else float(value))
    return values


def _lhs_choice_values(gene: Gene, n: int, rng: random.Random) -> List[Any]:
    """A balanced categorical analogue of stratification.

    Every choice is used ``n // k`` times (``k`` = number of choices), the
    ``n % k`` remainder fills additional *distinct* choices so no choice is
    ever used unevenly by more than one, and the whole assignment is
    shuffled so choice does not correlate with draw order.
    """
    choices = list(gene.choices)  # type: ignore[arg-type]
    k = len(choices)
    full_rounds, remainder = divmod(n, k)
    assignment = choices * full_rounds
    assignment.extend(rng.sample(choices, remainder))
    rng.shuffle(assignment)
    return assignment


def _latin_hypercube_genomes(schema: GenomeSchema, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    columns: Dict[str, List[Any]] = {
        name: (_lhs_choice_values(gene, n, rng) if gene.choices is not None else _lhs_bounded_values(gene, n, rng))
        for name, gene in schema.genes.items()
    }
    return [{name: columns[name][i] for name in schema.genes} for i in range(n)]


def _random_genomes(schema: GenomeSchema, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    return [{name: gene.sample(rng) for name, gene in schema.genes.items()} for _ in range(n)]


def propose(
    base_spec: ExperimentSpec,
    genome_schema: GenomeSchema,
    n: int,
    seed: int,
    *,
    method: str = "lhs",
    min_episode_length: Optional[int] = None,
    stage_budget_seconds: Optional[float] = None,
    cost_model: Callable[[Mapping[str, Any]], float] = project_cost_seconds,
) -> List[ExperimentSpec]:
    """Propose ``n`` sibling trials that vary only ``genome_schema``'s genes.

    ``base_spec`` (an already-resolved ``ExperimentSpec``, i.e. the output
    of ``spec.resolve()``) supplies everything a proposal batch must hold
    fixed: the same ``organism``, ``mode``, ``parent`` (fixed-parent search,
    epic §13.1), ``data``, ``model`` and ``evaluation`` blocks. Only
    ``base_spec.training`` is overridden, one dotted-path gene value at a
    time.

    The same ``seed`` always produces the same proposal set: every random
    draw goes through ``random.Random(seed)``, and a ``GenomeSchema``'s
    genes preserve declaration order (a plain ``dict``), so neither
    ``PYTHONHASHSEED`` nor process identity can perturb the result.

    Raises :class:`SearchError` for a malformed request (unknown ``method``
    or non-positive ``n``); a genome that fails ``genome.repair`` or a spec
    that fails ``spec.validate`` after that raises their own
    ``GenomeRepairError``/``SpecError`` -- neither is ever caught and
    silently replaced with a fresh sample, so any exception means the
    *declared* search space itself needs narrowing.
    """
    if method not in METHODS:
        raise SearchError(f"unknown method {method!r}; expected one of {METHODS}")
    if n < 1:
        raise SearchError(f"n must be a positive integer; got {n!r}")

    objective = base_spec.training.get("objective")
    rng = random.Random(seed)
    genomes = (
        _latin_hypercube_genomes(genome_schema, n, rng)
        if method == "lhs"
        else _random_genomes(genome_schema, n, rng)
    )

    proposals: List[ExperimentSpec] = []
    for genome in genomes:
        repaired = repair(
            genome_schema,
            genome,
            objective=objective,
            min_episode_length=min_episode_length,
            stage_budget_seconds=stage_budget_seconds,
            cost_model=cost_model,
        )
        training = _apply_genome(base_spec.training, repaired)
        for field in _FIXED_BUDGET_FIELDS:
            training[field] = base_spec.training.get(field)
        candidate = replace(base_spec, training=training)
        validate_spec(candidate)
        proposals.append(candidate)
    return proposals


__all__ = ["METHODS", "SearchError", "propose"]
