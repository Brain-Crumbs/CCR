"""Bounded random and Latin-hypercube trial proposals (epic #212, Phase D
step 1, §13.1, issue #230), plus checkpoint-based successive halving
(Phase D step 2, §13.2, issue #231, :func:`run_successive_halving`).

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

import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from cognitive_runtime.training.model_factory.checkpoint import read_factory_checkpoint_metadata
from cognitive_runtime.training.model_factory.comparison import compare_paired_episodes
from cognitive_runtime.training.model_factory.corpus import resolve_corpus
from cognitive_runtime.training.model_factory.genome import (
    Gene,
    GenomeSchema,
    project_cost_seconds,
    repair,
)
from cognitive_runtime.training.model_factory.promotion import (
    _rollout_beats_copy_last_gate,
    _rollout_health_gate,
)
from cognitive_runtime.training.model_factory.runner import (
    RUNS_ROOT_DEFAULT,
    _resolve_selection_metric,
    run_trial,
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


SUCCESSIVE_HALVING_FORMAT = "model-factory-successive-halving-v1"

#: successive_halving's own gate set (epic #212 §13.2): a candidate that
#: fails either one is eliminated at that rung regardless of its primary
#: metric. Mirrors the two rollout-based gates
#: :mod:`.promotion`/:mod:`.confirmation` already apply -- the corpus's own
#: data-quality/split-overlap gates and the representation-collapse probe are
#: stage-level checks already enforced once when the frozen corpus and
#: schema were built, not re-run per rung.
_HALVING_GATES = (_rollout_beats_copy_last_gate, _rollout_health_gate)

#: A candidate never launched because the campaign's own ``stage_budget_seconds``
#: was already spent -- distinct from ``run_trial``'s own per-trial
#: ``"budget_exceeded"`` (a trial that *started* but hit its individual
#: ``training.max_training_seconds``).
STATE_NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class RungCandidateResult:
    """One candidate's outcome at one rung.

    ``epochs_executed`` is exactly what
    :func:`~.budget.build_budget_report` recorded for *this* call
    (``budget_report["epochs_recorded"]``) -- the physical number of epochs
    trained this rung, never the rung's cumulative target. ``metric_value``
    and ``gates`` are only populated when ``state == "completed"``: a
    candidate that ran out of its training-time budget mid-rung (``state ==
    "budget_exceeded"``), or was never launched because the campaign's own
    ``stage_budget_seconds`` was already spent (``state ==
    "not_attempted"``, see :data:`STATE_NOT_ATTEMPTED`), has no trustworthy
    validation reading to rank or gate on, so it is eliminated outright.
    """

    candidate_index: int
    run_id: str
    mode: str
    budget: int
    epochs_executed: int
    state: str
    metric_value: Optional[float]
    gates: Tuple[Dict[str, Any], ...]
    gate_passed: bool
    advanced: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "run_id": self.run_id,
            "mode": self.mode,
            "budget": self.budget,
            "epochs_executed": self.epochs_executed,
            "state": self.state,
            "metric_value": self.metric_value,
            "gates": [dict(gate) for gate in self.gates],
            "gate_passed": self.gate_passed,
            "advanced": self.advanced,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RungReport:
    """Every candidate's result at one checkpoint budget, plus which
    candidate indices advance to the next rung (empty at the last rung --
    nothing advances *from* it, see :attr:`SuccessiveHalvingReport.final_survivor_run_id`)."""

    rung_index: int
    budget: int
    delta_epochs: int
    results: Tuple[RungCandidateResult, ...]
    survivor_indices: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rung_index": self.rung_index,
            "budget": self.budget,
            "delta_epochs": self.delta_epochs,
            "results": [result.to_dict() for result in self.results],
            "survivor_indices": list(self.survivor_indices),
        }


@dataclass(frozen=True)
class SuccessiveHalvingReport:
    """The full record of one checkpoint-based successive-halving campaign."""

    format: str
    seed: int
    method: str
    budgets: Tuple[int, ...]
    initial_candidates: int
    halving_factor: int
    rungs: Tuple[RungReport, ...]
    total_epochs_executed: int
    theoretical_halving_epochs: int
    from_scratch_epochs: int
    stage_budget_exceeded: bool
    final_survivor_run_id: Optional[str]
    final_survivor_metric_value: Optional[float]
    champion_comparison: Optional[Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "method": self.method,
            "budgets": list(self.budgets),
            "initial_candidates": self.initial_candidates,
            "halving_factor": self.halving_factor,
            "rungs": [rung.to_dict() for rung in self.rungs],
            "total_epochs_executed": self.total_epochs_executed,
            "theoretical_halving_epochs": self.theoretical_halving_epochs,
            "from_scratch_epochs": self.from_scratch_epochs,
            "stage_budget_exceeded": self.stage_budget_exceeded,
            "final_survivor_run_id": self.final_survivor_run_id,
            "final_survivor_metric_value": self.final_survivor_metric_value,
            "champion_comparison": dict(self.champion_comparison) if self.champion_comparison is not None else None,
        }


class _CandidateTrack:
    """Mutable per-candidate bookkeeping threaded across rungs.

    Never exposed to callers -- :class:`RungCandidateResult`/
    :class:`SuccessiveHalvingReport` are the public, immutable record of what
    happened. ``spec`` is the candidate's fixed genome-varied
    ``ExperimentSpec`` (identical ``data``/``model`` blocks and training
    genes at every rung, per :func:`propose`'s own contract); only
    ``training.epoch_budget``/``mode``/``parent`` are rebuilt fresh each
    rung from it.
    """

    __slots__ = ("index", "spec", "run_id", "parent", "total_epochs_executed", "last_metric_value")

    def __init__(self, index: int, spec: ExperimentSpec) -> None:
        self.index = index
        self.spec = spec
        self.run_id: Optional[str] = None
        self.parent: Optional[Dict[str, Any]] = None
        self.total_epochs_executed = 0
        self.last_metric_value: Optional[float] = None


def _restore_numeric_keys(value: Any) -> Any:
    """Undo the ``str(key)`` coercion a JSON round-trip applies to a
    validation report's horizon-keyed mappings.

    ``atomic_write_json`` (``artifacts._jsonable``) stringifies every
    mapping key it writes -- JSON object keys are always strings -- but
    ``_resolve_selection_metric`` looks a horizon up by its original
    int/float tick-or-frame key. A ``metrics/validation.json`` read back
    from disk (the champion comparison's only source for an out-of-process
    champion) needs those keys restored before it can be resolved exactly
    like the in-memory report ``run_trial`` itself produces.
    """
    if isinstance(value, Mapping):
        restored: Dict[Any, Any] = {}
        for key, item in value.items():
            new_key: Any = key
            if isinstance(key, str):
                try:
                    new_key = int(key)
                except ValueError:
                    try:
                        new_key = float(key)
                    except ValueError:
                        new_key = key
            restored[new_key] = _restore_numeric_keys(item)
        return restored
    if isinstance(value, list):
        return [_restore_numeric_keys(item) for item in value]
    return value


def _validate_budgets(budgets: Sequence[int]) -> Tuple[int, ...]:
    resolved = tuple(int(budget) for budget in budgets)
    if not resolved:
        raise SearchError("budgets must declare at least one checkpoint rung")
    if any(budget <= 0 for budget in resolved):
        raise SearchError(f"every budget must be a positive epoch count; got {resolved!r}")
    if list(resolved) != sorted(set(resolved)):
        raise SearchError(
            f"budgets must be strictly increasing (each rung trains more epochs than the last); got {resolved!r}"
        )
    return resolved


def run_successive_halving(
    base_spec: ExperimentSpec,
    genome_schema: GenomeSchema,
    budgets: Sequence[int],
    n: int,
    seed: int,
    *,
    method: str = "lhs",
    halving_factor: int = 2,
    root: Union[str, Path] = RUNS_ROOT_DEFAULT,
    corpus_root: Optional[Union[str, Path]] = None,
    run_id_prefix: Optional[str] = None,
    naming_seed: Optional[Union[str, int]] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    heartbeat_timeout_seconds: Optional[float] = None,
    champion_run_id: Optional[str] = None,
    min_episode_length: Optional[int] = None,
    stage_budget_seconds: Optional[float] = None,
    cost_model: Callable[[Mapping[str, Any]], float] = project_cost_seconds,
) -> SuccessiveHalvingReport:
    """Checkpoint-based successive halving over ``budgets`` (epic #212 Phase D
    step 2, §13.2, issue #231).

    Trains ``n`` siblings (:func:`propose`, so only ``genome_schema``'s
    training genes vary) fresh to ``budgets[0]``, evaluates each on the
    corpus's fixed validation split, and retains the best half by
    ``base_spec.evaluation["selection_metric"]`` **and** two safety gates
    (``rollout_beats_copy_last``, ``rollout_health``) -- a gate failure
    eliminates a candidate regardless of how good its primary metric is.
    Survivors *continue* from their own checkpoint (``mode="fine_tune"``
    against their own immediately-preceding rung's ``checkpoints/last.pt``,
    with ``training.epoch_budget`` set to only the rung's *incremental*
    epoch delta) to the next declared budget; §13.2's non-negotiable
    constraint is that this must never retrain a survivor from scratch, and
    :attr:`SuccessiveHalvingReport.total_epochs_executed` is the physical
    epoch count actually run (via each rung's own
    ``budget_report["epochs_recorded"]``) so that invariant is directly
    checkable rather than assumed. The last rung applies the gates only (no
    further rank-based elimination -- there is no next budget to advance
    to); whichever gate-passing candidate has the best primary metric
    (ties broken by ``run_id``, for a fixed ``seed``) is
    ``final_survivor_run_id``.

    ``run_id_prefix`` (default ``f"sh{seed}"``) plus each rung/candidate
    index deterministically name every run -- never the factory's own
    random ``new_run_id`` -- so ``final_survivor_run_id`` and every rung's
    elimination order are exactly reproducible for a fixed ``seed`` across
    processes, matching :func:`propose`'s own reproducibility guarantee.

    Every rung evaluates the corpus's one frozen validation split (fixed by
    ``base_spec.data["corpus_id"]``, never varied across siblings); this is
    asserted directly by comparing every rung's persisted
    ``metrics/validation.json["episode_ids"]`` against the first rung's.
    Only ``run_trial`` is ever called, and it never resolves the corpus's
    ``test`` split (see its own module docstring), so the sealed test split
    is never queried here either.

    ``champion_run_id``, when given, is paired-compared
    (:func:`~.comparison.compare_paired_episodes`) against the final
    survivor's validation reading; ``None`` when there is no survivor (every
    candidate was eliminated) or no champion was named.

    ``stage_budget_seconds``, when given, is a campaign-wide wall-clock cap
    -- distinct from each candidate's own ``training.max_training_seconds``
    (which every ``run_trial`` call already enforces per-trial, and which
    :func:`propose` holds fixed across every candidate). Without this cap,
    ``n`` siblings at rung 0 alone could consume up to ``n`` full per-trial
    budgets, and every later rung would add more, letting the campaign as a
    whole run arbitrarily long. Checked before launching *each* candidate:
    once the campaign's elapsed wall-clock time reaches
    ``stage_budget_seconds``, every remaining candidate (in the current rung
    and any later rung) is recorded with ``state="not_attempted"`` and never
    launched, :attr:`SuccessiveHalvingReport.stage_budget_exceeded` is
    ``True``, and no further rungs run.
    """
    resolved_budgets = _validate_budgets(budgets)
    if halving_factor < 2:
        raise SearchError(f"halving_factor must be at least 2; got {halving_factor!r}")

    candidates = propose(
        base_spec, genome_schema, n, seed, method=method,
        min_episode_length=min_episode_length, stage_budget_seconds=stage_budget_seconds,
        cost_model=cost_model,
    )
    tracks = [
        _CandidateTrack(index=index, spec=replace(spec, mode="fresh", parent=None))
        for index, spec in enumerate(candidates)
    ]

    corpus = resolve_corpus(
        base_spec.data["corpus_id"], allow_record=False, root=corpus_root, organism=base_spec.organism,
    )
    ticks_per_frame = float(corpus.data_contract.ticks_per_frame)
    selection_metric = base_spec.evaluation["selection_metric"]
    prefix = run_id_prefix or f"sh{seed}"

    alive: List[_CandidateTrack] = list(tracks)
    rungs: List[RungReport] = []
    final_candidates: List[Tuple[_CandidateTrack, RungCandidateResult]] = []
    reference_episode_ids: Optional[Tuple[str, ...]] = None
    total_epochs_executed = 0
    theoretical_halving_epochs = 0
    from_scratch_epochs = 0
    previous_budget = 0
    stage_started = time.monotonic()
    stage_budget_exceeded = False

    for rung_index, budget in enumerate(resolved_budgets):
        delta = budget - previous_budget
        participants = list(alive)
        is_last_rung = rung_index == len(resolved_budgets) - 1
        theoretical_halving_epochs += len(participants) * delta
        from_scratch_epochs += len(participants) * budget

        results: List[RungCandidateResult] = []
        for track in participants:
            run_id = f"{prefix}-r{rung_index}-c{track.index}"
            child_training = _thaw(track.spec.training)
            child_training["epoch_budget"] = delta
            if rung_index == 0:
                child_spec = replace(track.spec, mode="fresh", parent=None, training=child_training)
            else:
                assert track.parent is not None, "a surviving candidate must carry its prior rung's checkpoint"
                child_spec = replace(
                    track.spec, mode="fine_tune", parent=dict(track.parent), training=child_training,
                )
            validate_spec(child_spec)

            # The campaign-wide wall-clock cap (distinct from each
            # run_trial call's own per-trial `training.max_training_seconds`,
            # which propose() already holds fixed across every candidate):
            # `n` siblings at rung 0 alone can otherwise consume up to `n`
            # full per-trial budgets before a single elimination ever
            # happens. Checked *before* launching each call so an
            # already-exhausted stage never starts one more trial.
            if stage_budget_seconds is not None and (time.monotonic() - stage_started) >= stage_budget_seconds:
                stage_budget_exceeded = True
                results.append(RungCandidateResult(
                    candidate_index=track.index, run_id=run_id, mode=child_spec.mode, budget=budget,
                    epochs_executed=0, state=STATE_NOT_ATTEMPTED, metric_value=None, gates=(),
                    gate_passed=False, advanced=False,
                    reason=(
                        f"skipped: the campaign's stage_budget_seconds ({stage_budget_seconds!r}) was "
                        "already exhausted before this candidate could be attempted"
                    ),
                ))
                continue

            result = run_trial(
                child_spec, root=root, corpus_root=corpus_root, run_id=run_id,
                naming_seed=naming_seed, trace_dir=trace_dir,
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            )

            validation_path = Path(result.directory) / "metrics" / "validation.json"
            with validation_path.open(encoding="utf-8") as handle:
                validation_payload = json.load(handle)
            episode_ids = tuple(validation_payload["episode_ids"])
            if reference_episode_ids is None:
                reference_episode_ids = episode_ids
            elif episode_ids != reference_episode_ids:
                raise SearchError(
                    "successive halving requires the identical validation episode set at every "
                    f"rung; rung {rung_index} candidate {track.index} ({run_id!r}) evaluated "
                    f"{episode_ids!r}, expected {reference_episode_ids!r}"
                )

            epochs_executed = int(result.budget_report.get("epochs_recorded") or 0)
            track.total_epochs_executed += epochs_executed
            total_epochs_executed += epochs_executed
            if track.total_epochs_executed > resolved_budgets[-1]:
                raise SearchError(
                    f"candidate {track.index} ({run_id!r}) has executed "
                    f"{track.total_epochs_executed} epochs, exceeding the halving schedule's "
                    f"declared maximum budget ({resolved_budgets[-1]})"
                )

            last_checkpoint_path = Path(result.directory) / "checkpoints" / "last.pt"
            last_sha = read_factory_checkpoint_metadata(str(last_checkpoint_path))["checkpoint_sha256"]
            track.run_id = run_id
            track.parent = {"run_id": run_id, "checkpoint": "last.pt", "sha256": last_sha}

            completed = result.state == "completed"
            gate_dicts: Tuple[Dict[str, Any], ...] = ()
            metric_value: Optional[float] = None
            gate_passed = False
            if not completed:
                reason = f"rung did not complete within its training-time budget (state={result.state!r})"
            else:
                rollout_metrics = result.evaluation["rollout"]
                gate_results = tuple(gate(rollout_metrics) for gate in _HALVING_GATES)
                gate_dicts = tuple(gate.to_dict() for gate in gate_results)
                gate_passed = all(gate.passed for gate in gate_results)
                metric_value, _ = _resolve_selection_metric(result.evaluation, selection_metric, ticks_per_frame)
                track.last_metric_value = metric_value
                reason = (
                    "passed both safety gates"
                    if gate_passed
                    else "; ".join(gate.reason for gate in gate_results if not gate.passed)
                )

            results.append(RungCandidateResult(
                candidate_index=track.index, run_id=run_id, mode=child_spec.mode, budget=budget,
                epochs_executed=epochs_executed, state=result.state, metric_value=metric_value,
                gates=gate_dicts, gate_passed=gate_passed, advanced=False, reason=reason,
            ))

        gate_passing = [
            (track, result) for track, result in zip(participants, results) if result.gate_passed
        ]
        if is_last_rung:
            # Nothing advances *from* the last rung -- there is no further
            # budget to continue to (RungReport's own contract). Every
            # gate-passing candidate here is a *candidate* for
            # final_survivor_run_id, decided once below, after the loop --
            # not an intermediate rung survivor, so `survivors` (which
            # feeds `alive`/`advanced`/`survivor_indices`) stays empty.
            target_keep = 0
            survivors: List[Tuple[_CandidateTrack, RungCandidateResult]] = []
            final_candidates = gate_passing
        else:
            target_keep = max(1, len(participants) // halving_factor)
            ranked = sorted(gate_passing, key=lambda pair: (pair[1].metric_value, pair[1].run_id))
            survivors = ranked[:target_keep]

        survivor_ids = {track.index for track, _ in survivors}
        final_results = []
        for result in results:
            advanced = result.candidate_index in survivor_ids
            reason = result.reason
            if result.gate_passed and not advanced:
                reason = (
                    "passed both safety gates at the final rung; ranked among the final "
                    "survivors by primary metric (no further budget to advance to)"
                    if is_last_rung else
                    f"eliminated by successive-halving rank: kept only the best {target_keep} of "
                    f"{len(gate_passing)} gate-passing candidate(s) by {selection_metric!r} "
                    "(ties broken by run_id)"
                )
            final_results.append(replace(result, advanced=advanced, reason=reason))

        rungs.append(RungReport(
            rung_index=rung_index, budget=budget, delta_epochs=delta,
            results=tuple(final_results), survivor_indices=tuple(sorted(survivor_ids)),
        ))

        alive = [track for track, _ in survivors]
        previous_budget = budget
        if not alive or stage_budget_exceeded:
            break

    final_survivor_run_id: Optional[str] = None
    final_survivor_metric_value: Optional[float] = None
    if final_candidates:
        best_track, best_result = min(
            final_candidates, key=lambda pair: (pair[1].metric_value, pair[1].run_id),
        )
        final_survivor_run_id = best_track.run_id
        final_survivor_metric_value = best_result.metric_value

    champion_comparison: Optional[Dict[str, Any]] = None
    if champion_run_id is not None and final_survivor_run_id is not None:
        organism_root = Path(root) / base_spec.organism
        with (organism_root / champion_run_id / "metrics" / "validation.json").open(encoding="utf-8") as handle:
            champion_validation = _restore_numeric_keys(json.load(handle))
        with (organism_root / final_survivor_run_id / "metrics" / "validation.json").open(encoding="utf-8") as handle:
            survivor_validation = _restore_numeric_keys(json.load(handle))
        _, survivor_per_episode = _resolve_selection_metric(survivor_validation, selection_metric, ticks_per_frame)
        _, champion_per_episode = _resolve_selection_metric(champion_validation, selection_metric, ticks_per_frame)
        comparison = compare_paired_episodes(
            dict(zip(survivor_validation["episode_ids"], survivor_per_episode)),
            dict(zip(champion_validation["episode_ids"], champion_per_episode)),
            primary_metric=selection_metric, minimum_episode_count=1,
        )
        payload = comparison.to_dict()
        payload["champion_run_id"] = champion_run_id
        payload["decision"] = (
            "candidate_improves"
            if comparison.status == "evaluable" and comparison.mean_delta is not None and comparison.mean_delta < 0
            else "hold"
        )
        champion_comparison = payload

    return SuccessiveHalvingReport(
        format=SUCCESSIVE_HALVING_FORMAT, seed=seed, method=method, budgets=resolved_budgets,
        initial_candidates=n, halving_factor=halving_factor, rungs=tuple(rungs),
        total_epochs_executed=total_epochs_executed, theoretical_halving_epochs=theoretical_halving_epochs,
        from_scratch_epochs=from_scratch_epochs, stage_budget_exceeded=stage_budget_exceeded,
        final_survivor_run_id=final_survivor_run_id,
        final_survivor_metric_value=final_survivor_metric_value, champion_comparison=champion_comparison,
    )


__all__ = [
    "METHODS",
    "SearchError",
    "propose",
    "SUCCESSIVE_HALVING_FORMAT",
    "STATE_NOT_ATTEMPTED",
    "RungCandidateResult",
    "RungReport",
    "SuccessiveHalvingReport",
    "run_successive_halving",
]
