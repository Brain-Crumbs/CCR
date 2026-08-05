"""Versioned *architecture* genome schema for Model Factory's NAS-style search.

:mod:`.genome` declares the bounded set of **training-only** genes a
hyperparameter search may recombine, and enforces that discipline
structurally: :func:`.genome._assert_genes_are_training_contract_paths`
makes a gene reaching an :class:`~.contracts.ArchitectureContract` field
unrepresentable, so :func:`.search.propose` can never drift the architecture
a batch is supposed to hold fixed. That guardrail is exactly right for
comparing training procedures; it is also exactly what makes evolving an
architecture impossible through that module.

This module is the deliberate, separately-named counterpart: the same
:class:`~.genome.Gene`/:class:`~.genome.MutationSpec`/
:class:`~.genome.GenomeSchema` machinery (reused unchanged -- every one of
them is generic over dotted gene paths, with the single exception of that
one guardrail), pointed at a small allowlist of genuinely free
``ExperimentSpec.model`` fields instead. The two schema families are
mutually exclusive by construction:
:func:`_assert_genes_are_architecture_contract_paths` rejects any gene that
names a :class:`~.contracts.TrainingContract` field, exactly as
:mod:`.genome`'s own guardrail rejects the reverse. A campaign therefore
always knows which of the two axes it is varying; nothing can quietly vary
both at once inside one population (epic #212 §13.1: "Do not jointly vary
architecture, data split, objective, and loss weights").

**The allowlist is deliberately narrower than ``ArchitectureContract``.**
Most of that contract's fields are *derived* from the corpus and the task,
not free choices: ``pixel_shape``, ``action_vocabulary``,
``workspace_modalities``, ``semantic_classes``, ``horizons_ticks`` and
``direct_horizon_topology`` all follow from the resolved data contract, and
mutating them would produce a model that no longer matches its own
evidence. ``reconstruction_size`` is free in principle but excluded from v1
on purpose: changing the decoder's output resolution across candidates
changes what a raw pixel MSE even measures, so the campaign's own selection
metric would stop comparing like with like. That is a v2 question (it needs
a resolution-normalised metric first), not a limitation of this machinery.

**The ``backbone_preset`` composite gene.** ``model.backbone`` and
``model.backbone_kwargs`` are coupled: a transformer backbone needs
``n_heads``/``n_layers``, a GRU accepts neither, and an independently
mutated pair of genes would routinely produce combinations like
``gru + {n_heads: 4}`` that are silently ignored, or
``transformer + {kernel_size: 2}`` that are silently wrong. Rather than
inventing conditional or hierarchical genes (a real change to
:mod:`.genome`'s flat model, for one coupling), this module declares **one**
categorical gene whose choices are curated *whole* presets, expanded into
both spec fields by :func:`apply_architecture_genome`. Crossover and
mutation then move between known-good backbone configurations by
construction.

Pure Python -- no torch import -- so a campaign's whole architecture search
space is declarable, hashable, and dry-runnable in the core-only install.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.genome import (
    GenomeSchema,
    GenomeSchemaError,
    _build_gene,
    _verify_schema_integrity,
)
from cognitive_runtime.training.model_factory.spec import MODEL_KEYS, _thaw


class ArchitectureGenomeError(ValueError):
    """A malformed architecture genome, preset, or model block."""


#: The one composite gene (see the module docstring): a single categorical
#: choice over curated whole backbone configurations, expanded into
#: ``model.backbone`` *and* ``model.backbone_kwargs`` at merge time.
BACKBONE_PRESET_GENE = "backbone_preset"

#: The curated ``backbone_preset`` choices. Each names a complete, valid
#: ``(backbone, backbone_kwargs)`` pair for ``brain.cortex.backbones.
#: build_backbone``: the GRU takes no extra kwargs, ``dilated_conv``'s
#: ``kernel_size``/``n_layers`` defaults are already sensible for the
#: context lengths this schema searches, and the two transformer presets
#: bracket a cheap and an expensive attention configuration. Every preset's
#: head count divides every ``hidden_dim`` choice below, so no
#: gene combination this schema can produce is unbuildable.
BACKBONE_PRESETS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "gru": {"backbone": "gru", "backbone_kwargs": {}},
    "dilated_conv": {"backbone": "dilated_conv", "backbone_kwargs": {}},
    "transformer_h2_l1": {
        "backbone": "transformer", "backbone_kwargs": {"n_heads": 2, "n_layers": 1},
    },
    "transformer_h4_l2": {
        "backbone": "transformer", "backbone_kwargs": {"n_heads": 4, "n_layers": 2},
    },
})

#: The two ``ExperimentSpec.model`` fields :data:`BACKBONE_PRESET_GENE`
#: expands into.
BACKBONE_PRESET_FIELDS: Tuple[str, ...] = ("backbone", "backbone_kwargs")

#: Genes a given backbone *ignores*, keyed by backbone name.
#:
#: ``context_length`` is a build-time ring-buffer capacity for the windowed
#: backbones and is documented as ignored by the GRU in both
#: ``ActionWorldModelConfig.context_length`` (``action_world_model.py``) and
#: ``PredictiveCortex``'s own config (``brain/cortex/predictive.py``); the
#: structural statement of the same fact is
#: ``TemporalBackbone.context_length_max``, which only ``_WindowedBackbone``
#: ever sets (``brain/cortex/backbones.py``), and ``GRUBackbone.__init__``
#: swallows the kwarg via ``**_unused``.
#:
#: This matters here because an inert gene is not free -- it is *worse* than
#: free. Four GRU genomes differing only in ``context_length`` build four
#: identical models, but they would carry four distinct genomes, four
#: distinct ``architecture_hash`` values (``context_length`` is an
#: ``ArchitectureContract`` field), and, in
#: ``architecture_search``, four separate full inner campaigns -- so the
#: outer loop would spend four budgets re-measuring one architecture and
#: then rank the training noise between them as if it were architectural
#: signal. :func:`canonicalize_architecture_genome` collapses them instead.
#:
#: Mirrored by hand because ``brain.cortex.backbones`` imports torch and this
#: module must stay importable in the core-only install; a torch-gated test
#: asserts this declaration against each backbone's real
#: ``context_length_max`` so the mirror cannot drift silently.
INERT_GENES_BY_BACKBONE: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "gru": ("context_length",),
})

#: Which preset a model block declaring a bare ``{backbone, backbone_kwargs:
#: {}}`` pair is, for backbones whose preset above spells out the
#: constructor's *own* defaults rather than overriding them.
#: ``spec.DEFAULT_MODEL`` declares exactly ``backbone: "transformer"`` with
#: empty kwargs, and ``TransformerBackbone.__init__``'s defaults are
#: ``n_heads=2, n_layers=1`` (``brain/cortex/backbones.py``), so that spec
#: builds the identical module ``transformer_h2_l1`` does. Without this
#: declaration :func:`backbone_preset_for` would refuse the single most
#: common hand-written model block in the repository. Note the mapping is
#: deliberately one-way: expanding the preset back out writes the kwargs
#: explicitly, so a bred child records ``{n_heads: 2, n_layers: 1}`` (and a
#: correspondingly different ``architecture_hash``) for what is the same
#: built model -- an explicit architecture declaration is the right thing
#: for a run whose whole point is that its architecture was chosen.
BARE_BACKBONE_PRESETS: Mapping[str, str] = MappingProxyType({
    "transformer": "transformer_h2_l1",
})

#: The plain (non-composite) ``model`` fields an architecture schema may
#: declare a gene for -- the genuinely free ones (see the module docstring).
FREE_MODEL_FIELDS: FrozenSet[str] = frozenset({
    "latent_width", "hidden_dim", "action_embed_dim", "context_length",
})

#: Every legal architecture gene root: the free ``model`` fields plus the
#: one composite preset gene.
ARCHITECTURE_GENE_ROOTS: FrozenSet[str] = FREE_MODEL_FIELDS | {BACKBONE_PRESET_GENE}

_TRAINING_CONTRACT_FIELDS: FrozenSet[str] = frozenset(
    field.name for field in dataclass_fields(TrainingContract)
)

_ARCHITECTURE_CONTRACT_FIELDS: FrozenSet[str] = frozenset(
    field.name for field in dataclass_fields(ArchitectureContract)
)

# Import-time proof that the allowlist above is not a hand-maintained list
# that can drift: every field a gene can reach -- directly, or through a
# preset expansion -- must be both a real `ExperimentSpec.model` key (so the
# merge lands somewhere `spec.validate` accepts) and a real
# `ArchitectureContract` field (so changing it genuinely changes the run's
# architecture identity, rather than being recorded and ignored).
_REACHABLE_MODEL_FIELDS: FrozenSet[str] = FREE_MODEL_FIELDS | frozenset(BACKBONE_PRESET_FIELDS)
assert _REACHABLE_MODEL_FIELDS <= frozenset(MODEL_KEYS), (
    "architecture genes must name real ExperimentSpec model keys: "
    f"{sorted(_REACHABLE_MODEL_FIELDS - frozenset(MODEL_KEYS))}"
)
assert _REACHABLE_MODEL_FIELDS <= _ARCHITECTURE_CONTRACT_FIELDS, (
    "architecture genes must name real ArchitectureContract fields: "
    f"{sorted(_REACHABLE_MODEL_FIELDS - _ARCHITECTURE_CONTRACT_FIELDS)}"
)


def _assert_genes_are_architecture_contract_paths(
    version: str, gene_names: Tuple[str, ...],
) -> None:
    """The mirror of :func:`.genome._assert_genes_are_training_contract_paths`.

    A gene's root segment must name either one of the free ``model`` fields
    or the composite :data:`BACKBONE_PRESET_GENE`. The explicit
    ``TrainingContract`` check below is redundant with that allowlist today
    (the two contracts share no field name), and is kept for the same reason
    :mod:`.genome` keeps its own mirrored check: it states the invariant --
    *an architecture schema never varies the training procedure* -- directly,
    so the two guardrails read as an obvious pair rather than as one
    allowlist someone has to reason about backwards.
    """
    roots = {name.split(".", 1)[0] for name in gene_names}
    unknown_roots = roots - ARCHITECTURE_GENE_ROOTS
    if unknown_roots:
        raise GenomeSchemaError(
            f"architecture genome schema {version!r} declares gene(s) outside the free "
            f"architecture allowlist: {sorted(unknown_roots)!r} (allowed roots: "
            f"{sorted(ARCHITECTURE_GENE_ROOTS)!r}). Corpus-derived ArchitectureContract "
            "fields (pixel_shape, action_vocabulary, workspace_modalities, "
            "semantic_classes, horizons_ticks, direct_horizon_topology) and "
            "reconstruction_size are deliberately not evolvable in v1; see this "
            "module's docstring."
        )
    forbidden = (set(gene_names) | roots) & _TRAINING_CONTRACT_FIELDS
    if forbidden:
        raise GenomeSchemaError(
            f"architecture genome schema {version!r} declares gene(s) reaching a "
            f"TrainingContract field: {sorted(forbidden)!r}. Training-procedure genes "
            "belong in a genome.GenomeSchema, evolved by the inner search; an "
            "architecture schema varies only the architecture."
        )


@dataclass(frozen=True)
class ArchitectureGenomeSchema(GenomeSchema):
    """A versioned set of architecture genes.

    Identical to :class:`~.genome.GenomeSchema` in every respect --
    ``content_hash``, ``gene_names``, ``active_gene_names`` and every
    operator in :mod:`.genome` work on it unchanged -- except that it
    enforces the *architecture* guardrail instead of the training one. The
    override is the whole reason this is a distinct class rather than a
    convention: an architecture schema is not merely discouraged from
    declaring a training gene, it cannot be constructed with one.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "genes", MappingProxyType(dict(self.genes)))
        _assert_genes_are_architecture_contract_paths(self.version, tuple(self.genes))


def build_architecture_schema(
    version: str, gene_specs: Mapping[str, Mapping[str, Any]],
) -> ArchitectureGenomeSchema:
    """Build an :class:`ArchitectureGenomeSchema` from declarative gene specs.

    Uses :mod:`.genome`'s own gene builder, so every "a declaration missing
    any of these fails to load" check (type, bounds-or-choices, default,
    mutation distribution, log-scale/mutation agreement) applies identically
    to architecture genes.
    """
    genes = {name: _build_gene(name, spec) for name, spec in gene_specs.items()}
    return ArchitectureGenomeSchema(version=version, genes=genes)


# §14.2's per-stage discipline, applied to the architecture axis: a small,
# deliberately bounded set of choices per gene. This is a search-space
# *design* choice, not a technical ceiling -- the outer loop's cost is
# multiplicative (see `architecture_search.estimate_cost`), so a v1 campaign
# that a human will actually wait for needs a space measured in dozens of
# architectures, not thousands. Every gene is categorical: the interesting
# architectural moves here are discrete jumps between known-good
# configurations, not fine perturbation of a continuous width.
_ARCHITECTURE_SEARCH_V1_GENES: Dict[str, Dict[str, Any]] = {
    BACKBONE_PRESET_GENE: {
        "type": "categorical",
        "choices": tuple(BACKBONE_PRESETS),
        # Matches spec.DEFAULT_MODEL's own `backbone: "transformer"` with
        # empty kwargs: TransformerBackbone's constructor defaults are
        # exactly n_heads=2, n_layers=1, so this preset is the declared
        # default architecture spelled out rather than a different one.
        "default": "transformer_h2_l1",
        "mutation": {"distribution": "categorical_resample"},
    },
    "latent_width": {
        "type": "int",
        "choices": (64, 128, 192, 256),
        "default": 128,
        "mutation": {"distribution": "categorical_resample"},
    },
    "hidden_dim": {
        "type": "int",
        # Every value is divisible by both transformer presets' head counts
        # (2 and 4), so no (backbone_preset, hidden_dim) combination this
        # schema can produce fails MultiheadAttention's own divisibility
        # requirement at build time.
        "choices": (128, 256, 384, 512),
        "default": 256,
        "mutation": {"distribution": "categorical_resample"},
    },
    "action_embed_dim": {
        "type": "int",
        "choices": (4, 8, 16),
        "default": 8,
        "mutation": {"distribution": "categorical_resample"},
    },
    "context_length": {
        "type": "int",
        "choices": (4, 8, 12, 16),
        "default": 8,
        "mutation": {"distribution": "categorical_resample"},
    },
}

#: Pinned content hash for ``architecture_search_v1``, asserted at import
#: time exactly as :mod:`.genome` pins its own shipped schema: changing any
#: choice set, default or mutation above without renaming the version fails
#: to import. The fix is ``architecture_search_v2``, never editing this hash.
_EXPECTED_CONTENT_HASHES: Dict[str, str] = {
    "architecture_search_v1": (
        "74b4c7b0ed043587da360bd2990ea185dc463cd833c976412435092441b4fa1e"
    ),
}

ARCHITECTURE_SEARCH_V1: ArchitectureGenomeSchema = _verify_schema_integrity(
    build_architecture_schema("architecture_search_v1", _ARCHITECTURE_SEARCH_V1_GENES),
    _EXPECTED_CONTENT_HASHES["architecture_search_v1"],
)

#: Registry of every declared architecture schema, mirroring
#: :data:`.genome.GENOME_SCHEMAS`. Kept separate from that registry on
#: purpose: ``ccr factory search --schema`` and
#: ``--outer-schema`` must not be able to name each other's schemas.
ARCHITECTURE_GENOME_SCHEMAS: Dict[str, ArchitectureGenomeSchema] = {
    ARCHITECTURE_SEARCH_V1.version: ARCHITECTURE_SEARCH_V1,
}


def get_architecture_schema(version: str) -> ArchitectureGenomeSchema:
    """Look up a declared architecture schema; unknown versions are a hard error."""
    try:
        return ARCHITECTURE_GENOME_SCHEMAS[version]
    except KeyError:
        raise GenomeSchemaError(
            f"unknown architecture genome schema version {version!r}; expected one of "
            f"{sorted(ARCHITECTURE_GENOME_SCHEMAS)}"
        ) from None


def resolve_backbone_preset(preset: str) -> Dict[str, Any]:
    """The ``{backbone, backbone_kwargs}`` pair one preset name expands into."""
    try:
        resolved = BACKBONE_PRESETS[preset]
    except KeyError:
        raise ArchitectureGenomeError(
            f"unknown backbone preset {preset!r}; expected one of {sorted(BACKBONE_PRESETS)}"
        ) from None
    return {key: _thaw(value) for key, value in resolved.items()}


def _set_nested(mapping: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``mapping[a][b]... = value``, mirroring :func:`.search._set_nested`.

    Refuses to traverse *through* an existing non-mapping value for the same
    reason that function does: silently replacing a scalar model field with
    a fresh ``{}`` would produce a spec that ``spec.validate`` accepts
    (dataclass annotations are not enforced at runtime) and that then fails
    deep inside the model builder.
    """
    parts = dotted_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
        elif not isinstance(existing, Mapping):
            raise ArchitectureGenomeError(
                f"gene {dotted_path!r} traverses through {part!r}, an existing "
                f"non-mapping model value ({existing!r}); every intermediate segment "
                "of a dotted gene path must name a mapping-valued model field"
            )
        cursor[part] = dict(existing)
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_architecture_genome(
    model: Mapping[str, Any], genome: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge a repaired architecture genome onto a copy of a ``model`` block.

    Every gene except :data:`BACKBONE_PRESET_GENE` merges as an ordinary
    single dotted-path value, exactly as :func:`.search._apply_genome` merges
    a training gene. ``backbone_preset`` is the one explicit special case:
    it expands into both ``model.backbone`` and ``model.backbone_kwargs``
    (see the module docstring on why the two are one gene).
    """
    merged = _thaw(model)
    for dotted_path, value in genome.items():
        if dotted_path == BACKBONE_PRESET_GENE:
            merged.update(resolve_backbone_preset(str(value)))
            continue
        _set_nested(merged, dotted_path, value)
    return merged


def inert_genes_for(genome: Mapping[str, Any]) -> Tuple[str, ...]:
    """Which of ``genome``'s genes its chosen backbone actually ignores.

    Empty for a genome with no :data:`BACKBONE_PRESET_GENE` (a schema is
    free not to evolve the backbone at all, in which case nothing is
    conditional) and for any backbone that ignores nothing.
    """
    preset = genome.get(BACKBONE_PRESET_GENE)
    if preset is None:
        return ()
    backbone = resolve_backbone_preset(str(preset))["backbone"]
    return tuple(
        name for name in INERT_GENES_BY_BACKBONE.get(str(backbone), ()) if name in genome
    )


def canonicalize_architecture_genome(
    schema: ArchitectureGenomeSchema, genome: Mapping[str, Any],
) -> Dict[str, Any]:
    """Snap every gene the chosen backbone ignores to its declared default.

    The architecture analogue of :func:`.genome.repair`'s joint constraints:
    a gene value that cannot change the built model must not be allowed to
    make two identical architectures look different. Two genomes that build
    the same model canonicalize to the same genome, so
    :mod:`.architecture_search`'s duplicate rejection sees them as the one
    architecture they are, and the recorded genome never claims a
    ``context_length`` for a model that ignores it.

    The gene's *declared default* is the canonical value rather than, say,
    dropping the key: the genome must stay complete (every operator in
    :mod:`.genome` requires every declared gene to be present), and a
    default is by construction always among the gene's allowed choices.

    Deliberately separate from :func:`.genome.repair` rather than folded
    into it: repair is generic over any schema and rejects invalid genomes,
    while this encodes one specific schema family's knowledge of which
    backbone reads which field.
    """
    canonical = dict(genome)
    for name in inert_genes_for(canonical):
        gene = schema.genes.get(name)
        if gene is not None:
            canonical[name] = gene.default
    return canonical


def _get_nested(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = mapping
    consumed = []
    for part in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ArchitectureGenomeError(
                f"model block is missing architecture gene path {dotted_path!r} "
                f"(failed at {'.'.join(consumed + [part])!r})"
            )
        cursor = cursor[part]
        consumed.append(part)
    return cursor


def backbone_preset_for(model: Mapping[str, Any]) -> str:
    """The preset name a resolved ``model`` block's backbone configuration is.

    The inverse of :func:`resolve_backbone_preset`, needed to read an
    existing run's architecture back out as a genome (outer breeding reads
    both parents' *actual* architectures, never a resampled or default one).
    A model block whose ``(backbone, backbone_kwargs)`` pair is not one of
    the curated presets has no genome representation, and saying so is the
    honest answer: breeding from it would have to invent a preset.
    """
    backbone = model.get("backbone")
    kwargs = _thaw(model.get("backbone_kwargs") or {})
    for name, preset in BACKBONE_PRESETS.items():
        if preset["backbone"] == backbone and _thaw(preset["backbone_kwargs"]) == kwargs:
            return name
    if not kwargs and backbone in BARE_BACKBONE_PRESETS:
        return BARE_BACKBONE_PRESETS[backbone]
    raise ArchitectureGenomeError(
        f"model backbone {backbone!r} with kwargs {kwargs!r} does not match any declared "
        f"backbone preset ({sorted(BACKBONE_PRESETS)}); it has no architecture-genome "
        "representation and cannot be used as an architecture breeding parent"
    )


def extract_architecture_genome(
    schema: ArchitectureGenomeSchema, model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read every declared architecture gene out of a resolved ``model`` block."""
    genome: Dict[str, Any] = {}
    for name in schema.genes:
        if name == BACKBONE_PRESET_GENE:
            genome[name] = backbone_preset_for(model)
        else:
            genome[name] = _get_nested(model, name)
    return genome


def default_architecture_genome(schema: ArchitectureGenomeSchema) -> Dict[str, Any]:
    """Every architecture gene at its declared default."""
    return {name: gene.default for name, gene in schema.genes.items()}


def sample_architecture_genome(
    schema: ArchitectureGenomeSchema, rng: random.Random,
) -> Dict[str, Any]:
    """Draw one fresh architecture genome -- seeds an initial outer population."""
    return {name: gene.sample(rng) for name, gene in schema.genes.items()}


__all__ = [
    "ArchitectureGenomeError",
    "BACKBONE_PRESET_GENE",
    "BACKBONE_PRESETS",
    "BACKBONE_PRESET_FIELDS",
    "INERT_GENES_BY_BACKBONE",
    "FREE_MODEL_FIELDS",
    "ARCHITECTURE_GENE_ROOTS",
    "ArchitectureGenomeSchema",
    "build_architecture_schema",
    "ARCHITECTURE_SEARCH_V1",
    "ARCHITECTURE_GENOME_SCHEMAS",
    "get_architecture_schema",
    "resolve_backbone_preset",
    "inert_genes_for",
    "canonicalize_architecture_genome",
    "apply_architecture_genome",
    "backbone_preset_for",
    "extract_architecture_genome",
    "default_architecture_genome",
    "sample_architecture_genome",
]
