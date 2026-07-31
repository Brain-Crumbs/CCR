"""Model Factory bounded random/Latin-hypercube search tests (epic #212,
Phase D step 1, §13.1, issue #230).

Deliberately torch-free: these run in the ``factory-contracts`` CI tier
(see .github/workflows/ci.yml), matching test_model_factory_genome.py and
test_model_factory_spec.py.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Any, Dict

import pytest

from cognitive_runtime.training.model_factory.contracts import contract_hash
from cognitive_runtime.training.model_factory.genome import (
    GENERIC_ACTION_EFFECTS_V1,
    GenomeRepairError,
    build_schema,
)
from cognitive_runtime.training.model_factory.search import METHODS, SearchError, propose
from cognitive_runtime.training.model_factory.spec import (
    DOCUMENT_FORMAT,
    SpecError,
    dump_canonical_json,
    resolve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_raw(**block_overrides: Dict[str, Any]) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "format": DOCUMENT_FORMAT,
        "organism": "Crafter",
        "mode": "fresh",
        "data": {
            "corpus_id": "crafter-generic-action-effects-v1",
            "world": "crafter",
        },
        "model": {},
        "training": {"optimizer": {"lr": 0.0003}},
        "evaluation": {"selection_metric": "rollout.t+4.model_over_copy_last_mse"},
    }
    for block, override in block_overrides.items():
        raw[block] = {**raw.get(block, {}), **override}
    return raw


BASE_SPEC = resolve(_minimal_raw())
SCHEMA = GENERIC_ACTION_EFFECTS_V1


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------

def test_unknown_method_is_rejected():
    with pytest.raises(SearchError, match="unknown method"):
        propose(BASE_SPEC, SCHEMA, 4, seed=0, method="bayesian")


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_n_is_rejected(n):
    with pytest.raises(SearchError, match="positive"):
        propose(BASE_SPEC, SCHEMA, n, seed=0)


def test_methods_tuple_names_both_strategies():
    assert set(METHODS) == {"random", "lhs"}


# ---------------------------------------------------------------------------
# Acceptance: the same seed produces the same proposal set across processes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lhs", "random"])
def test_same_seed_reproducible_within_process(method):
    a = propose(BASE_SPEC, SCHEMA, 6, seed=7, method=method)
    b = propose(BASE_SPEC, SCHEMA, 6, seed=7, method=method)
    assert [dump_canonical_json(spec) for spec in a] == [dump_canonical_json(spec) for spec in b]


def test_different_seed_changes_the_proposal_set():
    a = propose(BASE_SPEC, SCHEMA, 6, seed=1, method="lhs")
    b = propose(BASE_SPEC, SCHEMA, 6, seed=2, method="lhs")
    assert [dump_canonical_json(spec) for spec in a] != [dump_canonical_json(spec) for spec in b]


_REPRODUCIBILITY_SCRIPT = """
import sys
sys.path.insert(0, {path!r})
from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V1
from cognitive_runtime.training.model_factory.search import propose
from cognitive_runtime.training.model_factory.spec import DOCUMENT_FORMAT, dump_canonical_json, resolve

raw = {{
    "format": DOCUMENT_FORMAT,
    "organism": "Crafter",
    "mode": "fresh",
    "data": {{"corpus_id": "crafter-generic-action-effects-v1", "world": "crafter"}},
    "model": {{}},
    "training": {{"optimizer": {{"lr": 0.0003}}}},
    "evaluation": {{"selection_metric": "rollout.t+4.model_over_copy_last_mse"}},
}}
base = resolve(raw)
proposals = propose(base, GENERIC_ACTION_EFFECTS_V1, 6, seed=99, method="lhs")
sys.stdout.buffer.write(b"|".join(dump_canonical_json(spec) for spec in proposals))
"""


def test_same_seed_reproducible_across_processes():
    env = dict(os.environ)
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", _REPRODUCIBILITY_SCRIPT.format(path=str(REPO_ROOT))],
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert len(outputs[0]) > 0


# ---------------------------------------------------------------------------
# Acceptance: all proposals in a batch share one architecture_hash and one
# data_contract_hash; only declared genes differ.
#
# The authoritative ArchitectureContract/DataContract hashes are computed
# downstream from a *built* model and a *resolved* corpus (runner.py,
# corpus.py) -- both require torch and on-disk corpus data, neither of
# which this torch-free module touches. What propose() actually guarantees,
# and what is checked here, is the structural property those downstream
# hashes are derived from: a batch's `model` and `data` blocks (the two
# blocks ArchitectureContract/DataContract are built from) are untouched,
# so hashing them directly (contract_hash, the same primitive
# ArchitectureContract/DataContract use internally) is identical across
# the whole batch.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lhs", "random"])
def test_batch_shares_one_architecture_and_data_identity(method):
    proposals = propose(BASE_SPEC, SCHEMA, 8, seed=3, method=method)
    architecture_hashes = {contract_hash(dict(spec.model)) for spec in proposals}
    data_hashes = {contract_hash(dict(spec.data)) for spec in proposals}
    assert len(architecture_hashes) == 1
    assert len(data_hashes) == 1
    assert architecture_hashes == {contract_hash(dict(BASE_SPEC.model))}
    assert data_hashes == {contract_hash(dict(BASE_SPEC.data))}
    for spec in proposals:
        assert spec.organism == BASE_SPEC.organism
        assert spec.mode == BASE_SPEC.mode
        assert spec.parent == BASE_SPEC.parent
        assert spec.evaluation == BASE_SPEC.evaluation
        assert spec.evolution == BASE_SPEC.evolution


def _leaf_paths(mapping: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        # `spec.training`'s nested blocks (optimizer, loss_weights, ...) are
        # frozen `MappingProxyType`, not `dict` -- recurse on any `Mapping`.
        if isinstance(value, ABCMapping):
            leaves.update(_leaf_paths(dict(value), path))
        else:
            leaves[path] = value
    return leaves


@pytest.mark.parametrize("method", ["lhs", "random"])
def test_only_declared_gene_leaves_differ_in_training_block(method):
    proposals = propose(BASE_SPEC, SCHEMA, 8, seed=11, method=method)
    base_leaves = _leaf_paths(dict(BASE_SPEC.training))
    for spec in proposals:
        candidate_leaves = _leaf_paths(dict(spec.training))
        # `loss_weights`/`transition_balance_policy` default to `{}`, so a
        # gene under either path is a *new* leaf in the candidate, not a
        # changed one -- both cases must still be limited to declared genes.
        added = set(candidate_leaves) - set(base_leaves)
        removed = set(base_leaves) - set(candidate_leaves)
        changed = {
            path
            for path in set(base_leaves) & set(candidate_leaves)
            if candidate_leaves[path] != base_leaves[path]
        }
        assert not removed
        assert added <= set(SCHEMA.gene_names)
        assert changed <= set(SCHEMA.gene_names)


# ---------------------------------------------------------------------------
# Acceptance: LHS with n samples places exactly one sample in each of the n
# strata per dimension.
# ---------------------------------------------------------------------------

def _stratum_index(value: float, low: float, high: float, log_scale: bool, n: int) -> int:
    if log_scale:
        value, low, high = math.log(value), math.log(low), math.log(high)
    unit = (value - low) / (high - low)
    return min(int(unit * n), n - 1)


@pytest.mark.parametrize(
    "gene_name", ["scheduled_sampling_p", "optimizer.lr", "loss_weights.closed_loop_pixel_loss_weight"]
)
def test_lhs_covers_every_stratum_exactly_once_for_bounded_genes(gene_name):
    n = 8
    proposals = propose(BASE_SPEC, SCHEMA, n, seed=5, method="lhs")
    gene = SCHEMA.genes[gene_name]
    low, high = gene.bounds
    values = []
    for spec in proposals:
        cursor: Any = dict(spec.training)
        for part in gene_name.split("."):
            cursor = cursor[part]
        values.append(cursor)
    strata = [_stratum_index(v, low, high, gene.log_scale, n) for v in values]
    assert sorted(strata) == list(range(n))


def test_lhs_covers_every_stratum_exactly_once_for_a_custom_int_bounded_gene():
    # The shipped schema declares no int+bounds gene (its int genes use
    # `choices`); build one to exercise int rounding against the same
    # exactly-one-per-stratum property, with a wide enough range that
    # rounding cannot collapse two strata onto the same integer.
    schema = build_schema(
        "search-test-int-bounds",
        {
            "batch_size": {
                "type": "int",
                "bounds": (0, 800),
                "default": 32,
                "mutation": {"distribution": "normal_perturb", "sigma": 10.0},
            }
        },
    )
    n = 8
    raw = _minimal_raw()
    base = resolve(raw)
    proposals = propose(base, schema, n, seed=13, method="lhs")
    values = [spec.training["batch_size"] for spec in proposals]
    strata = [_stratum_index(v, 0, 800, False, n) for v in values]
    assert sorted(strata) == list(range(n))


def test_lhs_covers_every_choice_evenly_for_categorical_genes():
    n = 12  # a multiple of len(choices) for rollout_frames (6 choices)
    proposals = propose(BASE_SPEC, SCHEMA, n, seed=17, method="lhs")
    values = [spec.training["rollout_frames"] for spec in proposals]
    gene = SCHEMA.genes["rollout_frames"]
    counts = {choice: values.count(choice) for choice in gene.choices}
    assert set(counts) == set(gene.choices)
    assert all(count == n // len(gene.choices) for count in counts.values())


# ---------------------------------------------------------------------------
# Acceptance: log-scale genes are uniform in log space (assert the
# distribution of log10(lr), not lr).
# ---------------------------------------------------------------------------

def test_random_method_log_scale_gene_is_uniform_in_log_space_not_linear_space():
    n = 4000
    proposals = propose(BASE_SPEC, SCHEMA, n, seed=23, method="random")
    lrs = [spec.training["optimizer"]["lr"] for spec in proposals]
    log_lrs = [math.log10(lr) for lr in lrs]

    low_log, high_log = math.log10(1e-5), math.log10(1e-2)
    assert min(log_lrs) >= low_log - 1e-9
    assert max(log_lrs) <= high_log + 1e-9

    # Uniform in log space => log10(lr)'s mean sits near the range midpoint.
    mean_log = sum(log_lrs) / len(log_lrs)
    midpoint = (low_log + high_log) / 2
    assert abs(mean_log - midpoint) < 0.15

    # The range spans 3 decades ([-5, -2]); a log-uniform draw lands in the
    # top decade ([-3, -2]) about 1/3 of the time. Naive *linear*-space
    # sampling over [1e-5, 1e-2] would instead put roughly 99.7% of draws in
    # that same top decade -- this distinguishes the two.
    top_decade_fraction = sum(1 for v in log_lrs if v >= -3.0) / n
    assert 0.2 < top_decade_fraction < 0.45


def test_lhs_method_log_scale_gene_strata_are_even_in_log_space():
    n = 10
    proposals = propose(BASE_SPEC, SCHEMA, n, seed=29, method="lhs")
    log_lrs = sorted(math.log10(spec.training["optimizer"]["lr"]) for spec in proposals)
    low_log, high_log = math.log10(1e-5), math.log10(1e-2)
    strata = [
        min(int((v - low_log) / (high_log - low_log) * n), n - 1) for v in log_lrs
    ]
    assert sorted(strata) == list(range(n))


# ---------------------------------------------------------------------------
# Acceptance: every proposal passes repair/validation before being
# returned -- an invalid spec is never emitted.
# ---------------------------------------------------------------------------

def test_propose_raises_instead_of_emitting_a_spec_that_fails_repair():
    # min_episode_length=6 is smaller than every warmup+rollout combination
    # the shipped schema can produce (min warmup=1, min rollout=4 => 5, but
    # every sampled combination across a real batch will regularly exceed
    # 6), so repair() must reject at least one candidate before any spec is
    # returned.
    with pytest.raises(GenomeRepairError):
        propose(BASE_SPEC, SCHEMA, 8, seed=31, method="lhs", min_episode_length=1)


def test_propose_raises_instead_of_emitting_a_spec_that_fails_spec_validation():
    # Not caught by genome.repair (no min_episode_length passed to propose),
    # but caught by spec.validate's own independent
    # warmup+rollout-fits-corpus check (spec.py::_validate_warmup_rollout_fits_corpus),
    # driven by `data.quality_policy.min_episode_length`. 13 is chosen so the
    # *base* spec's default warmup(4)+rollout(8)=12 still resolves cleanly,
    # while the schema's own wider warmup/rollout choices (up to 8+16=24)
    # let a search proposal exceed it. This exercises the second,
    # independent validation layer propose() relies on.
    raw = _minimal_raw(data={"quality_policy": {"min_episode_length": 13}})
    base = resolve(raw)
    with pytest.raises(SpecError):
        propose(base, SCHEMA, 8, seed=37, method="lhs")


def test_propose_returns_only_specs_that_pass_validation():
    proposals = propose(BASE_SPEC, SCHEMA, 8, seed=41, method="lhs")
    assert len(proposals) == 8
    for spec in proposals:
        # validate() raising would already have stopped propose() from
        # returning; re-running it here pins that every returned spec is
        # independently valid, not merely "didn't crash".
        from cognitive_runtime.training.model_factory.spec import validate

        validate(spec)


# ---------------------------------------------------------------------------
# Acceptance: imports cleanly without torch.
# ---------------------------------------------------------------------------

_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.search  # noqa: F401
print("OK")
"""


def test_search_module_imports_cleanly_without_torch():
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=str(REPO_ROOT))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
