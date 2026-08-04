"""Regression coverage for the JSON-backed champion registry: tiered
portfolio, atomic locked store, and lineage graph (epic #212 Phase C step 4,
§7, §15.1, issue #224).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from cognitive_runtime.training.model_factory.artifacts import LINEAGE_FORMAT
from cognitive_runtime.training.model_factory.registry import (
    REGISTRY_FORMAT,
    LineageCycleError,
    RegistryError,
    assemble_population_candidates,
    hold,
    leading_champion,
    lineage_graph,
    population,
    promote,
    record_test_use,
    registry_path,
)
from cognitive_runtime.training.model_factory.population import (
    ComputeLedger,
    PopulationCandidate,
    PopulationPolicy,
)
from cognitive_runtime.training.model_factory.state import (
    STATE_BUDGET_EXCEEDED,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_RUNNING,
    IncompleteRunError,
    create_state,
    state_path,
    transition,
)


def _mark_completed(root, run_id):
    """Drive a fresh run's persisted trial state to `completed` -- the gate
    `promote` requires before a run can enter the champion population."""
    path = state_path(root / run_id)
    create_state(path, run_id)
    transition(path, STATE_RUNNING)
    transition(path, STATE_COMPLETED)


def _write_lineage(
    root, run_id, *, parent=None, configuration_parents=(), weight_donor=None, mode="fresh",
):
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": LINEAGE_FORMAT,
        "mode": mode,
        "parent": (
            {"run_id": parent, "checkpoint": "best-validation.pt", "sha256": "deadbeef"}
            if parent else None
        ),
        "parent_checkpoint_sha": "deadbeef" if parent else None,
        "sibling_group": None,
        "configuration_parents": list(configuration_parents),
        "weight_donor": weight_donor,
    }
    (directory / "lineage.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------- AC1: tiered portfolio


def test_promoting_a_scale_champion_leaves_the_fast_champion_intact(tmp_path):
    _mark_completed(tmp_path, "run-fast-1")
    _mark_completed(tmp_path, "run-scale-1")
    promote(
        tmp_path, family="generic_action_effects_v1", tier="fast", objective="rollout.t+4",
        run_id="run-fast-1", checkpoint_path="checkpoints/best-validation.pt",
        checkpoint_sha256="a" * 64, metrics={"rollout.t+4.mse": 0.1},
    )
    promote(
        tmp_path, family="generic_action_effects_v1", tier="scale", objective="rollout.t+4",
        run_id="run-scale-1", checkpoint_path="checkpoints/best-validation.pt",
        checkpoint_sha256="b" * 64, metrics={"rollout.t+4.mse": 0.05},
    )

    fast = leading_champion(tmp_path, family="generic_action_effects_v1", tier="fast", objective="rollout.t+4")
    scale = leading_champion(tmp_path, family="generic_action_effects_v1", tier="scale", objective="rollout.t+4")
    assert fast.run_id == "run-fast-1"
    assert scale.run_id == "run-scale-1"


def test_promoting_a_new_leader_replaces_only_its_own_slots_leader(tmp_path):
    _mark_completed(tmp_path, "a")
    _mark_completed(tmp_path, "b")
    kwargs = dict(family="f", tier="fast", objective="obj", checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    promote(tmp_path, run_id="a", **kwargs)
    promote(tmp_path, run_id="b", **kwargs)

    champion = leading_champion(tmp_path, family="f", tier="fast", objective="obj")
    assert champion.run_id == "b"
    members = population(tmp_path, family="f", tier="fast", objective="obj")
    assert {entry.run_id for entry in members} == {"a", "b"}


def test_promote_can_add_a_population_member_without_becoming_the_leader(tmp_path):
    _mark_completed(tmp_path, "a")
    _mark_completed(tmp_path, "b")
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="b",
            checkpoint_path="p2", checkpoint_sha256="d" * 64, metrics={}, as_leading=False)

    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj").run_id == "a"
    members = {entry.run_id for entry in population(tmp_path, family="f", tier="fast", objective="obj")}
    assert members == {"a", "b"}


def test_policy_backed_promotion_replaces_the_persisted_population_atomically(tmp_path):
    """D5 replacement is enforced at the registry write boundary, not merely
    calculated by a caller and then ignored while appending the entry."""
    _mark_completed(tmp_path, "leader")
    _mark_completed(tmp_path, "near-worse")
    policy = PopulationPolicy(declared_genome_fields=("optimizer.lr",), capacity=4)

    def candidate(run_id, primary):
        return PopulationCandidate(
            run_id=run_id,
            resolved_spec={"training": {"seed": run_id}},
            genome={"optimizer.lr": 0.001},
            primary_metric=primary,
            runtime=1.0,
            retention_metric=primary,
            ledger=ComputeLedger.fresh(1.0),
            gates={"quality": True},
        )

    leader = candidate("leader", 1.0)
    promote(
        tmp_path, family="f", tier="fast", objective="obj", run_id="leader",
        checkpoint_path="leader.pt", checkpoint_sha256="a" * 64, metrics={},
        population_policy=policy, population_candidates=(leader,),
    )
    slot = promote(
        tmp_path, family="f", tier="fast", objective="obj", run_id="near-worse",
        checkpoint_path="near-worse.pt", checkpoint_sha256="b" * 64, metrics={},
        population_policy=policy, population_candidates=(leader, candidate("near-worse", 2.0)),
    )

    assert [member.run_id for member in slot.population] == ["leader"]
    assert slot.leading_champion == "leader"
    assert slot.history[-1]["retained"] is False


# --------------------------------------------------------------------- population assembly


def _write_population_run_artifacts(root, run_id, *, metric_value=0.1, runtime=10.0, ticks_per_frame=1.0):
    """The minimal on-disk artifacts assemble_population_candidates reads
    for one run: trial_spec.json (only the two genome fields this test's
    minimal schema declares need be present -- absent schema fields are
    skipped, not required), contracts.json (for ticks_per_frame),
    metrics/validation.json, metrics/budget_report.json."""
    directory = root / run_id
    (directory / "metrics").mkdir(parents=True, exist_ok=True)
    (directory / "trial_spec.json").write_text(json.dumps({
        "training": {"optimizer": {"lr": 0.001}, "scheduled_sampling_p": 0.25},
    }), encoding="utf-8")
    (directory / "contracts.json").write_text(json.dumps({
        "data_contract": {"ticks_per_frame": ticks_per_frame},
    }), encoding="utf-8")
    (directory / "metrics" / "validation.json").write_text(json.dumps({
        "rollout": {"horizons": {}, "per_episode_model_mse": {}},
        "direct": {
            "horizons": {"4": {"model_mse": metric_value}},
            "per_episode_model_mse": {"4": [metric_value]},
        },
        "per_episode_model_mse": {"4": [metric_value]},
    }), encoding="utf-8")
    (directory / "metrics" / "budget_report.json").write_text(json.dumps({
        "total_trial_seconds": runtime, "completion_status": "completed",
    }), encoding="utf-8")


def test_assemble_population_candidates_merges_existing_members_with_a_new_run(tmp_path):
    objective = "direct.t+4.model_mse"
    for run_id in ("existing-a", "existing-b", "new-run"):
        _mark_completed(tmp_path, run_id)
        _write_population_run_artifacts(tmp_path, run_id)
    for run_id in ("existing-a", "existing-b"):
        promote(
            tmp_path, family="f", tier="fast", objective=objective, run_id=run_id,
            checkpoint_path="p", checkpoint_sha256="a" * 64, metrics={}, as_leading=False,
        )

    policy, candidates = assemble_population_candidates(
        tmp_path, family="f", tier="fast", objective=objective,
        new_run_directory=tmp_path / "new-run", new_run_id="new-run",
        new_run_gates={"rollout_health": True},
    )

    assert set(policy.declared_genome_fields) == {"optimizer.lr", "scheduled_sampling_p"}
    assert [candidate.run_id for candidate in candidates][-1] == "new-run"
    assert {candidate.run_id for candidate in candidates} == {"existing-a", "existing-b", "new-run"}
    by_id = {candidate.run_id: candidate for candidate in candidates}
    assert by_id["new-run"].gates == {"rollout_health": True}
    # An existing member's candidate is rebuilt from its own persisted
    # artifacts, not carried over with whatever gates it happened to record
    # at its own original promotion time.
    assert by_id["existing-a"].gates == {}
    assert by_id["existing-a"].genome == {"optimizer.lr": 0.001, "scheduled_sampling_p": 0.25}


def test_assemble_population_candidates_excludes_the_new_run_from_the_existing_pass(tmp_path):
    """Re-promoting an already-registered run must rebuild it once (as the
    new run, with fresh gates), never twice."""
    objective = "direct.t+4.model_mse"
    for run_id in ("already-in-population",):
        _mark_completed(tmp_path, run_id)
        _write_population_run_artifacts(tmp_path, run_id, metric_value=0.2)
    promote(
        tmp_path, family="f", tier="fast", objective=objective, run_id="already-in-population",
        checkpoint_path="p", checkpoint_sha256="a" * 64, metrics={},
    )
    # Overwrite its artifacts as if it were re-evaluated with a better metric
    # before this second promotion call.
    _write_population_run_artifacts(tmp_path, "already-in-population", metric_value=0.05)

    _, candidates = assemble_population_candidates(
        tmp_path, family="f", tier="fast", objective=objective,
        new_run_directory=tmp_path / "already-in-population", new_run_id="already-in-population",
        new_run_gates={"rollout_health": True},
    )

    assert len(candidates) == 1
    assert candidates[0].gates == {"rollout_health": True}
    assert candidates[0].primary_metric == pytest.approx(0.05)


def test_unknown_tier_is_rejected(tmp_path):
    _mark_completed(tmp_path, "a")
    with pytest.raises(RegistryError, match="unknown runtime budget tier"):
        promote(tmp_path, family="f", tier="medium", objective="obj", run_id="a",
                checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})


def test_leading_champion_and_population_are_empty_for_an_unknown_slot(tmp_path):
    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj") is None
    assert population(tmp_path, family="f", tier="fast", objective="obj") == ()


# --------------------------------------------------------------------- AC2: no tensor payload


class _FakeTensor:
    """Stands in for a torch.Tensor without actually importing torch."""


def test_promote_rejects_a_non_json_safe_metric_value(tmp_path):
    _mark_completed(tmp_path, "a")
    with pytest.raises(RegistryError, match="tensors or model weights"):
        promote(
            tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64,
            metrics={"weights": _FakeTensor()},
        )
    assert not registry_path(tmp_path).exists()


def test_registry_document_never_contains_a_state_dict_style_key(tmp_path):
    _mark_completed(tmp_path, "a")
    promote(
        tmp_path, family="f", tier="fast", objective="obj", run_id="a",
        checkpoint_path="checkpoints/best-validation.pt", checkpoint_sha256="c" * 64,
        metrics={"rollout.t+4.mse": 0.1},
    )
    raw = registry_path(tmp_path).read_text(encoding="utf-8")
    for forbidden in ("model_state_dict", "optimizer_state_dict", "tensor("):
        assert forbidden not in raw
    document = json.loads(raw)
    assert document["format"] == REGISTRY_FORMAT


# --------------------------------------------------------------------- promotion gate (MF-B5 reuse)


def test_promote_rejects_a_run_with_no_persisted_trial_state(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
                checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    assert not registry_path(tmp_path).exists()


@pytest.mark.parametrize(
    "steps",
    [
        (),  # still queued
        (STATE_RUNNING,),  # interrupted mid-flight
        (STATE_RUNNING, STATE_FAILED),
        (STATE_RUNNING, STATE_BUDGET_EXCEEDED),
        (STATE_CANCELLED,),
    ],
)
def test_promote_rejects_a_run_that_has_not_completed(tmp_path, steps):
    path = state_path(tmp_path / "a")
    create_state(path, "a")
    for step in steps:
        transition(path, step)

    with pytest.raises(IncompleteRunError):
        promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
                checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    assert not registry_path(tmp_path).exists()


# --------------------------------------------------------------------- AC3: concurrent promotions


_PROMOTE_WORKER_SCRIPT = textwrap.dedent(
    """
    import sys, os, time, json
    sys.path.insert(0, {path!r})
    from cognitive_runtime.training.model_factory.registry import promote

    ready_file = {ready_file!r}
    go_file = {go_file!r}
    out_file = {out_file!r}

    with open(ready_file, "w") as handle:
        handle.write("ready")
    deadline = time.monotonic() + 30.0
    while not os.path.exists(go_file):
        if time.monotonic() > deadline:
            raise SystemExit("timed out waiting for go file")

    result = {{}}
    try:
        promote(
            {root!r}, family={family!r}, tier={tier!r}, objective={objective!r},
            run_id={run_id!r}, checkpoint_path="p", checkpoint_sha256={sha!r},
            metrics={{"m": 1.0}},
        )
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    with open(out_file, "w") as handle:
        json.dump(result, handle)
    """
)


def _run_promote_workers(tmp_path, jobs):
    """Launch one real subprocess per (label, family, tier, objective, run_id,
    sha) job, release them together, and return their JSON results.

    Each job's run_id is first driven to `completed` in this (parent)
    process, since the run directories live on the shared filesystem and
    `promote` now requires a completed trial state to accept a candidate.
    """
    for _label, _family, _tier, _objective, run_id, _sha in jobs:
        _mark_completed(tmp_path, run_id)

    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    procs, outcomes = [], []
    for label, family, tier, objective, run_id, sha in jobs:
        ready_file = tmp_path / f"ready-{label}"
        go_file = tmp_path / "go"
        out_file = tmp_path / f"out-{label}.json"
        script = _PROMOTE_WORKER_SCRIPT.format(
            path=repo_root, root=str(tmp_path), family=family, tier=tier, objective=objective,
            run_id=run_id, sha=sha, ready_file=str(ready_file), go_file=str(go_file), out_file=str(out_file),
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        procs.append(proc)
        outcomes.append((ready_file, out_file))

    deadline = time.monotonic() + 30.0
    for ready_file, _out_file in outcomes:
        while not ready_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    (tmp_path / "go").write_text("go", encoding="utf-8")

    for proc in procs:
        assert proc.wait(timeout=30) == 0

    return [json.loads(out_file.read_text(encoding="utf-8")) for _, out_file in outcomes]


def test_concurrent_promotions_to_different_slots_from_two_real_subprocesses_both_land(tmp_path):
    """AC: concurrent promotions serialise; the resulting document is valid
    and contains both effects."""
    results = _run_promote_workers(tmp_path, [
        ("a", "f", "fast", "obj", "run-a", "a" * 64),
        ("b", "f", "scale", "obj", "run-b", "b" * 64),
    ])
    assert all(r["ok"] for r in results), results

    document = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))
    assert document["format"] == REGISTRY_FORMAT
    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj").run_id == "run-a"
    assert leading_champion(tmp_path, family="f", tier="scale", objective="obj").run_id == "run-b"


def test_concurrent_promotions_to_the_same_slot_serialize_without_corruption(tmp_path):
    """AC: two racing promotions into the *same* slot never interleave into
    a corrupt document -- the lock serialises them and both candidates land
    in the population."""
    results = _run_promote_workers(tmp_path, [
        ("a", "f", "fast", "obj", "run-a", "a" * 64),
        ("b", "f", "fast", "obj", "run-b", "b" * 64),
    ])
    assert all(r["ok"] for r in results), results

    document = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))
    assert document["format"] == REGISTRY_FORMAT
    members = {entry.run_id for entry in population(tmp_path, family="f", tier="fast", objective="obj")}
    assert members == {"run-a", "run-b"}
    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj").run_id in {"run-a", "run-b"}


# --------------------------------------------------------------------- hold / record_test_use


def test_hold_records_a_decision_without_touching_leading_champion_or_population(tmp_path):
    _mark_completed(tmp_path, "a")
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    hold(tmp_path, family="f", tier="fast", objective="obj", run_id="b", reason="did not clear the margin")

    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj").run_id == "a"
    members = {entry.run_id for entry in population(tmp_path, family="f", tier="fast", objective="obj")}
    assert members == {"a"}


def test_record_test_use_increments_the_named_champions_counter(tmp_path):
    _mark_completed(tmp_path, "a")
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})

    updated = record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="a")
    assert updated.test_uses == 1
    updated_again = record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="a")
    assert updated_again.test_uses == 2
    assert leading_champion(tmp_path, family="f", tier="fast", objective="obj").test_uses == 2


def test_record_test_use_rejects_a_run_id_not_in_the_population(tmp_path):
    _mark_completed(tmp_path, "a")
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={})
    with pytest.raises(RegistryError, match="not a population member"):
        record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="unknown")


def test_record_test_use_rejects_an_unknown_slot(tmp_path):
    with pytest.raises(RegistryError, match="no registry slot"):
        record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="a")


def test_re_promoting_a_run_preserves_its_accumulated_test_uses(tmp_path):
    """Regression: re-promoting an existing population member (e.g. to
    refresh its checkpoint pointer or metrics) must not reset the sealed
    test-use ledger `record_test_use` already accumulated for it -- that
    would let the explicit test-use budget (epic §10.3) be bypassed by a
    re-promotion."""
    _mark_completed(tmp_path, "a")
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p", checkpoint_sha256="c" * 64, metrics={"m": 1.0})
    record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="a")
    record_test_use(tmp_path, family="f", tier="fast", objective="obj", run_id="a")

    # Re-promote the same run with an updated checkpoint pointer/metrics.
    promote(tmp_path, family="f", tier="fast", objective="obj", run_id="a",
            checkpoint_path="p2", checkpoint_sha256="d" * 64, metrics={"m": 2.0})

    champion = leading_champion(tmp_path, family="f", tier="fast", objective="obj")
    assert champion.checkpoint_path == "p2"
    assert champion.metrics == {"m": 2.0}
    assert champion.test_uses == 2


# --------------------------------------------------------------------- AC4/5: lineage graph


def test_lineage_graph_walks_a_fresh_to_clone_chain(tmp_path):
    _write_lineage(tmp_path, "root-run", mode="fresh")
    _write_lineage(tmp_path, "child-run", parent="root-run", mode="clone")

    graph = lineage_graph(tmp_path, "child-run")
    assert graph.root_run_id == "child-run"
    assert graph.ancestor_ids == ("root-run",)
    assert ("child-run", "root-run") in graph.edges


def test_lineage_graph_includes_both_configuration_parents_and_the_weight_donor(tmp_path):
    _write_lineage(tmp_path, "parent-a", mode="fresh")
    _write_lineage(tmp_path, "parent-b", mode="fresh")
    _write_lineage(
        tmp_path, "bred-child", mode="fine_tune",
        configuration_parents=["parent-a", "parent-b"], weight_donor="parent-b",
    )

    graph = lineage_graph(tmp_path, "bred-child")
    assert set(graph.ancestor_ids) == {"parent-a", "parent-b"}
    node = graph.nodes["bred-child"]
    assert node.configuration_parents == ("parent-a", "parent-b")
    assert node.weight_donor == "parent-b"
    # weight_donor is deduplicated against configuration_parents, not walked twice.
    assert node.parents == ("parent-a", "parent-b")


def test_lineage_graph_treats_a_missing_ancestor_manifest_as_a_leaf(tmp_path):
    _write_lineage(tmp_path, "child-run", parent="ghost-run")  # ghost-run has no manifest

    graph = lineage_graph(tmp_path, "child-run")
    assert graph.ancestor_ids == ("ghost-run",)
    assert graph.nodes["ghost-run"].parents == ()


def test_lineage_graph_handles_a_diamond_without_mistaking_it_for_a_cycle(tmp_path):
    _write_lineage(tmp_path, "grandparent", mode="fresh")
    _write_lineage(tmp_path, "parent-a", parent="grandparent")
    _write_lineage(tmp_path, "parent-b", parent="grandparent")
    _write_lineage(
        tmp_path, "child", configuration_parents=["parent-a", "parent-b"], weight_donor="parent-a",
    )

    graph = lineage_graph(tmp_path, "child")
    assert set(graph.ancestor_ids) == {"parent-a", "parent-b", "grandparent"}


def test_lineage_graph_terminates_on_an_injected_cycle_instead_of_recursing_forever(tmp_path):
    _write_lineage(tmp_path, "run-a", parent="run-b")
    _write_lineage(tmp_path, "run-b", parent="run-a")

    with pytest.raises(LineageCycleError):
        lineage_graph(tmp_path, "run-a")


def test_lineage_graph_rejects_a_run_with_no_lineage_manifest_at_all(tmp_path):
    with pytest.raises(RegistryError, match="no lineage manifest"):
        lineage_graph(tmp_path, "does-not-exist")


# --------------------------------------------------------------------- AC6: torch-free import


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.registry  # noqa: F401
print("OK")
"""


def test_registry_module_imports_cleanly_without_torch():
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=repo_root)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
