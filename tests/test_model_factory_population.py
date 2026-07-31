"""Champion population policy regression coverage (issue #233)."""

from __future__ import annotations

from cognitive_runtime.training.model_factory.population import (
    ComputeLedger,
    DuplicateSpecCache,
    PopulationCandidate,
    PopulationPolicy,
    configuration_distance,
    replace_population,
    select_parents,
)


def _policy(**overrides):
    values = {
        "declared_genome_fields": ("optimizer.lr", "rollout_frames"),
        "capacity": 4,
        "comparable_primary_margin": 0.05,
    }
    values.update(overrides)
    return PopulationPolicy(**values)


def _candidate(run_id, *, primary=1.0, runtime=10.0, retention=1.0, genome=None,
               gates=None, status="completed", trial_compute=10.0, ancestral_compute=None):
    trial = float(trial_compute)
    return PopulationCandidate(
        run_id=run_id,
        resolved_spec={"training": {"seed": run_id}},
        genome=genome or {"optimizer.lr": 0.001, "rollout_frames": 4},
        primary_metric=primary,
        runtime=runtime,
        retention_metric=retention,
        ledger=ComputeLedger(trial, trial if ancestral_compute is None else ancestral_compute),
        gates={"quality": True, "rollout_health": True, "representation": True, "time_budget": True}
        if gates is None else gates,
        completion_status=status,
    )


def test_replacement_is_deterministic_and_breaks_exact_ties_by_run_id():
    policy = _policy()
    candidates = [_candidate("z-run"), _candidate("a-run")]

    first = replace_population(candidates, policy)
    second = replace_population(reversed(candidates), policy)

    assert first.to_dict() == second.to_dict()
    assert first.leading_champion == "a-run"
    assert [member.run_id for member in first.members] == ["a-run"]
    assert first.rejected["z-run"] == "dominated near-duplicate of a-run"


def test_gate_failures_and_budget_exceeded_never_enter_population():
    decision = replace_population([
        _candidate("best-but-quality-failed", primary=0.0, gates={"quality": False}),
        _candidate("best-but-over-budget", primary=0.0, status="budget_exceeded"),
        _candidate("completed", primary=2.0),
    ], _policy())

    assert [member.run_id for member in decision.members] == ["completed"]
    assert "quality" in decision.rejected["best-but-quality-failed"]
    assert decision.rejected["best-but-over-budget"] == "budget_exceeded"


def test_dominated_near_duplicate_is_removed_but_distant_configuration_survives():
    policy = _policy(comparable_primary_margin=0.2)
    leader = _candidate("leader", primary=1.0, runtime=9.0, retention=1.0)
    near_worse = _candidate("near-worse", primary=1.1, runtime=10.0, retention=1.1)
    distant_comparable = _candidate(
        "distant", primary=1.1, runtime=10.0, retention=1.1,
        genome={"optimizer.lr": 0.01, "rollout_frames": 8},
    )

    decision = replace_population([near_worse, distant_comparable, leader], policy)

    assert [member.run_id for member in decision.members] == ["distant", "leader"]
    assert "dominated near-duplicate" in decision.rejected["near-worse"]


def test_configuration_distance_uses_only_declared_genome_fields():
    policy = _policy()
    left = _candidate("left", genome={"optimizer.lr": 0.001, "rollout_frames": 4, "runtime_secret": "a"})
    right = _candidate("right", genome={"optimizer.lr": 0.001, "rollout_frames": 4, "runtime_secret": "b"})

    assert configuration_distance(left, right, policy) == 0.0


def test_duplicate_resolved_spec_reuses_prior_result_without_running_again():
    cache = DuplicateSpecCache()
    calls = []
    spec = {"training": {"lr": 0.001}, "data": {"corpus_id": "fixed"}}

    result, reused = cache.reuse_or_run(spec, lambda: calls.append("ran") or {"run_id": "one"})
    duplicate, duplicate_reused = cache.reuse_or_run(dict(spec), lambda: calls.append("should-not-run"))

    assert result == duplicate == {"run_id": "one"}
    assert not reused and duplicate_reused
    assert calls == ["ran"]


def test_compute_ledger_exposes_trial_and_weight_lineage_costs_separately():
    donor = ComputeLedger.fresh(1_000.0)
    child = ComputeLedger.child(10.0, donor)

    assert child.trial_compute == 10.0
    assert child.ancestral_compute == 1_010.0
    assert child.inherited_compute == 1_000.0


def test_parent_tournament_is_seeded_and_uses_only_gate_passing_population_members():
    policy = _policy(tournament_size=3)
    members = [
        _candidate("a", primary=1.0, genome={"optimizer.lr": 0.001, "rollout_frames": 4}),
        _candidate("b", primary=1.1, genome={"optimizer.lr": 0.01, "rollout_frames": 8}),
        _candidate("c", primary=1.2, genome={"optimizer.lr": 0.005, "rollout_frames": 4}),
        _candidate("failed", primary=0.0, gates={"quality": False}),
    ]

    first = select_parents(members, policy, seed=73)
    second = select_parents(members, policy, seed=73)

    assert tuple(member.run_id for member in first) == tuple(member.run_id for member in second)
    assert all(member.run_id != "failed" for member in first)
