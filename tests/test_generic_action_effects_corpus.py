"""Acceptance tests for the ``generic_action_effects_v1`` corpus spec and
event-stratified evaluation (issue #237, MF-E4, epic #212 Sec 12.2/12.3/12.6).

Mirrors ``test_model_factory_corpus.py``'s fake-nursery fixture pattern so
the scenario-mix/DataContract-field machinery is exercised fast and
torch-free; only the couple of tests that need a real Crafter-world
DataContract (``semantic_vocabulary_version``, the label schema version)
import torch, and do so lazily inside the test itself so the rest of this
module still collects and runs in a torch-free environment (CI's ``core``
tier -- see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognitive_runtime.training.model_factory.corpus as corpus_module
from cognitive_runtime.training.action_effect_taxonomy import ACTION_EFFECT_LABEL_VERSION
from cognitive_runtime.training.model_factory.contracts import DataContract
from cognitive_runtime.training.model_factory.corpus import DEFAULT_SPLIT_ROLES, load_corpus_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_SPEC_PATH = REPO_ROOT / "specs" / "corpora" / "generic-action-effects-v1.yaml"

CANARIES = ("walk_forward_short", "blocked_forward", "turn", "approach_entity")
SCENARIOS = ("motor_babbling_open", "motor_babbling_walls") + CANARIES

DECLARED_MIX = {
    "motor_babbling_open": 0.60,
    "motor_babbling_walls": 0.30,
    "walk_forward_short": 0.025,
    "blocked_forward": 0.025,
    "turn": 0.025,
    "approach_entity": 0.025,
}


def _fake_scenario(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _install_fake_nursery(monkeypatch, cache: Path, *, failing_scenarios=frozenset()) -> None:
    """A fast, deterministic stand-in for real Crafter recording.

    ``failing_scenarios`` simulates a scenario's *own* specialized quality
    gate (``NurseryScenario``'s per-scenario thresholds in real code)
    catching a regression -- independent of the aggregate scenario-mix gate
    exercised elsewhere in this file.
    """
    monkeypatch.setattr(
        corpus_module, "_scenarios_for_world",
        lambda world: {name: _fake_scenario(name) for name in SCENARIOS},
    )

    def validate(_paths, scenario, *, expected_pixel_source=None):
        if scenario.name in failing_scenarios:
            return [f"{scenario.name}: failed its own specialized quality gate"]
        return []

    monkeypatch.setattr(corpus_module, "validate_nursery_recordings", validate)

    def record_or_reuse(_record_dir, session_id, seed, scenario, _cfg):
        session = cache / session_id
        session.mkdir(parents=True, exist_ok=True)
        motor_babbling = None
        if scenario.name.startswith("motor_babbling"):
            motor_babbling = {
                "generator_name": scenario.name,
                "generator_version": "v1",
                "action_subset": ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "NULL"],
                "burst_ticks_distribution": {"policy": "uniform_integer", "min_ticks": 1, "max_ticks": 4},
                "generator_seed": seed,
            }
            if scenario.name == "motor_babbling_walls":
                motor_babbling["layout_distribution"] = {
                    "layout_seed": seed,
                    "outer_wall": {"shape": "square_ring", "radius": 5},
                    "interior_barrier": {
                        "orientations": ["vertical", "horizontal"],
                        "offsets": [-2, 2],
                        "gap_offsets": [-2, -1, 0, 1, 2],
                        "guaranteed_recovery_gap": True,
                    },
                    "realised_layout": {"barrier_orientation": "vertical", "barrier_offset": -2, "gap_offset": 0},
                }
        (session / "session.json").write_text(json.dumps({
            "session_id": session_id,
            "program_config": {"motor_babbling": motor_babbling},
        }), encoding="utf-8")
        (session / "episode_00000.decisions.jsonl").write_text("{}\n", encoding="utf-8")
        # Different session ids have different content, so the real overlap
        # gate can prove the frozen splits do not leak into one another.
        (session / "episode_00000.streams.jsonl").write_text(
            json.dumps({"stream_id": "vision.frame.pixels", "frame_ref": f"frame-{session_id}"}) + "\n",
            encoding="utf-8",
        )
        return str(session)

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", record_or_reuse)


def _mix_spec(
    tmp_path: Path, *, world: str = "minecraft", corpus_id: str = "generic-v1",
    train_split=None, weights=None, tolerance: float = 0.05,
) -> dict:
    """A ``generic_action_effects_v1``-shaped spec: 40 train episodes
    matching the epic's declared 60/30/2.5x4 mix exactly, by default.

    ``world`` defaults to ``"minecraft"`` -- a label only, since every
    scenario here is faked via ``_install_fake_nursery`` -- purely to keep
    ``_data_contract``'s lazy, torch-importing ``nursery`` module untouched
    for the tests that don't need it (mirrors
    ``test_model_factory_corpus.py``'s own fixture convention).
    """
    train_split = train_split or {
        "motor_babbling_open": list(range(24)),
        "motor_babbling_walls": list(range(12)),
        "walk_forward_short": [0],
        "blocked_forward": [0],
        "turn": [0],
        "approach_entity": [0],
    }
    return {
        "root": str(tmp_path / "corpora"),
        "organism": "Crafter",
        "corpus_id": corpus_id,
        "generator": {
            "world": world,
            "backend": world,
            "episode_cache_dir": str(tmp_path / "episode-cache"),
        },
        "splits": {
            "train": train_split,
            "validation": {name: [1000] for name in SCENARIOS},
            "test": {},
        },
        "scenario_mix": {
            "split": "train",
            "weights": weights or DECLARED_MIX,
            "tolerance": tolerance,
        },
    }


# --------------------------------------------------------------------- spec


def test_generic_action_effects_yaml_declares_the_epic_mix():
    """The shipped corpus spec (epic Sec 12.2) parses and its train-split
    mix is exactly 60% motor_babbling_open / 30% motor_babbling_walls /
    2.5% each of the four canaries."""
    raw = load_corpus_spec(CORPUS_SPEC_PATH)
    cfg = corpus_module._config_from_spec(raw)
    splits = corpus_module._split_assignments(raw, cfg)
    policy = corpus_module._scenario_mix_policy(raw, splits)

    assert policy["split"] == "train"
    assert policy["weights"] == DECLARED_MIX
    assert sum(policy["weights"].values()) == pytest.approx(1.0)
    assert set(splits["train"]) == set(SCENARIOS)


def test_corpus_recording_config_ignores_legacy_horizons():
    raw = load_corpus_spec(CORPUS_SPEC_PATH)
    cfg = corpus_module._config_from_spec(raw)

    assert raw["generator"]["horizons"] == [1, 10, 100]
    assert not hasattr(cfg, "horizons")


def test_scenario_recording_config_treats_sealed_test_seeds_as_held_out():
    """A corpus's test split is not a third layout regime: scripted
    generators must use their held-out schedule for it, just as validation
    does.  This covers the shipped ``blocked_forward`` test seed 2000."""
    cfg = corpus_module.CorpusNurseryConfig(train_seeds=(0,), holdout_seeds=(1000,))
    splits = {
        "train": {"blocked_forward": (0,)},
        "validation": {"blocked_forward": (1000,)},
        "test": {"blocked_forward": (2000,)},
    }

    resolved = corpus_module._scenario_recording_config(cfg, splits, "blocked_forward")

    assert resolved.train_seeds == (0,)
    assert resolved.holdout_seeds == (1000, 2000)


# ------------------------------------------------------------- mix builds


def test_corpus_builds_end_to_end_and_freezes_manifest_within_declared_mix_tolerance(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    spec = _mix_spec(tmp_path)

    first = corpus_module.build_corpus(spec)
    quality = json.loads((first.directory / "quality_report.json").read_text(encoding="utf-8"))
    mix_report = quality["scenario_mix_report"]

    assert quality["passed"] is True
    assert mix_report["within_tolerance"] is True
    assert mix_report["realised"]["motor_babbling_open"] == pytest.approx(0.60)
    assert mix_report["realised"]["motor_babbling_walls"] == pytest.approx(0.30)
    for canary in CANARIES:
        assert mix_report["realised"][canary] == pytest.approx(0.025)

    # Rebuilding the same declaration reuses the frozen manifest rather than
    # re-recording (the existing corpus-identity guarantee), so this also
    # proves the mix-policy fields round-trip through corpus_spec.json.
    def unexpected_record(*args, **kwargs):
        pytest.fail("a completed corpus must not record replacement episodes")

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", unexpected_record)
    second = corpus_module.build_corpus(spec)
    assert second.manifest == first.manifest


def test_realised_mix_miss_is_reported_and_fails_the_quality_gate(tmp_path, monkeypatch):
    """Issue #237 acceptance criterion: a corpus that misses the declared
    weights fails its quality gate rather than being silently accepted."""
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    # One episode per scenario (~16.7% each) is nowhere near the declared
    # 60/30/2.5x4 mix.
    skewed_split = {name: [0] for name in SCENARIOS}
    spec = _mix_spec(tmp_path, corpus_id="skewed-v1", train_split=skewed_split, tolerance=0.05)

    with pytest.raises(ValueError, match="did not pass its gates"):
        corpus_module.build_corpus(spec)

    quality = json.loads(
        (tmp_path / "corpora" / "Crafter" / "skewed-v1" / "quality_report.json").read_text(encoding="utf-8")
    )
    mix_report = quality["scenario_mix_report"]
    assert quality["passed"] is False
    assert mix_report["within_tolerance"] is False
    assert mix_report["realised"]["motor_babbling_open"] == pytest.approx(1 / 6)
    assert any("motor_babbling_open" in issue for issue in quality["issues"])
    assert any("scenario mix" in issue for issue in quality["issues"])


def test_scenario_mix_is_opt_in_and_declared_weights_must_sum_to_one(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    no_mix_spec = _mix_spec(tmp_path, corpus_id="no-mix-v1")
    del no_mix_spec["scenario_mix"]
    corpus = corpus_module.build_corpus(no_mix_spec)
    quality = json.loads((corpus.directory / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["scenario_mix_report"] is None

    bad_weights_spec = _mix_spec(
        tmp_path, corpus_id="bad-weights-v1",
        weights={"motor_babbling_open": 0.5, "motor_babbling_walls": 0.4},
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        corpus_module.build_corpus(bad_weights_spec)


# ------------------------------------------------------ DataContract fields


def test_every_epic_datacontract_field_is_populated(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus = corpus_module.build_corpus(_mix_spec(tmp_path, corpus_id="fields-v1"))
    data = corpus.data_contract

    assert set(data.scenario_generators) == {"motor_babbling_open", "motor_babbling_walls"}
    open_generator = data.scenario_generators["motor_babbling_open"]
    assert open_generator["generator_name"] == "motor_babbling_open"
    assert open_generator["generator_version"] == "v1"
    assert tuple(open_generator["action_subset"]) == ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "NULL")
    assert dict(open_generator["burst_ticks_distribution"]) == {
        "policy": "uniform_integer", "min_ticks": 1, "max_ticks": 4,
    }
    assert "generator_seed" not in open_generator  # varies per episode; not a corpus-level constant

    walls_layout = data.scenario_generators["motor_babbling_walls"]["layout_distribution"]
    assert dict(walls_layout["outer_wall"]) == {"shape": "square_ring", "radius": 5}
    assert "layout_seed" not in walls_layout
    assert "realised_layout" not in walls_layout

    assert data.split_roles == DEFAULT_SPLIT_ROLES
    assert data.scenario_mix_policy["weights"] == DECLARED_MIX
    # world="minecraft" here (see ``_mix_spec``'s docstring), so the
    # Crafter-specific label schema version is deliberately not populated;
    # the crafter-world case is covered by the torch-guarded test below.
    assert data.action_effect_label_schema_version == ""


def test_crafter_world_datacontract_carries_the_real_label_and_vocabulary_versions(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    from cognitive_runtime.training.nursery import SEMANTIC_VOCABULARY_VERSION

    corpus = corpus_module.build_corpus(_mix_spec(tmp_path, world="crafter", corpus_id="crafter-fields-v1"))
    data = corpus.data_contract

    assert data.action_effect_label_schema_version == ACTION_EFFECT_LABEL_VERSION
    assert data.semantic_vocabulary_version == SEMANTIC_VOCABULARY_VERSION


def test_changing_any_epic_field_changes_the_data_contract_hash():
    """Issue #237 acceptance criterion: every Sec 12.6 field's hash
    contribution, checked directly against the frozen dataclass rather than
    a full corpus build."""
    base_kwargs = dict(
        world="crafter", backend="crafter", program_config={}, scenario_names=("motor_babbling_open",),
        scenario_code_version="v1", train_session_ids=("s1",), train_session_hashes=("h1",),
        validation_session_ids=(), validation_session_hashes=(), test_session_ids=(), test_session_hashes=(),
        seed_assignments={"s1": 0}, pixel_provenance="crafter", semantic_vocabulary_version="v1",
        preprocessing_version="native", horizons_ticks=(1, 10, 100), ticks_per_frame=1.0,
        scenario_generators={"motor_babbling_open": {"generator_version": "v1"}},
        action_effect_label_schema_version=ACTION_EFFECT_LABEL_VERSION,
        split_roles=dict(DEFAULT_SPLIT_ROLES),
        scenario_mix_policy={"split": "train", "weights": {"motor_babbling_open": 1.0}, "tolerance": 0.05},
    )
    baseline = DataContract(**base_kwargs).hash

    for field_name, changed_value in (
        ("scenario_generators", {"motor_babbling_open": {"generator_version": "v2"}}),
        ("action_effect_label_schema_version", "action-effect-v2"),
        ("split_roles", {"train": "generic_training", "validation": "validation", "test": "different"}),
        ("scenario_mix_policy", {"split": "train", "weights": {"motor_babbling_open": 1.0}, "tolerance": 0.10}),
    ):
        changed_kwargs = dict(base_kwargs)
        changed_kwargs[field_name] = changed_value
        assert DataContract(**changed_kwargs).hash != baseline, f"{field_name} did not change the hash"


# --------------------------------------------------------------- canaries


def test_canary_scenario_retains_its_own_quality_gate_even_when_the_aggregate_mix_is_healthy(tmp_path, monkeypatch):
    """Epic Sec 12.2: canaries "retain their specialized quality gates and
    allow regressions ... to be found even when a broad action-burst
    average is healthy." A ``turn`` regression must fail the corpus build
    even though the declared 60/30/2.5x4 mix is met exactly."""
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache", failing_scenarios={"turn"})
    spec = _mix_spec(tmp_path, corpus_id="canary-regression-v1")

    with pytest.raises(ValueError, match="did not pass its gates"):
        corpus_module.build_corpus(spec)

    quality = json.loads(
        (tmp_path / "corpora" / "Crafter" / "canary-regression-v1" / "quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    # The aggregate mix is exactly on-target -- the canary's own gate is
    # what caught this, not the mix gate.
    assert quality["scenario_mix_report"]["within_tolerance"] is True
    assert quality["passed"] is False
    assert any("turn" in issue and "specialized quality gate" in issue for issue in quality["issues"])
