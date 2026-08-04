"""Acceptance tests for the immutable Model Factory corpus boundary."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognitive_runtime.training.model_factory.corpus as corpus_module


def _fake_scenario() -> SimpleNamespace:
    return SimpleNamespace(name="synthetic")


def _install_fake_nursery(monkeypatch, cache: Path) -> None:
    monkeypatch.setattr(corpus_module, "_scenarios_for_world", lambda world: {"synthetic": _fake_scenario()})
    monkeypatch.setattr(corpus_module, "validate_nursery_recordings", lambda *args, **kwargs: [])

    def record_or_reuse(_record_dir, _session_id, seed, scenario, _cfg):
        session = cache / f"{scenario.name}-{seed}"
        session.mkdir(parents=True, exist_ok=True)
        (session / "session.json").write_text(json.dumps({"session_id": session.name}), encoding="utf-8")
        (session / "episode_00000.decisions.jsonl").write_text("{}\n", encoding="utf-8")
        # Different seeds have different content, so the real overlap gate can
        # prove that the three frozen splits do not leak into one another.
        (session / "episode_00000.streams.jsonl").write_text(
            json.dumps({"stream_id": "vision.frame.pixels", "frame_ref": f"frame-{seed}"}) + "\n",
            encoding="utf-8",
        )
        return str(session)

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", record_or_reuse)


def _spec(tmp_path: Path, corpus_id: str = "frozen-v1") -> dict:
    return {
        "root": str(tmp_path / "corpora"),
        "organism": "test-organism",
        "corpus_id": corpus_id,
        "generator": {
            "world": "minecraft",
            "backend": "simulated",
            "episode_cache_dir": str(tmp_path / "episode-cache"),
            "data_quality_gate": True,
        },
        "splits": {
            "train": {"synthetic": [1]},
            "validation": {"synthetic": [2]},
            "test": {"synthetic": [3]},
        },
    }


def test_default_generator_targets_crafter():
    config = corpus_module._config_from_spec({"generator": {}})

    assert config.world == "crafter"
    assert config.backend == "crafter"


def test_build_freezes_stable_cached_content_and_disjoint_splits(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    spec = _spec(tmp_path)

    first = corpus_module.build_corpus(spec)
    second = corpus_module.build_corpus(spec)

    assert first.data_contract_hash == second.data_contract_hash
    assert first.manifest["sessions"] == second.manifest["sessions"]
    report = json.loads((first.directory / "split_overlap_report.json").read_text(encoding="utf-8"))
    assert report["disjoint"] is True
    assert set(report["session_ids"]) == {"train", "validation", "test"}
    assert (first.directory / "quality_report.json").is_file()


def test_rebuilding_a_completed_corpus_resolves_without_recording(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    spec = _spec(tmp_path)
    first = corpus_module.build_corpus(spec)

    def unexpected_record(*args, **kwargs):
        pytest.fail("a completed corpus must not record replacement episodes")

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", unexpected_record)
    second = corpus_module.build_corpus(spec)

    assert second.manifest == first.manifest


def test_quality_gate_uses_the_declared_pixel_provenance_policy(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    expected_sources = []

    def validate(_paths, _scenario, *, expected_pixel_source=None):
        expected_sources.append(expected_pixel_source)
        return []

    monkeypatch.setattr(corpus_module, "validate_nursery_recordings", validate)
    spec = _spec(tmp_path)
    spec["quality_policy"] = {"enabled": True, "expected_pixel_source": "viewer"}
    corpus = corpus_module.build_corpus(spec)

    assert expected_sources == ["viewer", "viewer", "viewer"]
    assert corpus.data_contract.pixel_provenance == "viewer"


def test_data_contract_and_quality_report_expose_wall_generator_evidence(tmp_path, monkeypatch):
    """Issue #236: a frozen corpus must expose the wall-layout distribution,
    declared mix bounds, and each episode's realised mix as contract evidence,
    not leave them implicit in the generator source."""
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    original_record = corpus_module._record_or_reuse_scenario_episode

    def record_with_wall_evidence(*args, **kwargs):
        session_dir = original_record(*args, **kwargs)
        path = Path(session_dir) / "session.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["program_config"] = {
            "motor_babbling": {
                "generator_name": "motor_babbling_walls",
                "layout_distribution": {"outer_wall": {"shape": "square_ring"}},
                "outcome_mix_bounds": {"blocked": {"min_fraction": 0.05, "max_fraction": 0.7}},
            },
        }
        metadata["quality_report"] = {
            "accepted": True,
            "episodes": {"episode_00000": {"fractions": {"moved": 0.6, "blocked": 0.2}}},
        }
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return session_dir

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", record_with_wall_evidence)
    corpus = corpus_module.build_corpus(_spec(tmp_path))

    contract_evidence = corpus.data_contract.program_config["scenario_generator_evidence"]
    assert set(contract_evidence) == {"synthetic-train-1", "synthetic-validation-2", "synthetic-test-3"}
    assert contract_evidence["synthetic-train-1"]["generator"]["layout_distribution"]
    quality = json.loads((corpus.directory / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["action_effect_evidence"]["synthetic-train-1"]["quality_report"]["accepted"] is True


def test_resolve_refuses_a_missing_frozen_session_and_names_it(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus = corpus_module.build_corpus(_spec(tmp_path))
    session = Path(corpus.manifest["sessions"]["train"][0]["session_path"])
    for child in session.iterdir():
        child.unlink()
    session.rmdir()

    with pytest.raises(FileNotFoundError, match=r"frozen-v1.*synthetic-train-1"):
        corpus_module.resolve_corpus("frozen-v1", root=tmp_path / "corpora", organism="test-organism")


def test_resolve_refuses_changed_content_without_recording_a_replacement(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus = corpus_module.build_corpus(_spec(tmp_path))
    session = Path(corpus.manifest["sessions"]["validation"][0]["session_path"])
    (session / "episode_00000.decisions.jsonl").write_text('{"edited": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"frozen-v1.*synthetic-validation-2.*content hash changed"):
        corpus_module.resolve_corpus("frozen-v1", root=tmp_path / "corpora", organism="test-organism")


def test_resolve_tolerates_a_clinic_prediction_export_written_into_a_frozen_session(tmp_path, monkeypatch):
    """A Model Factory trial's clinic (viewer/) prediction export writes
    ``<experiment_id>-predictions_<episode>.json`` straight into the
    session's own directory (issue: auto-export Model Factory predictions),
    including for a frozen corpus's validation sessions. Since multiple
    trials against the same corpus each add their own such file there, it
    must never perturb the corpus's content-integrity hash -- otherwise the
    very first trial to export predictions would break every later resolve
    of that corpus for every other trial."""
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus = corpus_module.build_corpus(_spec(tmp_path))
    session = Path(corpus.manifest["sessions"]["validation"][0]["session_path"])

    (session / "run-1-predictions_episode_00000.json").write_text("{}", encoding="utf-8")
    (session / "run-2-predictions_episode_00000.json").write_text("{}", encoding="utf-8")
    (session / "predictions_episode_00000.json").write_text("{}", encoding="utf-8")

    resolved = corpus_module.resolve_corpus("frozen-v1", root=tmp_path / "corpora", organism="test-organism")
    assert resolved.data_contract_hash == corpus.data_contract_hash


def test_resolve_requires_manifest_sessions_to_match_the_data_contract(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus = corpus_module.build_corpus(_spec(tmp_path))
    manifest_path = corpus.directory / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sessions"]["test"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"test.*sessions do not match its data contract"):
        corpus_module.resolve_corpus("frozen-v1", root=tmp_path / "corpora", organism="test-organism")


def test_reusing_a_corpus_id_with_changed_generator_is_an_error(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    spec = _spec(tmp_path)
    corpus_module.build_corpus(spec)
    changed = copy.deepcopy(spec)
    changed["generator"]["episode_ticks"] = 17

    with pytest.raises(ValueError, match=r"different generator declaration.*new corpus_id"):
        corpus_module.build_corpus(changed)


def test_data_contract_hash_tracks_declared_content_not_corpus_location(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    first = corpus_module.build_corpus(_spec(tmp_path, "same-content-a"))
    same = corpus_module.build_corpus(_spec(tmp_path, "same-content-b"))
    changed_spec = _spec(tmp_path, "changed-content")
    changed_spec["generator"]["episode_ticks"] = 17
    changed = corpus_module.build_corpus(changed_spec)

    assert first.data_contract_hash == same.data_contract_hash
    assert changed.data_contract_hash != first.data_contract_hash


def test_explicit_corpus_root_wins_over_an_in_process_duplicate_id(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    first_spec = _spec(tmp_path / "first-root")
    first = corpus_module.build_corpus(first_spec)
    second_spec = _spec(tmp_path / "second-root")
    second = corpus_module.build_corpus(second_spec)

    resolved = corpus_module.resolve_corpus(
        "frozen-v1", root=tmp_path / "first-root" / "corpora", organism="test-organism",
    )

    assert resolved.directory == first.directory
    assert resolved.directory != second.directory


def test_list_corpora_summarizes_every_frozen_corpus_under_an_organism(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus_module.build_corpus(_spec(tmp_path, "corpus-a"))
    corpus_module.build_corpus(_spec(tmp_path, "corpus-b"))

    listed = corpus_module.list_corpora(root=tmp_path / "corpora", organism="test-organism")

    assert {entry["corpus_id"] for entry in listed} == {"corpus-a", "corpus-b"}
    for entry in listed:
        assert entry["organism"] == "test-organism"
        assert entry["data_contract_hash"]
        assert entry["session_counts"] == {"train": 1, "validation": 1, "test": 1}


def test_list_corpora_scans_every_organism_when_none_is_given(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    root = tmp_path / "corpora"
    spec_a = _spec(tmp_path, "corpus-a")
    spec_a["organism"] = "organism-a"
    spec_b = _spec(tmp_path, "corpus-b")
    spec_b["organism"] = "organism-b"
    corpus_module.build_corpus(spec_a)
    corpus_module.build_corpus(spec_b)

    listed = corpus_module.list_corpora(root=root)

    assert {(entry["organism"], entry["corpus_id"]) for entry in listed} == {
        ("organism-a", "corpus-a"), ("organism-b", "corpus-b"),
    }


def test_list_corpora_skips_a_corpus_directory_without_a_manifest_yet(tmp_path, monkeypatch):
    _install_fake_nursery(monkeypatch, tmp_path / "episode-cache")
    corpus_module.build_corpus(_spec(tmp_path, "corpus-a"))
    (tmp_path / "corpora" / "test-organism" / "still-building").mkdir(parents=True)

    listed = corpus_module.list_corpora(root=tmp_path / "corpora", organism="test-organism")

    assert [entry["corpus_id"] for entry in listed] == ["corpus-a"]


def test_list_corpora_returns_empty_list_for_a_root_that_does_not_exist(tmp_path):
    assert corpus_module.list_corpora(root=tmp_path / "no-such-root") == []
    assert corpus_module.list_corpora(root=tmp_path / "no-such-root", organism="none") == []


def test_corpus_module_imports_without_torch():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockTorch(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'")
        return None

sys.meta_path.insert(0, BlockTorch())
import cognitive_runtime.training.model_factory.corpus  # noqa: F401
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
