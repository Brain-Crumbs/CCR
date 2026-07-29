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
