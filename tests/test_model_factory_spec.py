"""Model Factory ExperimentSpec tests (issue #214).

Deliberately torch-free: these run in the ``factory-contracts`` CI tier
under two different ``PYTHONHASHSEED`` values (see .github/workflows/ci.yml),
same as tests/test_model_factory_contracts.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Any, Dict

import pytest

from cognitive_runtime.training.model_factory.contracts import TrainingContract
from cognitive_runtime.training.model_factory.spec import (
    DOCUMENT_FORMAT,
    ExperimentSpec,
    SpecError,
    apply_overrides,
    dump_canonical_json,
    load_spec,
    resolve,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC = REPO_ROOT / "specs" / "crafter-baseline.yaml"


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


def _leaf_diff(a: Dict[str, Any], b: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively collect ``{dotted_path: (a_value, b_value)}`` for leaves that differ."""
    diffs: Dict[str, Any] = {}
    keys = set(a) | set(b)
    for key in keys:
        path = f"{prefix}.{key}" if prefix else key
        av, bv = a.get(key), b.get(key)
        if isinstance(av, ABCMapping) and isinstance(bv, ABCMapping):
            diffs.update(_leaf_diff(dict(av), dict(bv), path))
        elif av != bv:
            diffs[path] = (av, bv)
    return diffs


# --------------------------------------------------------------------- resolve() basics


def test_resolve_materializes_every_default():
    spec = resolve(_minimal_raw())
    assert spec.training["batch_size"] == 32
    assert spec.training["device"] == "cpu"
    assert spec.model["backbone"] == "transformer"
    assert spec.evaluation["confidence"] == 0.95
    assert spec.data["expected_pixel_source"] == "crafter"


def test_resolve_returns_frozen_spec():
    spec = resolve(_minimal_raw())
    with pytest.raises(Exception):
        spec.organism = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.training["batch_size"] = 999  # type: ignore[index]


def test_example_template_resolves_successfully():
    spec = resolve(load_spec(EXAMPLE_SPEC))
    assert spec.organism == "Crafter"
    assert spec.mode == "fresh"


# --------------------------------------------------------------------- round trip


def test_yaml_to_resolved_to_json_to_reload_has_identical_training_contract_hash(tmp_path):
    raw = load_spec(EXAMPLE_SPEC)
    resolved = resolve(raw)
    original_hash = resolved.training_contract_hash

    json_path = tmp_path / "resolved.json"
    json_path.write_bytes(dump_canonical_json(resolved))

    reloaded_raw = load_spec(json_path)
    reloaded = resolve(reloaded_raw)

    assert reloaded.training_contract_hash == original_hash
    assert reloaded.training_contract_hash == TrainingContract(**dict(resolved.training)).hash


def test_two_specs_differing_only_in_key_order_resolve_to_byte_identical_json():
    raw_a = _minimal_raw()
    # Same content, different key/insertion order and nested block order.
    raw_b = {
        "evaluation": dict(raw_a["evaluation"]),
        "training": {"optimizer": dict(raw_a["training"]["optimizer"])},
        "model": dict(raw_a["model"]),
        "data": {"world": raw_a["data"]["world"], "corpus_id": raw_a["data"]["corpus_id"]},
        "mode": raw_a["mode"],
        "organism": raw_a["organism"],
        "format": raw_a["format"],
    }
    assert dump_canonical_json(resolve(raw_a)) == dump_canonical_json(resolve(raw_b))


# --------------------------------------------------------------------- apply_overrides


def test_set_override_changes_exactly_one_resolved_field():
    base_raw = _minimal_raw()
    base = resolve(base_raw)

    overridden_raw = apply_overrides(base_raw, ["training.optimizer.lr=0.0003"])
    # lr is already 0.0003 in the fixture; pick a value that's guaranteed to differ.
    overridden_raw = apply_overrides(base_raw, ["training.optimizer.lr=0.00075"])
    overridden = resolve(overridden_raw)

    diffs = _leaf_diff(base.to_dict(), overridden.to_dict())
    assert diffs == {"training.optimizer.lr": (0.0003, 0.00075)}


def test_apply_overrides_coerces_int_float_and_bool():
    raw = _minimal_raw()
    raw = apply_overrides(raw, ["training.batch_size=64"])
    raw = apply_overrides(raw, ["training.scheduled_sampling_p=0.5"])
    raw = apply_overrides(raw, ["training.determinism_policy.deterministic=false"])
    resolved = resolve(raw)
    assert resolved.training["batch_size"] == 64
    assert isinstance(resolved.training["batch_size"], int)
    assert resolved.training["scheduled_sampling_p"] == 0.5
    assert resolved.training["determinism_policy"]["deterministic"] is False


def test_apply_overrides_never_uses_eval():
    raw = _minimal_raw()
    raw = apply_overrides(raw, ["organism=__import__('os').system('true')"])
    # It must be treated as an opaque string, not executed.
    assert raw["organism"] == "__import__('os').system('true')"


def test_apply_overrides_rejects_malformed_override():
    with pytest.raises(SpecError):
        apply_overrides(_minimal_raw(), ["training.batch_size"])


def test_apply_overrides_keeps_string_fields_as_strings():
    raw = apply_overrides(_minimal_raw(), ["data.corpus_id=2026"])
    assert raw["data"]["corpus_id"] == "2026"
    assert isinstance(raw["data"]["corpus_id"], str)

    raw = apply_overrides(_minimal_raw(), ["organism=true"])
    assert raw["organism"] == "true"
    assert isinstance(raw["organism"], str)


def test_apply_overrides_rejects_traversal_through_an_existing_scalar():
    raw = _minimal_raw()
    raw["training"]["optimizer"]["lr"] = 0.0003
    with pytest.raises(SpecError, match="lr"):
        apply_overrides(raw, ["training.optimizer.lr.extra=7"])


# --------------------------------------------------------------------- mode / parent validation


def test_fresh_mode_rejects_parent():
    raw = _minimal_raw()
    raw["parent"] = {"run_id": "x", "checkpoint": "last.pt", "sha256": "abc"}
    with pytest.raises(SpecError, match="fresh"):
        resolve(raw)


def test_clone_without_sha256_is_rejected_naming_the_missing_field():
    raw = _minimal_raw()
    raw["mode"] = "clone"
    raw["parent"] = {"run_id": "crafter-rollout-0003", "checkpoint": "best-validation.pt"}
    with pytest.raises(SpecError, match="parent.sha256"):
        resolve(raw)


def test_clone_without_any_parent_is_rejected():
    raw = _minimal_raw()
    raw["mode"] = "clone"
    with pytest.raises(SpecError, match="parent"):
        resolve(raw)


def test_resume_with_full_parent_succeeds():
    raw = _minimal_raw()
    raw["mode"] = "resume"
    raw["parent"] = {
        "run_id": "crafter-rollout-0003",
        "checkpoint": "last.pt",
        "sha256": "deadbeef",
    }
    spec = resolve(raw)
    assert spec.parent["sha256"] == "deadbeef"


def test_invalid_mode_is_rejected():
    raw = _minimal_raw()
    raw["mode"] = "sideways"
    with pytest.raises(SpecError, match="sideways"):
        resolve(raw)


# --------------------------------------------------------------------- run_id / display_name


def test_run_id_in_template_is_rejected():
    raw = _minimal_raw()
    raw["run_id"] = "crafter-rollout-0099"
    with pytest.raises(SpecError, match="run_id"):
        resolve(raw)


def test_display_name_in_template_is_rejected():
    raw = _minimal_raw()
    raw["display_name"] = "vigorous-shannon"
    with pytest.raises(SpecError, match="display_name"):
        resolve(raw)


# --------------------------------------------------------------------- unknown keys


def test_unknown_top_level_key_names_key_and_suggests_nearest_field():
    raw = _minimal_raw()
    raw["oragnism"] = raw.pop("organism")
    with pytest.raises(SpecError) as excinfo:
        resolve(raw)
    assert "oragnism" in str(excinfo.value)
    assert "organism" in str(excinfo.value)


def test_unknown_nested_training_key_is_rejected():
    raw = _minimal_raw()
    raw["training"]["oprimizer"] = raw["training"].pop("optimizer")
    with pytest.raises(SpecError) as excinfo:
        resolve(raw)
    assert "oprimizer" in str(excinfo.value)
    assert "optimizer" in str(excinfo.value)


# --------------------------------------------------------------------- warmup_frames


def test_zero_warmup_frames_is_rejected():
    # train_action_world_model() rejects warmup_frames < 1 unconditionally, so
    # a spec that resolves with 0 would pass here and then fail at training.
    raw = _minimal_raw()
    raw["training"]["warmup_frames"] = 0
    with pytest.raises(SpecError, match="warmup_frames"):
        resolve(raw)


# --------------------------------------------------------------------- evaluation.selection_metric


@pytest.mark.parametrize(
    "metric",
    ["rollout.t+4.model_over_copy_last_mse", "validation_loss", "goal_navigation.success_rate"],
)
def test_valid_selection_metric_paths_are_accepted(metric):
    raw = _minimal_raw()
    raw["evaluation"]["selection_metric"] = metric
    resolve(raw)  # must not raise


@pytest.mark.parametrize("metric", ["", "###", "rollout..mse", ".leading_dot", "trailing."])
def test_invalid_selection_metric_paths_are_rejected(metric):
    raw = _minimal_raw()
    raw["evaluation"]["selection_metric"] = metric
    with pytest.raises(SpecError, match="selection_metric"):
        resolve(raw)


# --------------------------------------------------------------------- warmup/rollout vs corpus


def test_warmup_plus_rollout_exceeding_min_episode_length_is_rejected():
    raw = _minimal_raw()
    raw["data"]["quality_policy"] = {"min_episode_length": 10}
    raw["training"]["warmup_frames"] = 4
    raw["training"]["rollout_frames"] = 8  # 4 + 8 = 12 > 10
    with pytest.raises(SpecError, match="min_episode_length"):
        resolve(raw)


def test_warmup_plus_rollout_within_min_episode_length_is_accepted():
    raw = _minimal_raw()
    raw["data"]["quality_policy"] = {"min_episode_length": 20}
    raw["training"]["warmup_frames"] = 4
    raw["training"]["rollout_frames"] = 8
    resolve(raw)  # must not raise


# --------------------------------------------------------------------- evolution block


def test_evolution_requires_exactly_two_configuration_parents():
    raw = _minimal_raw()
    raw["evolution"] = {
        "configuration_parents": ["crafter-rollout-0018"],
        "weight_donor": "crafter-rollout-0018",
    }
    with pytest.raises(SpecError, match="configuration_parents"):
        resolve(raw)


def test_evolution_requires_weight_donor():
    raw = _minimal_raw()
    raw["evolution"] = {
        "configuration_parents": ["crafter-rollout-0018", "crafter-rollout-0021"],
    }
    with pytest.raises(SpecError, match="weight_donor"):
        resolve(raw)


def test_valid_evolution_block_is_accepted():
    raw = _minimal_raw()
    raw["evolution"] = {
        "generation": 3,
        "configuration_parents": ["crafter-rollout-0018", "crafter-rollout-0021"],
        "weight_donor": "crafter-rollout-0021",
        "genome_operator": "uniform_crossover_v1",
        "mutations": ["training.optimizer.lr"],
    }
    spec = resolve(raw)
    assert spec.evolution["weight_donor"] == "crafter-rollout-0021"


def test_configuration_parents_as_a_two_character_string_is_rejected():
    # len("ab") == 2, so a naive `len(parents) == 2` check would accept this
    # even though it names zero actual parent run IDs.
    raw = _minimal_raw()
    raw["evolution"] = {"configuration_parents": "ab", "weight_donor": "crafter-rollout-0021"}
    with pytest.raises(SpecError, match="configuration_parents"):
        resolve(raw)


def test_configuration_parents_as_a_two_key_mapping_is_rejected():
    raw = _minimal_raw()
    raw["evolution"] = {
        "configuration_parents": {"a": "crafter-rollout-0018", "b": "crafter-rollout-0021"},
        "weight_donor": "crafter-rollout-0021",
    }
    with pytest.raises(SpecError, match="configuration_parents"):
        resolve(raw)


def test_selection_metric_horizon_must_be_declared_by_trial_data():
    raw = _minimal_raw()
    raw["data"]["horizons_ticks"] = [1, 2]
    raw["evaluation"]["selection_metric"] = "rollout.t+4.model_over_copy_last_mse"

    with pytest.raises(SpecError, match=r"requires horizon tick.*4.*horizons_ticks.*1, 2"):
        resolve(raw)


# --------------------------------------------------------------------- training_contract_hash


def test_training_contract_hash_matches_a_manually_built_contract():
    spec = resolve(_minimal_raw())
    manual = TrainingContract(**dict(spec.training))
    assert spec.training_contract_hash == manual.hash


def test_changing_only_a_loss_weight_changes_training_contract_hash():
    base = resolve(_minimal_raw())
    changed_raw = apply_overrides(_minimal_raw(), ["training.loss_weights.pixel=2.0"])
    changed = resolve(changed_raw)
    assert base.training_contract_hash != changed.training_contract_hash


# --------------------------------------------------------------------- load_spec


def test_load_spec_rejects_unsupported_extension(tmp_path):
    bogus = tmp_path / "spec.txt"
    bogus.write_text("format: model-factory-spec-v1")
    with pytest.raises(SpecError):
        load_spec(bogus)


def test_load_spec_json_matches_load_spec_yaml(tmp_path):
    raw = load_spec(EXAMPLE_SPEC)
    json_path = tmp_path / "spec.json"
    json_path.write_text(json.dumps(raw))
    assert load_spec(json_path) == raw


# --------------------------------------------------------------------- torch-free import


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.spec  # noqa: F401
print("OK")
"""


def test_spec_module_imports_cleanly_without_torch():
    import os

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


_HASH_STABILITY_SCRIPT = """
import sys
sys.path.insert(0, {path!r})
from cognitive_runtime.training.model_factory.spec import load_spec, resolve

spec = resolve(load_spec({example_path!r}))
print(spec.training_contract_hash)
"""


def _hash_under_pythonhashseed(seed: str) -> str:
    import os

    env = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _HASH_STABILITY_SCRIPT.format(path=str(REPO_ROOT), example_path=str(EXAMPLE_SPEC)),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_training_contract_hash_is_identical_across_different_pythonhashseed_subprocesses():
    hash_seed_0 = _hash_under_pythonhashseed("0")
    hash_seed_other = _hash_under_pythonhashseed("12345")
    assert hash_seed_0 == hash_seed_other
    assert len(hash_seed_0) == 64
