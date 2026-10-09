"""Offline golden compatibility gate for the legacy Model Factory (#212/#297)."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    TrainingContract,
    canonical_json,
)
from cognitive_runtime.training.model_factory.state import load_state

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / ".github" / "schemas"
FIXTURES = ROOT / "tests" / "fixtures" / "factory"
EXPECTED_SCHEMAS = {"contracts", "lineage", "state"}
EXPECTED_FIXTURES = {
    "contracts.json", "lineage.fresh.json", "lineage.clone.json",
    "state.queued.json", "state.completed.json",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator_module():
    return _load_module("factory_schema_validator", ROOT / ".github/scripts/validate_artifact_schemas.py")


@pytest.fixture(scope="module")
def generator():
    return _load_module("factory_golden_generator", FIXTURES / "metadata/generate.py")


@pytest.fixture
def layout(tmp_path):
    schemas, fixtures = tmp_path / "schemas", tmp_path / "fixtures"
    shutil.copytree(SCHEMAS, schemas)
    shutil.copytree(FIXTURES, fixtures)
    return schemas, fixtures


def test_schema_cli_validates_all_checked_in_goldens():
    result = subprocess.run(
        [sys.executable, str(ROOT / ".github/scripts/validate_artifact_schemas.py")],
        cwd=ROOT, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 fixture(s) validated against 3 schema(s)" in result.stdout


def test_fixture_inventory_and_synthetic_rights_hashes(generator):
    manifest = json.loads((FIXTURES / "metadata/manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "ccr-synthetic-factory-fixtures-v1"
    assert manifest["status"] == "synthetic_only"
    assert manifest["seed"] == 297
    assert manifest["license"] == "MIT"
    assert "Project-authored" in manifest["source"]
    assert "no third-party payloads" in manifest["rights"]
    assert manifest["config_sha256"] == generator.sha256(FIXTURES / manifest["config"])
    assert manifest["generator_sha256"] == generator.sha256(FIXTURES / manifest["generator"])
    assert {entry["path"] for entry in manifest["artifacts"]} == EXPECTED_FIXTURES
    assert len(manifest["artifacts"]) == len(EXPECTED_FIXTURES)
    assert {path.name for path in FIXTURES.glob("*.json")} == EXPECTED_FIXTURES
    assert {path.name.removesuffix(".schema.json") for path in SCHEMAS.glob("*.schema.json")} == EXPECTED_SCHEMAS
    for entry in manifest["artifacts"]:
        assert entry["intended_edge_case"]
        assert entry["sha256"] == generator.sha256(FIXTURES / entry["path"])
        expected_schema = f".github/schemas/{entry['path'].split('.')[0]}.schema.json"
        assert entry["schema"] == expected_schema
        assert entry["schema_sha256"] == generator.sha256(ROOT / entry["schema"])


def test_production_serializers_reproduce_golden_bytes(generator, tmp_path):
    """Re-resolve independent inputs and use actual allocation/state APIs.

    This catches serialization drift that still passes a permissive loader,
    rather than merely comparing json.loads(json.dumps(the_same_dictionary)).
    """
    generator.generate_artifacts(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FIXTURES
    for name in EXPECTED_FIXTURES:
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name


@pytest.mark.parametrize(
    ("key", "hash_key", "contract_type"),
    [
        ("architecture_contract", "architecture_hash", ArchitectureContract),
        ("data_contract", "data_contract_hash", DataContract),
        ("training_contract", "training_contract_hash", TrainingContract),
    ],
)
def test_golden_contracts_rehydrate_with_identical_identity(key, hash_key, contract_type, tmp_path):
    golden = json.loads((FIXTURES / "contracts.json").read_text(encoding="utf-8"))
    contract = contract_type(**golden[key])
    assert contract.hash == golden[hash_key]
    assert canonical_json(contract.to_dict()) == canonical_json(golden[key])
    output = tmp_path / "contract.json"
    atomic_write_json(output, contract.to_dict())
    loaded_again = contract_type(**json.loads(output.read_text(encoding="utf-8")))
    assert loaded_again.hash == golden[hash_key]


@pytest.mark.parametrize("name", ["state.queued.json", "state.completed.json"])
def test_golden_state_round_trips_through_public_loader_and_serializer(name, tmp_path):
    golden_path = FIXTURES / name
    state = load_state(golden_path)
    assert state.is_terminal == (name == "state.completed.json")
    output = tmp_path / "state.json"
    atomic_write_json(output, state.to_dict())
    assert output.read_bytes() == golden_path.read_bytes()
    assert load_state(output) == state


@pytest.mark.parametrize("missing", ["both", "schemas", "fixtures", "pair", "fixture"])
def test_validator_never_skips_missing_inputs(validator_module, layout, missing, capsys):
    schemas, fixtures = layout
    if missing in {"both", "schemas"}:
        shutil.rmtree(schemas)
    if missing in {"both", "fixtures"}:
        shutil.rmtree(fixtures)
    if missing == "pair":
        (schemas / "contracts.schema.json").unlink()
        (fixtures / "contracts.json").unlink()
    if missing == "fixture":
        (fixtures / "contracts.json").unlink()
    assert validator_module.main(schemas, fixtures) == 1
    assert "schema failure(s)" in capsys.readouterr().err


def test_fixture_without_schema_is_an_error(validator_module, layout, capsys):
    schemas, fixtures = layout
    (fixtures / "unknown.json").write_text('{"format": "new-v1"}', encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert "unknown.json: no schema" in capsys.readouterr().err


@pytest.mark.parametrize("replacement", ['{', '{}', 'true', '{"type":"no-such-type"}'])
def test_malformed_or_vacuous_schema_fails_cleanly(validator_module, layout, replacement, capsys):
    schemas, fixtures = layout
    (schemas / "contracts.schema.json").write_text(replacement, encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert "contracts.schema.json: invalid schema" in capsys.readouterr().err


@pytest.mark.parametrize("reference", ["https://example.invalid/remote.json", "#/missing"])
def test_external_or_unresolved_references_fail_without_fetching(validator_module, layout, reference, capsys):
    schemas, fixtures = layout
    path = schemas / "contracts.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    schema["properties"]["architecture_contract"] = {"$ref": reference}
    path.write_text(json.dumps(schema), encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert "invalid schema" in capsys.readouterr().err


@pytest.mark.parametrize("replacement", ['{', '{"format":NaN}', '{"format":Infinity}', '{"format":1e999}', '{"format":"a","format":"b"}'])
def test_malformed_data_fails_cleanly(validator_module, layout, replacement, capsys):
    schemas, fixtures = layout
    (fixtures / "contracts.json").write_text(replacement, encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert "contracts.json: invalid artifact" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("fixture", "path", "replacement"),
    [
        ("contracts.json", ["architecture_hash"], "not-a-sha256"),
        ("contracts.json", ["architecture_contract", "hidden_dim"], 0),
        ("contracts.json", ["architecture_contract", "unexpected"], 1),
        ("contracts.json", ["data_contract", "ticks_per_frame"], "1.0"),
        ("contracts.json", ["training_contract", "optimizer", "lr"], "fast"),
        ("lineage.clone.json", ["parent"], None),
        ("lineage.fresh.json", ["parent_checkpoint_sha"], "a" * 64),
        ("state.queued.json", ["state"], "invented-state"),
        ("state.completed.json", ["history"], []),
        ("state.completed.json", ["updated_at"], "yesterday"),
        ("state.completed.json", ["reason"], 123),
    ],
)
def test_schema_rejects_semantic_shape_drift(fixture, path, replacement):
    schema = json.loads((SCHEMAS / f"{fixture.split('.')[0]}.schema.json").read_text(encoding="utf-8"))
    payload = copy.deepcopy(json.loads((FIXTURES / fixture).read_text(encoding="utf-8")))
    node = payload
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = replacement
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)


def test_required_contract_field_cannot_disappear():
    payload = json.loads((FIXTURES / "contracts.json").read_text(encoding="utf-8"))
    del payload["data_contract"]["train_session_hashes"]
    schema = json.loads((SCHEMAS / "contracts.schema.json").read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError, match="train_session_hashes"):
        jsonschema.Draft202012Validator(schema).validate(payload)


def test_datetime_validation_needs_no_optional_format_package(validator_module, layout, capsys):
    schemas, fixtures = layout
    path = fixtures / "state.completed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["updated_at"] = "2026-02-30T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert "date-time" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_manifest", "cannot read provenance manifest"),
        ("malformed_manifest", "cannot read provenance manifest"),
        ("missing_source", "missing nonempty source provenance"),
        ("wrong_seed", "seed does not match"),
        ("wrong_status", "status must equal 'synthetic_only'"),
        ("missing_inventory", "missing artifact inventory entry"),
        ("duplicate_inventory", "duplicate artifact inventory entry"),
        ("missing_config", "metadata/inputs.json: cannot read hash input"),
        ("changed_config", "metadata/inputs.json: SHA-256 mismatch"),
        ("changed_generator", "metadata/generate.py: SHA-256 mismatch"),
        ("changed_schema", ".github/schemas/contracts.schema.json: SHA-256 mismatch"),
        ("changed_artifact", "contracts.json: SHA-256 mismatch"),
        ("wrong_config_hash", "metadata/inputs.json: SHA-256 mismatch"),
        ("wrong_schema_hash", ".github/schemas/contracts.schema.json: SHA-256 mismatch"),
        ("wrong_artifact_hash", "contracts.json: SHA-256 mismatch"),
    ],
)
def test_standalone_validator_enforces_provenance_integrity(validator_module, layout, mutation, expected, capsys):
    schemas, fixtures = layout
    path = fixtures / "metadata/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_manifest":
        path.unlink()
    elif mutation == "malformed_manifest":
        path.write_text("{", encoding="utf-8")
    elif mutation == "missing_config":
        (fixtures / "metadata/inputs.json").unlink()
    elif mutation.startswith("changed_"):
        target = {
            "changed_config": fixtures / "metadata/inputs.json",
            "changed_generator": fixtures / "metadata/generate.py",
            "changed_schema": schemas / "contracts.schema.json",
            "changed_artifact": fixtures / "contracts.json",
        }[mutation]
        # Legal whitespace is enough: integrity protects the reviewed bytes,
        # independently of whether the altered artifact still passes its schema.
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        if mutation == "missing_source":
            del manifest["source"]
        elif mutation == "wrong_seed":
            manifest["seed"] += 1
        elif mutation == "wrong_status":
            manifest["status"] = "validated_live"
        elif mutation == "missing_inventory":
            manifest["artifacts"].pop()
        elif mutation == "duplicate_inventory":
            manifest["artifacts"].append(manifest["artifacts"][0])
        elif mutation == "wrong_config_hash":
            manifest["config_sha256"] = "0" * 64
        elif mutation == "wrong_schema_hash":
            manifest["artifacts"][0]["schema_sha256"] = "0" * 64
        elif mutation == "wrong_artifact_hash":
            manifest["artifacts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01 00:00:00+00:00",  # fromisoformat accepts a space separator
        "20260101T000000+0000",       # and compact dates/times/offsets
        "2026-01-01T00:00+00:00",     # seconds are mandatory in RFC3339
        "2026-01-01T00:00:00+00:00:01",
        "2026-01-01T00:00:00+00:60",
        "2026-01-01T00:60:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01T00:00:00",
        "2026-02-30T00:00:00Z",
        "2026-01-01T00:00:00Z\n",
    ],
)
def test_datetime_checker_rejects_non_rfc3339_iso_variants(validator_module, value):
    assert not validator_module._format_checker().conforms(value, "date-time")


@pytest.mark.parametrize("value", ["2026-01-01T00:00:00Z", "2026-01-01t00:00:00z", "2026-01-01T00:00:00.125+02:30"])
def test_datetime_checker_accepts_rfc3339(validator_module, value):
    assert validator_module._format_checker().conforms(value, "date-time")


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("path", "/tmp/contracts.json", "not a checked-in top-level JSON fixture"),
        ("path", "../contracts.json", "not a checked-in top-level JSON fixture"),
        ("schema", "https://example.invalid/schema.json", "schema must equal"),
        ("config", "../inputs.json", "config must equal"),
        ("generator", "/tmp/generate.py", "generator must equal"),
    ],
)
def test_provenance_paths_cannot_escape_checked_in_inventory(validator_module, layout, field, value, expected, capsys):
    schemas, fixtures = layout
    path = fixtures / "metadata/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target = manifest if field in {"config", "generator"} else manifest["artifacts"][0]
    target[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator_module.main(schemas, fixtures) == 1
    assert expected in capsys.readouterr().err
