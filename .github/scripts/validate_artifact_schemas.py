#!/usr/bin/env python3
"""Validate the checked-in, synthetic Model Factory golden artifacts offline.

Every ``.github/schemas/<name>.schema.json`` needs a matching
``tests/fixtures/factory/<name>[.<variant>].json`` and vice versa. The baseline
inventory is mandatory even when both directories (or one artifact pair) are
removed. Provenance/configuration files live in the fixtures' metadata directory,
not among the artifact instances. See issues #212, #242 and #297.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys

try:
    import jsonschema
    from referencing import Registry
    from referencing.exceptions import Unresolvable
except ImportError:  # pragma: no cover - provided by the dev extra
    sys.exit("jsonschema is required: install the project's dev extra")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / ".github" / "schemas"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "factory"
REQUIRED_ARTIFACTS = frozenset({"contracts", "lineage", "state"})
DRAFT = "https://json-schema.org/draft/2020-12/schema"
RFC3339 = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])[Tt]"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$"
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _load(path: pathlib.Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=_finite_float,
        )


def _check_contract_schema(schema: object) -> None:
    """Do not let a syntactically valid but vacuous schema disable this gate."""
    jsonschema.Draft202012Validator.check_schema(schema)
    if not isinstance(schema, dict) or schema.get("$schema") != DRAFT:
        raise ValueError(f"schema must explicitly declare {DRAFT}")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    version = properties.get("format", {})
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not properties
        or not required
        or not set(required).issubset(properties)
        or "format" not in required
        or not isinstance(version, dict)
        or not isinstance(version.get("const"), str)
        or not version["const"]
    ):
        raise ValueError("schema must be a closed object with required fields and a constant format")
    _check_references(schema, schema)


def _check_references(value: object, root: dict) -> None:
    if isinstance(value, list):
        for child in value:
            _check_references(child, root)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"}:
                # These fixtures are self-contained: never fetch a schema or
                # model over the network, even if a contributor adds a URL.
                if not isinstance(child, str) or not child.startswith("#/"):
                    raise ValueError("schema references must be local JSON pointers (#/...)")
                target = root
                try:
                    for part in child[2:].split("/"):
                        target = target[part.replace("~1", "/").replace("~0", "~")]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"unresolved local schema reference: {child}") from exc
            else:
                _check_references(child, root)


def _format_checker() -> jsonschema.FormatChecker:
    # jsonschema's date-time checker otherwise needs an optional RFC3339
    # package. Keep base installs deterministic with the standard library.
    checker = jsonschema.FormatChecker()

    @checker.checks("date-time", raises=(ValueError, TypeError))
    def is_datetime(value: object) -> bool:
        if not isinstance(value, str):
            return True  # the schema's type constraint handles non-strings
        if RFC3339.fullmatch(value) is None:
            return False
        normalized = value.upper().replace("Z", "+00:00")
        return dt.datetime.fromisoformat(normalized).utcoffset() is not None

    return checker


def _validate_metadata(schema_dir: pathlib.Path, fixture_dir: pathlib.Path) -> list[str]:
    """Verify provenance and byte identities without importing Factory code."""
    manifest_path = fixture_dir / "metadata" / "manifest.json"
    try:
        manifest = _load(manifest_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"metadata/manifest.json: cannot read provenance manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["metadata/manifest.json: provenance manifest must be an object"]

    failures = []
    expected = {
        "format": "ccr-synthetic-factory-fixtures-v1",
        "status": "synthetic_only",
        "license": "MIT",
        "config": "metadata/inputs.json",
        "generator": "metadata/generate.py",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(f"metadata/manifest.json: {key} must equal {value!r}")
    for key in ("source", "rights"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            failures.append(f"metadata/manifest.json: missing nonempty {key} provenance")
    if type(manifest.get("seed")) is not int:
        failures.append("metadata/manifest.json: seed must be an integer")

    def check_hash(path: pathlib.Path, declared: object, label: str) -> None:
        if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
            failures.append(f"metadata/manifest.json: missing or invalid SHA-256 for {label}")
            return
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"{label}: cannot read hash input: {exc}")
            return
        if declared != actual:
            failures.append(
                f"{label}: SHA-256 mismatch (manifest {declared}; actual {actual}); "
                "review the change before regenerating the fixture manifest"
            )

    # Resolve only fixed local paths, never paths supplied by fixture metadata.
    for key in ("config", "generator"):
        relative = expected[key]
        check_hash(fixture_dir / relative, manifest.get(f"{key}_sha256"), relative)
    try:
        inputs = _load(fixture_dir / expected["config"])
        if not isinstance(inputs, dict) or inputs.get("seed") != manifest.get("seed"):
            failures.append("metadata/inputs.json: seed does not match the provenance manifest")
    except (OSError, UnicodeError, ValueError) as exc:
        failures.append(f"metadata/inputs.json: cannot read fixture configuration: {exc}")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        failures.append("metadata/manifest.json: artifacts must be a nonempty inventory")
        return failures
    inventory = set()
    actual_files = {path.name for path in fixture_dir.glob("*.json")}
    for index, entry in enumerate(entries):
        label = f"metadata/manifest.json: artifacts[{index}]"
        if not isinstance(entry, dict):
            failures.append(f"{label} must be an object")
            continue
        name = entry.get("path")
        if not isinstance(name, str) or name not in actual_files:
            failures.append(f"{label}: artifact path {name!r} is not a checked-in top-level JSON fixture")
            continue
        if name in inventory:
            failures.append(f"{label}: duplicate artifact inventory entry {name}")
        inventory.add(name)
        if not isinstance(entry.get("intended_edge_case"), str) or not entry["intended_edge_case"].strip():
            failures.append(f"{label}: missing intended_edge_case for {name}")
        schema_name = f"{name.split('.')[0]}.schema.json"
        schema_relative = f".github/schemas/{schema_name}"
        if entry.get("schema") != schema_relative:
            failures.append(f"{label}: schema must equal {schema_relative!r}")
        check_hash(fixture_dir / name, entry.get("sha256"), name)
        check_hash(schema_dir / schema_name, entry.get("schema_sha256"), schema_relative)
    for name in sorted(actual_files - inventory):
        failures.append(f"metadata/manifest.json: missing artifact inventory entry for {name}")
    return failures


def main(
    schema_dir: pathlib.Path | None = None,
    fixture_dir: pathlib.Path | None = None,
) -> int:
    schema_dir = schema_dir if schema_dir is not None else SCHEMA_DIR
    fixture_dir = fixture_dir if fixture_dir is not None else FIXTURE_DIR
    schemas = {p.name.removesuffix(".schema.json"): p for p in schema_dir.glob("*.schema.json")}
    fixtures = sorted(fixture_dir.glob("*.json"))
    failures = _validate_metadata(schema_dir, fixture_dir)
    validated = 0

    if not schemas:
        failures.append(f"no schemas found in {schema_dir}")
    for name in sorted(REQUIRED_ARTIFACTS - schemas.keys()):
        failures.append(f"missing required artifact schema: {name}.schema.json")

    for name, schema_path in sorted(schemas.items()):
        matched = [p for p in fixtures if p.name == f"{name}.json" or p.name.startswith(f"{name}.")]
        if not matched:
            failures.append(f"{name}: schema has no golden fixture in {fixture_dir}")
        try:
            schema = _load(schema_path)
            _check_contract_schema(schema)
            # An empty registry has no remote retrieval callback. Format checks
            # also reject invalid ISO timestamps instead of treating them as annotations.
            validator = jsonschema.Draft202012Validator(
                schema, registry=Registry(), format_checker=_format_checker(),
            )
        except (OSError, UnicodeError, ValueError, jsonschema.SchemaError) as exc:
            failures.append(f"{schema_path.name}: invalid schema: {exc}")
            continue

        for fixture in matched:
            try:
                validator.validate(_load(fixture))
            except jsonschema.ValidationError as exc:
                location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
                failures.append(f"{fixture.name}: at {location}: {exc.message}")
            except (OSError, UnicodeError, ValueError, Unresolvable) as exc:
                failures.append(f"{fixture.name}: invalid artifact: {exc}")
            else:
                validated += 1
                print(f"ok  {fixture.name}  <-  {schema_path.name}")

    for fixture in fixtures:
        stem = fixture.name.removesuffix(".json").split(".")[0]
        if stem not in schemas:
            failures.append(f"{fixture.name}: no schema {stem}.schema.json declared")

    if failures:
        print(f"\n{len(failures)} schema failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\n{validated} fixture(s) validated against {len(schemas)} schema(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
