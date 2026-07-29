#!/usr/bin/env python3
"""Validate Model Factory artifact fixtures against their JSON Schemas.

For a provenance system, silent schema drift is data loss: an older run stops
being readable and nobody notices until they need it.  Committing a schema per
artifact plus a golden instance of each makes drift a red check on the PR that
caused it, instead of an archaeology problem six months later.

Layout:

    .github/schemas/<name>.schema.json      the schema
    tests/fixtures/factory/<name>.json      one or more golden instances

A schema with no fixture is an error -- an unexercised schema proves nothing.
A fixture with no schema is an error -- it means an artifact was added without
declaring its contract.

Run from the repository root.  See issue #242.
"""

from __future__ import annotations

import json
import pathlib
import sys

try:
    import jsonschema
except ImportError:  # pragma: no cover - CI installs it explicitly
    sys.exit("jsonschema is required: pip install jsonschema")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / ".github" / "schemas"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "factory"


def _load(path: pathlib.Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    schemas = {p.name.removesuffix(".schema.json"): p for p in SCHEMA_DIR.glob("*.schema.json")}
    if not schemas:
        print(f"no schemas found in {SCHEMA_DIR}")
        return 1

    failures: list[str] = []
    validated = 0

    for name, schema_path in sorted(schemas.items()):
        schema = _load(schema_path)
        # `<name>.json` plus any `<name>.<variant>.json` -- one artifact often
        # needs several golden instances (a fresh run and a bred child do not
        # carry the same lineage shape).
        fixtures = sorted(FIXTURE_DIR.glob(f"{name}.json")) + sorted(
            FIXTURE_DIR.glob(f"{name}.*.json")
        )
        if not fixtures:
            failures.append(f"{name}: schema has no golden fixture in {FIXTURE_DIR}")
            continue

        for fixture in fixtures:
            try:
                jsonschema.validate(instance=_load(fixture), schema=schema)
            except jsonschema.ValidationError as exc:
                location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
                failures.append(f"{fixture.name}: at {location}: {exc.message}")
            else:
                validated += 1
                print(f"ok  {fixture.relative_to(ROOT)}  <-  {schema_path.name}")

    # An artifact added without a declared contract is exactly the drift this
    # script exists to catch, so flag orphan fixtures too.
    for fixture in sorted(FIXTURE_DIR.glob("*.json")):
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
