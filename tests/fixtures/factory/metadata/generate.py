#!/usr/bin/env python3
"""Regenerate original synthetic Factory goldens; never train or fetch data.

Run from the repository root with the core + dev environment:
    python tests/fixtures/factory/metadata/generate.py

Only the reviewed inputs and ordinary production serializers determine artifact
bytes. Clocks are fixed for state records. Environment snapshots are irrelevant
to these contracts and are replaced by an explicitly synthetic execution record.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts
from cognitive_runtime.training.model_factory.contracts import ArchitectureContract, DataContract, TrainingContract
from cognitive_runtime.training.model_factory.naming import DisplayNameParent
from cognitive_runtime.training.model_factory.spec import resolve
from cognitive_runtime.training.model_factory.state import create_state, transition

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / "tests" / "fixtures" / "factory"
INPUTS = Path(__file__).with_name("inputs.json")
EDGE_CASES = {
    "contracts.json": "Nonempty disjoint split identities, canonical contract hashes, nullable budgets and absent optional policy maps.",
    "lineage.fresh.json": "Fresh run with no checkpoint parent, no configuration parents and a sibling group.",
    "lineage.clone.json": "Clone with a checkpoint parent, exact donor hash and two configuration parents.",
    "state.queued.json": "Initial queued run with null reason/retry and one history entry.",
    "state.completed.json": "Queued -> running -> checkpointing -> completed with append-only timestamped history.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_artifacts(destination: Path) -> None:
    """Exercise real contract/spec resolution, run allocation and state writes."""
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    architecture = ArchitectureContract(**inputs["architecture_contract"])
    data = DataContract(**inputs["data_contract"])
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for mode in ("fresh", "clone"):
            raw = dict(inputs["spec"])
            raw["mode"] = mode
            if mode == "clone":
                raw["parent"] = inputs["clone_parent"]
                raw["evolution"] = {
                    "configuration_parents": [inputs["clone_parent"]["run_id"], "synthetic-config-parent-298"],
                    "weight_donor": inputs["clone_parent"]["run_id"],
                }
            spec = resolve(raw)
            training = TrainingContract(**dict(spec.training))
            # Execution provenance is intentionally out of scope here. Avoid
            # importing optional torch or snapshotting host-specific packages.
            with patch(
                "cognitive_runtime.training.model_factory.artifacts.execution_manifest",
                return_value={"format": "model-factory-execution-v1", "source": "synthetic-only"},
            ):
                artifacts = allocate_run_artifacts(
                    workspace, spec, architecture, data, training,
                    run_id=f"{inputs['run_id']}-{mode}", naming_seed=inputs["seed"],
                    sibling_group=inputs["sibling_group"],
                    naming_parents=[
                        DisplayNameParent(parent, f"synthetic-parent{index}")
                        for index, parent in enumerate((raw.get("evolution") or {}).get("configuration_parents", []))
                    ],
                )
            if mode == "fresh":
                (destination / "contracts.json").write_bytes(artifacts.contracts_path.read_bytes())
            (destination / f"lineage.{mode}.json").write_bytes(artifacts.lineage_path.read_bytes())

        state_path = workspace / "state.json"
        with patch(
            "cognitive_runtime.training.model_factory.state._now_iso",
            side_effect=inputs["timestamps"],
        ):
            create_state(
                state_path, inputs["run_id"], devices=["cpu"], heartbeat_timeout_seconds=30.0,
            )
            (destination / "state.queued.json").write_bytes(state_path.read_bytes())
            transition(state_path, "running", reason="synthetic worker started")
            transition(state_path, "checkpointing", reason="synthetic checkpoint boundary")
            transition(state_path, "completed", reason="synthetic serialization smoke complete")
            (destination / "state.completed.json").write_bytes(state_path.read_bytes())


def main() -> None:
    generate_artifacts(FIXTURES)
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    manifest = {
        "format": "ccr-synthetic-factory-fixtures-v1",
        "seed": inputs["seed"],
        "status": "synthetic_only",
        "source": "Project-authored inputs serialized by the existing CCR Model Factory; no external dataset.",
        "license": "MIT",
        "rights": "Original synthetic fixtures contributed under this repository's MIT license; no third-party payloads.",
        "config": "metadata/inputs.json",
        "config_sha256": sha256(INPUTS),
        "generator": "metadata/generate.py",
        "generator_sha256": sha256(Path(__file__)),
        "artifacts": [],
    }
    for filename, edge_case in EDGE_CASES.items():
        schema = f".github/schemas/{filename.split('.')[0]}.schema.json"
        manifest["artifacts"].append({
            "path": filename,
            "sha256": sha256(FIXTURES / filename),
            "schema": schema,
            "schema_sha256": sha256(ROOT / schema),
            "intended_edge_case": edge_case,
        })
    (FIXTURES / "metadata" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
