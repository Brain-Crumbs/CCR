"""Missing, failed and over-budget lanes cannot produce a green total."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / ".github/scripts/check_core_budget.py"
spec = importlib.util.spec_from_file_location("core_budget", PATH)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def reports(tmp_path, duration=10):
    for name in budget.REPORTS:
        (tmp_path / name).write_text(json.dumps({"duration_seconds": duration, "returncode": 0, "network_attempts": 0, "model_cache_files": 0, "pythonhashseed": "12345" if name == "contracts-seed-12345.json" else "0", "collect_only": name == "core-collection.json", "executed_tests": 0 if name == "core-collection.json" else (1 if name == "contracts-seed-12345.json" else 2), "nodeids": ["tests/test_model_factory_example.py::test_case"] if name == "contracts-seed-12345.json" else ["tests/test_model_factory_example.py::test_case", "tests/test_other.py::test_case"]}))


def test_aggregate_budget_counts_every_lane(tmp_path):
    reports(tmp_path)
    assert budget.main(tmp_path) == 0
    reports(tmp_path, 41)
    assert budget.main(tmp_path) == 1


@pytest.mark.parametrize("mutation", ["missing", "malformed", "failed", "network", "cache", "nan"])
def test_incomplete_evidence_fails_closed(tmp_path, mutation):
    reports(tmp_path)
    path = tmp_path / budget.REPORTS[0]
    data = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "malformed":
        path.write_text("{")
    else:
        key, value = {"failed": ("returncode", 1), "network": ("network_attempts", 1), "cache": ("model_cache_files", 1), "nan": ("duration_seconds", float("nan"))}[mutation]
        data[key] = value
        path.write_text(json.dumps(data))
    assert budget.main(tmp_path) == 1


def test_pr_workflow_keeps_all_lightweight_tests_and_budget_gates():
    import yaml
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    for name in ("core", "factory-contracts", "aggregate-core-budget"):
        assert int(workflow["jobs"][name]["timeout-minutes"]) <= 5
    core = workflow["jobs"]["core"]
    commands = "\n".join(step.get("run", "") for step in core["steps"])
    assert 'run_core_tests.py --report' in commands
    assert 'run_core_tests.py --collect-only' in commands
    assert 'unshare --net' in commands
    assert '--run-market' not in commands and '--ignore' not in commands
    contracts = workflow["jobs"]["factory-contracts"]
    commands = "\n".join(step.get("run", "") for step in contracts["steps"])
    assert 'tests/test_model_factory_*.py' in commands
    assert 'validate_artifact_schemas.py' in commands
    assert not any('if' in step for step in contracts["steps"] if 'validate_artifact_schemas.py' in step.get('run', ''))
    assert 'if [ -d' not in commands and 'skipping' not in commands
    assert workflow["jobs"]["aggregate-core-budget"]["needs"] == ["core", "factory-contracts"]
    assert "workflow_dispatch" in workflow["jobs"]["neural"]["if"]


@pytest.mark.parametrize("field,value", [("pythonhashseed", "0"), ("nodeids", []), ("nodeids", ["tests/test_model_factory_other.py::test_case"])])
def test_second_seed_cannot_lose_seed_or_test_membership(tmp_path, field, value):
    reports(tmp_path)
    path = tmp_path / "contracts-seed-12345.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    assert budget.main(tmp_path) == 1


def test_collection_cannot_masquerade_as_execution(tmp_path):
    reports(tmp_path)
    path = tmp_path / "core-tests.json"
    data = json.loads(path.read_text())
    data["collect_only"] = True
    data["executed_tests"] = 0
    path.write_text(json.dumps(data))
    assert budget.main(tmp_path) == 1
