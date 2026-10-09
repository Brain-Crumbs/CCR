#!/usr/bin/env python3
"""Fail closed on missing/failed runtime evidence or >120s aggregate core."""
import json
from pathlib import Path
import sys

REPORTS = ("core-collection.json", "core-tests.json", "contracts-seed-12345.json")


def main(directory: Path) -> int:
    import math
    total = 0.0
    memberships = {}
    for name in REPORTS:
        try:
            report = json.loads((directory / name).read_text())
            duration = report["duration_seconds"]
            if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid duration")
            for key in ("returncode", "network_attempts", "model_cache_files"):
                if type(report[key]) is not int or report[key] != 0:
                    raise ValueError(f"{key} was not zero")
            expected_seed = "12345" if name == "contracts-seed-12345.json" else "0"
            if report["pythonhashseed"] != expected_seed:
                raise ValueError("incorrect or missing hash seed")
            nodeids = report["nodeids"]
            if not isinstance(nodeids, list) or not nodeids or not all(isinstance(node, str) for node in nodeids) or len(nodeids) != len(set(nodeids)):
                raise ValueError("missing/invalid selected node IDs")
            collecting = name == "core-collection.json"
            expected_executed = 0 if collecting else len(nodeids)
            if report["collect_only"] is not collecting or report["executed_tests"] != expected_executed:
                raise ValueError("collection-only or incomplete execution cannot satisfy a test lane")
            memberships[name] = set(nodeids)
            total += duration
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"invalid core evidence {name}: {exc}", file=sys.stderr)
            return 1
    core = memberships["core-tests.json"]
    factories = {node for node in core if node.startswith("tests/test_model_factory_")}
    if core != memberships["core-collection.json"] or not factories or factories != memberships["contracts-seed-12345.json"]:
        print("core collection/execution or Factory dual-seed test membership differs", file=sys.stderr)
        return 1
    print(f"aggregate core execution: {total:.3f}s / 120s (collection + full suite (seed 0) + Factory seed 12345)")
    if total > 120:
        print("optimize/split execution without removing existing lightweight coverage", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
