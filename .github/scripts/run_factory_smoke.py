#!/usr/bin/env python3
"""Run the opt-in, synthetic Factory CPU canary (issues #242, #297; epic #212).

Uses the installed interpreter's actual CLI, with a fresh output directory,
actual returned run IDs, bounded subprocesses, and no success-shaped skips.
No torch import is needed to inspect or test this orchestration contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / ".github" / "fixtures"


def result_field(output: str, key: str) -> str:
    values = re.findall(rf"^{re.escape(key)}: (.+)$", output, flags=re.MULTILINE)
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"expected exactly one {key!r} in successful Factory CLI output")
    return values[0].strip()


def factory_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "cognitive_runtime", "factory", *arguments, "--no-trace"]


def run_smoke(output: Path, *, timeout_seconds: float = 150.0) -> dict:
    if not 0 < timeout_seconds <= 180:
        raise ValueError("smoke timeout must be greater than zero and at most 180 seconds")
    for name in ("micro-corpus.yaml", "micro-baseline.yaml", "provenance.json"):
        if not (FIXTURES / name).is_file():
            raise FileNotFoundError(f"required checked-in Factory fixture is missing: {name}")
    output = output.resolve()
    # Reusing old artifacts could make a broken build appear successful.
    output.mkdir(parents=True, exist_ok=False)
    runs, corpora = output / "runs", output / "corpora"
    started = time.monotonic()
    deadline = started + timeout_seconds
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    summary = {
        "status": "running", "validation_status": "synthetic_only",
        "device": "cpu", "threads": 1, "run_ids": [], "stages": [],
        "fixture_sha256": {
            name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
            for name in ("micro-corpus.yaml", "micro-baseline.yaml", "provenance.json")
        },
    }

    def invoke(label: str, *arguments: str) -> str:
        command = factory_command(*arguments)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Factory smoke exceeded its total wall-clock bound")
        before = time.monotonic()
        process = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=min(60.0, remaining), check=False,
        )
        (output / f"{label}.stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / f"{label}.stderr.log").write_text(process.stderr, encoding="utf-8")
        summary["stages"].append({
            "name": label, "command": command, "returncode": process.returncode,
            "seconds": round(time.monotonic() - before, 3),
        })
        print(f"{label}: {summary['stages'][-1]['seconds']}s", flush=True)
        if process.returncode:
            raise RuntimeError(f"Factory {label} failed ({process.returncode}):\n{process.stderr}\n{process.stdout}")
        return process.stdout

    def trial(label: str, *arguments: str) -> str:
        stdout = invoke(label, *arguments, "--root", str(runs), "--corpus-root", str(corpora),
                        "--no-export-predictions")
        run_id = result_field(stdout, "run_id")
        if run_id in summary["run_ids"]:
            raise ValueError(f"Factory returned duplicate run_id: {run_id}")
        if result_field(stdout, "state") != "completed":
            raise ValueError(f"Factory {label} did not complete")
        directory = Path(result_field(stdout, "directory")).resolve()
        if not directory.is_relative_to(runs):
            raise ValueError(f"Factory {label} wrote outside its isolated runs directory")
        for name in ("trial_spec.json", "contracts.json", "lineage.json", "execution.json",
                     "metrics/validation.json", "metrics/budget_report.json", "experiment_report.json",
                     "checkpoints/last.pt", "checkpoints/best-validation.pt"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Factory {label} omitted required artifact: {name}")
        if (directory / "metrics" / "test.json").exists():
            raise ValueError("routine Factory smoke must not evaluate the sealed test split")
        summary["run_ids"].append(run_id)
        return run_id

    try:
        built = invoke("corpus-build", "corpus", "build", str(FIXTURES / "micro-corpus.yaml"),
                       "--root", str(corpora))
        corpus_directory = Path(result_field(built, "directory")).resolve()
        if not corpus_directory.is_relative_to(corpora):
            raise ValueError("Factory corpus escaped its isolated output directory")
        manifest_path = corpus_directory / "corpus_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if any(not manifest["sessions"][split] for split in ("train", "validation", "test")):
            raise ValueError("smoke corpus must contain all three nonempty splits")
        if not all(entry.get("sha256") for entries in manifest["sessions"].values() for entry in entries):
            raise ValueError("smoke corpus did not freeze session hashes")
        invoke("corpus-reuse", "corpus", "build", str(FIXTURES / "micro-corpus.yaml"), "--root", str(corpora))
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("rebuilding a frozen corpus changed its manifest")
        summary["corpus_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        baseline = trial("baseline", "baseline", str(FIXTURES / "micro-baseline.yaml"))
        first = trial("clone-a", "clone", baseline, "--set", "training.loss_weights.closed_loop_pixel_loss_weight=0.125")
        second = trial("clone-b", "clone", baseline, "--set", "training.loss_weights.closed_loop_pixel_loss_weight=0.5")
        comparison = invoke("compare", "compare", baseline, first, second, "--root", str(runs))
        if f"baseline: {baseline} " not in comparison or any(
            f"{run_id}: status=evaluable " not in comparison for run_id in (first, second)
        ):
            raise ValueError("Factory comparison did not evaluate the actual sibling run IDs")
        summary["status"] = "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["seconds"] = round(time.monotonic() - started, 3)
        (output / "smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "factory-smoke")
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    args = parser.parse_args()
    result = run_smoke(args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
