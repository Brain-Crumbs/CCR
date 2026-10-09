#!/usr/bin/env python3
"""Run the existing torch-free suite offline with empty model/data caches.

The script fails on caught network attempts and enforces the aggregate 120s
pytest execution budget. Dependency installation is a separate networked step.
Use --collect-only for an independently checked import/collection pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import xml.etree.ElementTree as ET
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_PACKAGES = ("torch", "transformers", "sentence_transformers", "alpaca", "openai", "anthropic")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(0, 2), default=2, help="two bounded CPU workers; 0 for sequential diagnostics")
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--report", type=Path)
    parser.add_argument("paths", nargs="*", default=["tests"])
    args = parser.parse_args(argv)
    installed = [name for name in OPTIONAL_PACKAGES if importlib.util.find_spec(name) is not None]
    if installed:
        parser.error("core environment must not contain optional frameworks/provider SDKs: " + ", ".join(installed))
    with tempfile.TemporaryDirectory(prefix="ccr-core-empty-") as temporary:
        cache = Path(temporary)
        attempts = cache / "network-attempts.log"
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT / ".github/scripts/offline"), str(ROOT)]),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0"),
            "CCR_OFFLINE_ATTEMPTS": str(attempts),
            "CCR_CORE_NODEIDS": str(cache / "nodeids.json"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        }
        command = [sys.executable, "-m", "pytest", "-p", "ccr_core_evidence", "-p", "xdist.plugin", *args.paths, "-q", "--durations=25", f"--junitxml={cache / 'results.xml'}"]
        if args.workers and not args.collect_only:
            command.extend(["-n", str(args.workers), "--dist=loadfile"])
        if args.collect_only:
            command.append("--collect-only")
        start = time.monotonic()
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            returncode = process.wait(timeout=args.max_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            print(f"core exceeded {args.max_seconds:g}s; optimize or split without dropping coverage", file=sys.stderr)
            returncode = 1
        elapsed = time.monotonic() - start
        attempted = attempts.read_text().splitlines() if attempts.exists() else []
        if attempted:
            print(f"offline policy failed: {len(attempted)} forbidden network attempt(s)", file=sys.stderr)
            returncode = 1
        # Empty caches prevent an accidental local-model pass. Any populated model
        # cache is an unexpected artifact, even if a download exception was caught.
        cached = [str(p.relative_to(cache)) for root in (cache / "huggingface", cache / "torch") if root.exists() for p in root.rglob("*") if p.is_file()]
        if cached:
            print("offline policy failed: unexpected model-cache files", file=sys.stderr)
            returncode = 1
        report = {"command": command, "duration_seconds": round(elapsed, 3), "max_seconds": args.max_seconds, "returncode": returncode, "network_attempts": len(attempted), "model_cache_files": len(cached), "python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count(), "threads_per_worker": 1, "workers": 0 if args.collect_only else args.workers, "pythonhashseed": env["PYTHONHASHSEED"], "optional_packages_absent": list(OPTIONAL_PACKAGES)}
        if (cache / "results.xml").exists():
            suites = ET.parse(cache / "results.xml").getroot()
            report["counts"] = {name: sum(int(suite.get(name, 0)) for suite in suites) for name in ("tests", "failures", "errors", "skipped")}
        evidence = json.loads((cache / "nodeids.json").read_text()) if (cache / "nodeids.json").exists() else {"nodeids": [], "executed_tests": 0}
        report.update(evidence)
        if args.workers and not args.collect_only:
            workers = evidence.get("worker_environments", {})
            if len(workers) != args.workers or any(value.get("pythonhashseed") != env["PYTHONHASHSEED"] or value.get("offline_guard") is not True for value in workers.values()):
                print("core worker seed/offline guard evidence missing or mismatched", file=sys.stderr)
                returncode = report["returncode"] = 1
        report["collect_only"] = args.collect_only
        print(json.dumps({key: value for key, value in report.items() if key != "nodeids"}, sort_keys=True), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
