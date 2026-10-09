"""Catch network-denial swallowing and inherited child network attempts."""
from pathlib import Path
import json
import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="clean-core runner requires a torch-free environment; covered by PR core lane")

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / ".github/scripts/run_core_tests.py"


@pytest.mark.parametrize("child", [False, True])
def test_caught_network_attempt_still_fails_runner(tmp_path, child):
    test = tmp_path / "test_attempt.py"
    attempt = "import socket\ntry:\n socket.getaddrinfo('example.invalid', 443)\nexcept RuntimeError:\n pass\n"
    body = f"import subprocess, sys\nsubprocess.run([sys.executable, '-c', {attempt!r}], check=True)\n" if child else attempt
    test.write_text("def test_attempt():\n" + "\n".join("    " + line for line in body.splitlines()) + "\n")
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(RUNNER), "--workers", "0", "--report", str(report), str(test)], text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["network_attempts"] == 1
    assert data["counts"]["failures"] == 0  # the pytest body caught the exception


def test_empty_caches_and_offline_flags_are_present(tmp_path):
    test = tmp_path / "test_environment.py"
    test.write_text('''import os
from pathlib import Path
def test_environment():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    for key in ("HF_HOME", "TORCH_HOME"):
        assert not Path(os.environ[key]).exists()
''')
    result = subprocess.run([sys.executable, str(RUNNER), "--workers", "0", str(test)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_caught_udp_send_is_recorded(tmp_path):
    test = tmp_path / "test_udp.py"
    test.write_text("import socket\ndef test_udp():\n    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n    try:\n        sock.sendto(b'probe', ('127.0.0.1', 9))\n    except RuntimeError:\n        pass\n    finally:\n        sock.close()\n")
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(RUNNER), "--workers", "0", "--report", str(report), str(test)], text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(report.read_text())["network_attempts"] == 1


def test_two_workers_inherit_seed_and_offline_guard(tmp_path):
    for index in range(2):
        (tmp_path / f"test_worker_{index}.py").write_text("import os\ndef test_worker():\n    assert os.environ['PYTHONHASHSEED'] == '0'\n    assert os.environ['HF_HUB_OFFLINE'] == '1'\n")
    report = tmp_path / "workers.json"
    result = subprocess.run([sys.executable, str(RUNNER), "--report", str(report), str(tmp_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["executed_tests"] == 2
    assert len(data["worker_environments"]) == 2
    assert all(value == {"pythonhashseed": "0", "offline_guard": True} for value in data["worker_environments"].values())
