"""The optional-module policy is tested before any test-module code executes."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("tier", ["extended", "text", "jepa", "live"])
def test_optional_markers_exclude_explicit_files_before_import(tmp_path, tier):
    (tmp_path / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    module = tmp_path / "test_optional.py"
    module.write_text(f"import pytest\npytestmark = pytest.mark.market_{tier}\nraise RuntimeError('optional module imported')\ndef test_example(): pass\n")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    command = [sys.executable, "-m", "pytest", str(module), "--collect-only", "-q"]
    default = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert default.returncode == 5, default.stdout + default.stderr
    assert "optional module imported" not in default.stdout + default.stderr
    enabled = subprocess.run(command + [f"--run-market={tier}"], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert enabled.returncode == 2
    assert "optional module imported" in enabled.stdout + enabled.stderr


def test_optional_path_excludes_unmarked_eager_import(tmp_path):
    (tmp_path / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    module = tmp_path / "tests" / "market" / "text" / "test_optional.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise RuntimeError('text framework imported')\n")
    command = [sys.executable, "-m", "pytest", str(module), "--collect-only", "-q"]
    default = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert default.returncode == 5, default.stdout + default.stderr
    assert "text framework imported" not in default.stdout + default.stderr


@pytest.mark.parametrize("declaration", ["import pytest as p\npytestmark = p.mark.market_text", "from pytest import mark as m\npytestmark = m.market_text", "import pytest\npytestmark = getattr(pytest.mark, 'market_text')"])
def test_marker_aliases_are_excluded_without_import(tmp_path, declaration):
    (tmp_path / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    module = tmp_path / "test_optional.py"
    module.write_text(declaration + "\nraise RuntimeError('optional imported')\n")
    result = subprocess.run([sys.executable, "-m", "pytest", str(module), "--collect-only", "-q"], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 5, result.stdout + result.stderr
    assert "optional imported" not in result.stdout + result.stderr


def test_checkout_ancestor_named_text_does_not_disable_core(tmp_path):
    checkout = tmp_path / "text"
    checkout.mkdir()
    (checkout / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    module = checkout / "test_core.py"
    module.write_text("def test_example(): pass\n")
    result = subprocess.run([sys.executable, "-m", "pytest", str(module), "-q"], cwd=checkout, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
