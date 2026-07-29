"""Regression tests for Torch remaining an optional motor dependency."""

from __future__ import annotations

import subprocess
import sys


def test_motor_top_level_cortex_api_is_exported():
    """The public cortex-MPC factory API remains available with Torch."""
    from motor import build_cortex_mpc, cortex_mpc_factory

    assert callable(build_cortex_mpc)
    assert callable(cortex_mpc_factory)


def test_motor_package_import_does_not_require_torch():
    """Core-only users can import the public motor API without Torch."""
    script = """
import builtins

_import = builtins.__import__

def block_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return _import(name, *args, **kwargs)

builtins.__import__ = block_torch
from motor import build_cortex_mpc, cortex_mpc_factory
assert callable(build_cortex_mpc)
assert callable(cortex_mpc_factory)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
