"""``programs.crafter.config`` (issue #202): translating a single
``world_size`` knob into Crafter's own ``area`` config key.
"""

from __future__ import annotations

from cognitive_runtime.programs.crafter.config import CrafterConfig, area_from_world_size


def test_area_from_world_size_returns_a_square_area():
    assert area_from_world_size(128) == (128, 128)
    assert area_from_world_size(64) == (64, 64)


def test_crafter_config_from_dict_ignores_unknown_world_size_key():
    """The bug this issue fixes: ``world_size`` is not a ``CrafterConfig``
    field, so passing it directly through ``from_dict`` is silently a
    no-op -- callers must translate it via ``area_from_world_size`` first."""
    cfg = CrafterConfig.from_dict({"world_size": 128})
    assert cfg.area == (64, 64)  # unchanged default, not (128, 128)


def test_crafter_config_from_dict_honors_area_directly():
    cfg = CrafterConfig.from_dict({"area": area_from_world_size(128)})
    assert cfg.area == (128, 128)
