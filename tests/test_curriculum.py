"""Curriculum presets (issue #30): world-config + reward-weight bundles.

Covers the preset registry itself (valid field names, deterministic seeds),
the CLI's `--curriculum` resolution helpers, and an end-to-end run per
preset in the simulated backend with the random policy -- the acceptance
criterion ("each preset runs ... and produces a recorded session tagged
with the curriculum name") kept fast with a small `episode_ticks` override.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from cognitive_runtime.cli import _reward_config_for, _resolve_world_args, build_parser
from cognitive_runtime.policies import RandomPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.curriculum import (
    CURRICULA,
    CURRICULUM_ORDER,
    get_curriculum,
)
from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime

_WORLD_FIELDS = set(SurvivalBoxConfig.__dataclass_fields__)
_REWARD_FIELDS = set(SurvivalRewardConfig.__dataclass_fields__)


def test_curriculum_order_matches_registry_keys():
    assert set(CURRICULUM_ORDER) == set(CURRICULA)


def test_get_curriculum_unknown_name_raises():
    with pytest.raises(KeyError):
        get_curriculum("does-not-exist")


def test_every_preset_only_overrides_valid_fields():
    """Guards against typos: every world_config/reward_config key must be a
    real SurvivalBoxConfig / SurvivalRewardConfig field."""
    for name in CURRICULUM_ORDER:
        preset = get_curriculum(name)
        assert set(preset.world_config) <= _WORLD_FIELDS, name
        assert set(preset.reward_config) <= _REWARD_FIELDS, name
        # Applying the overrides must not raise.
        SurvivalBoxConfig.from_dict(preset.world_config)
        dataclasses.replace(SurvivalRewardConfig(), **preset.reward_config)


def test_presets_have_distinct_seeds():
    seeds = [get_curriculum(name).seed for name in CURRICULUM_ORDER]
    assert len(seeds) == len(set(seeds))


def test_cli_curriculum_choice_rejects_unknown_name():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--curriculum", "not-a-preset"])


def test_resolve_world_args_fills_from_curriculum_but_respects_explicit_flags():
    parser = build_parser()
    args = parser.parse_args([
        "run", "--curriculum", "combat", "--policy", "random", "--world-size", "99",
    ])
    _resolve_world_args(args)
    preset = get_curriculum("combat")
    assert args.seed == preset.seed
    assert args.difficulty == preset.world_config["difficulty"]
    assert args.max_mobs == preset.world_config["max_mobs"]
    # The explicit --world-size flag wins over the curriculum's value.
    assert args.world_size == 99


def test_resolve_world_args_without_curriculum_keeps_historical_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "--policy", "random"])
    _resolve_world_args(args)
    assert args.seed == 0
    assert args.episode_ticks == 6000
    assert args.difficulty == 1.0
    assert args.world_size == 64
    assert args.day_length == 6000
    assert args.start_time == 0
    assert args.max_mobs == 3


def test_reward_config_for_none_without_curriculum():
    parser = build_parser()
    args = parser.parse_args(["run", "--policy", "random"])
    assert _reward_config_for(args) is None


@pytest.mark.parametrize("name", CURRICULUM_ORDER)
def test_each_preset_runs_in_the_simulated_backend_and_tags_the_session(name, tmp_path):
    preset = get_curriculum(name)
    world_config = dict(preset.world_config)
    world_config["episode_ticks"] = 30  # keep the acceptance run fast
    reward_config = dataclasses.replace(SurvivalRewardConfig(), **preset.reward_config)

    program = MinecraftSurvivalBox(config=world_config, reward_config=reward_config)
    config = RuntimeConfig(
        episodes=1,
        seed=preset.seed,
        max_ticks_per_episode=30,
        record_dir=str(tmp_path),
        session_id=f"curriculum-{name}",
        program_config=world_config,
        curriculum=name,
    )
    runtime = CognitiveRuntime(
        program=program,
        policy=RandomPolicy(ACTION_SPACE, seed=preset.seed),
        config=config,
        stream_registry=MINECRAFT_STREAM_REGISTRY,
    )
    summaries = runtime.run()

    assert len(summaries) == 1
    assert summaries[0].curriculum == name

    session_dir = tmp_path / f"curriculum-{name}"
    with open(session_dir / "session.json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["curriculum"] == name

    with open(session_dir / "episode_00000.summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["curriculum"] == name
