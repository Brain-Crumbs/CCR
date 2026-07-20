"""CLI wiring for repointing ``--async-trainer`` at cortex consolidation
(issue #175): the online learner used to be the CLI's own actor/critic split
(``--policy actor-critic``/the old ``--async-trainer``); both are gone now,
and ``--async-trainer`` means "consolidate the live predictive cortex" --
requiring a cortex world model, and running a short simulated session without
crashing when one is supplied.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.cli import main  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.programs.minecraft.streams import PIXEL_SHAPE  # noqa: E402
from cognitive_runtime.training.action_world_model import save_action_world_model  # noqa: E402


def _write_tiny_cortex_checkpoint(path, action_keys):
    torch.manual_seed(0)
    cortex = PredictiveCortex(
        PIXEL_SHAPE, action_keys,
        PredictiveCortexConfig(latent_width=8, hidden_dim=16, reconstruction_size=8, horizons_ticks=(1,)),
    )
    save_action_world_model(str(path), cortex, {})


def test_async_trainer_without_cortex_world_model_exits_with_actionable_message(tmp_path):
    with pytest.raises(SystemExit, match="--world-model cortex:"):
        main([
            "run", "--policy", "scripted", "--async-trainer",
            "--episodes", "1", "--episode-ticks", "5", "--world-size", "16",
            "--no-record",
        ])


def test_async_trainer_with_neural_world_model_path_exits_with_actionable_message(tmp_path):
    """A non-cortex ``--world-model`` (the memoryless bridge) also does not
    satisfy ``--async-trainer``'s requirement."""
    action_keys = [
        a.key() for a in MinecraftSurvivalBox(config={"world_size": 16}).metadata().action_space
    ]
    checkpoint = tmp_path / "cortex.pt"
    _write_tiny_cortex_checkpoint(checkpoint, action_keys)
    # A plain path (no `cortex:` prefix) routes to `NeuralWorldModel` instead,
    # which does not exist here -- but the async-trainer gate must fire
    # before that path is even attempted.
    with pytest.raises(SystemExit, match="--world-model cortex:"):
        main([
            "run", "--policy", "scripted", "--async-trainer",
            "--world-model", str(checkpoint),
            "--episodes", "1", "--episode-ticks", "5", "--world-size", "16",
            "--no-record",
        ])


def test_async_trainer_with_cortex_world_model_runs_without_crashing(tmp_path):
    program_config = {"episode_ticks": 20, "world_size": 16}
    action_keys = [
        a.key() for a in MinecraftSurvivalBox(config=program_config).metadata().action_space
    ]
    checkpoint = tmp_path / "cortex.pt"
    _write_tiny_cortex_checkpoint(checkpoint, action_keys)

    main([
        "run", "--policy", "scripted",
        "--world-model", f"cortex:{checkpoint}",
        "--async-trainer", "--async-wake-ticks", "5", "--async-consolidation-steps", "2",
        "--episodes", "1", "--episode-ticks", str(program_config["episode_ticks"]),
        "--world-size", str(program_config["world_size"]),
        "--no-record",
    ])
