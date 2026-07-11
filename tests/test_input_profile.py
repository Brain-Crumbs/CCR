"""Issue #32 acceptance: the "raw input" profile runs end to end in the
simulated backend, and the online policy's fused state actually shrinks to
just the agent_input-classified streams.
"""

import json

from cognitive_runtime.cli import main
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY

FAST_CONFIG = {"episode_ticks": 20, "world_size": 32}


def _run_cli_online(path, profile, ticks=20):
    main(
        [
            "run",
            "--policy", "online",
            "--input-profile", profile,
            "--episodes", "1",
            "--episode-ticks", str(ticks),
            "--world-size", "32",
            "--online-model", str(path),
            "--online-save-every", "5",
            "--no-record",
        ]
    )


def test_raw_input_profile_runs_end_to_end_in_the_simulated_backend(tmp_path):
    path = tmp_path / "online-q-raw.json"
    _run_cli_online(path, "raw")
    model = OnlineQModel.load(str(path))
    assert model.training_ticks > 0

    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    raw_registry = MINECRAFT_STREAM_REGISTRY.to_encoder_registry(classifications={"agent_input"})
    raw_fusion = TemporalFusion(program.stream_catalog(), raw_registry)
    assert model.latent_width == raw_fusion.width
    assert model.layout_hash == raw_fusion.layout_hash


def test_raw_input_profile_produces_a_narrower_fused_state_than_full(tmp_path):
    full_path = tmp_path / "online-q-full.json"
    raw_path = tmp_path / "online-q-raw.json"
    _run_cli_online(full_path, "full")
    _run_cli_online(raw_path, "raw")

    full_model = OnlineQModel.load(str(full_path))
    raw_model = OnlineQModel.load(str(raw_path))
    assert raw_model.latent_width < full_model.latent_width

    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    full_fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    assert full_model.latent_width == full_fusion.width


def test_full_input_profile_is_the_default_and_unchanged(tmp_path):
    """--input-profile full (the default) must not change pre-#32 behavior."""
    default_path = tmp_path / "online-q-default.json"
    explicit_path = tmp_path / "online-q-explicit-full.json"
    main([
        "run", "--policy", "online", "--episodes", "1", "--episode-ticks", "20",
        "--world-size", "32", "--online-model", str(default_path),
        "--online-save-every", "5", "--no-record",
    ])
    _run_cli_online(explicit_path, "full")

    default_model = OnlineQModel.load(str(default_path))
    explicit_model = OnlineQModel.load(str(explicit_path))
    assert default_model.latent_width == explicit_model.latent_width
    assert default_model.layout_hash == explicit_model.layout_hash


def test_session_metadata_still_records_every_stream_under_raw_profile(tmp_path):
    """Aux/debug and privileged streams keep publishing and recording under
    the raw profile -- only the policy's fused state narrows (issue #32:
    'recording/replay handles the classification')."""
    record_dir = tmp_path / "sessions"
    main([
        "run", "--policy", "scripted", "--input-profile", "raw",
        "--episodes", "1", "--episode-ticks", "20", "--world-size", "32",
        "--record-dir", str(record_dir), "--session-id", "raw-profile-session",
    ])
    with open(record_dir / "raw-profile-session" / "session.json", encoding="utf-8") as fh:
        session = json.load(fh)
    catalog_ids = {s["stream_id"] for s in session["stream_catalog"]}
    declared_ids = {d["stream_id"] for d in session["stream_registry"]}
    assert catalog_ids == declared_ids
    # aux_debug/privileged streams are still declared/recordable
    for still_present in ("world.front_block", "world.sheltered", "world.nearby_blocks_exact"):
        assert still_present in catalog_ids
    declared = {d["stream_id"]: d for d in session["stream_registry"]}
    assert declared["world.front_block"]["classification"] == "aux_debug"
    assert declared["world.nearby_blocks_exact"]["classification"] == "privileged"
    assert declared["vision.frame.pixels"]["classification"] == "agent_input"
