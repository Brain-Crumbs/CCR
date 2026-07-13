"""Pixel vision: deterministic RGB render, online/offline parity, end-to-end BC.

The determinism tests need no torch; the training/parity tests import torch and
are skipped when it is not installed, so the suite stays green either way.
"""

import os
from collections import deque

import numpy as np
import pytest

from cognitive_runtime.core.streams import TemporalBuffer, TemporalFusion
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.streams import PIXEL_SHAPE, PIXEL_STREAM
from cognitive_runtime.programs.minecraft.world import SimulatedWorld, pixels_from_frame
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.frame_store import open_frame_store
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
)
from cognitive_runtime.training.datasets import (
    _catalog,
    _motor_label,
    build_neural_dataset,
)

FAST_CONFIG = {"episode_ticks": 300, "world_size": 32}


def _record(tmp_path, session_id, episodes=2, seed=0, config=FAST_CONFIG):
    runtime_config = RuntimeConfig(
        episodes=episodes, seed=seed, max_ticks_per_episode=config["episode_ticks"],
        record_dir=str(tmp_path), session_id=session_id, program_config=config,
        record_frames=True,  # pixel frames must be in the log to train vision
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=1),
        config=runtime_config,
    )
    runtime.run()
    return os.path.join(str(tmp_path), session_id)


# --------------------------------------------------------------- determinism

def test_pixel_render_is_deterministic_and_correct_shape():
    a = SimulatedWorld(SurvivalBoxConfig.from_dict(FAST_CONFIG), seed=7)
    b = SimulatedWorld(SurvivalBoxConfig.from_dict(FAST_CONFIG), seed=7)
    pa, pb = a.render_pixels(), b.render_pixels()
    assert isinstance(pa, np.ndarray)
    assert np.array_equal(pa, pb), "same seed must render byte-identical pixels (replay-safe)"
    assert tuple(pa.shape) == PIXEL_SHAPE
    assert pa.dtype == np.uint8
    assert bool(((pa >= 0) & (pa <= 255)).all())


def test_pixel_render_changes_when_turning_in_place():
    world = SimulatedWorld(SurvivalBoxConfig.from_dict(FAST_CONFIG), seed=7)
    before = world.render_pixels()
    world.yaw = (world.yaw + 90.0) % 360.0
    after = world.render_pixels()
    assert not np.array_equal(before, after)


def test_pixels_from_frame_upscales_and_colors_generically():
    # A 1x2 grid frame of two frame codes -> scale*2 wide, scale tall, RGB.
    frame = [[1, 4]]  # grass, water
    img = pixels_from_frame(frame, scale=3)
    assert isinstance(img, np.ndarray)
    assert img.shape == (3, 6, 3)
    assert not np.array_equal(img[0][0], img[0][-1]), "different frame codes get different colors"


def test_pixel_stream_present_and_replayable(tmp_path):
    session_dir = _record(tmp_path, "pixels")
    episode = list_episodes(session_dir)[0]
    frame_store = open_frame_store(session_dir)
    frames = [
        stream_event_from_log(rec, frame_store=frame_store).payload
        for _d, sensory, _m in iter_cognitive_ticks(session_dir, episode)
        for rec in sensory
        if rec.get("stream_id") == PIXEL_STREAM and not rec.get("elided")
    ]
    assert frames, "pixel frames must be recorded under record_frames"
    assert all(isinstance(f, np.ndarray) and tuple(f.shape) == PIXEL_SHAPE for f in frames)


# ----------------------------------------------------- online/offline parity

def _non_vision_from_full_fusion(catalog, buffer):
    """Reconstruct the non-vision vector the way NeuralPolicy does at inference:
    the full fusion the runtime computes, minus every vision.* slice."""
    from cognitive_runtime.policies.neural_policy import non_vision_features

    full = TemporalFusion(catalog)  # default registry, includes vision encoders
    return non_vision_features(full.fuse(None, buffer))


def test_online_offline_pixel_and_feature_parity(tmp_path):
    pytest.importorskip("torch")
    session_dir = _record(tmp_path, "parity")
    dataset = build_neural_dataset([session_dir])
    assert len(dataset) == 600  # 2 episodes x 300 ticks
    assert tuple(dataset.pixel_shape) == PIXEL_SHAPE
    assert len(dataset.non_vision[0]) == len(dataset.non_vision_names)

    # Re-walk the first episode exactly as the builder does, but derive the
    # non-vision vector via the *online* code path (full fusion -> drop vision),
    # and confirm every sample matches the dataset built by the offline path.
    metadata_catalog = _catalog(load_session_metadata(session_dir))
    episode = list_episodes(session_dir)[0]
    frame_store = open_frame_store(session_dir)
    buffer = TemporalBuffer()
    key_to_label = {k: i for i, k in enumerate(dataset.action_keys)}
    i = 0
    for _decision, sensory, motor in iter_cognitive_ticks(session_dir, episode):
        for rec in sensory:
            if not rec.get("elided"):
                buffer.append(stream_event_from_log(rec, frame_store=frame_store))
        label = _motor_label(motor)
        latest = buffer.latest(PIXEL_STREAM)
        if label in key_to_label and latest is not None:
            online_vec, online_names = _non_vision_from_full_fusion(metadata_catalog, buffer)
            assert online_names == dataset.non_vision_names
            assert online_vec == dataset.non_vision[i], f"non-vision mismatch at sample {i}"
            assert np.array_equal(latest.payload, dataset.pixels[i]), f"pixel mismatch at sample {i}"
            i += 1
    assert i > 0


# -------------------------------------------------------------- training / io

def test_neural_bc_learns_and_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    from cognitive_runtime.models.vision import VisionBCModel
    from cognitive_runtime.training.neural import train_neural_bc

    session_dir = _record(tmp_path, "train", episodes=3, seed=100)
    dataset = build_neural_dataset([session_dir])
    torch.manual_seed(0)
    model, metrics = train_neural_bc(dataset, epochs=12, seed=0)

    # Class-balanced BC must learn context beyond a single dominant action.
    assert metrics["train_balanced_accuracy"] > metrics["random_class_baseline"]

    path = os.path.join(str(tmp_path), "vision_bc.pt")
    model.save(path)
    loaded = VisionBCModel.load(path)
    args = (dataset.pixels[0], dataset.non_vision[0], dataset.motor[0])
    assert loaded.logits(*args) == model.logits(*args), "reload must be bit-identical"


def test_build_neural_dataset_requires_frames(tmp_path):
    pytest.importorskip("torch")
    # Record WITHOUT frames: the pixel stream is elided, so training must refuse.
    runtime_config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=FAST_CONFIG["episode_ticks"],
        record_dir=str(tmp_path), session_id="noframes", program_config=FAST_CONFIG,
        record_frames=False,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=1), config=runtime_config,
    ).run()
    with pytest.raises(ValueError, match="hash-only"):
        build_neural_dataset([os.path.join(str(tmp_path), "noframes")])


# ------------------------------------------------------ input-profile ablation (issue #32)


def test_stream_profile_raw_narrows_the_non_vision_companion_vector(tmp_path):
    session_dir = _record(tmp_path, "profile-session", episodes=1)
    full = build_neural_dataset([session_dir])
    raw = build_neural_dataset([session_dir], stream_profile="raw")

    assert full.stream_profile == "full"
    assert raw.stream_profile == "raw"
    assert len(raw.non_vision_names) < len(full.non_vision_names)
    assert len(full) == len(raw) > 0  # same samples, narrower per-sample vector

    # semantic scalar names (e.g. world.front_block's one-hot slots) are gone
    assert not any(name.startswith("world.front_block") for name in raw.non_vision_names)
    assert any(name.startswith("body.health") for name in raw.non_vision_names)


def test_stream_profile_rejects_unknown_value(tmp_path):
    session_dir = _record(tmp_path, "profile-session-bad", episodes=1)
    with pytest.raises(ValueError, match="stream_profile"):
        build_neural_dataset([session_dir], stream_profile="bogus")
