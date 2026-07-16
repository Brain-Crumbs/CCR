"""Phase 4 recall: closed-loop generation, senses-off, and viewer export."""

from __future__ import annotations

import base64
import json

import pytest

torch = pytest.importorskip("torch")

from brain.cortex import PredictiveCortex, PredictiveCortexConfig
from brain.hippocampus import Seed, SeedTags
from sleep.dream import dream, export_dream_file


def _cortex():
    torch.manual_seed(3)
    return PredictiveCortex(
        (4, 4, 3),
        ["wait", "left"],
        PredictiveCortexConfig(latent_width=5, hidden_dim=8, reconstruction_size=4),
    )


def _seed(model, actions=("wait", "left", "wait")):
    return Seed(
        z=torch.randn(model.latent_width).tolist(),
        actions=list(actions),
        tags=SeedTags(),
        priority=1.0,
        tick_index=7,
        source="held-out",
    )


def test_dream_is_the_cortex_own_closed_loop_rollout():
    model = _cortex()
    seed = _seed(model)
    action_ids = torch.tensor([[0, 1, 0]])
    with torch.no_grad():
        expected_latents, _ = model.rollout(
            torch.tensor(seed.z).unsqueeze(0), action_ids, model.initial_state(1)
        )
        expected = model.decoder(expected_latents.squeeze(0))

    actual = torch.stack(list(dream(seed, 3, model)))
    # Recall reproduces precisely the cortex's own T+h prediction, including
    # whatever prediction error that model has against the held-out episode.
    assert torch.equal(actual, expected)


def test_dream_never_reads_live_senses(monkeypatch):
    model = _cortex()
    seed = _seed(model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a dream attempted to read the sensory bus")

    from cognitive_runtime.core.streams import bus

    monkeypatch.setattr(bus.StreamBus, "read_since", forbidden)
    monkeypatch.setattr(bus.StreamBus, "latest", forbidden)
    assert len(list(dream(seed, 2, model))) == 2


def test_dream_export_round_trips_viewer_schema_and_is_name_prefixed(tmp_path):
    model = _cortex()
    seed = _seed(model)
    (tmp_path / "session.json").write_text(json.dumps({"name": "Ada"}))
    actual = [torch.rand(3, 4, 4) for _ in range(4)]

    path = export_dream_file(
        seed, model, actual, [1, 3], session_dir=str(tmp_path), episode_id="episode_00001"
    )
    assert path.endswith("Ada-dream_episode_00001.json")
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["format"] == "pixel-predictions-v1"
    assert payload["horizons"] == [1, 3]
    assert payload["prediction_shape"] == [4, 4, 3]
    assert payload["n_frames"] == 4
    assert set(payload["predictions"]) == {"1", "3"}
    assert len(payload["targets"]) == 4
    expected_bytes = 4 * 4 * 3
    assert all(
        len(base64.b64decode(entry["frames"][0])) == expected_bytes
        for entry in payload["predictions"].values()
    )
