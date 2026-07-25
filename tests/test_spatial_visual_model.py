"""Regression coverage for issue #204's spatial residual visual path."""

import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig
from cognitive_runtime.neural.pixel_stream_encoder import SpatialVisualEncoder
from cognitive_runtime.training.action_world_model import (
    ActionWorldModelConfig,
    _spatial_pixel_loss,
    load_action_world_model,
    save_action_world_model,
)


def test_spatial_encoder_distinguishes_locations_with_equal_histograms():
    encoder = SpatialVisualEncoder((64, 64, 3), latent_width=8)
    left = torch.zeros(1, 3, 64, 64)
    right = torch.zeros_like(left)
    left[:, :, 8:16, 8:16] = 1
    right[:, :, 8:16, 48:56] = 1
    a, b = encoder.encode(left), encoder.encode(right)
    assert a.spatial.shape[-2:] == (8, 8)
    assert not torch.allclose(a.spatial, b.spatial)


def test_zero_residual_returns_reference_exactly():
    model = PredictiveCortex(
        (64, 64, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, reconstruction_size=64),
    )
    frame = torch.rand(2, 3, 64, 64)
    spatial = model.encode_visual(frame).spatial
    with torch.no_grad():
        model.decoder.mask_head.weight.zero_()
        model.decoder.mask_head.bias.fill_(-100.0)
    result = model.decode_prediction(torch.zeros(2, 8), reference_frame=frame, reference_spatial=spatial)
    assert torch.equal(result["vision"], frame)
    assert result["change_mask"].shape == (2, 1, 64, 64)


def test_closed_loop_rollout_uses_its_previous_predicted_frame_as_reference(monkeypatch):
    model = PredictiveCortex(
        (16, 16, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, hidden_dim=12, reconstruction_size=16),
    )
    frame = torch.rand(1, 3, 16, 16)
    spatial = model.encode_visual(frame).spatial
    references = []
    outputs = []
    real_decode = model.decode_prediction

    def tracked(latent, *, reference_frame, reference_spatial):
        references.append(reference_frame.detach().clone())
        output = real_decode(latent, reference_frame=reference_frame, reference_spatial=reference_spatial)
        outputs.append(output["vision"].detach().clone())
        return output

    monkeypatch.setattr(model, "decode_prediction", tracked)
    model.forward_horizons(
        torch.rand(1, 8), torch.zeros(1, 2, dtype=torch.long), model.initial_state(1),
        horizon_frames=[1, 2], reference_frame=frame, reference_spatial=spatial,
    )
    assert len(references) == 2
    assert torch.equal(references[1], outputs[0])


def test_legacy_global_pool_checkpoint_has_actionable_error(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"format": "action-world-model-v1"}, path)
    with pytest.raises(ValueError, match="global_pool_v1; start a new spatial_residual_v1"):
        load_action_world_model(str(path))
    inspected, stats = load_action_world_model(str(path), inspection_only=True)
    assert inspected is None
    assert stats["legacy_visual_architecture"] == "global_pool_v1"


def test_semantic_decoder_is_available_and_receives_gradients():
    model = PredictiveCortex(
        (64, 64, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, semantic_classes=5),
    )
    frame = torch.rand(1, 3, 64, 64)
    output = model.decode_prediction(
        torch.rand(1, 8, requires_grad=True), reference_frame=frame,
        reference_spatial=model.encode_visual(frame).spatial,
    )
    assert output["semantic_logits"].shape == (1, 5, 9, 9)
    output["semantic_logits"].mean().backward()
    assert model.encoder.features[0].weight.grad is not None
    assert model.decoder.semantic_head.weight.grad is not None


def test_scene_hud_loss_downweights_hud_only_changes():
    model = PredictiveCortex(
        (8, 8, 3), ["wait"],
        PredictiveCortexConfig(latent_width=4, reconstruction_size=8),
    )
    with torch.no_grad():
        model.decoder.mask_head.weight.zero_()
        model.decoder.mask_head.bias.fill_(-100.0)
    reference = torch.zeros(1, 3, 8, 8)
    target = reference.clone()
    target[:, :, 6:] = 1.0
    loss, _mask, _decoded = _spatial_pixel_loss(
        model, torch.zeros(1, 4), reference, target,
        ActionWorldModelConfig(scene_height=6, hud_loss_weight=0.25),
    )
    assert loss.item() == pytest.approx(0.25, abs=1e-6)


def test_semantic_spatial_checkpoint_round_trips(tmp_path):
    model = PredictiveCortex(
        (16, 16, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, hidden_dim=12, reconstruction_size=16, semantic_classes=4),
    )
    path = tmp_path / "spatial.pt"
    save_action_world_model(str(path), model, {})
    restored, _stats = load_action_world_model(str(path))
    assert restored.config.semantic_classes == 4
    frame = torch.rand(1, 3, 16, 16)
    assert restored.decode_prediction(
        torch.zeros(1, 8), reference_frame=frame,
        reference_spatial=restored.encode_visual(frame).spatial,
    )["semantic_logits"].shape == (1, 4, 9, 9)


def test_semantic_fixture_learns_deterministic_grid_location():
    torch.manual_seed(4)
    model = PredictiveCortex(
        (16, 16, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, hidden_dim=12, reconstruction_size=16, semantic_classes=3),
    )
    frame = torch.zeros(1, 3, 16, 16)
    latent = torch.zeros(1, 8)
    target = torch.zeros(1, 9, 9, dtype=torch.long)
    target[:, 4, 6] = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(80):
        optimizer.zero_grad()
        output = model.decode_prediction(
            latent, reference_frame=frame, reference_spatial=model.encode_visual(frame).spatial,
        )
        per_cell = torch.nn.functional.cross_entropy(output["semantic_logits"], target, reduction="none")
        weights = torch.ones_like(per_cell)
        weights[:, 4, 6] = 100.0
        loss = (per_cell * weights).mean()
        loss.backward()
        optimizer.step()
    assert output["semantic_logits"].argmax(1)[0, 4, 6].item() == 2


def test_static_square_stays_localized_without_a_predicted_change():
    model = PredictiveCortex(
        (32, 32, 3), ["wait"],
        PredictiveCortexConfig(latent_width=8, reconstruction_size=32),
    )
    frame = torch.zeros(1, 3, 32, 32)
    frame[:, :, 10:14, 18:22] = 1.0
    with torch.no_grad():
        model.decoder.mask_head.weight.zero_()
        model.decoder.mask_head.bias.fill_(-100.0)
    output = model.decode_prediction(
        torch.zeros(1, 8), reference_frame=frame,
        reference_spatial=model.encode_visual(frame).spatial,
    )["vision"]
    assert torch.equal(output, frame)
