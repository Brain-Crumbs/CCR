"""Generative recall from a hippocampal seed.

A dream deliberately has no sensory-bus dependency: its complete input is the
stored latent and replayed action sequence.  The cortex is advanced in one
closed-loop :meth:`PredictiveCortex.rollout`, then each predicted latent is
decoded into the regenerated experience.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import torch

from brain.hippocampus import Seed
from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.training.prediction_export import _b64_frame, _session_name
from cognitive_runtime.training.visual_representation import reconstruction_target


def _model_device_and_dtype(cortex: Any) -> tuple[torch.device, torch.dtype]:
    parameter = next(cortex.parameters())
    return parameter.device, parameter.dtype


def _action_indices(seed: Seed, length: int, cortex: Any, device: torch.device) -> torch.Tensor:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length!r}")
    if length > len(seed.actions):
        raise ValueError(
            f"seed contains {len(seed.actions)} replay actions, fewer than dream length {length}"
        )
    vocabulary = {key: index for index, key in enumerate(cortex.action_keys)}
    missing = [action for action in seed.actions[:length] if action not in vocabulary]
    if missing:
        raise ValueError(
            f"seed action {missing[0]!r} is outside the cortex vocabulary {cortex.action_keys!r}"
        )
    return torch.tensor(
        [[vocabulary[action] for action in seed.actions[:length]]],
        dtype=torch.long,
        device=device,
    )


def dream_latents(
    seed: Seed,
    length: int,
    cortex: Any,
    hidden: Optional[Any] = None,
) -> torch.Tensor:
    """The undecoded counterpart of :func:`dream`: ``length`` predicted
    latents (``Tensor[length, latent_width]``) from ``seed``'s stored latent
    and replayed actions, with no pixel decode.

    Generative replay (``sleep.replay_mix``) trains directly against these
    latents -- decoding to pixel space would be pure overhead for a loss that
    never looks at pixels -- so this is factored out of :func:`dream` rather
    than having callers decode-then-discard.
    """
    device, dtype = _model_device_and_dtype(cortex)
    actions = _action_indices(seed, length, cortex, device)
    if length == 0:
        return torch.empty(0, cortex.latent_width, device=device, dtype=dtype)

    latent = torch.as_tensor(seed.z, dtype=dtype, device=device).reshape(1, -1)
    if latent.shape[1] != cortex.latent_width:
        raise ValueError(
            f"seed latent width {latent.shape[1]} does not match cortex width {cortex.latent_width}"
        )
    state = cortex.initial_state(1) if hidden is None else hidden
    was_training = cortex.training
    cortex.eval()
    try:
        with torch.no_grad():
            latents, _ = cortex.rollout(latent, actions, state)
        return latents.squeeze(0)
    finally:
        cortex.train(was_training)


def dream(
    seed: Seed,
    length: int,
    cortex: Any,
    hidden: Optional[Any] = None,
) -> Iterator[torch.Tensor]:
    """Yield ``length`` decoded frames from ``seed`` using replayed actions.

    ``hidden`` is an optional cortex-backbone state captured at the seed.  When
    it is unavailable, a fresh state is used.  No observation, sensory bus, or
    encoder is consulted: after construction the rollout is wholly generative.
    Returned frames have shape ``[C, H, W]`` in the cortex reconstruction space.
    """
    if length == 0:
        return
    latents = dream_latents(seed, length, cortex, hidden)
    was_training = cortex.training
    cortex.eval()
    try:
        with torch.no_grad():
            decoded = cortex.decoder(latents)
        for frame in decoded:
            yield frame
    finally:
        cortex.train(was_training)


def _actual_targets(actual_frames: Sequence[Any], reconstruction_shape: Sequence[int]) -> torch.Tensor:
    tensors = []
    for frame in actual_frames:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().cpu().float()
            if tensor.ndim != 3:
                raise ValueError("actual frames must have three dimensions")
            # Accept both viewer-native HWC and model-native CHW tensors.
            if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = pixels_to_chw(frame)
        tensors.append(tensor)
    if not tensors:
        raise ValueError("actual_frames must not be empty")
    return reconstruction_target(torch.stack(tensors), tuple(reconstruction_shape))


def export_dream_file(
    seed: Seed,
    cortex: Any,
    actual_frames: Sequence[Any],
    horizons: Sequence[int],
    out_path: Optional[str] = None,
    *,
    session_dir: Optional[str] = None,
    episode_id: str = "dream",
    name: Optional[str] = None,
    hidden: Optional[Any] = None,
) -> str:
    """Write dreamed-vs-actual horizons in ``pixel-predictions-v1`` format.

    ``actual_frames[0]`` is the seed-time frame and ``actual_frames[h]`` is
    the comparison for dreamed horizon ``h``.  Thus each prediction horizon
    contains the single start index ``t=0`` understood by the existing pixel
    horizon viewer.
    """
    selected = sorted({int(h) for h in horizons if int(h) > 0})
    if not selected:
        raise ValueError("horizons must contain at least one positive offset")
    max_horizon = selected[-1]
    if len(actual_frames) <= max_horizon:
        raise ValueError(
            f"actual_frames has {len(actual_frames)} frames, too short for horizon {max_horizon}"
        )

    dreamed = list(dream(seed, max_horizon, cortex, hidden))
    targets = _actual_targets(actual_frames, cortex.reconstruction_shape)
    payload = {
        "format": "pixel-predictions-v1",
        "session_id": os.path.basename(os.path.normpath(session_dir)) if session_dir else seed.source,
        "episode_id": episode_id,
        "horizons": selected,
        "prediction_shape": list(cortex.reconstruction_shape),
        "n_frames": len(actual_frames),
        "predictions": {
            str(h): {"frames": [_b64_frame(dreamed[h - 1].cpu())]} for h in selected
        },
        "targets": [_b64_frame(frame) for frame in targets],
    }

    if out_path is None:
        if session_dir is None:
            raise ValueError("out_path or session_dir is required")
        resolved_name = name or _session_name(session_dir)
        filename = (
            f"{resolved_name}-dream_{episode_id}.json"
            if resolved_name
            else f"dream_{episode_id}.json"
        )
        out_path = os.path.join(session_dir, filename)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return out_path
