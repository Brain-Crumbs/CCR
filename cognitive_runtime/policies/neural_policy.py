"""Neural pixel-vision policy: the end-to-end BC model at inference.

Each tick it feeds the live ``vision.frame.pixels`` frame through the trained
CNN, alongside the fused non-vision vector and its own recent-motor history, and
emits the argmax action.  It reconstructs the non-vision vector from the latent
state the runtime already computed — concatenating every non-``vision.*`` stream
slice in the same stream-id order the trainer used — so train-time and
inference-time features come from the same fusion, with no catalog plumbing.

Imports torch (via :mod:`cognitive_runtime.models.vision`), so it is imported
lazily by the CLI only when the neural policy is selected; the rest of the
runtime stays torch-free.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.streams.fusion import LatentState
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.models.vision import VisionBCModel
from cognitive_runtime.programs.minecraft.streams import PIXEL_STREAM
from cognitive_runtime.training.features import motor_history_features

_VISION_PREFIX = "vision."


def non_vision_features(latent: LatentState) -> Tuple[List[float], List[str]]:
    """The non-vision half of a fused latent: (vector, slot names).

    Mirrors ``TemporalFusion.feature_names``/layout ordering, dropping every
    ``vision.*`` stream so the CNN is the sole visual pathway — the exact
    counterpart of ``datasets._non_vision_fusion`` used at train time.
    """
    vector: List[float] = []
    names: List[str] = []
    for stream_id in sorted(latent.slices):
        if stream_id.startswith(_VISION_PREFIX):
            continue
        lo, hi = latent.slices[stream_id]
        vector.extend(latent.vector[lo:hi])
        width = hi - lo
        if width == 1:
            names.append(stream_id)
        else:
            names.extend(f"{stream_id}[{i}]" for i in range(width))
    return vector, names


class NeuralPolicy(SingleActionPolicy):
    name = "neural"

    def __init__(self, model: VisionBCModel | str, history: int = 8):
        self.model = VisionBCModel.load(model) if isinstance(model, str) else model
        self.history = history
        self._recent: Deque[str] = deque(maxlen=history)
        self._expected_names = self.model.meta.get("non_vision_names")

    def reset(self) -> None:
        self._recent.clear()

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        latent = memory.fused_latent()
        if latent is None:
            raise RuntimeError(
                "neural policy needs the fused LatentState, but the runtime "
                "produced none this tick"
            )
        non_vision, names = non_vision_features(latent)
        if self._expected_names is not None and names != self._expected_names:
            raise ValueError(
                "non-vision layout mismatch: the model was trained on a different "
                "stream catalog; re-train or align the program config"
            )

        latest_pixels = memory.buffer.latest(PIXEL_STREAM)
        if latest_pixels is None:
            raise RuntimeError(
                f"neural policy needs the {PIXEL_STREAM} stream; none has arrived. "
                "Is the program publishing pixel frames?"
            )
        frame = latest_pixels.payload
        if isinstance(frame, np.ndarray):
            shape = tuple(frame.shape)
        else:
            shape = (len(frame), len(frame[0]) if frame else 0,
                     len(frame[0][0]) if frame and frame[0] else 0)
        if shape != self.model.pixel_shape:
            raise ValueError(
                f"pixel-frame shape {shape} != model's {self.model.pixel_shape}; "
                "re-train or align the render geometry"
            )

        key = self.model.predict_key(frame, non_vision, motor_history_features(list(self._recent)))
        self._recent.append(key)
        return Action.from_key(key)
