"""Trainable per-stream encoder contract (Phase A: interface only).

:class:`StreamEncoderModule` is the neural counterpart of
``cognitive_runtime.core.streams.trainable.TrainableStreamModule``: it is a
:class:`torch.nn.Module` that also satisfies the ``StreamEncoder`` contract
(``encode``, ``width``, ``neutral``) the fixed-layout ``TemporalFusion`` uses
today, plus the trainable hooks (``predict_next``, ``update``, ``state_dict``,
``train_mode``/``eval_mode``) ``TrainableStreamModule`` reserves for future
learned modules.

No concrete encoder (CNN, RNN, transformer, ...) is implemented here; this is
the abstract shape every future stream encoder must fill in.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn

from cognitive_runtime.core.streams.encoder_registry import LatentToken
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.trainable import TrainableStreamModule


class StreamEncoderModule(nn.Module, TrainableStreamModule, abc.ABC):
    """A learned, fixed-width encoder for one stream.

    Input/output shapes
    --------------------
    - :meth:`encode_latent` takes the same ``events``/``spec`` window a fixed
      ``StreamEncoder`` gets, and returns a 1-D ``Tensor[width(spec)]`` (or
      ``None`` if the window has nothing usable) -- the per-stream latent
      slice ``TemporalFusion`` will concatenate into the fused vector.
    - :meth:`predict_next_latent` takes that same ``Tensor[width]`` latent
      slice and returns a dict of named tensors describing this stream's own
      local forecast, e.g. ``{"next": Tensor[width], "risk": Tensor[]}``
      (scalar tensors are 0-D). Concrete subclasses document the exact keys
      and shapes they produce; an encoder with nothing to predict may return
      an empty dict.

    Checkpoint keys
    ---------------
    :meth:`state_dict`/:meth:`load_state_dict` resolve to
    :class:`torch.nn.Module`'s own (registered parameters and buffers), which
    satisfies ``TrainableStreamModule``'s reserved hooks with real trainable
    state instead of the empty dict ``FixedStreamModule`` returns.
    :meth:`checkpoint_payload` (inherited from ``TrainableStreamModule``)
    wraps that ``state_dict()`` alongside ``checkpoint_metadata()`` --
    concrete subclasses should extend :meth:`checkpoint_metadata` with
    whatever shape/config fields (e.g. ``width``, ``spec`` hash) a loader
    needs to validate compatibility before restoring weights.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        """Fixed latent width this encoder emits for ``spec``."""

    @abc.abstractmethod
    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        """Encode a stream event window into a ``Tensor[width(spec)]`` latent,
        or ``None`` if the window carries nothing usable (mirrors
        ``StreamEncoder.encode`` returning ``None``)."""

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        latent = self.encode_latent(events, spec)
        if latent is None:
            return None
        expected = self.width(spec)
        if latent.shape != (expected,):
            raise ValueError(
                f"{type(self).__name__}.encode_latent produced shape "
                f"{tuple(latent.shape)}, expected ({expected},)"
            )
        latest = events[-1]
        return LatentToken(
            stream_id=latest.stream_id,
            modality=latest.modality,
            timestamp=latest.timestamp,
            vector=latent.detach().to("cpu").tolist(),
        )

    @abc.abstractmethod
    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stream-local forecast from this stream's own latent slice.

        Returns named tensors (e.g. ``{"next": Tensor[width], "risk":
        Tensor[]}``); return an empty dict if this encoder predicts nothing.
        """

    def predict_next(self, latent_slice) -> Dict[str, Any]:
        tensor = torch.as_tensor(latent_slice, dtype=torch.float32)
        prediction = self.predict_next_latent(tensor)
        return {key: value.detach().to("cpu").tolist() for key, value in prediction.items()}

    def update(self, loss_signal: Mapping[str, Any]) -> Dict[str, float]:
        raise NotImplementedError(
            f"{type(self).__name__}.update is not used in Phase A: gradient "
            "steps for neural stream modules are owned by an OnlineOptimizer, "
            "which holds the optimizer(s) and calls .backward()/.step() over "
            "this module's parameters directly rather than routing through "
            "a per-module loss_signal."
        )

    def train_mode(self) -> None:
        self.train()

    def eval_mode(self) -> None:
        self.eval()
