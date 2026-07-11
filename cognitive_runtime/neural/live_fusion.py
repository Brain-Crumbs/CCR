"""Wires trainable stream encoders + :class:`LatentFusionModel` into the live
policy path (issue #57, ``docs/neural-stream-agent.md`` Phase C's remaining
bridge).

``TemporalFusion`` (``core.streams.fusion``) concatenates each stream's fixed,
hand-written encoder output into one versioned vector -- that stays the
``--fusion fixed`` default. :class:`LiveLearnedFusion` is the ``--fusion
learned`` alternative: it builds a second ``TemporalFusion``-shaped layout,
but binds each stream to the trainable module its
:class:`~cognitive_runtime.core.streams.registry.StreamDeclaration` already
names (``neural_encoder``, issue #21) instead of the legacy fixed encoder,
then runs :class:`LatentFusionModel` over those trainable latents to produce
the fused agent state ``ActorCriticPolicy`` consumes -- exactly the same shape
``memory.fused_latent()`` already has, so no downstream consumer needs to
know which mode produced it.

Training (issue #37's async trainer eventually owns this) happens here as a
small synchronous, self-supervised online update: each tick, a reward-
prediction head regresses the fused state against the window's training
reward, backpropagating into both the fusion model and every trainable
stream encoder feeding it. This is deliberately modest -- a placeholder
online objective, not the full representation-learning loss suite
``docs/neural-stream-agent.md`` describes -- kept behind the same
``ActorCriticLearner``/``NeuralAgentCheckpoint`` interfaces so the async
trainer can replace it later without another interface change.
"""

from __future__ import annotations

import importlib
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cognitive_runtime.core.hashing import canonical_json
from cognitive_runtime.core.streams.encoder_registry import StreamEncoderRegistry
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.fusion import LatentState, TemporalFusion
from cognitive_runtime.core.streams.registry import StreamRegistry
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer
from cognitive_runtime.neural.fusion import LatentFusionModel, latent_fusion_inputs_from_buffer


def _resolve_neural_encoder_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid neural_encoder path {dotted_path!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"neural_encoder {dotted_path!r} has no class {class_name!r}") from exc


def build_trainable_encoder_registry(
    catalog: Iterable[StreamSpec],
    stream_registry: StreamRegistry,
) -> Tuple[StreamEncoderRegistry, Dict[str, nn.Module]]:
    """A :class:`StreamEncoderRegistry` mirroring
    ``stream_registry.to_encoder_registry()`` except every stream declared
    ``train_eval_behavior == "trainable"`` binds to the real module its
    ``neural_encoder`` names (``cognitive_runtime.neural.BodyStateEncoder``
    etc.) instead of the legacy fixed encoder. "fixed" declarations keep
    their existing encoder unchanged; "raw" declarations keep having no
    fusion-layout slot -- identical to the fixed path in both cases.

    Returns ``(registry, encoders)`` where ``encoders`` maps each trainable
    stream's checkpoint key to its module instance, ready to hand to
    :class:`~cognitive_runtime.neural.checkpoint.NeuralAgentCheckpoint`'s
    ``encoders=`` argument. One module instance is created per concrete
    stream id (each stream learns its own weights), keyed and cached by
    ``StreamDeclaration.resolve_checkpoint_key``.
    """
    registry = StreamEncoderRegistry()
    encoders: Dict[str, nn.Module] = {}
    for spec in sorted(catalog, key=lambda s: s.stream_id):
        decl = stream_registry.declaration_for(spec.stream_id)
        if decl is None:
            raise ValueError(
                f"stream {spec.stream_id!r} has no StreamDeclaration; call "
                "stream_registry.assert_complete(catalog) before building "
                "the trainable encoder registry"
            )
        if decl.train_eval_behavior == "trainable" and decl.neural_encoder:
            key = decl.resolve_checkpoint_key(spec.stream_id)
            module = encoders.get(key)
            if module is None:
                cls = _resolve_neural_encoder_class(decl.neural_encoder)
                kwargs = {"latent_width": decl.neural_latent_width} if decl.neural_latent_width else {}
                module = cls(**kwargs)
                encoders[key] = module
            registry.register(spec.stream_id, module)
        elif decl.train_eval_behavior == "fixed":
            registry.register(spec.stream_id, decl.encoder())
        else:
            registry.register(spec.stream_id, None)
    return registry, encoders


def learned_fusion_layout_hash(base_layout_hash: str, learned_layout_hash: str, fused_width: int) -> str:
    """Compatibility hash for a learned-fusion actor/critic checkpoint.

    Deliberately distinct from the plain ``TemporalFusion.layout_hash`` a
    ``--fusion fixed`` run uses (folds in the fixed layout, the trainable
    layout, the fusion mode tag, and the fused width) so a checkpoint trained
    under one fusion mode fails loudly (``CheckpointCompatibilityError``) if
    loaded under the other, rather than silently loading a policy/critic that
    expects a differently-shaped, differently-meaning input vector.
    """
    import hashlib

    blob = canonical_json(
        ["learned-fusion-actor-critic-v1", base_layout_hash, learned_layout_hash, int(fused_width)]
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class LiveLearnedFusionModule(nn.Module):
    """Checkpointable bundle: the fusion model plus its online reward-
    prediction head, saved/loaded as one ``NeuralAgentCheckpoint`` module."""

    def __init__(self, fusion: LatentFusionModel) -> None:
        super().__init__()
        self.fusion = fusion
        self.reward_head = nn.Linear(fusion.fused_width(), 1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.fusion(*args, **kwargs)

    def fused_width(self) -> int:
        return self.fusion.fused_width()

    def checkpoint_metadata(self) -> Dict[str, object]:
        metadata = dict(self.fusion.checkpoint_metadata())
        metadata["aux_heads"] = ["reward_head"]
        return metadata


class LiveLearnedFusion:
    """The live ``--fusion learned`` pipeline: trainable stream encoders feed
    :class:`LatentFusionModel`, producing the fused agent state the runtime
    loop stores into ``Memory`` every tick exactly where the fixed
    ``TemporalFusion`` output used to go (``CognitiveRuntime.learned_fusion``,
    ``runtime/loop.py``).
    """

    def __init__(
        self,
        catalog: Iterable[StreamSpec],
        stream_registry: StreamRegistry,
        *,
        base_layout_hash: str,
        fused_width: Optional[int] = None,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
        lr: float = 1e-3,
        window: int = 8,
        half_life_seconds: float = 1.0,
    ) -> None:
        catalog = list(catalog)
        self.encoder_registry, self.encoders = build_trainable_encoder_registry(
            catalog, stream_registry
        )
        self.temporal_fusion = TemporalFusion(
            catalog, self.encoder_registry, window=window, half_life_seconds=half_life_seconds
        )
        fusion_model = LatentFusionModel.from_temporal_fusion(
            self.temporal_fusion,
            fused_width=fused_width,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.module = LiveLearnedFusionModule(fusion_model)
        self.layout_hash = learned_fusion_layout_hash(
            base_layout_hash, self.temporal_fusion.layout_hash, self.module.fused_width()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.training = True
        self.last_metrics: Dict[str, float] = {}

    def fused_width(self) -> int:
        return self.module.fused_width()

    def parameters(self) -> Iterator[nn.Parameter]:
        return chain(self.module.parameters(), *(m.parameters() for m in self.encoders.values()))

    def train_mode(self) -> None:
        self.training = True
        self.module.train()
        for module in self.encoders.values():
            train_mode = getattr(module, "train_mode", None)
            (train_mode or module.train)()

    def eval_mode(self) -> None:
        self.training = False
        self.module.eval()
        for module in self.encoders.values():
            eval_mode = getattr(module, "eval_mode", None)
            (eval_mode or module.eval)()

    def fuse(
        self,
        window: Optional[TickWindow],
        buffer: TemporalBuffer,
        attention_weights: Optional[Dict[str, float]] = None,
    ) -> LatentState:
        """Fused agent state for the current tick (no gradient -- inference
        only; see :meth:`maybe_train_step` for the online update).

        `attention_weights` (issue #59) is the same per-stream hook
        `LatentFusionModel.forward` takes; omitting it defaults every stream
        to `1.0` (uniform), byte-equivalent to no attention controller.
        """
        inputs = latent_fusion_inputs_from_buffer(
            self.temporal_fusion, buffer, tick_window=window, attention_weights=attention_weights
        )
        was_training = self.module.training
        self.module.eval()
        with torch.no_grad():
            fused = self.module(
                inputs.latents, inputs.presence_mask, inputs.recency, inputs.staleness, inputs.attention
            )
        if was_training:
            self.module.train()
        return LatentState(vector=fused[0].tolist(), slices={}, layout_hash=self.layout_hash)

    def _grad_latents(self, buffer: TemporalBuffer) -> torch.Tensor:
        """Per-stream latents with gradient preserved through every
        trainable encoder, for :meth:`maybe_train_step` -- unlike
        :meth:`fuse` (and unlike ``TemporalFusion.fuse``, which always
        detaches into plain floats), this is the one path that actually lets
        gradients reach the stream encoders."""
        fusion = self.temporal_fusion
        reference_time = fusion._reference_time(buffer)
        parts: List[torch.Tensor] = []
        for entry in fusion.layout:
            events = buffer.window(entry.stream_id, fusion.window)
            if entry.modality == "event":
                recency = fusion._event_recency(events[-1].timestamp, reference_time) if events else 0.0
                tensor = torch.tensor([recency], dtype=torch.float32)
            else:
                encode_latent = getattr(entry.encoder, "encode_latent", None)
                tensor = encode_latent(events, entry.spec) if callable(encode_latent) and events else None
                if tensor is None:
                    if events:
                        token = entry.encoder.encode(events, entry.spec)
                        vec = token.vector if token is not None else entry.encoder.neutral(entry.spec)
                    else:
                        vec = entry.encoder.neutral(entry.spec)
                    tensor = torch.tensor(vec, dtype=torch.float32)
            parts.append(tensor.reshape(entry.width))
        if not parts:
            return torch.zeros((1, 0), dtype=torch.float32)
        return torch.cat(parts).unsqueeze(0)

    def maybe_train_step(
        self,
        window: Optional[TickWindow],
        buffer: TemporalBuffer,
        *,
        reward: float,
        attention_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """One synchronous online gradient step (issue #57's stand-in for the
        async trainer, #37): regress the reward-prediction head over the
        freshly-fused state against this tick's training reward, updating
        every trainable stream encoder and the fusion model together.

        No-ops (returns ``{}``) when not in training mode.
        """
        if not self.training:
            return {}
        inputs = latent_fusion_inputs_from_buffer(
            self.temporal_fusion, buffer, tick_window=window, attention_weights=attention_weights
        )
        latents = self._grad_latents(buffer)
        self.module.train()
        fused = self.module(latents, inputs.presence_mask, inputs.recency, inputs.staleness, inputs.attention)
        reward_pred = self.module.reward_head(fused).squeeze(-1)
        target = torch.tensor([reward], dtype=torch.float32)
        loss = F.mse_loss(reward_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_metrics = {"live_fusion_reward_loss": float(loss.detach())}
        return self.last_metrics
