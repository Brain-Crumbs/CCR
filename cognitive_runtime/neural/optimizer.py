"""Online-optimizer contract (Phase A: interface only).

:class:`OnlineOptimizer` owns everything the neural modules in this package
need to actually learn online: computing losses across whatever combination
of :class:`~cognitive_runtime.neural.encoder.StreamEncoderModule`,
:class:`~cognitive_runtime.neural.fusion.LatentFusionModel`,
:class:`~cognitive_runtime.neural.world_model.WorldModel`,
:class:`~cognitive_runtime.neural.policy.PolicyModel`, and
:class:`~cognitive_runtime.neural.value.ValueModel` a concrete agent wires
together, taking the gradient steps, clipping gradients, keeping target
networks in sync, and owning the optimizer checkpoint state -- the neural
counterpart of ``OnlineQModel``'s ``td_update``/``save``/``load``, generalized
to (possibly several) ``torch.optim.Optimizer`` instances over neural-network
parameters instead of one linear weight matrix.

No concrete loss/optimizer wiring is implemented here.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Mapping

import torch


class OnlineOptimizer(abc.ABC):
    """Owns losses, gradient steps, clipping, target networks, and optimizer
    checkpoint state for a concrete set of neural modules.

    Input/output shapes
    --------------------
    - :meth:`step` takes a ``batch`` mapping of named tensors (e.g.
      ``fused_latent``, ``action_onehot``, ``reward``, ``next_fused_latent``,
      ``done`` -- concrete subclasses document the exact keys and shapes
      their loss needs) and returns a ``Dict[str, float]`` of scalar loss /
      metric values (mirrors ``OnlineQModel.td_update``'s return of
      ``q_before``/``target``/``td_error``).
    - :meth:`sync_target_networks` takes and returns nothing; it copies (or
      Polyak-averages) online-network weights into whatever target-network
      copies the subclass keeps for training stability.

    Checkpoint keys
    ---------------
    :meth:`state_dict` returns:

    - ``"modules"``: ``{name: module.state_dict()}`` for every wrapped
      ``nn.Module`` (encoders, fusion, world model, policy, value).
    - ``"optimizers"``: ``{name: optimizer.state_dict()}`` for every wrapped
      ``torch.optim.Optimizer``.
    - ``"target_modules"``: ``{name: module.state_dict()}`` for target-network
      copies, if any.
    - ``"step"``: ``int``, the number of gradient steps taken so far.
    - ``"grad_clip_norm"``: ``float``, the gradient-clipping threshold in
      effect.

    :meth:`load_state_dict` restores exactly what :meth:`state_dict` returns.
    ``NeuralAgentCheckpoint`` serializes this to/from disk alongside
    compatibility metadata (layout hash, action keys, ...) as the neural
    analogue of ``OnlineQModel.save``/``OnlineQModel.load``.
    """

    @abc.abstractmethod
    def step(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        """Run one training step over a batch; return scalar loss metrics."""

    @abc.abstractmethod
    def sync_target_networks(self) -> None:
        """Copy (or Polyak-average) online weights into target networks."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Optimizer + module + target-network + step-count checkpoint
        state; see the class docstring for the exact keys."""

    @abc.abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state previously returned by :meth:`state_dict`."""
