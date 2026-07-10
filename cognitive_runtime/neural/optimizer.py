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
import copy
import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F

from cognitive_runtime.neural.policy import PolicyModel
from cognitive_runtime.neural.value import ValueModel
from cognitive_runtime.neural.world_model import WorldModel


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


class _RunningMeanStd:
    """Welford running mean/variance, used for reward normalization.

    Checkpointable via ``state_dict``/``load_state_dict`` so resuming a
    training run keeps the same normalization scale instead of resetting it.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, values: torch.Tensor) -> None:
        batch = values.detach().float().reshape(-1)
        if batch.numel() == 0:
            return
        batch_mean = float(batch.mean())
        batch_var = float(batch.var(unbiased=False)) if batch.numel() > 1 else 0.0
        batch_count = float(batch.numel())

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        std = (self.var + 1e-8) ** 0.5
        return (values - self.mean) / std

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mean = float(state.get("mean", self.mean))
        self.var = float(state.get("var", self.var))
        self.count = float(state.get("count", self.count))


class ActorCriticOptimizer(OnlineOptimizer):
    """Phase-E concrete online update for an actor/critic pair over the fused
    latent + world-model features (``docs/neural-stream-agent.md`` "Add
    Actor/Critic Online Learning").

    Batch keys
    ----------
    :meth:`step` expects, on top of the base contract's ``fused_latent``,
    ``action_onehot``, ``reward``, ``next_fused_latent``, ``done``:

    - ``world_features``/``next_world_features``: ``Tensor[batch,
      world_feature_width]`` -- the same world-model-derived features the
      policy/critic condition on
      (``cognitive_runtime.policies.actor_critic.world_features_vector``).
      Transitions replayed from :class:`~cognitive_runtime.neural.replay_buffer.ReplayBuffer`
      don't carry the live decision's world features (the buffer only stores
      fused-latent references), so callers fill these with zeros for
      replay-sourced batches -- the same graceful-degradation convention
      ``transition_priority`` uses for signals a transition doesn't carry.

    Update
    ------
    - Critic: TD(0) target ``reward_norm + gamma * (1 - done) * V_target(next)``,
      MSE value loss, advantage = target - value (optionally standardized).
    - Actor: ``-log pi(a) * advantage`` (REINFORCE with a learned baseline),
      minus an entropy bonus for exploration.
    - Optional joint world-model loss (next-latent + reward + terminal) when
      constructed with a ``world_model``, so encoders/fusion feeding it still
      receive *some* training signal from live transitions even though this
      optimizer does not itself hold/train separate encoder or fusion modules
      (transitions only carry fused-latent vectors, not raw per-stream
      latents, so per-encoder joint training needs richer transition storage
      than issue #28's replay buffer keeps -- future work).
    - Reward normalization (running mean/std), gradient clipping, and a
      Polyak-averaged target critic for a stable bootstrap target.
    """

    def __init__(
        self,
        policy: PolicyModel,
        critic: ValueModel,
        *,
        world_model: Optional[WorldModel] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        world_model_coef: float = 1.0,
        grad_clip_norm: float = 5.0,
        target_tau: float = 0.01,
        normalize_reward: bool = True,
        normalize_advantage: bool = False,
        seed: int = 0,
    ) -> None:
        """``seed`` seeds subsequent stochastic ops this optimizer performs
        (e.g. dropout masks during training) -- it does *not* retroactively
        seed ``policy``/``critic``/``world_model`` construction, which has
        already happened by the time they are passed in here; callers that
        need reproducible weight initialization must call
        ``torch.manual_seed(seed)`` before constructing those modules."""
        if not 0.0 < target_tau <= 1.0:
            raise ValueError(f"target_tau must be in (0, 1], got {target_tau!r}")
        if grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive, got {grad_clip_norm!r}")

        self.policy = policy
        self.critic = critic
        self.world_model = world_model
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.world_model_coef = float(world_model_coef)
        self.grad_clip_norm = float(grad_clip_norm)
        self.target_tau = float(target_tau)
        self.normalize_reward = normalize_reward
        self.normalize_advantage = normalize_advantage

        torch.manual_seed(seed)
        self.target_critic = copy.deepcopy(critic)
        self.target_critic.load_state_dict(critic.state_dict())
        self.target_critic.eval()

        self.optimizer = torch.optim.Adam(self._trainable_parameters(), lr=lr)
        self.reward_normalizer = _RunningMeanStd()
        self.step_count = 0

    def _trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        modules: List[torch.nn.Module] = [self.policy, self.critic]
        if self.world_model is not None:
            modules.append(self.world_model)
        return itertools.chain.from_iterable(module.parameters() for module in modules)

    def step(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        fused = batch["fused_latent"].float()
        next_fused = batch["next_fused_latent"].float()
        world_features = batch["world_features"].float()
        next_world_features = batch["next_world_features"].float()
        action_onehot = batch["action_onehot"].float()
        reward = batch["reward"].float()
        done = batch["done"].float()

        self.reward_normalizer.update(reward)
        reward_for_target = (
            self.reward_normalizer.normalize(reward) if self.normalize_reward else reward
        )

        with torch.no_grad():
            next_value = self.target_critic(next_fused, next_world_features)
            target = reward_for_target + self.gamma * (1.0 - done) * next_value

        value = self.critic(fused, world_features)
        value_loss = F.mse_loss(value, target)

        advantage = (target - value).detach()
        if self.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)

        logits = self.policy(fused, world_features)
        log_probs = F.log_softmax(logits, dim=-1)
        action_index = action_onehot.argmax(dim=-1)
        action_log_prob = log_probs.gather(1, action_index.unsqueeze(1)).squeeze(1)
        policy_loss = -(action_log_prob * advantage).mean()

        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        world_model_loss = torch.zeros(())
        if self.world_model is not None:
            wm_out = self.world_model(fused, action_onehot)
            next_latent_loss = F.mse_loss(wm_out.next_latent, next_fused)
            reward_loss = F.mse_loss(wm_out.reward, reward)
            terminal_loss = F.binary_cross_entropy_with_logits(wm_out.terminal_logit, done)
            world_model_loss = next_latent_loss + reward_loss + terminal_loss

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
            + self.world_model_coef * world_model_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self._trainable_parameters()), self.grad_clip_norm
        )
        self.optimizer.step()
        self.sync_target_networks()
        self.step_count += 1

        return {
            "total_loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "world_model_loss": float(world_model_loss.detach()),
            "advantage_mean": float(advantage.mean().detach()),
            "reward_mean": float(reward.mean().detach()),
            "target_mean": float(target.mean().detach()),
            "grad_norm": float(grad_norm),
        }

    def sync_target_networks(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.mul_(1.0 - self.target_tau).add_(param, alpha=self.target_tau)

    def state_dict(self) -> Dict[str, Any]:
        modules: Dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
        }
        if self.world_model is not None:
            modules["world_model"] = self.world_model.state_dict()
        return {
            "modules": modules,
            "optimizers": {"adam": self.optimizer.state_dict()},
            "target_modules": {"critic": self.target_critic.state_dict()},
            "step": self.step_count,
            "grad_clip_norm": self.grad_clip_norm,
            "reward_normalizer": self.reward_normalizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        modules = state["modules"]
        self.policy.load_state_dict(modules["policy"])
        self.critic.load_state_dict(modules["critic"])
        if self.world_model is not None and "world_model" in modules:
            self.world_model.load_state_dict(modules["world_model"])
        self.target_critic.load_state_dict(state["target_modules"]["critic"])
        self.optimizer.load_state_dict(state["optimizers"]["adam"])
        self.step_count = int(state.get("step", self.step_count))
        self.grad_clip_norm = float(state.get("grad_clip_norm", self.grad_clip_norm))
        if "reward_normalizer" in state:
            self.reward_normalizer.load_state_dict(state["reward_normalizer"])
