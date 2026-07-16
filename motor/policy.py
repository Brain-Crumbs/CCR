"""Alternative voluntary controllers behind the ``motor.voluntary`` seam
(docs/v2/phases/phase-6-motor-system.md, issue #103): active-inference
decoding, a DreamerV3-style imagination actor trained inside dreams, and
the existing actor/critic policy head
(:mod:`cognitive_runtime.policies.actor_critic`).

Per the phase doc's design commitment, MPC (``motor/voluntary.py``) stays
the online spine; everything here is an A/B experiment kept off the
critical path -- ``choose`` is always pure inference under
``torch.no_grad``, and any learning (:meth:`ImaginationActor.train_on_dream`)
happens separately, driven from the sleep cycle (Phase 4/5), never from a
motor tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from cognitive_runtime.core.action import NULL_ACTION, Action
from motor.voluntary import CallableController, VoluntaryController

# --------------------------------------------------------------------------
# Active-inference decoding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveInferenceState:
    """The ``state`` a chooser built by :func:`build_active_inference_controller`
    expects: the cortex's current latent plus its backbone hidden state,
    threaded the same way ``PredictiveCortex.step`` takes them."""

    latent: torch.Tensor
    hidden: Any


def _encode_goal(cortex: Any, goal: Any) -> torch.Tensor:
    """Encode ``goal`` into a target latent -- the "encoder" leg of "T+1
    output -> encoder -> motor inverse path". Accepts either an
    already-encoded latent, passed through unchanged, or a raw H x W x C
    RGB pixel frame (``0..255`` per channel, matching
    ``pixel_stream_encoder.pixels_to_chw``'s convention), permuted into the
    encoder's expected N x C x H x W layout and normalized before
    encoding. Stays on ``goal``'s own device/dtype throughout -- no forced
    CPU round-trip, so a CUDA-resident goal encodes on-device."""
    if not isinstance(goal, torch.Tensor):
        raise ValueError("active-inference goal must be a tensor (pixel frame or latent)")
    if goal.dim() <= 2 and goal.shape[-1] == cortex.latent_width:
        return goal.reshape(1, -1)
    with torch.no_grad():
        chw = goal.permute(2, 0, 1).float() / 255.0
        return cortex.encoder(chw.unsqueeze(0))


def build_active_inference_controller(
    cortex: Any,
    action_keys: Sequence[str],
) -> VoluntaryController:
    """Decode the forecast into the action that fulfils it: for each
    candidate action, roll the cortex forward one step and score it by how
    close the *predicted* latent lands to the encoded *preferred* state
    (``goal``) -- active inference's expected free energy (minimizing
    predicted surprise relative to a preference), not MPC's maximized
    reward. Deterministic given the cortex and goal; runs under
    ``torch.no_grad`` -- nothing here learns.
    """
    vocabulary = {key: index for index, key in enumerate(action_keys)}

    def choose(state: ActiveInferenceState, actions: Sequence[Action], goal: Any = None) -> Action:
        if not actions:
            raise ValueError("voluntary action space must not be empty")
        if goal is None:
            raise ValueError("active-inference controller requires a preferred-state goal")
        target = _encode_goal(cortex, goal)
        with torch.no_grad():
            surprise = []
            for action in actions:
                index = vocabulary[action.key()]
                action_idx = torch.tensor([index], dtype=torch.long, device=state.latent.device)
                predicted, _ = cortex.step(state.latent, action_idx, state.hidden)
                surprise.append(float(F.mse_loss(predicted, target)))
        return actions[min(range(len(actions)), key=surprise.__getitem__)]

    return CallableController("active", choose)


# --------------------------------------------------------------------------
# DreamerV3-style imagination actor
# --------------------------------------------------------------------------


class ImaginationActor(torch.nn.Module):
    """A small actor/critic pair trained entirely on imagined (dreamed)
    rollouts through the frozen cortex -- Dreamer's "learn behaviour in
    latent imagination" applied to this cortex's ``rollout``/``heads``.

    It never touches a real observation or reward: only
    :meth:`train_on_dream`, called from the sleep cycle against a
    hippocampal seed (Phase 4/5 dreams), updates its weights. The cortex
    itself is never trained by this path -- its parameters are walked
    forward under ``torch.no_grad`` every imagined step.
    """

    def __init__(self, latent_width: int, n_actions: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(latent_width, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_actions),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(latent_width, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def act(self, latent: torch.Tensor) -> int:
        """Deterministic (argmax) action choice -- the inference path used
        by :func:`build_imagination_controller`; never mutates weights."""
        with torch.no_grad():
            logits = self.actor(latent)
            return int(torch.argmax(logits[0]).item())

    def train_on_dream(
        self,
        cortex: Any,
        seed_latent: torch.Tensor,
        hidden: Any,
        horizon: int,
        *,
        gamma: float = 0.99,
        generator: Optional[torch.Generator] = None,
    ) -> float:
        """One imagination rollout plus one actor/critic policy-gradient
        step. Samples its own actions from the current actor, rolls the
        cortex forward under ``torch.no_grad`` (frozen world model), reads
        the cortex's predicted reward at each imagined step
        (``cortex.heads``), and updates the actor toward higher-return
        imagined trajectories with a critic baseline -- Dreamer's
        imagined-rollout actor/critic loop, minus world-model learning.
        Returns the scalar loss for the step.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon!r}")
        latent = seed_latent
        state = hidden
        log_probs = []
        values = []
        rewards = []
        for _ in range(horizon):
            logits = self.actor(latent)
            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs[0], 1, generator=generator)
            log_probs.append(torch.log(probs[0, index].clamp_min(1e-8)))
            values.append(self.critic(latent)[0])
            with torch.no_grad():
                next_latent, state = cortex.step(latent, index.reshape(1), state)
                reward, _terminal, _risk, _uncertainty = cortex.heads(state)
            rewards.append(reward.detach())
            latent = next_latent

        returns = []
        running = torch.zeros_like(rewards[-1])
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.insert(0, running)
        returns_t = torch.cat(returns).detach()
        values_t = torch.cat(values)
        advantage = (returns_t - values_t).detach()

        policy_loss = -torch.stack(
            [log_prob * adv for log_prob, adv in zip(log_probs, advantage)]
        ).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def build_imagination_controller(
    actor: ImaginationActor,
    action_keys: Sequence[str],
) -> VoluntaryController:
    """The imagination actor as a ``voluntary`` controller. ``choose`` is
    pure inference (:meth:`ImaginationActor.act`) over the current latent
    ``state`` -- all learning happens in
    :meth:`ImaginationActor.train_on_dream`, off the online critical path.
    """

    def choose(state: torch.Tensor, actions: Sequence[Action], goal: Any = None) -> Action:
        if not actions:
            raise ValueError("voluntary action space must not be empty")
        key = action_keys[actor.act(state)]
        for action in actions:
            if action.key() == key:
                return action
        raise ValueError(f"imagination actor chose {key!r}, outside the offered action space")

    return CallableController("imagination", choose)


# --------------------------------------------------------------------------
# Actor/critic policy head
# --------------------------------------------------------------------------


def build_policy_controller(policy: Any) -> VoluntaryController:
    """Wire the existing actor/critic policy head
    (:class:`cognitive_runtime.policies.actor_critic.ActorCriticPolicy`)
    into the ``motor.voluntary`` seam. ``state`` is the
    ``(state, memory, prediction)`` triple ``ActorCriticPolicy.emit``
    already takes; ``actions``/``goal`` are unused -- the policy head
    reasons over its own configured action space, not a per-call
    candidate list. An empty ``emit`` result (the policy chose ``NULL``)
    maps to ``NULL_ACTION``, matching the runtime motor contract.
    """

    def choose(state: tuple, actions: Sequence[Action], goal: Any = None) -> Action:
        cognitive_state, memory, prediction = state
        chosen = policy.emit(cognitive_state, memory, prediction)
        return chosen[0] if chosen else NULL_ACTION

    return CallableController("policy", choose)
