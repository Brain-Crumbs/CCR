"""Async actor/learner split: the background trainer (issue #37).

``AsyncTrainer`` is the whole "learner" side of the split: it owns the
actor/critic optimizer, the replay buffer minibatches are drawn from, and
checkpoint/weight-publication writes.  It reads transitions from up to two
sources through the *same* :class:`~cognitive_runtime.neural.replay_buffer.ReplayBuffer`
-- "the same dataloader interface reads (a) the live experience queue and
(b) recorded sessions on disk. Pretraining from recordings and live
training are the same code path with different source mixes":

- :meth:`AsyncTrainer.load_recorded_sessions` replays session directories
  into the buffer once, up front (offline pretraining source).
- :meth:`AsyncTrainer.ingest_live` drains a
  :class:`~cognitive_runtime.neural.experience_queue.SharedExperienceRing`
  into the same buffer, called every loop iteration (live source).

Neither source is required: a trainer built with only ``session_dirs`` and
no ``live_ring_handle`` performs pure offline pretraining ("the same
trainer, pointed only at recorded sessions with no live actor"); a trainer
built with only a live ring trains purely online.

``spawn_trainer_process`` runs an ``AsyncTrainer`` in its own OS process
(not a thread -- the GIL would serialize its gradient steps against the
realtime loop's Python bytecode) via :mod:`multiprocessing`.  Objects that
cross the process boundary are either plain data (the ``ActorCriticArch``
dataclass, paths, floats) or the explicit ``SharedExperienceRing.handle()``
protocol (see ``neural/experience_queue.py``) -- nothing here relies on
pickling live ``nn.Module``/``SharedMemory`` objects across ``spawn``.

Failure isolation: the actor never talks to the trainer process directly.
It writes to shared memory (``SharedExperienceRing.push``, non-blocking)
and polls a checkpoint file on disk
(:class:`~cognitive_runtime.neural.weight_publisher.WeightSubscriber`) --
neither operation depends on the trainer being alive.  If the trainer is
``kill -9``'d, those calls keep working exactly as before (empty polls,
full ring); :class:`TrainerSupervisor` restarts it, and
:meth:`AsyncTrainer.resume_if_checkpoint_exists` makes the fresh process
pick up training where the last published checkpoint left off.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint
from cognitive_runtime.neural.experience_queue import MP_CONTEXT, SharedExperienceRing
from cognitive_runtime.neural.optimizer import ActorCriticOptimizer
from cognitive_runtime.neural.policy import MLPPolicyModel
from cognitive_runtime.neural.replay_buffer import ReplayBuffer, load_session_into_buffer
from cognitive_runtime.neural.value import MLPValueModel
from sleep.weight_publisher import EMAWeightPublisher, WeightPublisher
from cognitive_runtime.neural.world_model import MLPWorldModel


@dataclass(frozen=True)
class ActorCriticArch:
    """Everything needed to rebuild the exact actor/critic module shapes a
    checkpoint was trained with -- the same fields
    ``cli._make_actor_critic_policy_and_learner`` stores under
    ``extra_metadata={"actor_critic": arch}``, so a checkpoint written by
    the CLI's synchronous learner and one written by ``AsyncTrainer`` are
    interchangeable."""

    fused_width: int
    world_feature_width: int
    n_actions: int
    action_keys: Tuple[str, ...]
    layout_hash: str
    hidden_dim: int = 128
    has_world_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fused_width": self.fused_width,
            "world_feature_width": self.world_feature_width,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "has_world_model": self.has_world_model,
        }


def build_actor_critic_modules(
    arch: ActorCriticArch,
    *,
    lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    grad_clip_norm: float = 5.0,
    seed: int = 0,
) -> Tuple[MLPPolicyModel, MLPValueModel, Optional[MLPWorldModel], ActorCriticOptimizer]:
    """Fresh (untrained, deterministically-seeded) actor/critic modules plus
    the optimizer wired over them -- mirrors the construction order
    ``cli._make_actor_critic_policy_and_learner`` uses so weights trained
    online by ``AsyncTrainer`` and weights trained by the synchronous CLI
    learner are bit-for-bit reproducible given the same seed."""
    torch.manual_seed(seed)
    policy_model = MLPPolicyModel(
        arch.fused_width, arch.world_feature_width, arch.n_actions,
        hidden_dim=arch.hidden_dim, layout_hash=arch.layout_hash,
        action_keys=arch.action_keys,
    )
    critic_model = MLPValueModel(
        arch.fused_width, arch.world_feature_width,
        hidden_dim=arch.hidden_dim, layout_hash=arch.layout_hash,
        action_keys=arch.action_keys,
    )
    world_model = None
    if arch.has_world_model:
        world_model = MLPWorldModel(
            arch.fused_width, arch.n_actions,
            hidden_dim=arch.hidden_dim, layout_hash=arch.layout_hash,
            action_keys=arch.action_keys,
        )
    optimizer = ActorCriticOptimizer(
        policy_model, critic_model, world_model=world_model,
        lr=lr, gamma=gamma, entropy_coef=entropy_coef,
        grad_clip_norm=grad_clip_norm, seed=seed,
    )
    return policy_model, critic_model, world_model, optimizer


class AsyncTrainer:
    """The learner half of the actor/learner split: owns the optimizer, the
    replay buffer, and checkpoint/weight-publication writes.  Runs entirely
    outside the realtime loop -- see module docstring."""

    def __init__(
        self,
        arch: ActorCriticArch,
        checkpoint_path: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip_norm: float = 5.0,
        seed: int = 0,
        replay_buffer: Optional[ReplayBuffer] = None,
        live_ring_handle: Optional[Dict[str, Any]] = None,
        session_dirs: Optional[Sequence[str]] = None,
        max_transitions_from_sessions: Optional[int] = None,
        min_episode_reward: Optional[float] = None,
        batch_size: int = 32,
        min_buffer_size: int = 1,
        publish_every_steps: int = 20,
        drain_max_items: Optional[int] = None,
        ema_decay: Optional[float] = None,
    ):
        if publish_every_steps <= 0:
            raise ValueError(f"publish_every_steps must be positive, got {publish_every_steps!r}")
        self.arch = arch
        self.policy_model, self.critic_model, self.world_model, self.optimizer = (
            build_actor_critic_modules(
                arch, lr=lr, gamma=gamma, entropy_coef=entropy_coef,
                grad_clip_norm=grad_clip_norm, seed=seed,
            )
        )
        self.checkpoint = NeuralAgentCheckpoint(
            checkpoint_path,
            layout_hash=arch.layout_hash,
            action_keys=arch.action_keys,
            policy=self.policy_model,
            critic=self.critic_model,
            online_optimizer=self.optimizer,
            extra_metadata={"actor_critic": arch.to_dict()},
        )
        # Concurrent schedule (issue #100): publish an EMA-averaged snapshot
        # -- a slow-moving target -- instead of raw in-training weights, so a
        # continuously-polling actor doesn't see tick-to-tick oscillation.
        # Phasic consolidation never publishes mid-update, so it has no
        # staleness/oscillation problem to begin with and leaves this unset.
        self.publisher = (
            EMAWeightPublisher(self.checkpoint, decay=ema_decay)
            if ema_decay is not None
            else WeightPublisher(self.checkpoint)
        )
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer()
        self.live_ring: Optional[SharedExperienceRing] = (
            SharedExperienceRing.attach(**live_ring_handle) if live_ring_handle else None
        )
        self.session_dirs = list(session_dirs) if session_dirs else []
        self.max_transitions_from_sessions = max_transitions_from_sessions
        self.min_episode_reward = min_episode_reward
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.publish_every_steps = publish_every_steps
        self.drain_max_items = drain_max_items

        self.resumed = False
        self.pretrain_transitions_loaded = 0
        self.total_live_ingested = 0
        self.last_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------- startup

    def resume_if_checkpoint_exists(self) -> bool:
        """Checkpoint ownership (issue #20): the trainer is the only writer,
        so on startup it is also the only thing that needs to check for a
        prior bundle and resume from it -- "restarted trainer resumes from
        checkpoint"."""
        import os

        if os.path.exists(self.checkpoint.path):
            self.checkpoint.load()
            self.resumed = True
        return self.resumed

    def load_recorded_sessions(self) -> int:
        """Offline-pretraining / warm-start source: replay recorded
        sessions into the same buffer live transitions feed."""
        if not self.session_dirs:
            return 0
        added = load_session_into_buffer(
            self.replay_buffer, self.session_dirs,
            max_transitions=self.max_transitions_from_sessions,
            min_episode_reward=self.min_episode_reward,
        )
        self.pretrain_transitions_loaded += added
        return added

    # --------------------------------------------------------------- steps

    def ingest_live(self) -> int:
        """Drain whatever the actor has pushed since the last call into the
        replay buffer.  A no-op (returns 0) for an offline-only trainer."""
        if self.live_ring is None:
            return 0
        transitions = self.live_ring.drain(self.drain_max_items)
        for transition in transitions:
            self.replay_buffer.add(transition)
        self.total_live_ingested += len(transitions)
        return len(transitions)

    def train_step(self) -> Optional[Dict[str, float]]:
        """One minibatch gradient step, or ``None`` if the buffer doesn't
        have ``min_buffer_size`` transitions yet."""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample_batch(batch_size, self.arch.n_actions)
        # The buffer only stores fused-latent references (issue #28), not
        # the live decision's world features; degrade to zero, the same
        # convention `ActorCriticLearner._replay_update` uses.
        zeros = torch.zeros(batch_size, self.arch.world_feature_width)
        batch["world_features"] = zeros
        batch["next_world_features"] = zeros.clone()
        self.last_metrics = self.optimizer.step(batch)
        return self.last_metrics

    def consolidate(self, steps: int) -> int:
        """Run one heavy, phasic sleep pass and publish only its final weights.

        The actor must be paused by :class:`sleep.PhasicSleepSchedule` while
        this method runs.  Intermediate optimizer states are deliberately not
        published, so an actor can observe either the pre-sleep snapshot or
        the complete post-consolidation snapshot, never a mid-update model.
        """
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps!r}")
        self.ingest_live()
        for _ in range(steps):
            if self.train_step() is None:
                break
        return self.publish(reason="consolidation")

    def publish(self, *, reason: str = "interval") -> int:
        """Weight publication + checkpoint write in one atomic file (see
        ``sleep/weight_publisher.py``)."""
        self.checkpoint.training_ticks = self.optimizer.step_count
        self.checkpoint.replay_metadata = self.replay_buffer.state_dict()
        self.checkpoint.training_stats = self.stats()
        return self.publisher.publish(reason=reason)

    def stats(self) -> Dict[str, Any]:
        return {
            "step_count": self.optimizer.step_count,
            "buffer_size": len(self.replay_buffer),
            "resumed": self.resumed,
            "pretrain_transitions_loaded": self.pretrain_transitions_loaded,
            "total_live_ingested": self.total_live_ingested,
            "live_ring": self.live_ring.stats().__dict__ if self.live_ring else None,
            "last_metrics": dict(self.last_metrics),
        }

    # ---------------------------------------------------------------- loop

    def run_forever(
        self,
        stop_event: "multiprocessing.synchronize.Event",
        *,
        max_steps: Optional[int] = None,
        idle_sleep_seconds: float = 0.05,
    ) -> Dict[str, Any]:
        """Train until ``stop_event`` is set or, for an offline-only trainer
        (no live ring -- nothing will ever grow the buffer on its own),
        until ``max_steps`` further gradient steps have run *this call*
        (relative to ``optimizer.step_count`` on entry, so re-running the
        same ``--steps N`` against a resumed checkpoint does another N
        steps rather than stopping immediately once the absolute count
        already exceeds ``N``).  Always publishes once more on the way out
        so a graceful stop leaves the latest weights on disk
        (``reason="shutdown"``); a ``kill -9`` skips this, which is exactly
        the failure the last periodic publish (every ``publish_every_steps``
        steps) exists to bound."""
        target_step_count = (
            None if max_steps is None else self.optimizer.step_count + max_steps
        )
        try:
            while not stop_event.is_set():
                ingested = self.ingest_live()
                metrics = self.train_step()
                if metrics is not None:
                    if self.optimizer.step_count % self.publish_every_steps == 0:
                        self.publish(reason="interval")
                    if (
                        target_step_count is not None
                        and self.optimizer.step_count >= target_step_count
                    ):
                        break
                elif self.live_ring is None:
                    # Offline pretraining with an empty/exhausted buffer and
                    # nothing left to arrive: there is nothing to wait for.
                    break
                if metrics is None and ingested == 0:
                    time.sleep(idle_sleep_seconds)
        finally:
            self.publish(reason="shutdown")
        return self.stats()


# ------------------------------------------------------------- process entry


def _trainer_process_main(
    arch: ActorCriticArch,
    checkpoint_path: str,
    live_ring_handle: Optional[Dict[str, Any]],
    session_dirs: Optional[List[str]],
    trainer_kwargs: Dict[str, Any],
    run_kwargs: Dict[str, Any],
    stop_event: "multiprocessing.synchronize.Event",
) -> None:
    trainer = AsyncTrainer(
        arch, checkpoint_path,
        live_ring_handle=live_ring_handle,
        session_dirs=session_dirs,
        **trainer_kwargs,
    )
    trainer.resume_if_checkpoint_exists()
    trainer.load_recorded_sessions()
    trainer.run_forever(stop_event, **run_kwargs)


def spawn_trainer_process(
    arch: ActorCriticArch,
    checkpoint_path: str,
    *,
    live_ring_handle: Optional[Dict[str, Any]] = None,
    session_dirs: Optional[Sequence[str]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
    idle_sleep_seconds: float = 0.05,
) -> Tuple[multiprocessing.Process, "multiprocessing.synchronize.Event"]:
    """Launch an :class:`AsyncTrainer` in its own process (issue #37:
    "separate process, not thread -- GIL").  Returns ``(process,
    stop_event)``: set ``stop_event`` and ``process.join()`` for a graceful
    shutdown, or ``process.kill()`` to simulate/handle a crash -- the actor
    process is unaffected either way (see module docstring)."""
    # Always "spawn" -- must match the context `SharedExperienceRing`'s own
    # `Lock`/`Value` were created under (see `neural/experience_queue.py`'s
    # `MP_CONTEXT`), and avoids inheriting any fork-unsafe thread state
    # (torch/OpenMP worker threads, a test runner's own threads) from
    # whatever process happens to be calling this.
    ctx = MP_CONTEXT
    stop_event = ctx.Event()
    process = ctx.Process(
        target=_trainer_process_main,
        args=(
            arch,
            checkpoint_path,
            live_ring_handle,
            list(session_dirs) if session_dirs else None,
            dict(trainer_kwargs or {}),
            {"max_steps": max_steps, "idle_sleep_seconds": idle_sleep_seconds},
            stop_event,
        ),
        daemon=True,
    )
    process.start()
    return process, stop_event


@dataclass
class TrainerSupervisor:
    """Restarts the trainer process if it dies unexpectedly (issue #37:
    "trainer crash must not kill the actor ... restarted trainer resumes
    from checkpoint").  The actor never depends on this -- it is purely so
    a live run keeps getting *new* weights after a trainer crash instead of
    forever serving the last snapshot; the actor itself already tolerates a
    trainer that never comes back (`WeightSubscriber.maybe_reload` just
    keeps returning ``None``).
    """

    arch: ActorCriticArch
    checkpoint_path: str
    live_ring_handle: Optional[Dict[str, Any]] = None
    session_dirs: Optional[Sequence[str]] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    restart_backoff_seconds: float = 1.0
    process: Optional[multiprocessing.Process] = field(default=None, init=False, repr=False)
    stop_event: Optional["multiprocessing.synchronize.Event"] = field(
        default=None, init=False, repr=False
    )
    restart_count: int = field(default=0, init=False)
    _stopping: bool = field(default=False, init=False, repr=False)
    _died_at: Optional[float] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._stopping = False
        self._died_at = None
        self.process, self.stop_event = spawn_trainer_process(
            self.arch, self.checkpoint_path,
            live_ring_handle=self.live_ring_handle,
            session_dirs=self.session_dirs,
            trainer_kwargs=self.trainer_kwargs,
        )

    def poll(self) -> bool:
        """Call periodically from a supervising process/thread -- including
        every tick of a realtime loop (issue #100 review: a concurrent
        schedule's whole point is that acting never pauses for the trainer,
        so this must never block; a synchronous ``restart_backoff_seconds``
        sleep here would itself cause missed ticks). Restarts the trainer
        once ``restart_backoff_seconds`` have elapsed since it was first
        observed dead -- each call is O(1) and returns immediately either
        way; returns whether a restart just happened."""
        if self._stopping or self.process is None:
            return False
        if self.process.is_alive():
            return False
        if self._died_at is None:
            self.process.join(timeout=0)
            self._died_at = time.time()
            return False
        if time.time() - self._died_at < self.restart_backoff_seconds:
            return False
        self.restart_count += 1
        self.start()
        return True

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stopping = True
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.terminate()
