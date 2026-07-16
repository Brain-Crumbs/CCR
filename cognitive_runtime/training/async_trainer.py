"""Compatibility shim for the Phase 5 sleep trainer.

The implementation moved to :mod:`sleep.async_trainer`.  This facade keeps
pre-Phase-5 imports working while new code uses the wake/sleep vocabulary.
"""

from sleep.async_trainer import (
    ActorCriticArch,
    AsyncTrainer,
    TrainerSupervisor,
    build_actor_critic_modules,
    spawn_trainer_process,
)

__all__ = [
    "ActorCriticArch",
    "AsyncTrainer",
    "TrainerSupervisor",
    "build_actor_critic_modules",
    "spawn_trainer_process",
]
