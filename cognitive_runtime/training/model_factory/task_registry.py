"""Explicit lazy backend registry. Import paths come from code, never a spec."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Any
from .task_contracts import TaskIdentity

LEGACY_BACKEND = "crafter"


@dataclass(frozen=True)
class BackendRegistration:
    identity: TaskIdentity
    factory: str | Callable[[], Any]
    capabilities: frozenset[str]
    extra: str | None = None


_REGISTRY: dict[str, BackendRegistration] = {}


def register_backend(registration: BackendRegistration) -> None:
    name = registration.identity.backend
    if name in _REGISTRY:
        raise ValueError(f"backend {name!r} already registered")
    _REGISTRY[name] = registration


def backend_registration(name: str) -> BackendRegistration:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown task/model backend {name!r}; registered: {sorted(_REGISTRY)}") from None


def load_backend(name: str):
    entry = backend_registration(name)
    factory = entry.factory
    try:
        if isinstance(factory, str):
            module, attribute = factory.split(":")
            factory = getattr(import_module(module), attribute)
        backend = factory()
    except ImportError as exc:
        hint = f"; install cognitive-runtime[{entry.extra}]" if entry.extra else ""
        raise ImportError(f"backend {name!r} is unavailable{hint}") from exc
    if backend.identity != entry.identity or backend.capabilities != entry.capabilities:
        raise ValueError(f"backend {name!r} declarations differ from its registry entry")
    return backend


register_backend(BackendRegistration(
    TaskIdentity("interactive-world", "action-world-model", LEGACY_BACKEND),
    "cognitive_runtime.training.model_factory.legacy_adapter:CrafterBackend",
    frozenset({"prepare", "fit", "predict", "evaluate", "save", "load", "resume", "clone", "fine_tune"}),
    "neural",
))
