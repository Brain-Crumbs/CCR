"""Thin dispatch wrapper around the unchanged legacy Factory lifecycle.

All pixel/action/nursery types stay on the existing execution path. Legacy
specs and checkpoint definitions acquire no new serialized default fields.
"""
from .task_registry import backend_registration, LEGACY_BACKEND


class CrafterBackend:
    identity = backend_registration(LEGACY_BACKEND).identity
    capabilities = backend_registration(LEGACY_BACKEND).capabilities

    def run(self, spec, **options):
        from .runner import _run_legacy_trial
        return _run_legacy_trial(spec, **options)

    @staticmethod
    def build_legacy_model(definition):
        from .checkpoint import _build_legacy_model
        return _build_legacy_model(definition)
