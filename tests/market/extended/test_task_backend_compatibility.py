"""Opt-in bounded real Factory/Crafter and neutral temporal compatibility.

Project-authored MIT synthetic inputs, seed 285. No downloads or live data.
"""
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.market_extended


def test_actual_legacy_factory_cli(tmp_path):
    # Same six-stage real integration canary as #297, through the new dispatch.
    pytest.importorskip("torch")
    subprocess.run([sys.executable, ".github/scripts/run_factory_smoke.py", "--output",
                    str(tmp_path / "smoke")], check=True, timeout=180)


def test_real_temporal_backbone_through_neutral_factory(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from brain.cortex.backbones import GRUBackbone
    from tests.test_model_factory_task_backends import FakeBackend, spec, continuation
    from cognitive_runtime.training.model_factory import task_registry as registry
    from cognitive_runtime.training.model_factory.runner import run_trial
    from cognitive_runtime.training.model_factory.checkpoint import load_factory_checkpoint
    from brain.cortex.model_contracts import require_inference_input

    class TemporalBackend(FakeBackend):
        def build(self, definition):
            with torch.random.fork_rng():
                torch.manual_seed(285)
                return GRUBackbone(input_dim=2, hidden_dim=2)

        def fit(self, model, data, configuration, control, *, resume):
            control.check()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.tensor(data.inputs.features).unsqueeze(1)
            hidden = model.forward_sequence(x)
            loss = (hidden.sum(-1) - torch.tensor(data.targets)).square().mean()
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}

        def predict(self, model, inputs):
            require_inference_input(inputs)
            x = torch.tensor(inputs.features)
            state = model.initial_state(len(x))
            hidden, state = model.step(x, state)
            assert torch.allclose(hidden, model.readout(state))
            assert torch.allclose(hidden, model.forward_sequence(x.unsqueeze(1))[:, -1])
            return tuple((float(v),) for v in hidden.detach().sum(-1))

        def save(self, model):
            return {k: v.tolist() for k, v in model.state_dict().items()}

        def load(self, definition, state, *, resume):
            model = self.build(definition)
            model.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
            return model

    backend = TemporalBackend()
    # This bounded smoke does not promise optimizer continuation: clone only.
    backend.capabilities = backend.capabilities - {"resume"}
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    registry.register_backend(registry.BackendRegistration(backend.identity, lambda: backend, backend.capabilities))
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = run_trial(spec(backend), root=tmp_path, run_id="temporal")
        assert result.state == "completed"
        restored = load_factory_checkpoint(result.checkpoint_path)
        assert isinstance(restored.model, GRUBackbone)
        clone = run_trial(continuation(spec(backend), result), root=tmp_path, run_id="temporal-clone")
        assert clone.state == "completed"
    finally:
        torch.set_num_threads(prior_threads)
