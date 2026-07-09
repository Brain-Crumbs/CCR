"""cognitive_runtime.neural: torch-optional contracts (issue #19).

Two things must hold regardless of whether torch is installed in the test
environment:

1. The default runtime surface (``import cognitive_runtime``, the runtime
   loop, replay, and the CLI's ``run --policy scripted``) never imports
   torch — verified in a subprocess with a real, unmodified environment,
   mirroring ``test_core_and_runtime_do_not_import_programs`` in
   ``tests/test_runtime_streams.py``.
2. Importing ``cognitive_runtime.neural`` without torch installed raises a
   clear, actionable ``ImportError`` — verified in a subprocess that shadows
   ``torch`` via ``sys.modules`` so the test is meaningful even when this
   environment does have torch installed.

The contract classes themselves (``StreamEncoderModule``, etc.) need torch to
import at all, so their shape/ABC-enforcement smoke tests are skipped when
torch is unavailable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_core_runtime_and_scripted_cli_never_import_torch(tmp_path):
    code = (
        "import sys; "
        "import cognitive_runtime; "
        "import cognitive_runtime.runtime.loop; "
        "import cognitive_runtime.runtime.replay; "
        "from cognitive_runtime.cli import main; "
        f"main(['run', '--policy', 'scripted', '--episodes', '1', "
        f"'--episode-ticks', '20', '--world-size', '16', '--no-record']); "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)


def test_neural_package_import_error_is_actionable_without_torch():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # forces `import torch` to raise ImportError
        "try:\n"
        "    import cognitive_runtime.neural\n"
        "except ImportError as exc:\n"
        "    msg = str(exc)\n"
        "    assert 'pip install' in msg and '.[neural]' in msg, msg\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------ contract shape (torch)

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.neural import (  # noqa: E402
    LatentFusionModel,
    OnlineOptimizer,
    PolicyModel,
    StreamEncoderModule,
    ValueModel,
    WorldModel,
    WorldModelOutput,
)


class _ToyEncoder(StreamEncoderModule):
    def width(self, spec=None):
        return 3

    def encode_latent(self, events, spec=None):
        if not events:
            return None
        return torch.zeros(3)

    def predict_next_latent(self, latent_slice):
        return {"next": latent_slice}


def test_stream_encoder_module_bridges_to_latent_token_and_predict_next():
    encoder = _ToyEncoder()
    event = StreamEvent("body.health", "body", 0.0, 0, 20.0)

    token = encoder.encode([event])
    assert token is not None
    assert token.stream_id == "body.health"
    assert token.vector == [0.0, 0.0, 0.0]
    assert encoder.encode([]) is None

    prediction = encoder.predict_next([0.0, 0.0, 0.0])
    assert prediction["next"] == [0.0, 0.0, 0.0]

    encoder.train_mode()
    assert encoder.training is True
    encoder.eval_mode()
    assert encoder.training is False

    assert encoder.state_dict() == {}  # no registered parameters in this toy
    with pytest.raises(NotImplementedError):
        encoder.update({"loss": 1.0})


def test_stream_encoder_module_rejects_wrong_shaped_latent():
    class BadEncoder(_ToyEncoder):
        def encode_latent(self, events, spec=None):
            return torch.zeros(2)  # width() says 3

    with pytest.raises(ValueError, match="shape"):
        BadEncoder().encode([StreamEvent("body.health", "body", 0.0, 0, 20.0)])


@pytest.mark.parametrize(
    "abstract_cls",
    [StreamEncoderModule, LatentFusionModel, WorldModel, PolicyModel, ValueModel],
)
def test_neural_module_contracts_are_not_directly_instantiable(abstract_cls):
    with pytest.raises(TypeError):
        abstract_cls()


def test_online_optimizer_is_abstract():
    with pytest.raises(TypeError):
        OnlineOptimizer()


def test_world_model_output_is_a_plain_named_tuple_of_tensors():
    zeros = torch.zeros(4)
    scalar = torch.zeros(())
    output = WorldModelOutput(
        next_latent=zeros, reward=scalar, terminal_logit=scalar,
        risk=scalar, prediction_error=scalar,
    )
    assert torch.equal(output.next_latent, zeros)
    assert output.reward.shape == ()
