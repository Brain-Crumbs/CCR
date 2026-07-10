"""Neural world-model bridge: the trained ``MLPWorldModel`` behind the loop's
existing ``core.world_model.WorldModel`` seam (issue #26).

The runtime calls ``world_model.predict(state, memory)`` once per cognitive
tick, *before* the policy chooses this tick's action
(``runtime/loop.py``).  So, like the heuristic ``TrendWorldModel`` it can
replace, this bridge cannot condition on the action about to be taken -- it
conditions on the last action the runtime actually emitted (steady-state:
"if we keep doing what we just did, what happens next"), the same
information a curiosity/novelty consumer has available this tick.  No prior
action (episode start) falls back to an all-zero action one-hot.

Imports torch (via ``cognitive_runtime.neural``), so it is imported lazily by
the CLI, mirroring ``policies/neural_policy.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.core.world_model import WorldModel as CoreWorldModel
from cognitive_runtime.neural.world_model import MLPWorldModel
from cognitive_runtime.training.world_model import load_world_model_checkpoint


class NeuralWorldModel(CoreWorldModel):
    """Bridges a trained :class:`MLPWorldModel` into the loop's ``WorldModel``
    seam, so any policy already reading ``Prediction`` -- risk, and now the
    additive Phase-D fields -- can receive learned predictions with no loop
    changes.
    """

    def __init__(
        self,
        model: Union[MLPWorldModel, str],
        action_keys: Optional[Sequence[str]] = None,
    ):
        if isinstance(model, str):
            model, _metadata = load_world_model_checkpoint(model)
        self.model = model
        self.model.eval()
        keys = list(action_keys) if action_keys is not None else model.action_keys
        if not keys:
            raise ValueError(
                "NeuralWorldModel needs action_keys (the checkpoint carried none); "
                "pass the program's ordered action space explicitly"
            )
        self.action_keys = list(keys)
        self._action_index = {key: i for i, key in enumerate(self.action_keys)}

    def reset(self) -> None:
        pass

    def predict(self, state: State, memory: Memory) -> Prediction:
        latent = memory.fused_latent()
        if latent is None:
            # Nothing fused yet (first tick of an episode): no learned signal.
            return Prediction()

        fused_width = self.model.fused_width()
        if len(latent.vector) != fused_width:
            raise ValueError(
                f"fused latent width {len(latent.vector)} != world model's "
                f"{fused_width}; re-train or align the stream catalog"
            )
        fused = torch.tensor([latent.vector], dtype=torch.float32)

        action_onehot = torch.zeros(1, len(self.action_keys))
        last_actions = memory.last_actions(1)
        if last_actions:
            index = self._action_index.get(last_actions[-1].key())
            if index is not None:
                action_onehot[0, index] = 1.0

        with torch.no_grad():
            out = self.model(fused, action_onehot)

        return Prediction(
            risk=float(torch.sigmoid(out.risk)[0]),
            p_death=float(torch.sigmoid(out.terminal_logit)[0]),
            predicted_reward=float(out.reward[0]),
            next_latent=out.next_latent[0].tolist(),
            prediction_error=float(out.prediction_error[0]),
        )
