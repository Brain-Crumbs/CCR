"""Dependency-free linear Q model for the first online-learning core.

Status: baseline only.  The target online learner is a neural actor/critic
(see docs/neural-stream-agent.md); this model stays as the comparison
baseline until the actor/critic reliably beats it, then gets deprecated.

The runtime loop does not use this model yet.  It is the pure model layer:
fixed layout checks, feature construction from fused latent state plus recent
motor history, epsilon-greedy action selection, TD updates, and atomic JSON
checkpoints.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from cognitive_runtime.core.streams.fusion import LatentState

FORMAT_VERSION = "online-q-v1"
BIAS_FEATURE_NAME = "bias"


def motor_history_features_for_actions(
    recent_action_keys: Sequence[str], action_keys: Sequence[str]
) -> List[float]:
    """One-hot of the most recent motor emission over this model's actions."""
    last_key = recent_action_keys[-1] if recent_action_keys else None
    return [1.0 if key == last_key else 0.0 for key in action_keys]


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _jsonable_rng_state(state: object) -> object:
    if isinstance(state, tuple):
        return [_jsonable_rng_state(v) for v in state]
    return state


def _tuple_rng_state(state: object) -> object:
    if isinstance(state, list):
        return tuple(_tuple_rng_state(v) for v in state)
    return state


@dataclass
class OnlineQModel:
    action_keys: List[str]
    latent_width: int
    layout_hash: str
    weights: List[List[float]]
    bias: List[float]
    lr: float = 0.02
    gamma: float = 0.99
    epsilon_start: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay_ticks: int = 50000
    td_clip: float = 5.0
    training_ticks: int = 0
    seed: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.action_keys:
            raise ValueError("OnlineQModel requires at least one action")
        if self.latent_width < 0:
            raise ValueError("latent_width must be non-negative")
        self._rng = random.Random(self.seed)
        expected_width = self.feature_width
        if not self.feature_names:
            self.feature_names = (
                [f"latent[{i}]" for i in range(self.latent_width)]
                + [f"last_action:{key}" for key in self.action_keys]
                + [BIAS_FEATURE_NAME]
            )
        if len(self.feature_names) != expected_width:
            raise ValueError(
                f"feature_names width {len(self.feature_names)} != expected {expected_width}"
            )
        if len(self.weights) != len(self.action_keys):
            raise ValueError(
                f"weights action rows {len(self.weights)} != action space "
                f"{len(self.action_keys)}"
            )
        bad_rows = [i for i, row in enumerate(self.weights) if len(row) != expected_width]
        if bad_rows:
            raise ValueError(
                f"weight row {bad_rows[0]} width {len(self.weights[bad_rows[0]])} "
                f"!= feature width {expected_width}"
            )
        if len(self.bias) != len(self.action_keys):
            raise ValueError(f"bias width {len(self.bias)} != action space {len(self.action_keys)}")

    @property
    def feature_width(self) -> int:
        return self.latent_width + len(self.action_keys) + 1

    @classmethod
    def initialize(
        cls,
        action_keys: Sequence[str],
        latent_width: int,
        layout_hash: str,
        *,
        lr: float = 0.02,
        gamma: float = 0.99,
        epsilon_start: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_decay_ticks: int = 50000,
        td_clip: float = 5.0,
        seed: int = 0,
        latent_feature_names: Optional[Sequence[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "OnlineQModel":
        actions = list(action_keys)
        width = latent_width + len(actions) + 1
        if latent_feature_names is None:
            feature_names = [f"latent[{i}]" for i in range(latent_width)]
        else:
            feature_names = list(latent_feature_names)
            if len(feature_names) != latent_width:
                raise ValueError(
                    f"latent_feature_names width {len(feature_names)} != latent_width "
                    f"{latent_width}"
                )
        feature_names += [f"last_action:{key}" for key in actions] + [BIAS_FEATURE_NAME]
        return cls(
            action_keys=actions,
            latent_width=latent_width,
            layout_hash=layout_hash,
            weights=[[0.0] * width for _ in actions],
            bias=[0.0] * len(actions),
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay_ticks=epsilon_decay_ticks,
            td_clip=td_clip,
            seed=seed,
            meta=dict(meta or {}),
            feature_names=feature_names,
        )

    @classmethod
    def from_latent(
        cls,
        latent: LatentState,
        action_keys: Sequence[str],
        **kwargs: Any,
    ) -> "OnlineQModel":
        return cls.initialize(
            action_keys=action_keys,
            latent_width=latent.width,
            layout_hash=latent.layout_hash,
            **kwargs,
        )

    def check_compatible(
        self,
        *,
        action_keys: Optional[Sequence[str]] = None,
        layout_hash: Optional[str] = None,
        latent_width: Optional[int] = None,
    ) -> None:
        if action_keys is not None and list(action_keys) != self.action_keys:
            raise ValueError(
                "online Q action-space mismatch: checkpoint has "
                f"{self.action_keys}, runtime has {list(action_keys)}"
            )
        if layout_hash is not None and layout_hash != self.layout_hash:
            raise ValueError(
                "online Q latent layout mismatch: checkpoint was trained on "
                f"{self.layout_hash} but runtime produced {layout_hash}"
            )
        if latent_width is not None and latent_width != self.latent_width:
            raise ValueError(
                f"online Q latent width mismatch: checkpoint expects {self.latent_width}, "
                f"runtime produced {latent_width}"
            )

    def current_epsilon(self) -> float:
        if self.epsilon_decay_ticks <= 0:
            return max(self.epsilon_min, self.epsilon_start)
        progress = min(self.training_ticks / float(self.epsilon_decay_ticks), 1.0)
        value = self.epsilon_start + (self.epsilon_min - self.epsilon_start) * progress
        return max(self.epsilon_min, value)

    def features(
        self,
        latent_vector: Sequence[float],
        recent_action_keys: Sequence[str],
    ) -> List[float]:
        if len(latent_vector) != self.latent_width:
            raise ValueError(
                f"online Q latent width mismatch: model expects {self.latent_width}, "
                f"got {len(latent_vector)}"
            )
        return (
            [float(v) for v in latent_vector]
            + motor_history_features_for_actions(recent_action_keys, self.action_keys)
            + [1.0]
        )

    def features_from_latent(
        self,
        latent: LatentState,
        recent_action_keys: Sequence[str],
    ) -> List[float]:
        self.check_compatible(layout_hash=latent.layout_hash, latent_width=latent.width)
        return self.features(latent.vector, recent_action_keys)

    def q_values(
        self,
        latent_vector: Sequence[float],
        recent_action_keys: Sequence[str],
    ) -> List[float]:
        x = self.features(latent_vector, recent_action_keys)
        return [
            sum(w * v for w, v in zip(row, x)) + b
            for row, b in zip(self.weights, self.bias)
        ]

    def q_values_from_latent(
        self,
        latent: LatentState,
        recent_action_keys: Sequence[str],
    ) -> List[float]:
        x = self.features_from_latent(latent, recent_action_keys)
        return [
            sum(w * v for w, v in zip(row, x)) + b
            for row, b in zip(self.weights, self.bias)
        ]

    def q_value(
        self,
        action_key: str,
        latent_vector: Sequence[float],
        recent_action_keys: Sequence[str],
    ) -> float:
        return self.q_values(latent_vector, recent_action_keys)[self._action_index(action_key)]

    def select_action_key(
        self,
        latent_vector: Sequence[float],
        recent_action_keys: Sequence[str],
        *,
        epsilon: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        eps = self.current_epsilon() if epsilon is None else epsilon
        chooser = rng or self._rng
        if eps > 0.0 and chooser.random() < eps:
            return self.action_keys[chooser.randrange(len(self.action_keys))]
        scores = self.q_values(latent_vector, recent_action_keys)
        return self.action_keys[max(range(len(scores)), key=scores.__getitem__)]

    def select_action_key_from_latent(
        self,
        latent: LatentState,
        recent_action_keys: Sequence[str],
        *,
        epsilon: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        self.check_compatible(layout_hash=latent.layout_hash, latent_width=latent.width)
        return self.select_action_key(latent.vector, recent_action_keys, epsilon=epsilon, rng=rng)

    def td_update(
        self,
        previous_latent_vector: Sequence[float],
        previous_recent_action_keys: Sequence[str],
        action_key: str,
        reward: float,
        current_latent_vector: Sequence[float],
        current_recent_action_keys: Sequence[str],
        *,
        done: bool = False,
    ) -> Dict[str, float]:
        action_index = self._action_index(action_key)
        prev_features = self.features(previous_latent_vector, previous_recent_action_keys)
        current_q = sum(
            w * v for w, v in zip(self.weights[action_index], prev_features)
        ) + self.bias[action_index]
        if done:
            target = float(reward)
        else:
            target = float(reward) + self.gamma * max(
                self.q_values(current_latent_vector, current_recent_action_keys)
            )
        raw_td_error = target - current_q
        td_error = _clip(raw_td_error, -self.td_clip, self.td_clip)
        row = self.weights[action_index]
        for i, value in enumerate(prev_features):
            if value != 0.0:
                row[i] += self.lr * td_error * value
        self.bias[action_index] += self.lr * td_error
        self.training_ticks += 1
        return {
            "q_before": current_q,
            "target": target,
            "td_error": td_error,
            "raw_td_error": raw_td_error,
        }

    def td_update_from_latents(
        self,
        previous_latent: LatentState,
        previous_recent_action_keys: Sequence[str],
        action_key: str,
        reward: float,
        current_latent: LatentState,
        current_recent_action_keys: Sequence[str],
        *,
        done: bool = False,
    ) -> Dict[str, float]:
        self.check_compatible(
            layout_hash=previous_latent.layout_hash,
            latent_width=previous_latent.width,
        )
        self.check_compatible(
            layout_hash=current_latent.layout_hash,
            latent_width=current_latent.width,
        )
        return self.td_update(
            previous_latent.vector,
            previous_recent_action_keys,
            action_key,
            reward,
            current_latent.vector,
            current_recent_action_keys,
            done=done,
        )

    def _action_index(self, action_key: str) -> int:
        try:
            return self.action_keys.index(action_key)
        except ValueError as exc:
            raise ValueError(f"unknown online Q action {action_key!r}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "action_keys": self.action_keys,
            "latent_width": self.latent_width,
            "layout_hash": self.layout_hash,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "hyperparameters": {
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_ticks": self.epsilon_decay_ticks,
                "td_clip": self.td_clip,
            },
            "training_ticks": self.training_ticks,
            "epsilon_state": {
                "current_epsilon": self.current_epsilon(),
                "rng_seed": self.seed,
                "rng_state": _jsonable_rng_state(self._rng.getstate()),
            },
            "meta": self.meta,
        }

    def save(self, path: str, *, atomic: bool = True) -> None:
        payload = self.to_dict()
        if not atomic:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            return

        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def load(
        path: str,
        *,
        expected_action_keys: Optional[Sequence[str]] = None,
        expected_layout_hash: Optional[str] = None,
        expected_latent_width: Optional[int] = None,
    ) -> "OnlineQModel":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported online Q checkpoint format {raw.get('format')!r}; "
                f"expected {FORMAT_VERSION}"
            )
        hp = raw.get("hyperparameters", {})
        epsilon_state = raw.get("epsilon_state", {})
        model = OnlineQModel(
            action_keys=list(raw["action_keys"]),
            latent_width=int(raw["latent_width"]),
            layout_hash=str(raw["layout_hash"]),
            feature_names=list(raw.get("feature_names") or []),
            weights=[[float(v) for v in row] for row in raw["weights"]],
            bias=[float(v) for v in raw["bias"]],
            lr=float(hp.get("lr", 0.02)),
            gamma=float(hp.get("gamma", 0.99)),
            epsilon_start=float(hp.get("epsilon_start", 0.2)),
            epsilon_min=float(hp.get("epsilon_min", 0.05)),
            epsilon_decay_ticks=int(hp.get("epsilon_decay_ticks", 50000)),
            td_clip=float(hp.get("td_clip", 5.0)),
            training_ticks=int(raw.get("training_ticks", 0)),
            seed=int(epsilon_state.get("rng_seed", 0)),
            meta=dict(raw.get("meta", {})),
        )
        if "rng_state" in epsilon_state:
            model._rng.setstate(_tuple_rng_state(epsilon_state["rng_state"]))  # type: ignore[arg-type]
        model.check_compatible(
            action_keys=expected_action_keys,
            layout_hash=expected_layout_hash,
            latent_width=expected_latent_width,
        )
        return model
