"""Behavioral cloning: a small softmax-regression policy head.

Deliberately tiny and dependency-free: input features -> linear layer ->
softmax over the discrete action space, trained with minibatch SGD and
cross-entropy.  Good enough to beat the random baseline; easy to replace
with a real network later.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_runtime.training.datasets import Dataset


@dataclass
class BCModel:
    feature_names: List[str]
    action_keys: List[str]
    weights: List[List[float]]  # [n_actions][n_features]
    bias: List[float]
    meta: Dict[str, object] = field(default_factory=dict)

    def logits(self, features: List[float]) -> List[float]:
        return [
            sum(w * x for w, x in zip(row, features)) + b
            for row, b in zip(self.weights, self.bias)
        ]

    def predict_index(self, features: List[float]) -> int:
        scores = self.logits(features)
        return max(range(len(scores)), key=scores.__getitem__)

    def predict_key(self, features: List[float]) -> str:
        return self.action_keys[self.predict_index(features)]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "action_keys": self.action_keys,
                    "weights": self.weights,
                    "bias": self.bias,
                    "meta": self.meta,
                },
                fh,
            )

    @staticmethod
    def load(path: str) -> "BCModel":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return BCModel(
            feature_names=raw["feature_names"],
            action_keys=raw["action_keys"],
            weights=raw["weights"],
            bias=raw["bias"],
            meta=raw.get("meta", {}),
        )


def _softmax(logits: List[float]) -> List[float]:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def train_bc(
    dataset: Dataset,
    epochs: int = 10,
    lr: float = 0.5,
    l2: float = 1e-4,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 0,
    class_balance: bool = True,
    max_class_weight: float = 25.0,
) -> Tuple[BCModel, Dict[str, float]]:
    """class_balance weights samples inversely to their action's frequency.

    Demonstration data is heavily skewed toward the dominant action (e.g.
    MOVE_FORWARD); without balancing, cloning collapses to the majority
    class and never learns the rare context-dependent actions (turning when
    blocked, attacking threats, eating).
    """
    if len(dataset) == 0:
        raise ValueError("dataset is empty; record sessions with observations first")
    rng = random.Random(seed)
    n_features = len(dataset.feature_names)
    n_actions = len(dataset.action_keys)

    counts: Dict[int, int] = {}
    for y in dataset.labels:
        counts[y] = counts.get(y, 0) + 1
    if class_balance:
        # Square-root inverse-frequency: lifts rare context-dependent actions
        # without drowning out the dominant one entirely.
        mean_count = len(dataset) / len(counts)
        class_weight = {
            y: min(math.sqrt(mean_count / c), max_class_weight) for y, c in counts.items()
        }
    else:
        class_weight = {y: 1.0 for y in counts}

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_fraction)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if not train_idx:
        train_idx, val_idx = indices, []

    weights = [[0.0] * n_features for _ in range(n_actions)]
    bias = [0.0] * n_actions

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        step_lr = lr / (1.0 + epoch * 0.3)
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            grad_w = [[0.0] * n_features for _ in range(n_actions)]
            grad_b = [0.0] * n_actions
            for i in batch:
                x = dataset.features[i]
                y = dataset.labels[i]
                sample_weight = class_weight[y]
                logits = [
                    sum(w * v for w, v in zip(weights[k], x)) + bias[k]
                    for k in range(n_actions)
                ]
                probs = _softmax(logits)
                for k in range(n_actions):
                    err = (probs[k] - (1.0 if k == y else 0.0)) * sample_weight
                    if err == 0.0:
                        continue
                    grad_b[k] += err
                    row = grad_w[k]
                    for j, v in enumerate(x):
                        if v != 0.0:
                            row[j] += err * v
            scale = step_lr / len(batch)
            for k in range(n_actions):
                wk = weights[k]
                gk = grad_w[k]
                for j in range(n_features):
                    wk[j] -= scale * (gk[j] + l2 * wk[j])
                bias[k] -= scale * grad_b[k]

    def accuracy(idx: List[int]) -> float:
        if not idx:
            return 0.0
        model_tmp = BCModel(dataset.feature_names, dataset.action_keys, weights, bias)
        hits = sum(
            1 for i in idx if model_tmp.predict_index(dataset.features[i]) == dataset.labels[i]
        )
        return hits / len(idx)

    majority = max(counts.values()) / len(dataset)

    def balanced_accuracy(idx: List[int]) -> float:
        if not idx:
            return 0.0
        model_tmp = BCModel(dataset.feature_names, dataset.action_keys, weights, bias)
        per_class: Dict[int, List[int]] = {}
        for i in idx:
            per_class.setdefault(dataset.labels[i], []).append(i)
        recalls = [
            sum(1 for i in members if model_tmp.predict_index(dataset.features[i]) == y)
            / len(members)
            for y, members in per_class.items()
        ]
        return sum(recalls) / len(recalls)

    metrics = {
        "samples": float(len(dataset)),
        "train_accuracy": round(accuracy(train_idx), 4),
        "val_accuracy": round(accuracy(val_idx), 4) if val_idx else float("nan"),
        "train_balanced_accuracy": round(balanced_accuracy(train_idx), 4),
        "majority_class_baseline": round(majority, 4),
        "random_class_baseline": round(1.0 / max(len(counts), 1), 4),
        "epochs": float(epochs),
    }
    model = BCModel(
        feature_names=list(dataset.feature_names),
        action_keys=list(dataset.action_keys),
        weights=weights,
        bias=bias,
        meta={
            "metrics": metrics,
            "sources": dataset.sources,
            "representation": dataset.representation,
            "layout_hash": dataset.layout_hash,
        },
    )
    return model, metrics
