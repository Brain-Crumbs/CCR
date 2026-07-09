"""End-to-end behavioral cloning for pixel vision (PyTorch).

Trains :class:`VisionPolicyNet` jointly — CNN over ``vision.frame.pixels`` plus
an MLP head over the fused non-vision vector and motor history — to predict the
demonstrated action.  The gradient flows through the CNN, so the agent learns
its own visual features from the pixel stream.

Mirrors ``training.imitation.train_bc``: cross-entropy with the same
square-root inverse-frequency class balancing (demonstrations are dominated by
one action, so unbalanced cloning collapses to it), a held-out split, and
accuracy / balanced-accuracy reporting against the majority baseline.  Imported
only when neural training is requested, so torch stays optional.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from cognitive_runtime.models.vision import (
    REPRESENTATION,
    VisionBCModel,
    VisionPolicyNet,
    pixels_to_chw,
)
from cognitive_runtime.training.datasets import NeuralDataset


def _class_weights(labels: List[int], n_actions: int, max_class_weight: float) -> torch.Tensor:
    counts: Dict[int, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    mean_count = len(labels) / max(len(counts), 1)
    weights = [1.0] * n_actions
    for y, c in counts.items():
        weights[y] = min(math.sqrt(mean_count / c), max_class_weight)
    return torch.tensor(weights, dtype=torch.float32)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return (logits.argmax(dim=1) == targets).float().mean().item()


def _balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    recalls = []
    for y in targets.unique():
        mask = targets == y
        recalls.append((preds[mask] == y).float().mean().item())
    return sum(recalls) / len(recalls)


def train_neural_bc(
    dataset: NeuralDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 0,
    class_balance: bool = True,
    max_class_weight: float = 25.0,
    embed_dim: int = 64,
    hidden_dim: int = 64,
) -> Tuple[VisionBCModel, Dict[str, float]]:
    if len(dataset) == 0:
        raise ValueError("neural dataset is empty; record sessions with --record-frames first")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    n_actions = len(dataset.action_keys)
    n_non_vision = len(dataset.non_vision[0])
    n_motor = len(dataset.motor[0])
    pixel_shape = tuple(dataset.pixel_shape or _infer_shape(dataset.pixels[0]))

    pixels = torch.stack([pixels_to_chw(p) for p in dataset.pixels])
    aux = torch.tensor(
        [nv + mv for nv, mv in zip(dataset.non_vision, dataset.motor)], dtype=torch.float32
    )
    targets = torch.tensor(dataset.labels, dtype=torch.long)

    n = len(dataset)
    perm = torch.randperm(n, generator=generator)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if train_idx.numel() == 0:
        train_idx, val_idx = perm, perm[:0]

    net = VisionPolicyNet(pixel_shape, n_non_vision, n_motor, n_actions, embed_dim, hidden_dim)
    weight = _class_weights(dataset.labels, n_actions, max_class_weight) if class_balance else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for _epoch in range(epochs):
        epoch_perm = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        for start in range(0, epoch_perm.numel(), batch_size):
            batch = epoch_perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = net(pixels[batch], aux[batch])
            loss_fn(logits, targets[batch]).backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        train_logits = net(pixels[train_idx], aux[train_idx])
        val_logits = net(pixels[val_idx], aux[val_idx]) if val_idx.numel() else train_logits[:0]
    counts = torch.bincount(targets, minlength=n_actions)
    majority = counts.max().item() / n

    metrics = {
        "samples": float(n),
        "train_accuracy": round(_accuracy(train_logits, targets[train_idx]), 4),
        "val_accuracy": round(_accuracy(val_logits, targets[val_idx]), 4)
        if val_idx.numel()
        else float("nan"),
        "train_balanced_accuracy": round(_balanced_accuracy(train_logits, targets[train_idx]), 4),
        "majority_class_baseline": round(majority, 4),
        "random_class_baseline": round(1.0 / max(int((counts > 0).sum().item()), 1), 4),
        "epochs": float(epochs),
    }
    model = VisionBCModel(
        net,
        action_keys=list(dataset.action_keys),
        meta={
            "metrics": metrics,
            "sources": dataset.sources,
            "representation": REPRESENTATION,
            "layout_hash": dataset.layout_hash,
            "pixel_shape": list(pixel_shape),
            "non_vision_names": dataset.non_vision_names,
        },
    )
    return model, metrics


def _infer_shape(frame: Any) -> Tuple[int, int, int]:
    if isinstance(frame, np.ndarray):
        return tuple(frame.shape)  # type: ignore[return-value]
    h = len(frame)
    w = len(frame[0]) if h else 0
    c = len(frame[0][0]) if w else 0
    return (h, w, c)
