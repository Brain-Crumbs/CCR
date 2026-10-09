"""Executable Factory configuration checks, separate from legacy artifact reads."""
from __future__ import annotations

import math
from typing import Any, Mapping

from cognitive_runtime.training.optimizer_config import resolve_optimizer

LOSS_ALIASES = {"pixel": "pixel_loss_weight", "latent": "latent_loss_weight", "semantic": "semantic_loss_weight"}
COMMON_LOSSES = frozenset({
    "pixel_loss_weight", "latent_loss_weight", "semantic_loss_weight", "reward_loss_weight",
    "terminal_loss_weight", "risk_loss_weight", "uncertainty_loss_weight", "motion_pixel_loss_weight",
    "hud_loss_weight", "change_mask_sparsity_weight", "change_mask_supervision_weight",
})
WINDOWED_LOSSES = frozenset({"closed_loop_pixel_loss_weight", "closed_loop_latent_loss_weight", "direct_horizon_loss_weight"})
AUTOREGRESSIVE_LOSSES = frozenset({"autoregressive_rollout_weight", "autoregressive_rollout_latent_weight", "autoregressive_rollout_pixel_weight"})


def resolve_loss_weights(training: Mapping[str, Any]) -> dict[str, float]:
    objective = training["objective"]
    if objective not in ("windowed_rollout", "autoregressive"):
        raise ValueError(f"unsupported training objective {objective!r}")
    allowed = COMMON_LOSSES | (WINDOWED_LOSSES if objective == "windowed_rollout" else AUTOREGRESSIVE_LOSSES)
    weights = training.get("loss_weights") or {}
    if not isinstance(weights, Mapping):
        raise ValueError("training.loss_weights must be a mapping")
    resolved = {}
    for key, value in weights.items():
        field = LOSS_ALIASES.get(key, key)
        if field not in allowed:
            raise ValueError(f"unsupported or inactive loss weight {key!r} for {objective}")
        if field in resolved:
            raise ValueError(f"training.loss_weights declares both aliases for {field!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss weight {key!r} must be finite and non-negative")
        resolved[field] = float(value)
    if "autoregressive_rollout_weight" in resolved and "autoregressive_rollout_latent_weight" in resolved:
        raise ValueError("training.loss_weights declares both autoregressive latent rollout aliases")
    return resolved


def validate_execution_spec(spec: Any) -> None:
    """Fail before data/model allocation, without importing torch.

    Historical spec documents remain structurally readable; executing them
    uses the explicitly versioned optimizer migration, never the old lie.
    """
    training = spec.training
    resolve_optimizer(training["optimizer"])
    losses = resolve_loss_weights(training)
    if "semantic_loss_weight" in losses:
        raise ValueError("semantic loss is inactive: Factory models have no semantic head; remove the semantic loss gene")
    if training["objective"] == "autoregressive" and training["batch_size"] != 1:
        raise ValueError("autoregressive training uses one episode per step; set batch_size=1")
    if training.get("scheduler") is not None:
        raise ValueError("training.scheduler is unsupported; only null (no scheduler) is executable")
    if training.get("precision") != "fp32":
        raise ValueError("training.precision supports only fp32")
    metric = spec.evaluation["selection_metric"]
    mode = "max" if metric in ("goal_navigation.success_rate", "goal_navigation.geodesic_efficiency", "goal_navigation.replan_recovery") else "min"
    selection = dict(training.get("checkpoint_selection_policy") or {})
    if selection not in ({"metric": "validation_loss", "mode": "min"}, {"metric": metric, "mode": mode}):
        raise ValueError("unsupported checkpoint_selection_policy; Factory selects evaluation.selection_metric")
    if training.get("step_budget") is not None:
        raise ValueError("training.step_budget is unsupported; use epoch_budget")
    if training.get("early_stopping_policy"):
        raise ValueError("training.early_stopping_policy is unsupported")
    policy = training.get("transition_balance_policy") or {}
    if set(policy) - {"stationary_cap"}:
        raise ValueError("unsupported training.transition_balance_policy parameters")
    if policy and training["objective"] == "autoregressive":
        raise ValueError("transition_balance_policy is inactive for autoregressive training")
    determinism = training.get("determinism_policy") or {}
    if set(determinism) - {"deterministic", "seed"} or not isinstance(determinism.get("deterministic", True), bool):
        raise ValueError("unsupported training.determinism_policy")
    if determinism.get("seed", training["seed"]) != training["seed"]:
        raise ValueError("determinism_policy.seed must equal training.seed")
    backbone = spec.model["backbone"]
    supported = {"gru": set(), "dilated_conv": {"kernel_size", "n_layers"}, "transformer": {"n_heads", "n_layers"}}
    if backbone not in supported:
        raise ValueError(f"unsupported model backbone {backbone!r}")
    kwargs = spec.model.get("backbone_kwargs") or {}
    if set(kwargs) - supported[backbone]:
        raise ValueError(f"unsupported or inactive backbone_kwargs for {backbone}")
    for key in ("n_layers", "kernel_size"):
        if key in kwargs and (isinstance(kwargs[key], bool) or not isinstance(kwargs[key], int) or kwargs[key] < (2 if key == "kernel_size" else 1)):
            raise ValueError(f"backbone_kwargs.{key} must be a positive supported integer")
    if backbone == "transformer":
        heads = kwargs.get("n_heads", 2)
        if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0 or spec.model["hidden_dim"] % heads:
            raise ValueError("transformer hidden_dim must be divisible by n_heads; implicit head fallback is unsupported")
