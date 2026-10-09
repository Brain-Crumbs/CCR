"""Torch-free optimizer resolution and observed runtime identity (#284).

Only this module chooses optimizer constructors. Importing it never imports
optional ML dependencies. Callers supply their lazily imported torch module.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


OPTIMIZER_FORMAT = "torch-optimizer-v2"
LEGACY_BEHAVIOR = "legacy-adam-v1"
OPTIMIZER_DEFAULTS = {
    "format": OPTIMIZER_FORMAT, "name": "adamw", "lr": 3e-4,
    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5,
    "amsgrad": False,
}


def resolve_optimizer(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("optimizer must be a mapping")
    unknown = set(config) - set(OPTIMIZER_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported optimizer parameters: {sorted(unknown)}")
    result = {**OPTIMIZER_DEFAULTS, **config}
    if result["format"] != OPTIMIZER_FORMAT:
        raise ValueError(f"unsupported optimizer format {result['format']!r}")
    if result["name"] not in ("adam", "adamw"):
        raise ValueError(f"unsupported optimizer name {result['name']!r}; use adam or adamw")
    for key in ("lr", "eps", "weight_decay"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"optimizer.{key} must be finite and non-negative")
        result[key] = float(value)
    betas = result["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
        isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1
        for b in betas
    ):
        raise ValueError("optimizer.betas must contain two finite numbers in [0, 1)")
    result["betas"] = [float(b) for b in betas]
    if not isinstance(result["amsgrad"], bool):
        raise ValueError("optimizer.amsgrad must be a boolean")
    return result


def action_optimizer_config(cfg: Any) -> dict[str, Any]:
    """Preserve direct AWM's historical Adam only under its named version."""
    if cfg.optimizer_behavior == LEGACY_BEHAVIOR:
        if cfg.optimizer is not None:
            raise ValueError("legacy-adam-v1 cannot specify optimizer; use torch-optimizer-v2")
        return resolve_optimizer({"name": "adam", "lr": cfg.lr, "weight_decay": 0.0})
    if cfg.optimizer_behavior != OPTIMIZER_FORMAT or cfg.optimizer is None:
        raise ValueError("configured optimizer requires optimizer_behavior='torch-optimizer-v2'")
    resolved = resolve_optimizer(cfg.optimizer)
    if resolved["lr"] != cfg.lr:
        raise ValueError("optimizer.lr and ActionWorldModelConfig.lr disagree")
    return resolved


def build_optimizer(torch: Any, parameters: Any, config: Mapping[str, Any]) -> Any:
    resolved = resolve_optimizer(config)
    constructor = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[resolved["name"]]
    kwargs = {k: v for k, v in resolved.items() if k not in ("format", "name")}
    kwargs["betas"] = tuple(kwargs["betas"])
    return constructor(parameters, **kwargs)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise ValueError(f"optimizer has unsupported non-JSON group setting {type(value).__name__}")


def optimizer_manifest(model: Any, optimizer: Any) -> dict[str, Any]:
    """Observe actual constructor and groups, including ordered parameter names."""
    names = {id(p): (name, list(p.shape)) for name, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        parameters = []
        for p in group["params"]:
            if id(p) not in names:
                raise ValueError("optimizer parameter is not part of the model")
            name, shape = names[id(p)]
            parameters.append({"name": name, "shape": shape})
        groups.append({**_plain({k: v for k, v in group.items() if k != "params"}), "parameters": parameters})
    return {"format": OPTIMIZER_FORMAT, "class": type(optimizer).__name__, "parameter_groups": groups}


def verify_optimizer_manifest(model: Any, optimizer: Any, expected: Any) -> None:
    if expected is None:
        raise ValueError("cannot resume without effective optimizer evidence; use clone/fine_tune for a new run")
    if optimizer_manifest(model, optimizer) != expected:
        raise ValueError("effective optimizer configuration differs; use clone/fine_tune for a new run")


def restore_optimizer(model: Any, optimizer: Any, state: Mapping[str, Any], expected: Any) -> None:
    # Compare before load_state_dict, which would overwrite requested lr/betas/decay.
    verify_optimizer_manifest(model, optimizer, expected)
    groups = state.get("param_groups", [])
    evidence_groups = expected["parameter_groups"]
    if len(groups) != len(evidence_groups):
        raise ValueError("optimizer state groups disagree with effective configuration")
    canonical_groups = optimizer.state_dict()["param_groups"]
    for group, evidence, canonical in zip(groups, evidence_groups, canonical_groups):
        settings = _plain({k: v for k, v in group.items() if k != "params"})
        declared = {k: v for k, v in evidence.items() if k != "parameters"}
        if settings != declared or group["params"] != canonical["params"] or len(group["params"]) != len(evidence["parameters"]):
            raise ValueError("optimizer state groups disagree with effective configuration")
    optimizer.load_state_dict(state)
    verify_optimizer_manifest(model, optimizer, expected)


def effective_hash(config: Mapping[str, Any]) -> str:
    from cognitive_runtime.training.model_factory.contracts import contract_hash

    return contract_hash(config)
