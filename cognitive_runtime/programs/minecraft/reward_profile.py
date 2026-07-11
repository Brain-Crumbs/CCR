"""Reward profile schema, loader and validation (issue #41).

A reward profile is a YAML/JSON document describing the reward components
for an episode as a **declarative rule set** instead of hand-written Python:
components are grouped into tiers (``survival``, ``capability``, ``quest``,
plus a ``shaping`` tier for anti-stagnation controls), each with a `kind`
selected from a small closed vocabulary (see :data:`KNOWN_KINDS`) and
anti-farming controls (`cap`, `decay`, `cooldown_ticks`).  A separate
``intrinsic`` section holds named slots (`learning_progress`, `safe_novelty`,
`predicted_risk_aversion`, ...) that read their signal from an
`internal.*` stream rather than recomputing anything reward-side (issue
#58/#61 supply the streams and the components; this module only supplies the
schema slot they plug into).

Profiles are loaded once at startup with :func:`load_reward_profile`, which
raises :class:`RewardProfileError` on anything malformed -- the whole point
being a bad profile fails fast before a run starts, not mid-episode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from cognitive_runtime.core.hashing import canonical_json

#: Recommended tier names for the three-tier "reward compass" plus the
#: cross-cutting engagement-shaping bucket (anti-stagnation penalties are
#: not reward *content*, so they don't fit survival/capability/quest).
#: Tier names are otherwise free-form; this is guidance, not a restriction.
RECOMMENDED_TIERS: Tuple[str, ...] = ("survival", "capability", "quest", "shaping")

#: Closed vocabulary of component "kind"s the engine knows how to evaluate.
#: See `docs/reward_profiles.md` for the full semantics of each.
KNOWN_KINDS = frozenset(
    {
        "tick",
        "death",
        "event_count",
        "delta_decrease",
        "threshold_enter",
        "periodic_no_event",
        "capped_novelty",
        "distance_ladder",
        "once_predicate",
        "once_event",
        "decaying_repeat",
        "streak_penalty",
        "idle_penalty",
        "spinning_penalty",
        "no_novelty_penalty",
    }
)

#: Milestone scopes: "life" resets every episode (`reset()`); "brain"
#: persists across episodes/checkpoint resume until explicitly cleared.
MILESTONE_SCOPES = frozenset({"life", "brain"})

_NORMALIZATION_METHODS = frozenset({"running", "none"})

#: Required `params` keys (and their expected type) per kind, checked at
#: load time so a missing/mistyped param fails at startup with a clear
#: message instead of an AttributeError/KeyError mid-episode.  Kinds not
#: listed here take no required params.
_REQUIRED_PARAMS: Dict[str, Dict[str, Any]] = {
    "event_count": {"event_prefix": str},
    "delta_decrease": {"field": str},
    "threshold_enter": {"field": str, "threshold": (int, float)},
    "periodic_no_event": {
        "event_prefix": str, "window": int, "min_field": str, "min_value": (int, float),
    },
    "capped_novelty": {"source": str},
    "distance_ladder": {"unit": (int, float)},
    "once_predicate": {"source": str, "predicate": str},
    "once_event": {"event": str},
    "decaying_repeat": {"source": str},
    "streak_penalty": {"threshold": int},
    "idle_penalty": {"threshold": int},
    "spinning_penalty": {"window": int, "actions": list},
    "no_novelty_penalty": {"ticks": int},
}

#: Kinds whose milestone/novelty bookkeeping is unbounded unless capped --
#: a missing `cap` is very likely an authoring mistake (unbounded reward
#: farming), so these two require an explicit `cap` at load time.
_REQUIRES_CAP = frozenset({"capped_novelty", "distance_ladder"})

_VALID_FIELDS = frozenset({"health", "hunger"})
_VALID_PREDICATES = frozenset({"is_tool", "is_food"})


class RewardProfileError(ValueError):
    """A reward profile is malformed. Always raised at load time."""


@dataclass(frozen=True)
class ComponentSpec:
    """One reward rule. `kind` selects the evaluation rule; `params` carries
    kind-specific knobs (event names, fields, thresholds, ...)."""

    kind: str
    value: float = 0.0
    cap: Optional[float] = None
    decay: float = 1.0
    decay_floor: float = 0.0
    cooldown_ticks: int = 0
    scope: str = "life"
    disabled: bool = False
    #: Intrinsic slots only: source stream id (e.g. "internal.learning_progress").
    stream: Optional[str] = None
    #: Intrinsic slots only: multiplier applied to the raw stream value.
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "cap": self.cap,
            "decay": self.decay,
            "decay_floor": self.decay_floor,
            "cooldown_ticks": self.cooldown_ticks,
            "scope": self.scope,
            "disabled": self.disabled,
            "stream": self.stream,
            "weight": self.weight,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class NormalizationSpec:
    """Two-scale rewards: raw components are always logged; `method`
    controls how the *training*-facing scalar is derived from them."""

    method: str = "running"
    clip: Optional[float] = 5.0
    epsilon: float = 1e-4
    warmup_ticks: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "warmup_ticks": self.warmup_ticks,
        }


@dataclass(frozen=True)
class RewardProfile:
    name: str
    description: str = ""
    tiers: Dict[str, Dict[str, ComponentSpec]] = field(default_factory=dict)
    intrinsic: Dict[str, ComponentSpec] = field(default_factory=dict)
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)

    def components(self) -> Dict[str, Tuple[str, ComponentSpec]]:
        """Flat map of component name -> (tier, spec), tiers then intrinsic.

        Component names must be unique across tiers *and* intrinsic slots so
        milestone/state keys and logged component names never collide.
        """
        flat: Dict[str, Tuple[str, ComponentSpec]] = {}
        for tier, components in self.tiers.items():
            for name, spec in components.items():
                flat[name] = (tier, spec)
        for name, spec in self.intrinsic.items():
            flat[name] = ("intrinsic", spec)
        return flat

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tiers": {
                tier: {name: spec.to_dict() for name, spec in components.items()}
                for tier, components in self.tiers.items()
            },
            "intrinsic": {name: spec.to_dict() for name, spec in self.intrinsic.items()},
            "normalization": self.normalization.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        """Stable hash of profile content, for session metadata / dashboard
        grouping ("group like with like") and milestone-state compatibility
        checks across resume."""
        blob = canonical_json(self.to_dict())
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        """Session-metadata summary: name + content hash (dashboard
        grouping), plus each intrinsic slot's stream/weight/cap/disabled
        (issue #61: "intrinsic weights ... in session metadata so #44's
        harness can compare drives") -- the full profile is already
        reproducible from `content_hash`, this just spells out the
        intrinsic knobs for direct inspection without a profile-file lookup.
        """
        return {
            "name": self.name,
            "content_hash": self.content_hash,
            "intrinsic": {
                name: {
                    "stream": spec.stream,
                    "weight": spec.weight,
                    "cap": spec.cap,
                    "disabled": spec.disabled,
                }
                for name, spec in self.intrinsic.items()
            },
        }


def _err(source: str, message: str) -> "RewardProfileError":
    return RewardProfileError(f"reward profile {source!r}: {message}")


def _require_type(source: str, path: str, value: Any, types: Any, optional: bool = False) -> Any:
    if value is None and optional:
        return None
    if not isinstance(value, types):
        raise _err(source, f"{path} must be {types}, got {type(value).__name__}")
    return value


def _validate_kind_params(
    source: str, path: str, kind: str, params: Mapping[str, Any], cap: Optional[float]
) -> None:
    if kind in _REQUIRES_CAP and cap is None:
        raise _err(source, f"{path} (kind {kind!r}) requires a 'cap' to bound its reward")
    required = _REQUIRED_PARAMS.get(kind, {})
    for key, expected_type in required.items():
        if key not in params:
            raise _err(source, f"{path}.params is missing required key {key!r} for kind {kind!r}")
        value = params[key]
        if not isinstance(value, expected_type):
            raise _err(
                source,
                f"{path}.params.{key} must be {expected_type}, got {type(value).__name__}",
            )
    if kind in ("delta_decrease", "threshold_enter") and params.get("field") not in _VALID_FIELDS:
        raise _err(
            source,
            f"{path}.params.field must be one of {sorted(_VALID_FIELDS)}, "
            f"got {params.get('field')!r}",
        )
    if kind == "once_predicate" and params.get("predicate") not in _VALID_PREDICATES:
        raise _err(
            source,
            f"{path}.params.predicate must be one of {sorted(_VALID_PREDICATES)}, "
            f"got {params.get('predicate')!r}",
        )
    if kind == "spinning_penalty":
        actions = params.get("actions")
        if not actions or not all(isinstance(a, str) for a in actions):
            raise _err(source, f"{path}.params.actions must be a non-empty list of strings")
    for int_key in ("window", "threshold", "ticks"):
        if int_key in required and int_key in params and params[int_key] <= 0:
            raise _err(source, f"{path}.params.{int_key} must be > 0, got {params[int_key]}")
    if "unit" in required and params.get("unit", 0) <= 0:
        raise _err(source, f"{path}.params.unit must be > 0, got {params.get('unit')}")


def _component_from_dict(
    source: str, tier: str, name: str, raw: Mapping[str, Any], *, is_intrinsic: bool = False
) -> ComponentSpec:
    path = f"{tier}.{name}"
    if not isinstance(raw, Mapping):
        raise _err(source, f"{path} must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {
        "kind", "value", "cap", "decay", "decay_floor", "cooldown_ticks",
        "scope", "disabled", "stream", "weight", "params",
    }
    if unknown:
        raise _err(source, f"{path} has unknown field(s): {sorted(unknown)}")

    kind = raw.get("kind", "intrinsic_stream" if is_intrinsic else None)
    if is_intrinsic:
        if "stream" not in raw:
            raise _err(source, f"{path} (intrinsic) is missing required field 'stream'")
    else:
        if kind is None:
            raise _err(source, f"{path} is missing required field 'kind'")
        if kind not in KNOWN_KINDS:
            raise _err(
                source,
                f"{path} has unknown kind {kind!r}; expected one of {sorted(KNOWN_KINDS)}",
            )

    value = _require_type(source, f"{path}.value", raw.get("value", 0.0), (int, float))
    cap = raw.get("cap")
    if cap is not None:
        cap = _require_type(source, f"{path}.cap", cap, (int, float))
        if cap < 0:
            raise _err(source, f"{path}.cap must be >= 0, got {cap}")
    decay = _require_type(source, f"{path}.decay", raw.get("decay", 1.0), (int, float))
    decay_floor = _require_type(
        source, f"{path}.decay_floor", raw.get("decay_floor", 0.0), (int, float)
    )
    cooldown_ticks = _require_type(
        source, f"{path}.cooldown_ticks", raw.get("cooldown_ticks", 0), int
    )
    if cooldown_ticks < 0:
        raise _err(source, f"{path}.cooldown_ticks must be >= 0, got {cooldown_ticks}")
    scope = _require_type(source, f"{path}.scope", raw.get("scope", "life"), str)
    if scope not in MILESTONE_SCOPES:
        raise _err(
            source, f"{path}.scope must be one of {sorted(MILESTONE_SCOPES)}, got {scope!r}"
        )
    disabled = _require_type(source, f"{path}.disabled", raw.get("disabled", False), bool)
    stream = raw.get("stream")
    if stream is not None:
        _require_type(source, f"{path}.stream", stream, str)
    weight = _require_type(source, f"{path}.weight", raw.get("weight", 1.0), (int, float))
    params = raw.get("params", {})
    _require_type(source, f"{path}.params", params, Mapping)
    if not is_intrinsic:
        _validate_kind_params(source, path, kind, params, cap)

    return ComponentSpec(
        kind=kind,
        value=float(value),
        cap=float(cap) if cap is not None else None,
        decay=float(decay),
        decay_floor=float(decay_floor),
        cooldown_ticks=int(cooldown_ticks),
        scope=scope,
        disabled=bool(disabled),
        stream=stream,
        weight=float(weight),
        params=dict(params),
    )


def _normalization_from_dict(source: str, raw: Mapping[str, Any]) -> NormalizationSpec:
    if not isinstance(raw, Mapping):
        raise _err(source, f"normalization must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"method", "clip", "epsilon", "warmup_ticks"}
    if unknown:
        raise _err(source, f"normalization has unknown field(s): {sorted(unknown)}")
    method = _require_type(source, "normalization.method", raw.get("method", "running"), str)
    if method not in _NORMALIZATION_METHODS:
        raise _err(
            source,
            f"normalization.method must be one of {sorted(_NORMALIZATION_METHODS)}, "
            f"got {method!r}",
        )
    clip = raw.get("clip", 5.0)
    if clip is not None:
        clip = _require_type(source, "normalization.clip", clip, (int, float))
        if clip <= 0:
            raise _err(source, f"normalization.clip must be > 0 (or null), got {clip}")
    epsilon = _require_type(
        source, "normalization.epsilon", raw.get("epsilon", 1e-4), (int, float)
    )
    warmup_ticks = _require_type(
        source, "normalization.warmup_ticks", raw.get("warmup_ticks", 100), int
    )
    if warmup_ticks < 0:
        raise _err(source, f"normalization.warmup_ticks must be >= 0, got {warmup_ticks}")
    return NormalizationSpec(
        method=method,
        clip=float(clip) if clip is not None else None,
        epsilon=float(epsilon),
        warmup_ticks=int(warmup_ticks),
    )


def reward_profile_from_dict(data: Mapping[str, Any], source: str = "<dict>") -> RewardProfile:
    """Validate and build a :class:`RewardProfile` from a parsed mapping.

    Raises :class:`RewardProfileError` with a message that names the exact
    offending field, so a malformed profile fails at load time with a
    diagnosis instead of an obscure `KeyError` mid-run.
    """
    if not isinstance(data, Mapping):
        raise _err(source, f"top level must be a mapping, got {type(data).__name__}")
    unknown = set(data) - {"name", "description", "tiers", "intrinsic", "normalization"}
    if unknown:
        raise _err(source, f"unknown top-level field(s): {sorted(unknown)}")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise _err(source, "top-level 'name' is required and must be a non-empty string")
    description = data.get("description", "")
    _require_type(source, "description", description, str)

    tiers_raw = data.get("tiers", {})
    _require_type(source, "tiers", tiers_raw, Mapping)
    tiers: Dict[str, Dict[str, ComponentSpec]] = {}
    seen_names: Dict[str, str] = {}
    for tier, components_raw in tiers_raw.items():
        _require_type(source, f"tiers.{tier}", components_raw, Mapping)
        components: Dict[str, ComponentSpec] = {}
        for cname, craw in components_raw.items():
            if cname in seen_names:
                raise _err(
                    source,
                    f"component {cname!r} is defined in both {seen_names[cname]!r} and "
                    f"{tier!r} tiers; component names must be unique",
                )
            seen_names[cname] = tier
            components[cname] = _component_from_dict(source, tier, cname, craw)
        tiers[tier] = components

    intrinsic_raw = data.get("intrinsic", {})
    _require_type(source, "intrinsic", intrinsic_raw, Mapping)
    intrinsic: Dict[str, ComponentSpec] = {}
    for cname, craw in intrinsic_raw.items():
        if cname in seen_names:
            raise _err(
                source,
                f"component {cname!r} is defined in both {seen_names[cname]!r} and "
                "'intrinsic'; component names must be unique",
            )
        seen_names[cname] = "intrinsic"
        intrinsic[cname] = _component_from_dict(source, "intrinsic", cname, craw, is_intrinsic=True)

    normalization = _normalization_from_dict(source, data.get("normalization", {}))

    return RewardProfile(
        name=name, description=description, tiers=tiers, intrinsic=intrinsic,
        normalization=normalization,
    )


def default_profile() -> RewardProfile:
    """The built-in default profile: the "survival foundation" tier only,
    reproducing the values `SurvivalRewardConfig` has always shipped
    (issue #41: "existing hard-coded components become the default
    survival.yaml profile").  See `goals/survival.yaml` for the on-disk copy
    loadable via `--reward-profile`.
    """
    return reward_profile_from_dict(
        {
            "name": "survival",
            "description": (
                "Survival foundation tier: stay alive, keep vitals up, learn "
                "basic exploration/items/shelter without dying or stagnating."
            ),
            "tiers": {
                "survival": {
                    "tick_alive": {"kind": "tick", "value": 0.01},
                    "death": {"kind": "death", "value": -10.0},
                    "damage_taken": {
                        "kind": "event_count", "value": -0.5,
                        "params": {"event_prefix": "damage"},
                    },
                    "health_maintained": {
                        "kind": "periodic_no_event", "value": 0.05,
                        "params": {
                            "event_prefix": "damage", "window": 100,
                            "min_field": "health", "min_value": 16.0,
                        },
                    },
                    "hunger_decrease": {
                        "kind": "delta_decrease", "value": -0.25,
                        "params": {"field": "hunger"},
                    },
                    "critical_health": {
                        "kind": "threshold_enter", "value": -1.0,
                        "params": {"field": "health", "threshold": 4.0},
                    },
                    "critical_hunger": {
                        "kind": "threshold_enter", "value": -1.0,
                        "params": {"field": "hunger", "threshold": 4.0},
                    },
                },
                "capability": {
                    "new_block_type": {
                        "kind": "capped_novelty", "value": 0.1, "cap": 2.0,
                        "params": {"source": "nearby_blocks"},
                    },
                    "new_biome": {
                        "kind": "capped_novelty", "value": 0.2, "cap": 1.0,
                        "params": {"source": "biome"},
                    },
                    "distance": {
                        "kind": "distance_ladder", "value": 0.1, "cap": 2.0,
                        "params": {"unit": 10.0},
                    },
                    "new_chunk": {
                        "kind": "capped_novelty", "value": 0.1, "cap": 2.0,
                        "params": {"source": "position_chunk", "chunk_size": 8.0},
                    },
                    "new_item": {
                        "kind": "capped_novelty", "value": 0.5, "cap": 5.0,
                        "params": {"source": "event:new_item"},
                    },
                    "first_tool": {
                        "kind": "once_predicate", "value": 1.0,
                        "params": {"source": "event:new_item", "predicate": "is_tool"},
                    },
                    "first_food": {
                        "kind": "once_predicate", "value": 1.0,
                        "params": {"source": "event:new_item", "predicate": "is_food"},
                    },
                    "first_block_placed": {
                        "kind": "once_event", "value": 1.0,
                        "params": {"event": "placed_block"},
                    },
                    "tool_used": {
                        "kind": "capped_novelty", "value": 0.3, "cap": 1.5,
                        "params": {"source": "event:used_tool"},
                    },
                    "craft_progress": {
                        "kind": "capped_novelty", "value": 0.5, "cap": 2.0,
                        "params": {"source": "event:crafted"},
                    },
                    "shelter": {
                        "kind": "once_event", "value": 1.0,
                        "params": {"event": "entered_shelter"},
                    },
                    "light_source": {
                        "kind": "once_event", "value": 1.0,
                        "params": {"event": "created_light_source"},
                    },
                    "survived_night": {
                        "kind": "once_event", "value": 1.0,
                        "params": {"event": "survived_night"},
                    },
                },
                "shaping": {
                    "repeated_action": {
                        "kind": "streak_penalty", "value": -0.01,
                        "params": {"threshold": 20},
                    },
                    "idle": {
                        "kind": "idle_penalty", "value": -0.05,
                        "params": {"threshold": 40, "low_health_threshold": 10.0},
                    },
                    "spinning": {
                        "kind": "spinning_penalty", "value": -0.1,
                        "params": {"window": 24, "actions": ["LOOK_LEFT", "LOOK_RIGHT"]},
                    },
                    "no_novelty": {
                        "kind": "no_novelty_penalty", "value": -0.1,
                        "params": {"ticks": 200},
                    },
                },
            },
        },
        source="<default_profile>",
    )


def load_reward_profile(path: str) -> RewardProfile:
    """Load and validate a reward profile from a `.yaml`/`.yml`/`.json` file.

    Fails fast with :class:`RewardProfileError` (or a wrapped parse error)
    at startup rather than mid-run.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise _err(path, f"could not read profile file: {exc}") from exc

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a core dep
            raise _err(
                path, "PyYAML is required to load .yaml/.yml reward profiles"
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _err(path, f"invalid YAML: {exc}") from exc
    elif ext == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _err(path, f"invalid JSON: {exc}") from exc
    else:
        raise _err(path, f"unsupported extension {ext!r}; expected .yaml, .yml or .json")

    if data is None:
        raise _err(path, "profile file is empty")
    return reward_profile_from_dict(data, source=path)
