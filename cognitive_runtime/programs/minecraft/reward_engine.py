"""Profile-driven reward engine (issue #41).

`ProfileRewardEngine` evaluates a `RewardProfile` (loaded via
`reward_profile.load_reward_profile`) against a tick's stream events,
producing the same `RewardSignal` shape `SurvivalReward` does -- the
adapter can swap one engine for the other without changing anything else.

Where `SurvivalReward` hard-codes each component's math, this engine reads
`kind` off each `ComponentSpec` and dispatches to one of a small set of
generic evaluation rules (`KNOWN_KINDS` in `reward_profile.py`).  A handful
of *sources* pull "the set of interesting keys this tick" out of the same
places `SurvivalReward` does (nearby blocks, biome, position chunk, semantic
events) so novelty/first-of-kind/keyed-repeat components share one
extraction path regardless of kind.

Milestone state (once-only components, capped-novelty seen-sets, running
totals, ...) lives in `self._state`, keyed by component name and tagged with
the component's `scope`.  `reset()` (a new life/episode) clears `life`-scoped
state only; `state_dict()`/`load_state_dict()` (a checkpoint bundle,
interrupt/resume) carry the `brain`-scoped state across process restarts so
"first_diamond" never re-fires just because training was interrupted.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.programs.minecraft.reward_profile import ComponentSpec, RewardProfile
from cognitive_runtime.programs.minecraft.rewards import (
    FOOD_ITEM_NAMES,
    SEMANTIC_EVENT_TRANSLATORS,
)
from cognitive_runtime.programs.minecraft.world import is_tool_or_weapon

_PREDICATES = {
    "is_tool": is_tool_or_weapon,
    "is_food": lambda item: item in FOOD_ITEM_NAMES,
}


class _TickContext:
    """Everything a kind evaluator can read for the current tick."""

    __slots__ = (
        "health", "hunger", "nearby_blocks", "biome", "distance", "mobs_visible",
        "events", "action", "novelty_hash", "position", "died", "latest",
    )

    def __init__(
        self,
        health: float,
        hunger: float,
        nearby_blocks: List[List[str]],
        biome: Optional[str],
        distance: float,
        mobs_visible: bool,
        events: List[str],
        action: Action,
        novelty_hash: str,
        position: Optional[Dict[str, float]],
        latest: Dict[str, Any],
    ) -> None:
        self.health = health
        self.hunger = hunger
        self.nearby_blocks = nearby_blocks
        self.biome = biome
        self.distance = distance
        self.mobs_visible = mobs_visible
        self.events = events
        self.action = action
        self.novelty_hash = novelty_hash
        self.position = position
        self.died = "died" in events
        self.latest = latest


class RunningNormalizer:
    """Welford running mean/std, used to normalize+clip raw reward totals
    before they reach an optimizer (issue #41: "the linear Q TD clip already
    assumes small errors -- huge raw values must never hit an optimizer
    directly")."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return math.sqrt(max(self._m2 / self.count, 0.0)) + self.epsilon

    def normalize(self, value: float) -> float:
        return value / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "m2": self._m2}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.count = int(state.get("count", 0))
        self.mean = float(state.get("mean", 0.0))
        self._m2 = float(state.get("m2", 0.0))


class ProfileRewardEngine:
    def __init__(self, profile: RewardProfile):
        self.profile = profile
        self._components = profile.components()  # name -> (tier, spec)
        self._normalizer = RunningNormalizer(profile.normalization.epsilon)
        self.reset(clear_brain_state=True)

    # ------------------------------------------------------------- lifecycle

    def reset(self, clear_brain_state: bool = False) -> None:
        """Start a new life/episode: clears `scope="life"` milestone state.

        `clear_brain_state=True` additionally wipes `scope="brain"` state --
        used for a genuinely fresh brain, never for an ordinary episode
        boundary.
        """
        if not hasattr(self, "_state"):
            self._state: Dict[str, Dict[str, Any]] = {}
        for name, (_tier, spec) in self._components.items():
            if spec.scope == "life" or clear_brain_state:
                self._state[name] = {}
        if clear_brain_state:
            self._tick = 0
            self._normalizer = RunningNormalizer(self.profile.normalization.epsilon)
        self._latest_streams: Dict[str, Any] = {}
        self._spawn: Optional[Tuple[float, float]] = None
        self._recent_actions: List[str] = []
        self._action_streak = 0
        self._last_action_key: Optional[str] = None
        self._null_streak = 0
        self._seen_obs_hashes: Set[str] = set()
        self._ticks_since_novel = 0
        # Reward-by-tier accounting (issue #44): cumulative reward per tier
        # this life/episode, for the statistical evaluation harness. Always
        # cleared at an episode boundary, like the anti-stagnation trackers
        # above -- it is not brain-scoped milestone state.
        self._tier_totals: Dict[str, float] = {}

    def tier_totals(self) -> Dict[str, float]:
        """Cumulative reward per tier so far this life/episode."""
        return dict(self._tier_totals)

    # --------------------------------------------------------- persistence

    def state_dict(self) -> Dict[str, Any]:
        """`scope="brain"` milestone state + the running normalizer, for the
        checkpoint bundle -- so resume doesn't re-grant one-time rewards."""
        brain_state = {
            name: dict(self._state.get(name, {}))
            for name, (_tier, spec) in self._components.items()
            if spec.scope == "brain"
        }
        return {
            "profile_content_hash": self.profile.content_hash,
            "brain_state": brain_state,
            "normalizer": self._normalizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("profile_content_hash") != self.profile.content_hash:
            raise ValueError(
                "reward engine state was saved for a different reward profile "
                f"(saved={state.get('profile_content_hash')!r}, "
                f"current={self.profile.content_hash!r}); milestone state cannot "
                "be safely reused across incompatible profiles"
            )
        for name, component_state in state.get("brain_state", {}).items():
            self._state[name] = dict(component_state)
        if "normalizer" in state:
            self._normalizer.load_state_dict(state["normalizer"])

    # ------------------------------------------------------------------ eval

    def evaluate(
        self,
        obs_data: Dict[str, Any],
        events: List[str],
        action: Action,
        observation_hash: str,
    ) -> RewardSignal:
        """Legacy pull-style entry point, mirroring `SurvivalReward.evaluate`."""
        ctx = _TickContext(
            health=float(obs_data.get("health", 0.0)),
            hunger=float(obs_data.get("hunger", 0.0)),
            nearby_blocks=obs_data.get("nearby_blocks", []),
            biome=obs_data.get("biome"),
            distance=float(obs_data.get("distance_from_spawn", 0.0)),
            mobs_visible=bool(obs_data.get("mobs")),
            events=list(events),
            action=action,
            novelty_hash=observation_hash,
            position=obs_data.get("position"),
            latest=self._latest_streams,
        )
        return self._evaluate(ctx)

    def prime_stream_state(self, stream_events: List[StreamEvent]) -> None:
        for event in stream_events:
            self._latest_streams[event.stream_id] = event.payload
            if self._spawn is None and event.stream_id == "spatial.position":
                self._spawn = (event.payload["x"], event.payload["z"])

    def observe_external_streams(self, payloads: Dict[str, Any]) -> None:
        """Merge runtime-computed stream payloads (issue #58/#61's
        `internal.*`) into the same latest-value cache `_eval_intrinsic`
        reads from. Unlike `prime_stream_state`, these never arrive as
        `StreamEvent`s the Program itself published -- `CognitiveRuntime`
        calls this directly with the raw `{stream_id: payload}` map."""
        self._latest_streams.update(payloads)

    def evaluate_stream_window(
        self, stream_events: List[StreamEvent], action: Action
    ) -> RewardSignal:
        import hashlib

        semantic_events: List[str] = []
        for event in stream_events:
            self.prime_stream_state([event])
            translated = SEMANTIC_EVENT_TRANSLATORS.get(event.stream_id)
            if translated is not None:
                semantic_events.append(translated(event.payload))

        latest = self._latest_streams
        position = latest.get("spatial.position")
        distance = 0.0
        if position is not None and self._spawn is not None:
            distance = round(math.dist((position["x"], position["z"]), self._spawn), 2)
        window_digest = hashlib.sha1(
            "".join(e.hash() for e in stream_events).encode("utf-8")
        ).hexdigest()

        ctx = _TickContext(
            health=float(latest.get("body.health", 0.0)),
            hunger=float(latest.get("body.hunger", 0.0)),
            nearby_blocks=latest.get("world.nearby_blocks", []),
            biome=latest.get("world.biome"),
            distance=distance,
            mobs_visible=bool(latest.get("vision.entities")),
            events=semantic_events,
            action=action,
            novelty_hash=window_digest,
            position=position,
            latest=latest,
        )
        return self._evaluate(ctx)

    def _evaluate(self, ctx: _TickContext) -> RewardSignal:
        self._tick += 1
        # Anti-stagnation trackers (action streak, null streak, novelty) must
        # reflect *this* tick's action/hash before streak/spinning/no-novelty
        # kinds read them below -- mirrors SurvivalReward's update-then-check
        # ordering.
        self._update_anti_stagnation_trackers(ctx)

        components: Dict[str, float] = {}
        for name, (_tier, spec) in self._components.items():
            if spec.disabled:
                continue
            if spec.stream is not None:
                self._eval_intrinsic(name, spec, ctx, components)
            else:
                getattr(self, f"_kind_{spec.kind}")(name, spec, ctx, components)

        for name, value in components.items():
            tier = self._components[name][0]
            self._tier_totals[tier] = round(self._tier_totals.get(tier, 0.0) + value, 6)

        raw_value = round(sum(components.values()), 6)
        self._normalizer.update(raw_value)
        norm = self.profile.normalization
        if norm.method == "running" and self._normalizer.count >= norm.warmup_ticks:
            training_value = self._normalizer.normalize(raw_value)
        else:
            training_value = raw_value
        if norm.clip is not None:
            training_value = max(-norm.clip, min(norm.clip, training_value))

        return RewardSignal.from_components(
            components, events=tuple(ctx.events), training_value=training_value
        )

    # ------------------------------------------------------------ bookkeeping

    def _capped_add(self, name: str, spec: ComponentSpec, raw: float) -> float:
        """Add `raw` to `name`'s running total, bounded by `spec.cap`.

        Caps only bound *positive* reward (anti-farming); penalties
        (negative `raw`, e.g. damage/critical-vitals components) pass
        through uncapped -- there is no farming concern to bound, and
        capping a penalty's magnitude would silently blunt it.
        """
        if raw == 0:
            return 0.0
        if raw < 0:
            return raw
        state = self._state.setdefault(name, {})
        if spec.cap is None:
            state["_total"] = state.get("_total", 0.0) + raw
            return raw
        total = state.get("_total", 0.0)
        bonus = min(raw, spec.cap - total)
        if bonus <= 0:
            return 0.0
        state["_total"] = total + bonus
        return bonus

    def _cooldown_ok(self, name: str, spec: ComponentSpec) -> bool:
        if spec.cooldown_ticks <= 0:
            return True
        last = self._state.get(name, {}).get("_last_trigger_tick")
        return last is None or (self._tick - last) >= spec.cooldown_ticks

    def _mark_triggered(self, name: str) -> None:
        self._state.setdefault(name, {})["_last_trigger_tick"] = self._tick

    def _extract_keys(self, source: str, ctx: _TickContext, params: Dict[str, Any]) -> List[str]:
        if source == "nearby_blocks":
            return [block for row in ctx.nearby_blocks for block in row]
        if source == "biome":
            return [ctx.biome] if ctx.biome else []
        if source == "position_chunk":
            if ctx.position is None:
                return []
            chunk_size = float(params.get("chunk_size", 8.0))
            chunk = (
                math.floor(ctx.position["x"] / chunk_size),
                math.floor(ctx.position["z"] / chunk_size),
            )
            return [f"{chunk[0]},{chunk[1]}"]
        if source.startswith("event:"):
            prefix = source.split(":", 1)[1]
            keys = []
            for event in ctx.events:
                if not event.startswith(f"{prefix}:"):
                    continue
                raw = event.split(":", 1)[1]
                if prefix == "crafted":
                    raw = json.loads(raw)["recipe"]
                keys.append(raw)
            return keys
        return []

    # ---------------------------------------------------------------- kinds

    def _kind_tick(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        if ctx.died:
            return
        bonus = self._capped_add(name, spec, spec.value)
        if bonus:
            components[name] = bonus

    def _kind_death(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        if ctx.died:
            components[name] = spec.value

    def _kind_event_count(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        prefix = spec.params["event_prefix"]
        count = sum(1 for e in ctx.events if e == prefix or e.startswith(f"{prefix}:"))
        if count == 0 or not self._cooldown_ok(name, spec):
            return
        bonus = self._capped_add(name, spec, spec.value * count)
        if bonus:
            components[name] = bonus
            self._mark_triggered(name)

    def _kind_delta_decrease(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        field = spec.params["field"]
        cur = getattr(ctx, field)
        state = self._state.setdefault(name, {})
        prev = state.get("_prev")
        if prev is not None:
            lost = int(prev) - int(cur)
            if lost > 0:
                bonus = self._capped_add(name, spec, spec.value * lost)
                if bonus:
                    components[name] = bonus
        state["_prev"] = cur

    def _kind_threshold_enter(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        field = spec.params["field"]
        threshold = spec.params["threshold"]
        cur = getattr(ctx, field)
        state = self._state.setdefault(name, {})
        was_below = state.get("_below", False)
        is_below = cur < threshold
        if is_below and not was_below and not ctx.died:
            bonus = self._capped_add(name, spec, spec.value)
            if bonus:
                components[name] = bonus
        state["_below"] = is_below

    def _kind_periodic_no_event(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        prefix = spec.params["event_prefix"]
        window = spec.params["window"]
        min_field = spec.params["min_field"]
        min_value = spec.params["min_value"]
        state = self._state.setdefault(name, {})
        matched = any(e == prefix or e.startswith(f"{prefix}:") for e in ctx.events)
        if matched:
            state["_streak"] = 0
            return
        streak = state.get("_streak", 0) + 1
        state["_streak"] = streak
        if streak % window == 0 and getattr(ctx, min_field) >= min_value:
            bonus = self._capped_add(name, spec, spec.value)
            if bonus:
                components[name] = bonus

    def _kind_capped_novelty(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        source = spec.params["source"]
        keys = self._extract_keys(source, ctx, spec.params)
        if not keys:
            return
        state = self._state.setdefault(name, {})
        seen: Set[str] = state.setdefault("_seen", set())
        total_bonus = 0.0
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            total_bonus += self._capped_add(name, spec, spec.value)
        if total_bonus:
            components[name] = round(total_bonus, 6)

    def _kind_distance_ladder(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        unit = spec.params["unit"]
        state = self._state.setdefault(name, {})
        max_rewarded = state.get("_max_rewarded", 0.0)
        total_bonus = 0.0
        while ctx.distance >= max_rewarded + unit and state.get("_total", 0.0) < spec.cap:
            max_rewarded += unit
            total_bonus += self._capped_add(name, spec, spec.value)
        state["_max_rewarded"] = max_rewarded
        if total_bonus:
            components[name] = round(total_bonus, 6)

    def _kind_once_predicate(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        source = spec.params["source"]
        predicate = _PREDICATES[spec.params["predicate"]]
        state = self._state.setdefault(name, {})
        if state.get("_fired"):
            return
        for key in self._extract_keys(source, ctx, spec.params):
            if predicate(key):
                state["_fired"] = True
                components[name] = spec.value
                return

    def _kind_once_event(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        event_name = spec.params["event"]
        state = self._state.setdefault(name, {})
        if state.get("_fired"):
            return
        if event_name in ctx.events:
            state["_fired"] = True
            components[name] = spec.value

    def _kind_decaying_repeat(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        source = spec.params["source"]
        keys = self._extract_keys(source, ctx, spec.params)
        if not keys or not self._cooldown_ok(name, spec):
            return
        state = self._state.setdefault(name, {})
        counts: Dict[str, int] = state.setdefault("_counts", {})
        total_bonus = 0.0
        for key in keys:
            n = counts.get(key, 0)
            raw = max(spec.value * (spec.decay ** n), spec.decay_floor)
            total_bonus += self._capped_add(name, spec, raw)
            counts[key] = n + 1
        if total_bonus:
            components[name] = round(total_bonus, 6)
            self._mark_triggered(name)

    def _kind_streak_penalty(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        if self._action_streak > spec.params["threshold"]:
            components[name] = spec.value

    def _kind_idle_penalty(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        threshold = spec.params["threshold"]
        low_health = spec.params.get("low_health_threshold", 10.0)
        threatened = ctx.mobs_visible or ctx.health < low_health
        if self._null_streak > threshold and not threatened:
            components[name] = spec.value

    def _kind_spinning_penalty(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        window = spec.params["window"]
        allowed = set(spec.params["actions"])
        recent = self._recent_actions[-window:]
        if len(recent) == window and all(k in allowed for k in recent):
            components[name] = spec.value
            self._recent_actions.clear()

    def _kind_no_novelty_penalty(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        if self._ticks_since_novel >= spec.params["ticks"]:
            components[name] = spec.value
            self._ticks_since_novel = 0

    def _eval_intrinsic(self, name: str, spec: ComponentSpec, ctx: _TickContext, components: Dict[str, float]) -> None:
        payload = ctx.latest.get(spec.stream)
        if payload is None:
            return
        raw_value = payload.get("value", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_value, (int, float)):
            return
        bonus = self._capped_add(name, spec, spec.weight * float(raw_value))
        if bonus:
            components[name] = round(bonus, 6)

    # -------------------------------------------------- anti-stagnation state

    def _update_anti_stagnation_trackers(self, ctx: _TickContext) -> None:
        key = ctx.action.key()
        if key == self._last_action_key:
            self._action_streak += 1
        else:
            self._action_streak = 1
            self._last_action_key = key
        self._recent_actions.append(key)

        if ctx.action.is_null:
            self._null_streak += 1
        else:
            self._null_streak = 0

        if ctx.novelty_hash in self._seen_obs_hashes:
            self._ticks_since_novel += 1
        else:
            self._seen_obs_hashes.add(ctx.novelty_hash)
            self._ticks_since_novel = 0
