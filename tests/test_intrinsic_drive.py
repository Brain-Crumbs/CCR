"""Intrinsic-drive acceptance tests (issue #61): "risk-gated surprise-
seeking intrinsic drive: reward learning progress and safe novelty, punish
predicted pain."

Covers the issue's acceptance criteria not already exercised by
``tests/test_modulation.py`` (the risk-gate sigmoid shape) or
``tests/test_reward_engine.py``/``tests/test_reward_profile.py`` (the
intrinsic-slot wiring itself):

- the noisy-TV problem: irreducible random values do not dominate intrinsic
  reward once the learning-progress signal has plateaued;
- predicted-risk aversion fires *before* any damage event, driven purely by
  predicted (not realized) risk;
- the simulated three-region test world named in
  ``docs/neural-stream-agent.md``'s success criteria: "the intrinsic drive
  demonstrably prefers novel low-risk situations over both boring and
  dangerous ones" -- with the drive enabled, occupancy ranks
  novel-low-risk > boring and novel-low-risk > novel-high-risk; with the
  drive disabled, no such preference. Statistical comparison over N
  episodes uses the same mean +/- CI methodology as issue #44's harness
  (``training.statistical_evaluation``), not a single seed;
- end-to-end runtime wiring: `internal.*` payloads actually reach a
  profile's intrinsic components in a live `CognitiveRuntime` run (not just
  in hand-built `ProfileRewardEngine` unit tests above), and replaying a
  profile-driven session reproduces `reward.scalar` exactly, including its
  intrinsic components.
"""

from __future__ import annotations

import os
import random

import pytest

from cognitive_runtime.core.action import NULL_ACTION
from cognitive_runtime.core.modulation import ModulationTracker, safe_gate
from cognitive_runtime.core.novelty import combine_novelty
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.reward_engine import ProfileRewardEngine
from cognitive_runtime.programs.minecraft.reward_profile import (
    load_reward_profile,
    reward_profile_from_dict,
)
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
from cognitive_runtime.tools.replay_runner import replay_session
from cognitive_runtime.training.statistical_evaluation import _mean_ci

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- noisy-TV


def _learning_progress_only_profile():
    return reward_profile_from_dict({
        "name": "lp-only",
        "tiers": {},
        "intrinsic": {
            "learning_progress": {"stream": "internal.learning_progress", "weight": 1.0},
        },
        "normalization": {"method": "none", "clip": None},
    })


def _feed_learning_progress(engine, tracker, errors):
    """Run `errors` through `tracker`/`engine` and return the per-tick raw
    reward totals (the same `learning_progress` component every tick, since
    the profile has nothing else in it)."""
    totals = []
    for tick, error in enumerate(errors):
        signals = tracker.update(Prediction(risk=0.0, prediction_error=error), None, 0.0)
        events = [
            StreamEvent(
                "internal.learning_progress", "event", float(tick), tick,
                {"value": signals.learning_progress},
            ),
        ]
        reward = engine.evaluate_stream_window(events, NULL_ACTION)
        totals.append(reward.value)
    return totals


def test_noisy_tv_does_not_dominate_intrinsic_reward_after_plateau():
    rng = random.Random(0)
    noisy_errors = [0.5 + rng.uniform(-0.15, 0.15) for _ in range(500)]
    # A genuinely learnable signal over the same span, for contrast: error
    # steadily decreasing instead of oscillating around a fixed mean.
    learning_errors = [max(0.01, 0.9 - 0.0015 * t) for t in range(500)]

    noisy_totals = _feed_learning_progress(
        ProfileRewardEngine(_learning_progress_only_profile()), ModulationTracker(), noisy_errors,
    )
    learning_totals = _feed_learning_progress(
        ProfileRewardEngine(_learning_progress_only_profile()), ModulationTracker(),
        learning_errors,
    )

    # "After the learning-progress signal plateaus": for constant noise
    # around a fixed mean, the EMA settles almost immediately (issue #58's
    # own `test_learning_progress_is_near_zero_when_error_is_noisy_but_static`
    # confirms this), so ticks [200, 500) are well past any startup
    # transient in both sequences.
    noisy_tail = noisy_totals[200:]
    learning_tail = learning_totals[200:]

    noisy_tail_total = sum(noisy_tail)
    learning_tail_total = sum(learning_tail)

    # The learnable signal keeps earning real reward; irreducible noise
    # nets out close to zero and stays a small fraction of it -- it does
    # not dominate.
    assert learning_tail_total > 5.0
    assert abs(noisy_tail_total) < 0.1 * learning_tail_total


# ----------------------------------------------------- predicted-risk aversion


def test_predicted_risk_aversion_fires_before_any_damage_event():
    """Rising `internal.risk` produces negative intrinsic shaping *before*
    any damage event lands -- no damage event is ever emitted in this test,
    only a climbing predicted-risk stream, isolating the anticipatory
    (pre-damage) nature of the term."""
    profile = reward_profile_from_dict({
        "name": "risk-aversion-only",
        "tiers": {},
        "intrinsic": {
            "predicted_risk_aversion": {
                "stream": "internal.predicted_risk_aversion", "weight": 1.0,
            },
        },
        "normalization": {"method": "none", "clip": None},
    })
    engine = ProfileRewardEngine(profile)
    tracker = ModulationTracker()

    rewards = []
    for tick, risk in enumerate([0.0, 0.1, 0.2, 0.4, 0.6, 0.8]):
        signals = tracker.update(Prediction(risk=risk), None, 0.0)
        events = [
            StreamEvent(
                "internal.predicted_risk_aversion", "event", float(tick), tick,
                {"value": signals.predicted_risk_aversion},
            ),
        ]
        reward = engine.evaluate_stream_window(events, NULL_ACTION)
        assert "died" not in reward.events  # no damage/death ever occurs
        rewards.append(reward.value)

    assert rewards[0] == pytest.approx(0.0)  # zero predicted risk -> no shaping
    # Monotonically more negative as predicted risk climbs.
    assert all(a >= b for a, b in zip(rewards, rewards[1:]))
    assert rewards[-1] < -0.5


# --------------------------------------------------- three-region test world

#: (risk, prediction_error) the world model would report in each region.
#: "a" is fully known (nothing left to predict, no danger); "b" and "c" are
#: equally novel, but "c" is dangerous. `entity_surprise` is left out of
#: `combine_novelty` throughout (`None`) -- irrelevant to this formula.
_REGIONS = {
    "a_boring": (0.02, 0.02),
    "b_novel_low_risk": (0.05, 0.6),
    "c_novel_high_risk": (0.85, 0.6),
}


def _region_intrinsic_score(risk: float, prediction_error: float) -> float:
    """`w_nov * safe_novelty + w_risk * predicted_risk_aversion` (both
    weights 1.0) for a region, straight off the production `safe_gate`/
    `combine_novelty` functions -- `learning_progress` is deliberately left
    out (it is not region-conditioned in this synthetic world and is
    already covered by the noisy-TV test above)."""
    novelty = combine_novelty(prediction_error, None)
    gate = safe_gate(risk)
    safe_novelty = novelty * gate
    predicted_risk_aversion = -risk
    return safe_novelty + predicted_risk_aversion


def _simulate_occupancy(seed: int, ticks: int, *, drive_enabled: bool, epsilon: float = 0.15):
    """One episode in the three-region world: at each tick the agent either
    explores to a uniformly random region (probability `epsilon`, or always
    when the drive is disabled -- there is no signal to prefer one region
    over another) or greedily moves to the region with the highest
    intrinsic score. Returns occupancy fraction per region."""
    rng = random.Random(seed)
    names = list(_REGIONS)
    counts = {name: 0 for name in names}
    current = names[0]
    for _ in range(ticks):
        if not drive_enabled or rng.random() < epsilon:
            current = rng.choice(names)
        else:
            current = max(names, key=lambda n: _region_intrinsic_score(*_REGIONS[n]))
        counts[current] += 1
    return {name: count / ticks for name, count in counts.items()}


def _occupancy_statistics(*, drive_enabled: bool, episodes: int = 20, ticks: int = 40):
    per_region = {name: [] for name in _REGIONS}
    for episode in range(episodes):
        fractions = _simulate_occupancy(seed=episode, ticks=ticks, drive_enabled=drive_enabled)
        for name, frac in fractions.items():
            per_region[name].append(frac)
    return {name: _mean_ci(values) for name, values in per_region.items()}


def test_three_region_world_prefers_novel_low_risk_when_drive_enabled():
    stats = _occupancy_statistics(drive_enabled=True)
    boring, low_risk, high_risk = stats["a_boring"], stats["b_novel_low_risk"], stats["c_novel_high_risk"]

    assert low_risk.mean > boring.mean
    assert low_risk.mean > high_risk.mean
    # Statistically significant, not just a higher point estimate (issue
    # #44's own convention: non-overlapping CIs on the better side).
    assert not low_risk.overlaps(boring)
    assert not low_risk.overlaps(high_risk)


def test_three_region_world_shows_no_preference_when_drive_disabled():
    stats = _occupancy_statistics(drive_enabled=False)
    boring, low_risk, high_risk = stats["a_boring"], stats["b_novel_low_risk"], stats["c_novel_high_risk"]

    # No intrinsic signal to prefer one region over another -> occupancy
    # stays statistically indistinguishable across all three.
    assert low_risk.overlaps(boring)
    assert low_risk.overlaps(high_risk)
    assert low_risk.mean == pytest.approx(1.0 / 3.0, abs=0.15)


# ----------------------------------------------- end-to-end runtime wiring


def test_live_runtime_feeds_internal_streams_into_intrinsic_reward(tmp_path):
    """`internal.*` payloads are computed by `CognitiveRuntime` itself,
    outside the Program -- unlike the hand-built `ProfileRewardEngine`
    tests above, this drives a real `CognitiveRuntime` episode and checks
    the intrinsic component actually shows up in the recorded reward, i.e.
    `Program.observe_external_streams` is really being called."""
    profile = load_reward_profile(os.path.join(REPO_ROOT, "goals", "intrinsic_only.yaml"))
    config = {"episode_ticks": 40, "world_size": 32, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=40,
        record_dir=str(tmp_path), session_id="intrinsic-live", program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config, reward_profile=profile),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
    ).run()

    session_dir = os.path.join(str(tmp_path), "intrinsic-live")
    saw_nonzero_reward_component = False
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        for record in sensory:
            if record.get("stream_id") != "reward.scalar" or record.get("elided"):
                continue
            components = record.get("payload", {}).get("components", {})
            if any(v != 0 for v in components.values()):
                saw_nonzero_reward_component = True
    assert saw_nonzero_reward_component


def test_profile_driven_session_replays_reward_scalar_exactly(tmp_path):
    """Replaying a session recorded with an intrinsic-slot profile must
    reproduce `reward.scalar` byte-for-byte, including the intrinsic
    components fed in from `internal.*` (issue #61 found this broken: the
    reward engine only ever saw the Program's own tick events, never the
    runtime-computed streams)."""
    profile = load_reward_profile(os.path.join(REPO_ROOT, "goals", "intrinsic_only.yaml"))
    config = {"episode_ticks": 40, "world_size": 32, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=40,
        record_dir=str(tmp_path), session_id="intrinsic-replay", program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config, reward_profile=profile),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
    ).run()

    session_dir = os.path.join(str(tmp_path), "intrinsic-replay")
    results = replay_session(session_dir, reward_profile=profile)
    assert len(results) == 1
    assert results[0].matched, results[0].notes

    # Replaying without the matching profile (or the wrong one) must fail
    # loudly instead of silently scoring against the wrong reward function.
    with pytest.raises(ValueError, match="reward profile"):
        replay_session(session_dir)
    other_profile = load_reward_profile(os.path.join(REPO_ROOT, "goals", "survival.yaml"))
    with pytest.raises(ValueError, match="content_hash"):
        replay_session(session_dir, reward_profile=other_profile)
