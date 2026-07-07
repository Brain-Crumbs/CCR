"""Milestone 5+6: demonstrations -> dataset -> behavioral cloning -> learned policy."""

import os

from cognitive_runtime.policies import LearnedPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.training.datasets import build_dataset
from cognitive_runtime.training.features import ACTION_KEYS, FEATURE_NAMES, featurize
from cognitive_runtime.training.imitation import BCModel, train_bc

FAST_CONFIG = {"episode_ticks": 300, "world_size": 32}
# Includes two nights so demonstrations contain combat, fleeing and eating,
# and so weak policies actually die during evaluation.
NIGHT_CONFIG = {
    "episode_ticks": 1200, "world_size": 32, "day_length": 800, "start_time": 300,
}


def _record_session(tmp_path, policy, session_id, config, episodes=2, seed=0):
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=config["episode_ticks"],
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
        record_frames=True,  # keep vision.frame in the log for the latent path
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config), policy=policy, config=runtime_config
    )
    return runtime.run(), os.path.join(str(tmp_path), session_id)


def test_featurizer_is_fixed_width():
    features = featurize({"health": 20, "hunger": 20}, ["MOVE_FORWARD"])
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features)


def test_stream_reconstruction_derives_in_water():
    """in_water is not a stream; it must reconstruct from the centre cell of
    the world.nearby_blocks patch so offline handcrafted features match the
    online observation (train/inference parity)."""
    from cognitive_runtime.training.features import observation_data_from_streams

    dry_patch = [["grass"] * 5 for _ in range(5)]
    wet_patch = [row[:] for row in dry_patch]
    wet_patch[2][2] = "water"  # the agent's own cell

    dry = observation_data_from_streams({"world.nearby_blocks": dry_patch})
    wet = observation_data_from_streams({"world.nearby_blocks": wet_patch})
    empty = observation_data_from_streams({})
    assert dry["in_water"] is False
    assert wet["in_water"] is True
    assert empty["in_water"] is False

    in_water_index = FEATURE_NAMES.index("in_water")
    assert featurize(wet, [])[in_water_index] == 1.0
    assert featurize(dry, [])[in_water_index] == 0.0


def test_dataset_built_from_recorded_traces(tmp_path):
    _, session_dir = _record_session(
        tmp_path, ScriptedSurvivalPolicy(seed=1), "scripted-data", FAST_CONFIG
    )
    # Default representation is the latent path.
    dataset = build_dataset([session_dir])
    assert dataset.representation == "latent"
    assert dataset.layout_hash
    assert len(dataset) == 600  # 2 episodes x 300 ticks
    assert len(dataset.features[0]) == len(dataset.feature_names)
    assert set(dataset.label_counts()) <= set(ACTION_KEYS)

    handcrafted = build_dataset([session_dir], representation="handcrafted")
    assert len(handcrafted.features[0]) == len(FEATURE_NAMES)


def test_bc_learns_to_imitate_scripted_policy(tmp_path):
    _, session_dir = _record_session(
        tmp_path, ScriptedSurvivalPolicy(seed=1), "bc-data", NIGHT_CONFIG,
        episodes=3, seed=100,
    )
    dataset = build_dataset([session_dir])
    model, metrics = train_bc(dataset, epochs=8, seed=0)
    # Must clearly beat both naive baselines.
    assert metrics["train_accuracy"] > metrics["majority_class_baseline"]
    assert metrics["train_balanced_accuracy"] > metrics["random_class_baseline"]

    path = os.path.join(str(tmp_path), "bc.json")
    model.save(path)
    loaded = BCModel.load(path)
    assert loaded.predict_index(dataset.features[0]) == model.predict_index(dataset.features[0])


def test_learned_policy_outperforms_random_baseline(tmp_path):
    """Milestone 6 success criterion: BC beats the random baseline."""
    summaries, session_dir = _record_session(
        tmp_path, ScriptedSurvivalPolicy(seed=1), "clone-data", NIGHT_CONFIG,
        episodes=4, seed=100,
    )
    assert all(s.success for s in summaries), "teacher must survive its own demos"
    dataset = build_dataset([session_dir])
    model, _ = train_bc(dataset, epochs=8, seed=0)

    def evaluate(policy):
        config = RuntimeConfig(
            episodes=3, seed=500, max_ticks_per_episode=NIGHT_CONFIG["episode_ticks"],
            record=False, program_config=NIGHT_CONFIG,
        )
        runtime = CognitiveRuntime(
            program=MinecraftSurvivalBox(config=NIGHT_CONFIG), policy=policy, config=config
        )
        results = runtime.run()
        return sum(s.total_reward for s in results), sum(s.duration_ticks for s in results)

    learned_reward, learned_ticks = evaluate(LearnedPolicy(model))
    random_reward, random_ticks = evaluate(RandomPolicy(ACTION_SPACE, seed=0))
    assert learned_ticks >= random_ticks, "learned policy must survive at least as long"
    assert learned_reward > random_reward, (
        f"learned {learned_reward:.2f} must beat random {random_reward:.2f}"
    )


def _evaluate(policy):
    config = RuntimeConfig(
        episodes=3, seed=500, max_ticks_per_episode=NIGHT_CONFIG["episode_ticks"],
        record=False, program_config=NIGHT_CONFIG,
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=NIGHT_CONFIG), policy=policy, config=config
    )
    results = runtime.run()
    return sum(s.total_reward for s in results), sum(s.duration_ticks for s in results)


def test_latent_bc_matches_or_beats_handcrafted_and_beats_random(tmp_path, capsys):
    """Milestone-6 parity on latent state: BC on the fused latent vector beats
    random and is within noise of (or better than) the hand-featurizer BC."""
    summaries, session_dir = _record_session(
        tmp_path, ScriptedSurvivalPolicy(seed=1), "ab-data", NIGHT_CONFIG,
        episodes=4, seed=100,
    )
    assert all(s.success for s in summaries), "teacher must survive its own demos"

    latent_ds = build_dataset([session_dir], representation="latent")
    hand_ds = build_dataset([session_dir], representation="handcrafted")
    latent_model, _ = train_bc(latent_ds, epochs=8, seed=0)
    hand_model, _ = train_bc(hand_ds, epochs=8, seed=0)

    latent_reward, latent_ticks = _evaluate(LearnedPolicy(latent_model))
    hand_reward, hand_ticks = _evaluate(LearnedPolicy(hand_model))
    random_reward, random_ticks = _evaluate(RandomPolicy(ACTION_SPACE, seed=0))

    with capsys.disabled():
        print(
            f"\n[A/B] latent: reward={latent_reward:.2f} ticks={latent_ticks} "
            f"(dim {len(latent_ds.feature_names)}) | "
            f"handcrafted: reward={hand_reward:.2f} ticks={hand_ticks} "
            f"(dim {len(hand_ds.feature_names)}) | "
            f"random: reward={random_reward:.2f} ticks={random_ticks}"
        )

    # Hard requirement: latent BC beats random on survival ticks and reward.
    assert latent_ticks > random_ticks
    assert latent_reward > random_reward
    # Within noise of the handcrafted baseline: survives at least as long, and
    # earns a comparable share of its reward.
    assert latent_ticks >= hand_ticks
    assert latent_reward >= 0.5 * hand_reward
