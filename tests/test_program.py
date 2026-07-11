"""MinecraftSurvivalBox Program interface tests: determinism, snapshots."""

from cognitive_runtime.core.action import Action, NULL_ACTION
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE

FAST_CONFIG = {"episode_ticks": 300, "world_size": 32}


def _run_hashes(seed, actions):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=seed)
    hashes = []
    for action in actions:
        hashes.append(program.observe().hash())
        program.act(action)
        program.reward()
    return hashes


def test_same_seed_same_actions_is_deterministic():
    actions = [Action("MOVE_FORWARD"), Action("LOOK_RIGHT"), Action("ATTACK")] * 30
    assert _run_hashes(7, actions) == _run_hashes(7, actions)


def test_different_seeds_differ():
    actions = [Action("MOVE_FORWARD")] * 20
    assert _run_hashes(1, actions) != _run_hashes(2, actions)


def test_observation_contains_mvp_fields():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=0)
    obs = program.observe()
    for key in ("health", "hunger", "oxygen", "position", "yaw", "pitch",
                "inventory", "selected_slot", "nearby_blocks"):
        assert key in obs.data, key
    assert obs.frame is not None and len(obs.frame) == 11
    assert obs.data["health"] == 20


def test_action_space_matches_mvp_plan():
    names = {a.name for a in ACTION_SPACE}
    expected = {
        "NULL", "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
        "JUMP", "SNEAK", "SPRINT", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP",
        "LOOK_DOWN", "ATTACK", "USE", "SELECT_HOTBAR_SLOT",
        # Issue #42: crafting, inventory, equip, targeted placement/use, interact.
        "INTERACT", "OPEN_INVENTORY", "CLOSE_INVENTORY", "EQUIP_ITEM",
        "PLACE_BLOCK", "USE_ITEM", "MOVE_INVENTORY_ITEM", "CRAFT",
    }
    assert names == expected


def test_invalid_action_rejected():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=0)
    assert not program.act(Action("EXPLODE")).ok  # wholly unknown action name
    assert not program.act(Action("CRAFT")).ok  # missing/unknown recipe param
    assert not program.act(Action.make("SELECT_HOTBAR_SLOT", slot=99)).ok
    assert program.act(NULL_ACTION).ok


def test_world_level_action_rejected_events():
    """Actions that pass the adapter's shape check (a valid name/slot) but
    are invalid given world state (empty slot, no materials, ...) still
    apply -- ``ok=True`` -- but surface an ``action_rejected:`` event
    (issue #42), the same pattern crafting-without-materials already used."""
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=0)
    result = program.act(Action.make("EQUIP_ITEM", slot=0))  # slot 0 starts empty
    assert result.ok
    assert any(e.startswith("action_rejected:") for e in result.info["events"])


def test_snapshot_restore():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=3)
    for _ in range(10):
        program.act(Action("MOVE_FORWARD"))
        program.reward()
    snap = program.snapshot()
    hash_at_snap = program.observe().hash()
    for _ in range(10):
        program.act(Action("MOVE_FORWARD"))
        program.reward()
    assert program.observe().hash() != hash_at_snap
    program.restore(snap)
    assert program.observe().hash() == hash_at_snap


def test_episode_completes_at_tick_limit():
    program = MinecraftSurvivalBox(config={"episode_ticks": 50, "world_size": 32})
    program.reset(seed=0)
    ticks = 0
    while not program.is_complete() and ticks < 1000:
        program.act(NULL_ACTION)
        program.reward()
        ticks += 1
    assert program.is_complete()
    stats = program.episode_stats()
    assert stats["final_tick"] == 50
    assert stats["success"] is True


# --------------------------------------------- backend capabilities (issue #14)


def test_simulated_backend_is_deterministic_and_snapshottable():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    meta = program.metadata()
    assert meta.deterministic is True
    assert "simulated" in meta.tags


def test_remote_backend_is_registered_and_non_deterministic():
    # The remote backend is implemented (see tests/test_remote_backend.py for
    # the wire-protocol coverage); here we only assert it is registered and
    # advertises live-world semantics without spawning a bridge.
    from cognitive_runtime.programs.minecraft.adapter import BACKENDS
    from cognitive_runtime.programs.minecraft.remote import RemoteMinecraftBackend

    assert BACKENDS["remote"] is RemoteMinecraftBackend
    assert RemoteMinecraftBackend.deterministic is False
    assert RemoteMinecraftBackend.supports_snapshots is False


def test_snapshot_guard_for_backends_without_snapshot_support():
    import pytest

    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program._backend.supports_snapshots = False  # e.g. a live-server backend
    with pytest.raises(NotImplementedError, match="does not support snapshots"):
        program.snapshot()
    with pytest.raises(NotImplementedError, match="does not support snapshots"):
        program.restore("snap-0-0")
