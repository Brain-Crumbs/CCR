"""Remote (real-Minecraft) backend tests, driven through the Python fake bridge.

The fake bridge (``bridge/fake/sim_bridge.py``) speaks the exact JSON protocol
the Node mineflayer bridge speaks, backed by the deterministic simulated world.
That lets these tests exercise the whole remote path — subprocess management,
JSON framing, event translation, status caching, error handling — with no
Minecraft and no Node, and cross-check that the remote path reproduces the
in-process backend byte-for-byte.
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cognitive_runtime.core.action import Action
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.backend import SimulatedBackend
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.remote import (
    BridgeError,
    RemoteBridge,
    RemoteMinecraftBackend,
)
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import NonDeterministicSessionError
from cognitive_runtime.tools.replay_runner import replay_session

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_BRIDGE = REPO_ROOT / "bridge" / "fake" / "sim_bridge.py"
BRIDGE_DIR = REPO_ROOT / "bridge" / "mineflayer"

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32}
NIGHT_CONFIG = {"episode_ticks": 600, "world_size": 32, "day_length": 400, "start_time": 300}


def _fake_bridge_cmd():
    return [sys.executable, str(FAKE_BRIDGE)]


def _remote_backend(config: SurvivalBoxConfig) -> RemoteMinecraftBackend:
    return RemoteMinecraftBackend(config, bridge=RemoteBridge(_fake_bridge_cmd()))


# ---------------------------------------------------- determinism cross-check


@pytest.mark.parametrize("cfg", [FAST_CONFIG, NIGHT_CONFIG])
def test_remote_via_fake_bridge_matches_simulated_backend(cfg):
    """remote-over-fake-bridge must reproduce the in-process SimulatedBackend
    tick for tick: identical events, observation hashes, tick, death and
    stats — proving the whole wire path is faithful."""
    config = SurvivalBoxConfig.from_dict(cfg)
    remote = _remote_backend(config)
    sim = SimulatedBackend(config)
    # A fixed action sequence, independent of observations (identical anyway).
    rng = random.Random(1234)
    actions = [rng.choice(ACTION_SPACE) for _ in range(400)]

    try:
        remote.reset(7)
        sim.reset(7)
        assert remote.observe(0.0).hash() == sim.observe(0.0).hash()
        for i, action in enumerate(actions):
            assert remote.step(action) == sim.step(action), f"events differ at {i}"
            assert remote.tick() == sim.tick()
            assert remote.is_dead() == sim.is_dead()
            assert remote.death_reason() == sim.death_reason()
            r_obs = remote.observe(remote.tick() * 0.05)
            s_obs = sim.observe(sim.tick() * 0.05)
            assert r_obs.hash() == s_obs.hash(), f"observation differs at tick {i}"
            if remote.is_dead():
                break
        assert remote.stats() == sim.stats()
    finally:
        remote.close()


# ---------------------------------------------------------- protocol surface


def test_reset_step_observe_and_status_caching():
    backend = _remote_backend(SurvivalBoxConfig.from_dict(FAST_CONFIG))
    try:
        backend.reset(3)
        assert backend.tick() == 0 and backend.is_dead() is False
        events = backend.step(Action("MOVE_FORWARD"))
        assert isinstance(events, list)
        assert backend.tick() == 1  # cached from the step response, no round-trip
        obs = backend.observe(backend.tick() * 0.05)
        assert obs.tick == 1
        assert "health" in obs.data and obs.frame is not None
        assert isinstance(backend.stats(), dict)
    finally:
        backend.close()


def test_snapshot_and_restore_are_unsupported():
    backend = _remote_backend(SurvivalBoxConfig.from_dict(FAST_CONFIG))
    try:
        backend.reset(0)
        with pytest.raises(NotImplementedError):
            backend.snapshot()
        with pytest.raises(NotImplementedError):
            backend.restore("snap")
    finally:
        backend.close()


def test_backend_capability_flags():
    assert RemoteMinecraftBackend.deterministic is False
    assert RemoteMinecraftBackend.supports_snapshots is False


# ---------------------------------------- richer event streams over the wire (#40)


def test_fake_bridge_emits_the_new_rich_event_streams_over_the_wire():
    """``SimBridge.handle`` is the exact code the subprocess entrypoint runs;
    driving it in-process proves the new event strings (issue #40) survive a
    real JSON round-trip, so the streams are testable with no server and no
    subprocess."""
    import json

    from bridge.fake.sim_bridge import SimBridge

    bridge = SimBridge()
    reset_resp = json.loads(json.dumps(bridge.handle(
        {"cmd": "reset", "seed": 0, "config": {"world_size": 32}}
    )))
    assert reset_resp["ok"]

    world = bridge._world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "crafting_table"
    world.inventory["log"] = 1

    step_resp = json.loads(json.dumps(bridge.handle(
        {"cmd": "step", "action": {"name": "USE"}}
    )))
    assert step_resp["ok"]
    events = step_resp["events"]
    assert any(e.startswith("container_interact:") for e in events)
    assert any(e.startswith("crafted:") for e in events)
    assert any(e.startswith("item_collected_exact:") for e in events)
    assert any(e == "advancement:sim.craft_item" for e in events)


# --------------------------------------------------------- error handling


def test_missing_bridge_command_fails_clearly():
    bridge = RemoteBridge(["definitely-not-a-real-binary-xyz"])
    with pytest.raises(BridgeError, match="CCR_MINECRAFT_BRIDGE_CMD|bridge"):
        bridge.start()


def test_crashing_bridge_reports_clearly():
    bridge = RemoteBridge([sys.executable, "-c", "import sys; sys.exit(2)"])
    bridge.start()
    with pytest.raises(BridgeError, match="exited|not responding"):
        bridge.request({"cmd": "reset", "seed": 0, "config": {}})


def test_bridge_error_response_becomes_bridge_error():
    bridge = RemoteBridge(_fake_bridge_cmd())
    bridge.start()
    try:
        # 'step' before 'reset' → the fake bridge returns {"ok": false, ...}.
        with pytest.raises(BridgeError, match="step before reset"):
            bridge.request({"cmd": "step", "action": {"name": "NULL"}})
    finally:
        bridge.close()


# ------------------------------------------------- end-to-end through the CLI path


def test_runtime_runs_over_remote_backend(tmp_path, monkeypatch):
    """The full CognitiveRuntime drives the remote backend (spawned the way the
    CLI spawns it, via CCR_MINECRAFT_BRIDGE_CMD) and records a session flagged
    non-deterministic; replay then refuses to re-simulate it."""
    monkeypatch.setenv(
        "CCR_MINECRAFT_BRIDGE_CMD", f"{sys.executable} {FAKE_BRIDGE}"
    )
    program = MinecraftSurvivalBox(config=FAST_CONFIG, backend="remote")
    config = RuntimeConfig(
        episodes=1, seed=5, max_ticks_per_episode=120,
        record_dir=str(tmp_path), session_id="remote", program_config=FAST_CONFIG,
        record_frames=True,
    )
    try:
        summaries = CognitiveRuntime(program=program, policy=_scripted(), config=config).run()
    finally:
        program.close()
    assert summaries[0].duration_ticks == 120
    assert program.metadata().deterministic is False

    session_dir = str(tmp_path / "remote")
    with pytest.raises(NonDeterministicSessionError):
        replay_session(session_dir)


def _scripted():
    from cognitive_runtime.policies import ScriptedSurvivalPolicy
    return ScriptedSurvivalPolicy(seed=1)


# --------------------------------------------------------- node bridge health


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_bridge_syntax_checks():
    for name in ("index.js", "world.js", "actions.js", "observation.js", "blocks.js"):
        result = subprocess.run(
            ["node", "--check", str(BRIDGE_DIR / name)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_frame_codes_match_python():
    """The Node bridge's frame codes must equal world.py BLOCK_IDS, or the
    vision encoder's legend would mis-read remote frames."""
    import json

    from cognitive_runtime.programs.minecraft.world import (
        AGENT_FRAME_ID,
        BLOCK_IDS,
        MOB_FRAME_ID,
    )

    script = (
        "const b=require('./blocks');"
        "process.stdout.write(JSON.stringify("
        "{BLOCK_IDS:b.BLOCK_IDS,MOB:b.MOB_FRAME_ID,AGENT:b.AGENT_FRAME_ID}));"
    )
    result = subprocess.run(
        ["node", "-e", script], cwd=str(BRIDGE_DIR), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    js = json.loads(result.stdout)
    assert js == {"BLOCK_IDS": BLOCK_IDS, "MOB": MOB_FRAME_ID, "AGENT": AGENT_FRAME_ID}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_semantic_mapper_common_minecraft_names():
    import json

    cases = {
        "oak_log": "tree",
        "mangrove_leaves": "tree",
        "wheat": "berry_bush",
        "diamond_ore": "coal_ore",
        "deepslate": "stone",
        "sand": "sand",
        "ice": "water",
        "lava": "barrier",
        "oak_door": "placed_block",
        "glass": "placed_block",
        "chest": "chest",
        "furnace": "furnace",
        "crafting_table": "crafting_table",
        "torch": "placed_block",
        "nether_portal": "portal",
    }
    script = (
        "const b=require('./blocks');"
        f"const cases={json.dumps(cases)};"
        "const out={};"
        "for (const [name,_] of Object.entries(cases)) "
        "out[name]=b.blockToVocab({name,boundingBox:'block'});"
        "out.unknownSolid=b.blockToVocab({name:'modded_dense_block',boundingBox:'block'});"
        "out.unknownOpen=b.blockToVocab({name:'modded_flower',boundingBox:'empty'});"
        "out.zombie=b.entityToSemantic('zombie');"
        "out.cow=b.entityToSemantic('cow');"
        "out.wolf=b.entityToSemantic('wolf');"
        "out.pickaxe=b.itemToVocab('stone_pickaxe');"
        "process.stdout.write(JSON.stringify(out));"
    )
    result = subprocess.run(
        ["node", "-e", script], cwd=str(BRIDGE_DIR), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    for name, vocab in cases.items():
        assert out[name] == vocab
    assert out["unknownSolid"] == "stone"
    assert out["unknownOpen"] == "grass"
    assert out["zombie"] == "hostile_mob"
    assert out["cow"] == "passive_mob"
    assert out["wolf"] == "neutral_mob"
    assert out["pickaxe"] == "stone_pickaxe"
