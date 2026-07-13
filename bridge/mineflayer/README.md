# Mineflayer bridge (real-Minecraft backend)

This bridge lets the Continuous Cognitive Runtime inhabit a **real Minecraft
server**. It is a small Node.js program that drives a headless
[mineflayer](https://github.com/PrismarineJS/mineflayer) client and speaks the
line-delimited JSON protocol the runtime's `RemoteMinecraftBackend` expects
(`cognitive_runtime/programs/minecraft/remote.py`). Nothing in the Python
runtime changes — you only select `--backend remote`.

```
runtime  ──JSON over stdio──►  bridge (this)  ──mineflayer──►  Minecraft server
```

## What the bridge does

- **Actions.** Maps the SurvivalBox action space (`MOVE_*`, `JUMP`, `SPRINT`,
  `SNEAK`, `LOOK_*`, `ATTACK`, `USE`, `SELECT_HOTBAR_SLOT`) onto mineflayer
  controls, `bot.dig`, `bot.attack`, `bot.placeBlock`, `bot.consume`.
  Issue #42 adds crafting, inventory management, equip, targeted
  placement/use, and generic interaction: `CRAFT(recipe)` (`bot.recipesFor`/
  `bot.craft`/`bot.openFurnace`, keyed to the same recipe ids as `world.py`'s
  `RECIPES`), `OPEN_INVENTORY`/`CLOSE_INVENTORY` (a tracked flag -- mineflayer
  has no inventory-GUI concept of its own), `EQUIP_ITEM(slot)`/
  `PLACE_BLOCK(slot)`/`USE_ITEM(slot)` (act on a specific hotbar slot via
  `bot.setQuickBarSlot`, independent of what's currently selected),
  `MOVE_INVENTORY_ITEM(from_slot, to_slot)` (`bot.moveSlotItem`), and
  `INTERACT` (containers behave like `USE`'s container branch but never
  auto-craft; `_door`-suffixed blocks toggle via `bot.activateBlock`;
  passive/neutral entities -- villagers included -- via `bot.activateEntity`).
  Every rejection path (empty slot, wrong container, insufficient materials,
  nothing to interact with, ...) calls `hooks.onRejected(reason)`, which
  `world.js` turns into an `action_rejected:<reason>` event -- the same
  `event.action_rejected` stream the simulated backend and the adapter's
  shape validation already publish.
- **Observations.** Builds the exact observation shape the runtime expects
  (vitals, position, yaw/pitch, 5×5 nearby-block patch, `front_block`, hostile
  `mobs` as distance/bearing, an 11×11 top-down `frame`) with block/biome/item
  names mapped into the SurvivalBox vocabulary (`blocks.js`).
- **Semantic events.** Synthesizes the event vocabulary the reward function
  consumes (`damage:<reason>`, `new_item:<item>`, `broke_block:<block>`,
  `placed_block`, `ate_food`, `entered_shelter`, `survived_night`, `died`) by
  diffing state across ticks and watching mineflayer activity callbacks.
- **Richer event streams.** Also emits the exact-identity / progression event
  vocabulary from issue #40: `item_collected_exact:<item>:<count>` (JSON),
  `block_broken_exact` / `block_placed_exact` (block id + position, JSON),
  `container_interact` (crafting table / furnace / chest, JSON),
  `crafted:<recipe>` (JSON inputs/outputs) from USE-ing a crafting table or
  furnace, `advancement:<vanilla id>` forwarded from the server's advancement
  packets, `dimension_changed:<from>:<to>` on respawn, `biome_entered:<biome>`,
  and a best-effort `structure_discovered:<name>` heuristic (nearby
  structure-typical marker blocks — see `STRUCTURE_MARKERS` in `blocks.js`).
  These map onto the runtime's `event.*` streams in
  `cognitive_runtime/programs/minecraft/streams.py`; see that module for the
  full schema. The simulated backend exercises the same streams with its own
  minimal, deterministic mechanics (a fixed crafting table/furnace/chest/
  portal and three structure markers placed at fixed world coordinates, and
  `sim.*` advancement ids) so they are covered by tests with no server.

Day/night (`time_of_day`, `is_night`) is **synthesized from the tick and
`--day-length`/`--start-time`**, exactly like the simulated world, so those
flags behave identically regardless of the server clock.

- **Mouse/look control history (issue #32).** The `input.mouse_look` stream
  ({`d_yaw`, `d_pitch`} per tick) is published for *both* backends by the
  Python `MinecraftSurvivalBox` adapter
  (`cognitive_runtime/programs/minecraft/adapter.py`), derived from the
  `LOOK_*` action taken that tick -- the bridge needs no changes for it,
  since the sim and this bridge apply the same `LOOK_STEP`/`PITCH_STEP`
  magnitudes (`world.py` / `actions.js`).

## First-person pixels (optional)

Remote runs request first-person viewer pixels by default
(`pixel_source="viewer"`). The top-down semantic `frame` still publishes as
`vision.frame.grid`, but pixel-vision training consumes `vision.frame.pixels`.

This is **best-effort**: `prismarine-viewer` plus `node-canvas-webgl` pull in
native/headless-GL pieces that many hosts (containers, CI, a sandbox with no
GPU/X server) cannot build or run. Any failure to install, initialize, or
capture a frame disables it for the session and silently falls back to the
compact colorized-grid pixels -- nothing here ever breaks a run. Force that
fallback with `--pixel-source grid` or `CCR_MINECRAFT_PIXELS=grid`.

To try it:

```bash
cd bridge/mineflayer
npm install --include=optional     # prismarine-viewer + node-canvas-webgl
# On headless Linux you likely also need a virtual display, e.g.:
#   xvfb-run -a node index.js   (or wrap the whole `ccr run` invocation)

python -m cognitive_runtime run --backend remote --policy scripted \
    --episodes 1 --episode-ticks 200 --realtime --record-frames
```

Watch stderr for `[mc-bridge:pixels]` lines: `first-person capture enabled` on
success, or a reason it fell back (module not installed, headless-GL init
failure, a bad frame). `bridge/mineflayer/pixels.js` is the whole
integration (`PixelViewer`).

Storage note: `--record-frames` does not embed PNG/JPEG/frame JSON in the
stream log. Python converts bridge frames to `uint8` arrays immediately, and
the recorder writes them to the bounded binary frame store under
`<session>/frames/`. Training loads frames from there, learns/predicts compact
latents, and old unpinned segments roll off under `--frame-disk-budget-mb`.

## Setup

1. **Install Node deps** (Node ≥ 18):

   ```bash
   cd bridge/mineflayer
   npm install
   ```

2. **Run a Minecraft Java server** the agent can join. For measurable,
   repeatable episodes, use a constrained world:
   - a **fixed level seed** and a **world border** (`/worldborder set 128`),
   - `online-mode=false` (offline auth) for a local test server, and
   - **op the agent** (`/op CCRAgent`) so per-episode `reset` can run
     `/gamemode survival`, `/effect clear`, and `/time set` — without op these
     are best-effort and simply skipped.

   A local [PaperMC](https://papermc.io/) or vanilla server on `localhost:25565`
   is the simplest starting point.

3. **Point the runtime at your server** via environment variables and run with
   `--backend remote`:

   ```bash
   export CCR_MINECRAFT_HOST=localhost
   export CCR_MINECRAFT_PORT=25565
   export CCR_MINECRAFT_USERNAME=CCRAgent
   # export CCR_MINECRAFT_VERSION=1.20.4   # pin if auto-detect misfires
   # export CCR_MINECRAFT_AUTH=offline     # or 'microsoft' for online-mode

   python -m cognitive_runtime run --backend remote \
       --policy scripted --episodes 1 --episode-ticks 400 --realtime --record-frames
   ```

   By default the runtime launches `node bridge/mineflayer/index.js`. Override
   the command with `CCR_MINECRAFT_BRIDGE_CMD` (e.g. to add Node flags or use a
   remote shim).

Recorded remote sessions are viewable and trainable exactly like simulated
ones; only `replay --verify` is skipped for them (a live server is not
deterministic, so it cannot be re-simulated — the runtime says so and moves on).

## Realtime is the natural fit

Run with `--realtime`: the bridge advances roughly one server tick per `step`,
and the runtime paces vision/body streams to wall-clock rates with bounded,
overflow-counted queues. `dashboard` then reports realtime health (rates,
staleness, overflows) for the session.

## Live smoke checklist (richer event streams, issue #40)

The exact-identity/progression streams above are the parts of #40 that only
a live server can really exercise (crafting, advancements, dimensions,
structures — the simulated backend approximates them for tests, not for
fidelity). After connecting the bridge to a real server, drive the bot
through this checklist and watch `dashboard` / the recorded session for the
corresponding `event.*` streams:

1. **Mine a log.** Face a tree and `ATTACK` until it breaks.
   Expect: `event.block_broken_exact` (block id + position) and
   `event.item_collected_exact` (`{"item": "oak_log", "count": 1}` or similar).
2. **Place a block.** Select a placeable item and `USE` facing an empty
   space. Expect: `event.block_placed_exact` alongside the existing
   `event.block_placed`.
3. **Craft a table / craft planks.** Give the bot logs, stand facing a
   crafting table, and `USE`. Expect: `event.container_interaction`
   (`"container": "crafting_table"`) and, if the bot holds a log,
   `event.crafted` with the resulting recipe/inputs/outputs. `event.advancement`
   should follow if the server awards a crafting-related advancement.
4. **Smelt something.** Give the bot cobblestone + coal, face a furnace,
   `USE`. Expect `event.container_interaction` (`"furnace"`) immediately, and
   `event.crafted` (`smelt_cobblestone`) after the ~10s smelt completes —
   this is the one interaction that is not instantaneous, so give it time.
5. **Open a chest.** Face a chest and `USE`. Expect `event.container_interaction`
   (`"chest"`) with no accompanying `event.crafted` (chests don't craft).
6. **Cross a portal.** Walk the bot through a nether portal. Expect
   `event.dimension_changed` (`"from": "overworld", "to": "the_nether"` or
   similar, from `bot.game.dimension`) on the following `respawn`.
7. **Change biome.** Walk across a biome boundary. Expect `event.biome_entered`
   with the new biome's SurvivalBox vocab name.
8. **Find a structure.** Get within ~16 blocks of a village/stronghold/nether
   fortress/ocean monument. Expect a best-effort `event.structure_discovered`
   — this is a marker-block heuristic (`STRUCTURE_MARKERS` in `blocks.js`),
   not a real "structure generated here" signal, so misses are expected;
   extend the marker list for structures your world/version features.
9. **Earn any vanilla advancement.** Expect `event.advancement` with the
   server's real advancement id (e.g. `minecraft:story/mine_wood`). The raw
   `advancements` packet shape is version-sensitive (see `world.js`); if this
   never fires on your server/mineflayer version, that parsing is the first
   thing to check.

## Live smoke checklist (expanded action space, issue #42)

The simulated backend exercises every one of these through the fake bridge
(`tests/test_expanded_actions.py`, `tests/test_remote_backend.py`'s
whole-action-space fuzz cross-check), but the live-only richness (real
vanilla recipe requirements, door toggling, villager trade dialogs) needs a
real server to actually verify. After connecting the bridge, drive the bot
through this checklist:

1. **Craft explicitly.** Give the bot a log, stand facing a crafting table,
   and send `CRAFT(recipe="log_to_planks")`. Expect `event.crafted`
   (`log_to_planks`) and no auto-craft of anything else. Repeat facing a
   furnace with `smelt_cobblestone` (needs cobblestone + coal) and
   `smelt_torch` (needs coal) — the furnace recipes take the same ~10s a
   real smelt does.
2. **Craft rejections.** Send `CRAFT(recipe="log_to_planks")` with no log in
   inventory, and again facing a furnace instead of a crafting table.
   Expect `event.action_rejected` both times, no `event.crafted`.
3. **Open/close inventory.** Send `OPEN_INVENTORY` then `OPEN_INVENTORY`
   again. Expect `body.inventory_open` to flip true once, then a rejected
   second open; `CLOSE_INVENTORY` mirrors it.
4. **Equip from a slot.** Put an item in hotbar slot 3 (not the selected
   slot) and send `EQUIP_ITEM(slot=3)`. Expect `body.hotbar.selected == 3`
   and no rejection; sending it again on an empty slot should reject.
5. **Place/use from a specific slot.** Put a placeable item in a
   non-selected slot and send `PLACE_BLOCK(slot=N)`; put food in another
   slot and send `USE_ITEM(slot=M)`. Both should act on that slot without
   changing what was previously selected once they finish (the bridge
   restores the prior quickbar slot).
6. **Move inventory items.** Put items in two hotbar slots and send
   `MOVE_INVENTORY_ITEM(from_slot=A, to_slot=B)`. Expect `body.hotbar` to
   show the two slots swapped.
7. **Interact with a door.** Face a wooden/iron door and send `INTERACT`.
   Expect the door to open/close in-game; the bridge emits
   `event.container_interaction` (`"container": "door"`) — a live-only
   value the simulated backend never produces.
8. **Interact with a villager.** Face a villager and send `INTERACT`.
   Expect the trade UI to open server-side and
   `event.container_interaction` (`"container": "villager"`).
9. **Interact with nothing.** Face open air and send `INTERACT`. Expect
   `event.action_rejected` (`"nothing to interact with"`).

## Verifying without a server

The protocol itself is covered without any Minecraft by the Python **fake
bridge** (`bridge/fake/sim_bridge.py`), which speaks the same JSON protocol
backed by the deterministic simulated world. `tests/test_remote_backend.py`
drives the whole remote path through it and asserts it reproduces the
in-process backend byte-for-byte. This bridge's own JavaScript is syntax-checked
in CI (`npm run check`); its live behaviour (block mapping fidelity, action
timing) is what you tune against your server and mineflayer version.

## Files

| File | Role |
|---|---|
| `index.js` | stdio JSON protocol loop; serializes commands, one response per line |
| `world.js` | `WorldSession`: connect, per-tick stepping, event synthesis, stats |
| `actions.js` | SurvivalBox action → mineflayer controls/activities |
| `observation.js` | mineflayer state → SurvivalBox observation + 11×11 frame |
| `blocks.js` | block/biome/item vocabulary + frame codes (kept in sync with `world.py`) |
| `pixels.js` | optional higher-fidelity pixel capture via prismarine-viewer (issue #32) |

## Tuning notes

The block→vocabulary table (`blocks.js`) and action timing are the two things
most likely to need adjustment for your server/version:

- **Block mapping** is deliberately conservative (unknown collidable blocks read
  as `stone`, open/air as `grass`). Extend `NAME_MAP` for blocks your world
  features prominently.
- **Action timing**: `step` advances one server tick; multi-tick activities
  (digging especially) complete across several steps, emitting their event when
  they finish. If your server ticks slowly or the agent lags, raise the tick
  wait in `world.js` (`_awaitTick`).
