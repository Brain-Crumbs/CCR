'use strict';

// Translate one SurvivalBox action into mineflayer controls/activities.
// Movement is expressed as control states the caller clears after one server
// tick; dig/attack/place/consume are multi-tick activities started here and
// reported (broke_block/placed_block/ate_food) when they complete via the
// callbacks in `hooks`.

const Vec3 = require('vec3');
const {
  FOOD_ITEMS, PLACEABLE_ITEMS, LIGHT_ITEMS, itemToVocab, blockToVocab, HOSTILE,
  containerType, mcDataFor,
} = require('./blocks');

const LOOK_STEP_RAD = (15 * Math.PI) / 180;
const PITCH_STEP_RAD = (10 * Math.PI) / 180;
const HALF_PI = Math.PI / 2;
const MOVE_CONTROLS = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'];

function clearControls(bot) {
  for (const c of MOVE_CONTROLS) bot.setControlState(c, false);
}

// Nearest hostile within reach+cone, mirroring the sim's _attack targeting.
function targetMob(bot) {
  const self = bot.entity.position;
  const yaw = bot.entity.yaw;
  const fx = -Math.sin(yaw);
  const fz = Math.cos(yaw);
  let best = null;
  let bestDist = Infinity;
  for (const id of Object.keys(bot.entities)) {
    const e = bot.entities[id];
    if (!e || e === bot.entity || !e.position) continue;
    if (!HOSTILE.has((e.name || '').toLowerCase())) continue;
    const vx = e.position.x - self.x;
    const vz = e.position.z - self.z;
    const dist = Math.hypot(vx, vz);
    if (dist > 2.0 || dist === 0) continue;
    const dot = (vx * fx + vz * fz) / dist;
    if (dot >= Math.cos((60 * Math.PI) / 180) && dist < bestDist) {
      best = e;
      bestDist = dist;
    }
  }
  return best;
}

function frontBlockObj(bot) {
  const yaw = bot.entity.yaw;
  const dx = -Math.sin(yaw);
  const dz = Math.cos(yaw);
  const p = bot.entity.position;
  return bot.blockAt(new Vec3(Math.floor(p.x + dx), Math.floor(p.y), Math.floor(p.z + dz)));
}

// Apply the action.  `hooks` collects async completions:
//   hooks.onBroke(vocabName, position), hooks.onPlaced(vocabName, exactName, position),
//   hooks.onAte(), hooks.onKilled(),
//   hooks.onContainer(kind, position), hooks.onCrafted(recipe, inputs, outputs)
function applyAction(bot, action, hooks) {
  const name = action.name;
  const params = action.params || {};
  switch (name) {
    case 'NULL':
      return;
    case 'MOVE_FORWARD':
      bot.setControlState('forward', true); return;
    case 'MOVE_BACKWARD':
      bot.setControlState('back', true); return;
    case 'MOVE_LEFT':
      bot.setControlState('left', true); return;
    case 'MOVE_RIGHT':
      bot.setControlState('right', true); return;
    case 'JUMP':
      bot.setControlState('jump', true); bot.setControlState('forward', true); return;
    case 'SPRINT':
      bot.setControlState('sprint', true); bot.setControlState('forward', true); return;
    case 'SNEAK':
      bot.setControlState('sneak', true); bot.setControlState('forward', true); return;
    case 'LOOK_LEFT':
      bot.look(bot.entity.yaw - LOOK_STEP_RAD, bot.entity.pitch, true); return;
    case 'LOOK_RIGHT':
      bot.look(bot.entity.yaw + LOOK_STEP_RAD, bot.entity.pitch, true); return;
    case 'LOOK_UP':
      bot.look(bot.entity.yaw, Math.min(HALF_PI, bot.entity.pitch + PITCH_STEP_RAD), true); return;
    case 'LOOK_DOWN':
      bot.look(bot.entity.yaw, Math.max(-HALF_PI, bot.entity.pitch - PITCH_STEP_RAD), true); return;
    case 'ATTACK':
      return attack(bot, hooks);
    case 'USE':
      return use(bot, hooks);
    case 'SELECT_HOTBAR_SLOT': {
      const slot = Number(params.slot) || 0;
      if (slot >= 0 && slot < 9) bot.setQuickBarSlot(slot);
      return;
    }
    default:
      return;
  }
}

function attack(bot, hooks) {
  const mob = targetMob(bot);
  if (mob) {
    bot.attack(mob);
    // Best-effort kill detection: gone next tick after we hit it.
    const id = mob.id;
    setTimeout(() => {
      if (!bot.entities[id]) hooks.onKilled();
    }, 100);
    return;
  }
  const block = frontBlockObj(bot);
  if (block && block.boundingBox === 'block' && block.name !== 'bedrock' && block.name !== 'barrier') {
    const vocab = blockToVocab(block);
    const pos = block.position;
    bot.dig(block, true).then(() => hooks.onBroke(vocab, pos)).catch(() => {});
  }
}

function use(bot, hooks) {
  const front = frontBlockObj(bot);
  const kind = front ? containerType(front.name) : null;
  if (kind) {
    hooks.onContainer(kind, front.position);
    tryCraft(bot, kind, front, hooks);
    return;
  }

  const held = bot.heldItem;
  if (!held) return;
  const vocab = itemToVocab(held.name);
  if (FOOD_ITEMS.has(vocab)) {
    bot.consume().then(() => hooks.onAte()).catch(() => {});
    return;
  }
  if (PLACEABLE_ITEMS.has(vocab) || LIGHT_ITEMS.has(held.name)) {
    const ref = frontBlockObj(bot);
    if (ref && ref.boundingBox === 'block') {
      const placedPos = ref.position.offset(0, 1, 0);
      bot.placeBlock(ref, new Vec3(0, 1, 0))
        .then(() => hooks.onPlaced(vocab, held.name, placedPos))
        .catch(() => {});
    }
  }
}

// Best-effort real-Minecraft crafting/smelting, mirroring the sim's two
// fixed recipes (RECIPES in world.py) so `event.crafted` is exercisable live.
// mcData/recipe shape varies by mineflayer & server version -- failures are
// swallowed, matching the rest of this file's dig/place/consume calls; verify
// against your server per the mineflayer bridge README's smoke checklist.
function tryCraft(bot, kind, block, hooks) {
  try {
    if (kind === 'crafting_table') {
      const log = bot.inventory.items().find((i) => /_log$/.test(i.name));
      if (!log) return;
      const mcData = mcDataFor(bot.version);
      const planksName = log.name.replace(/_log$/, '_planks');
      const planksItem = mcData && mcData.itemsByName[planksName];
      if (!planksItem) return;
      const recipes = bot.recipesFor(planksItem.id, null, 1, block);
      if (!recipes || !recipes.length) return;
      bot.craft(recipes[0], 1, block)
        .then(() => hooks.onCrafted('log_to_planks', { log: 1 }, { [planksName]: 4 }))
        .catch(() => {});
    } else if (kind === 'furnace') {
      const cobble = bot.inventory.items().find((i) => i.name === 'cobblestone');
      const coal = bot.inventory.items().find((i) => i.name === 'coal');
      if (!cobble || !coal) return;
      bot.openFurnace(block).then((furnace) => {
        let done = false;
        const finish = () => {
          if (done) return;
          done = true;
          furnace.removeListener('update', onUpdate);
          try { furnace.close(); } catch (e) { /* ignore */ }
        };
        const onUpdate = () => {
          const output = furnace.outputItem && furnace.outputItem();
          if (output && output.count > 0) {
            finish();
            hooks.onCrafted('smelt_cobblestone', { cobblestone: 1, coal: 1 }, { stone: 1 });
          }
        };
        furnace.on('update', onUpdate);
        furnace.putInput(cobble.type, null, 1).catch(() => {});
        furnace.putFuel(coal.type, null, 1).catch(() => {});
        setTimeout(finish, 15000); // vanilla smelting is ~10s; give it headroom
      }).catch(() => {});
    }
  } catch (e) { /* best-effort: never let a bad recipe/version crash the tick */ }
}

module.exports = { applyAction, clearControls, MOVE_CONTROLS };
