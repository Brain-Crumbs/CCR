'use strict';

// Translate one SurvivalBox action into mineflayer controls/activities.
// Movement is expressed as control states the caller clears after one server
// tick; dig/attack/place/consume are multi-tick activities started here and
// reported (broke_block/placed_block/ate_food) when they complete via the
// callbacks in `hooks`.

const Vec3 = require('vec3');
const {
  FOOD_ITEMS, PLACEABLE_ITEMS, LIGHT_ITEMS, itemToVocab, blockToVocab, HOSTILE,
  PASSIVE, NEUTRAL, containerType, mcDataFor,
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

// Nearest passive/neutral entity within reach+cone -- INTERACT's target for
// "villagers" (issue #42); doors/containers go through frontBlockObj instead.
function targetEntity(bot) {
  const self = bot.entity.position;
  const yaw = bot.entity.yaw;
  const fx = -Math.sin(yaw);
  const fz = Math.cos(yaw);
  let best = null;
  let bestDist = Infinity;
  for (const id of Object.keys(bot.entities)) {
    const e = bot.entities[id];
    if (!e || e === bot.entity || !e.position) continue;
    const name = (e.name || '').toLowerCase();
    if (!PASSIVE.has(name) && !NEUTRAL.has(name)) continue;
    const vx = e.position.x - self.x;
    const vz = e.position.z - self.z;
    const dist = Math.hypot(vx, vz);
    if (dist > 3.0 || dist === 0) continue;
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
  const dz = -Math.cos(yaw);
  const p = bot.entity.position;
  return bot.blockAt(new Vec3(Math.floor(p.x + dx), Math.floor(p.y), Math.floor(p.z + dz)));
}

// The item mineflayer currently has in hotbar slot `slot` (0..8), or null.
function hotbarItem(bot, slot) {
  const start = bot.inventory.hotbarStart != null ? bot.inventory.hotbarStart : 36;
  return bot.inventory.slots[start + slot] || null;
}

// Run `fn` with hotbar slot `slot` selected, then restore whatever was
// selected before -- lets PLACE_BLOCK/USE_ITEM act on a specific slot
// without permanently changing what SELECT_HOTBAR_SLOT/EQUIP_ITEM chose.
async function withHotbarSlot(bot, slot, fn) {
  const previous = bot.quickBarSlot;
  bot.setQuickBarSlot(slot);
  try {
    return await fn();
  } finally {
    bot.setQuickBarSlot(previous);
  }
}

// Apply the action.  `hooks` collects async completions:
//   hooks.onBroke(vocabName, position), hooks.onPlaced(vocabName, exactName, position),
//   hooks.onAte(), hooks.onKilled(),
//   hooks.onContainer(kind, position), hooks.onCrafted(recipe, inputs, outputs),
//   hooks.onRejected(reason) -- issue #42: feedback for an invalid parameterized
//   action (craft without materials, equip/place/use an empty slot, ...),
//   mirrored on the Python side as `event.action_rejected`.
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
      bot.look(bot.entity.yaw + LOOK_STEP_RAD, bot.entity.pitch, true); return;
    case 'LOOK_RIGHT':
      bot.look(bot.entity.yaw - LOOK_STEP_RAD, bot.entity.pitch, true); return;
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
    case 'INTERACT':
      return interact(bot, hooks);
    case 'OPEN_INVENTORY':
      if (bot._ccrInventoryOpen) { hooks.onRejected('inventory already open'); return; }
      bot._ccrInventoryOpen = true;
      return;
    case 'CLOSE_INVENTORY':
      if (!bot._ccrInventoryOpen) { hooks.onRejected('inventory already closed'); return; }
      bot._ccrInventoryOpen = false;
      return;
    case 'EQUIP_ITEM': {
      const slot = Number(params.slot);
      if (!(slot >= 0 && slot < 9)) { hooks.onRejected(`invalid slot ${slot}`); return; }
      if (!hotbarItem(bot, slot)) { hooks.onRejected(`cannot equip empty slot ${slot}`); return; }
      bot.setQuickBarSlot(slot);
      return;
    }
    case 'PLACE_BLOCK':
      return placeBlockFromSlot(bot, Number(params.slot), hooks);
    case 'USE_ITEM':
      return useItemFromSlot(bot, Number(params.slot), hooks);
    case 'MOVE_INVENTORY_ITEM':
      return moveInventoryItem(bot, Number(params.from_slot), Number(params.to_slot), hooks);
    case 'CRAFT':
      return craftRecipe(bot, String(params.recipe || ''), hooks);
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

// Generic block/entity interaction (issue #42): containers/furnace behave
// exactly like USE's container branch (but never auto-crafts -- CRAFT is the
// explicit trigger for that); doors toggle open/closed; villagers/passive
// mobs open their trade/interact dialog.  None of this exists in the
// simulated world beyond containers -- it is the live-server-only richness
// the sim's docstring defers to a real backend for.
function interact(bot, hooks) {
  const front = frontBlockObj(bot);
  const kind = front ? containerType(front.name) : null;
  if (kind) {
    hooks.onContainer(kind, front.position);
    return;
  }
  if (front && /_door$/.test(front.name || '')) {
    try { bot.activateBlock(front); } catch (e) { /* best-effort */ }
    hooks.onContainer('door', front.position);
    return;
  }
  const entity = targetEntity(bot);
  if (entity) {
    try { bot.activateEntity(entity); } catch (e) { /* best-effort */ }
    hooks.onContainer((entity.name || 'entity').toLowerCase(), entity.position);
    return;
  }
  hooks.onRejected('nothing to interact with');
}

function placeBlockFromSlot(bot, slot, hooks) {
  if (!(slot >= 0 && slot < 9)) { hooks.onRejected(`invalid slot ${slot}`); return; }
  const item = hotbarItem(bot, slot);
  if (!item) { hooks.onRejected(`no item to place in slot ${slot}`); return; }
  const vocab = itemToVocab(item.name);
  if (!(PLACEABLE_ITEMS.has(vocab) || LIGHT_ITEMS.has(item.name))) {
    hooks.onRejected(`${vocab} is not placeable`);
    return;
  }
  const ref = frontBlockObj(bot);
  if (!ref || ref.boundingBox !== 'block') {
    hooks.onRejected('target cell is not placeable');
    return;
  }
  const placedPos = ref.position.offset(0, 1, 0);
  withHotbarSlot(bot, slot, () => bot.placeBlock(ref, new Vec3(0, 1, 0)))
    .then(() => hooks.onPlaced(vocab, item.name, placedPos))
    .catch(() => hooks.onRejected(`failed to place from slot ${slot}`));
}

function useItemFromSlot(bot, slot, hooks) {
  if (!(slot >= 0 && slot < 9)) { hooks.onRejected(`invalid slot ${slot}`); return; }
  const item = hotbarItem(bot, slot);
  if (!item) { hooks.onRejected(`no item to use in slot ${slot}`); return; }
  const vocab = itemToVocab(item.name);
  if (!FOOD_ITEMS.has(vocab)) { hooks.onRejected(`${vocab} is not usable`); return; }
  withHotbarSlot(bot, slot, () => bot.consume())
    .then(() => hooks.onAte())
    .catch(() => hooks.onRejected(`failed to use item in slot ${slot}`));
}

function moveInventoryItem(bot, fromSlot, toSlot, hooks) {
  if (!(fromSlot >= 0 && fromSlot < 9 && toSlot >= 0 && toSlot < 9) || fromSlot === toSlot) {
    hooks.onRejected(`invalid slots ${fromSlot},${toSlot}`);
    return;
  }
  if (!hotbarItem(bot, fromSlot) && !hotbarItem(bot, toSlot)) {
    hooks.onRejected(`both slots ${fromSlot},${toSlot} are empty`);
    return;
  }
  const start = bot.inventory.hotbarStart != null ? bot.inventory.hotbarStart : 36;
  bot.moveSlotItem(start + fromSlot, start + toSlot)
    .catch(() => hooks.onRejected(`failed to move slots ${fromSlot},${toSlot}`));
}

// recipe id -> the container type it needs, mirroring RECIPE_CONTAINER in
// cognitive_runtime/programs/minecraft/world.py.
const RECIPE_CONTAINER = {
  log_to_planks: 'crafting_table',
  planks_to_pickaxe: 'crafting_table',
  smelt_cobblestone: 'furnace',
  smelt_torch: 'furnace',
};

// Explicit, parameterized craft (issue #42) -- unlike `tryCraft` (USE's
// implicit "try every recipe" fallback), this targets exactly `recipe` and
// rejects (not silently no-ops) when the container or materials are wrong.
// Vanilla recipes are richer than the sim's fixed inputs/outputs (e.g. a
// real wooden pickaxe also needs sticks); this asks mineflayer's own recipe
// book for the real recipe of the same target item, so it stays correct
// against a live server even where it diverges from world.py's simplified
// version -- verify against your server per the bridge README's smoke
// checklist.
function craftRecipe(bot, recipe, hooks) {
  const container = RECIPE_CONTAINER[recipe];
  if (!container) { hooks.onRejected(`unknown recipe ${recipe}`); return; }
  const front = frontBlockObj(bot);
  const kind = front ? containerType(front.name) : null;
  if (kind !== container) { hooks.onRejected(`recipe ${recipe} needs a ${container}`); return; }

  try {
    if (container === 'crafting_table') {
      craftAtTable(bot, recipe, front, hooks);
    } else {
      smeltAtFurnace(bot, recipe, front, hooks);
    }
  } catch (e) {
    hooks.onRejected(`failed to craft ${recipe}`);
  }
}

function craftAtTable(bot, recipe, table, hooks) {
  const mcData = mcDataFor(bot.version);
  let targetName;
  let inputsDescription;
  if (recipe === 'log_to_planks') {
    const log = bot.inventory.items().find((i) => /_log$/.test(i.name));
    if (!log) { hooks.onRejected(`insufficient materials for ${recipe}`); return; }
    targetName = log.name.replace(/_log$/, '_planks');
    inputsDescription = { log: 1 };
  } else {
    targetName = 'wooden_pickaxe';
    inputsDescription = { planks: 3 };
  }
  const targetItem = mcData && mcData.itemsByName[targetName];
  if (!targetItem) { hooks.onRejected(`insufficient materials for ${recipe}`); return; }
  const recipes = bot.recipesFor(targetItem.id, null, 1, table);
  if (!recipes || !recipes.length) { hooks.onRejected(`insufficient materials for ${recipe}`); return; }
  bot.craft(recipes[0], 1, table)
    .then(() => hooks.onCrafted(recipe, inputsDescription, { [targetName]: recipe === 'log_to_planks' ? 4 : 1 }))
    .catch(() => hooks.onRejected(`failed to craft ${recipe}`));
}

function smeltAtFurnace(bot, recipe, furnaceBlock, hooks) {
  const cobble = bot.inventory.items().find((i) => i.name === 'cobblestone');
  const coal = bot.inventory.items().find((i) => i.name === 'coal' || i.name === 'charcoal');
  if (recipe === 'smelt_cobblestone' && (!cobble || !coal)) {
    hooks.onRejected(`insufficient materials for ${recipe}`);
    return;
  }
  if (recipe === 'smelt_torch' && !coal) {
    hooks.onRejected(`insufficient materials for ${recipe}`);
    return;
  }
  bot.openFurnace(furnaceBlock).then((furnace) => {
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
        if (recipe === 'smelt_cobblestone') {
          hooks.onCrafted(recipe, { cobblestone: 1, coal: 1 }, { stone: 1 });
        } else {
          hooks.onCrafted(recipe, { coal: 1 }, { torch: 4 });
        }
      }
    };
    furnace.on('update', onUpdate);
    if (recipe === 'smelt_cobblestone') {
      furnace.putInput(cobble.type, null, 1).catch(() => hooks.onRejected(`failed to craft ${recipe}`));
    }
    furnace.putFuel(coal.type, null, 1).catch(() => hooks.onRejected(`failed to craft ${recipe}`));
    setTimeout(finish, 15000); // vanilla smelting is ~10s; give it headroom
  }).catch(() => hooks.onRejected(`failed to craft ${recipe}`));
}

// Best-effort real-Minecraft crafting/smelting, mirroring the sim's two
// fixed recipes (RECIPES in world.py) so `event.crafted` is exercisable live.
// This is USE's implicit auto-craft (issue #40), kept for backward
// compatibility; CRAFT(recipe) (issue #42, `craftRecipe` above) is the
// explicit, parameterized, rejection-on-failure alternative. mcData/recipe
// shape varies by mineflayer & server version -- failures are swallowed,
// matching the rest of this file's dig/place/consume calls; verify against
// your server per the mineflayer bridge README's smoke checklist.
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
