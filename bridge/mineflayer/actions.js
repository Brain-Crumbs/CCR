'use strict';

// Translate one SurvivalBox action into mineflayer controls/activities.
// Movement is expressed as control states the caller clears after one server
// tick; dig/attack/place/consume are multi-tick activities started here and
// reported (broke_block/placed_block/ate_food) when they complete via the
// callbacks in `hooks`.

const Vec3 = require('vec3');
const {
  FOOD_ITEMS, PLACEABLE_ITEMS, LIGHT_ITEMS, itemToVocab, blockToVocab, HOSTILE,
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
//   hooks.onBroke(vocabName), hooks.onPlaced(vocabName, exactName),
//   hooks.onAte(), hooks.onKilled()
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
    bot.dig(block, true).then(() => hooks.onBroke(vocab)).catch(() => {});
  }
}

function use(bot, hooks) {
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
      bot.placeBlock(ref, new Vec3(0, 1, 0))
        .then(() => hooks.onPlaced(vocab, held.name))
        .catch(() => {});
    }
  }
}

module.exports = { applyAction, clearControls, MOVE_CONTROLS };
