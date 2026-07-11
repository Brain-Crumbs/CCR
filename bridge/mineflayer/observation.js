'use strict';

// Build a SurvivalBox observation (data keys + 11x11 frame) from a mineflayer
// bot.  The shape must match cognitive_runtime/programs/minecraft/observations.py
// so the stream publisher, featurizer and reward run unchanged.

const {
  MOB_FRAME_ID, AGENT_FRAME_ID,
  blockToVocab, blockToFrameCode, blockExactName, biomeToVocab, itemToVocab, HOSTILE,
} = require('./blocks');

const RAD2DEG = 180 / Math.PI;

function round(value, digits) {
  const f = Math.pow(10, digits);
  return Math.round(value * f) / f;
}

// SurvivalBox yaw: degrees, 0 = +z (south), increasing clockwise, like the
// sim's _facing_vector (dx=-sin(yaw), dz=cos(yaw)).  mineflayer yaw is radians
// with the same convention (0 = south), so degrees = yaw * 180/pi mod 360.
function yawDegrees(bot) {
  return ((bot.entity.yaw * RAD2DEG) % 360 + 360) % 360;
}

// mineflayer pitch: radians, positive = looking up.  SurvivalBox pitch:
// degrees, positive = looking down (LOOK_DOWN increments it), range -90..90.
function pitchDegrees(bot) {
  return round(-bot.entity.pitch * RAD2DEG, 1);
}

function facingVector(bot) {
  const yaw = bot.entity.yaw;
  return { dx: -Math.sin(yaw), dz: Math.cos(yaw) };
}

// A block at an integer column, sampled at the player's feet Y.
function blockAtColumn(bot, x, z) {
  const y = Math.floor(bot.entity.position.y);
  return bot.blockAt(new bot.vec3(x, y, z));
}

function nearbyBlocks(bot, radius) {
  const ix = Math.floor(bot.entity.position.x);
  const iz = Math.floor(bot.entity.position.z);
  const patch = [];
  for (let dx = -radius; dx <= radius; dx++) {
    const row = [];
    for (let dz = -radius; dz <= radius; dz++) {
      row.push(blockToVocab(blockAtColumn(bot, ix + dx, iz + dz)));
    }
    patch.push(row);
  }
  return patch;
}

function nearbyBlocksExact(bot, radius) {
  const ix = Math.floor(bot.entity.position.x);
  const iz = Math.floor(bot.entity.position.z);
  const patch = [];
  for (let dx = -radius; dx <= radius; dx++) {
    const row = [];
    for (let dz = -radius; dz <= radius; dz++) {
      row.push(blockExactName(blockAtColumn(bot, ix + dx, iz + dz)));
    }
    patch.push(row);
  }
  return patch;
}

function frontBlock(bot) {
  const { dx, dz } = facingVector(bot);
  const x = Math.floor(bot.entity.position.x + dx);
  const z = Math.floor(bot.entity.position.z + dz);
  return blockToVocab(blockAtColumn(bot, x, z));
}

function frontBlockExact(bot) {
  const { dx, dz } = facingVector(bot);
  const x = Math.floor(bot.entity.position.x + dx);
  const z = Math.floor(bot.entity.position.z + dz);
  return blockExactName(blockAtColumn(bot, x, z));
}

function isSheltered(bot) {
  const ix = Math.floor(bot.entity.position.x);
  const iz = Math.floor(bot.entity.position.z);
  let solid = 0;
  for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const block = blockAtColumn(bot, ix + dx, iz + dz);
    if (block && block.boundingBox === 'block') solid += 1;
  }
  return solid >= 3;
}

function inWater(bot) {
  const b = bot.blockAt(bot.entity.position);
  return Boolean((b && b.name === 'water') || bot.entity.isInWater);
}

// Hostile mobs as {distance, angle}, bearing relative to facing in degrees
// (-180..180), nearest first — matching world.py mob_summary.
function mobs(bot, limit) {
  const self = bot.entity.position;
  const { dx, dz } = facingVector(bot);
  const facingDeg = Math.atan2(-dx, dz) * RAD2DEG;
  const out = [];
  for (const id of Object.keys(bot.entities)) {
    const e = bot.entities[id];
    if (!e || e === bot.entity || !e.position) continue;
    const name = (e.name || '').toLowerCase();
    if (!HOSTILE.has(name)) continue;
    const vx = e.position.x - self.x;
    const vz = e.position.z - self.z;
    const dist = Math.hypot(vx, vz);
    if (dist > 16) continue;
    const bearing = Math.atan2(-vx, vz) * RAD2DEG;
    let rel = ((bearing - facingDeg + 180) % 360 + 360) % 360 - 180;
    out.push({ distance: round(dist, 2), angle: round(rel, 1) });
  }
  out.sort((a, b) => a.distance - b.distance);
  return out.slice(0, limit);
}

function inventorySummary(bot) {
  const counts = {};
  for (const item of bot.inventory.items()) {
    const name = itemToVocab(item.name);
    counts[name] = (counts[name] || 0) + item.count;
  }
  // Sorted keys so the payload (and its hash) is order-stable, like the sim.
  const sorted = {};
  for (const key of Object.keys(counts).sort()) sorted[key] = counts[key];
  return sorted;
}

function inventoryExactSummary(bot) {
  const counts = {};
  for (const item of bot.inventory.items()) {
    const name = String(item.name || '').toLowerCase();
    counts[name] = (counts[name] || 0) + item.count;
  }
  const sorted = {};
  for (const key of Object.keys(counts).sort()) sorted[key] = counts[key];
  return sorted;
}

function hotbarSlots(bot) {
  const slots = [];
  for (let i = 0; i < 9; i++) {
    const item = bot.inventory.slots[bot.inventory.hotbarStart + i];
    slots.push(item ? itemToVocab(item.name) : null);
  }
  return slots;
}

// Coarse top-down frame (radius r -> (2r+1) square) of frame codes; the agent
// at the centre, hostile mobs as MOB_FRAME_ID.
function renderFrame(bot, radius) {
  const ix = Math.floor(bot.entity.position.x);
  const iz = Math.floor(bot.entity.position.z);
  const mobCells = new Set();
  for (const id of Object.keys(bot.entities)) {
    const e = bot.entities[id];
    if (!e || e === bot.entity || !e.position) continue;
    if (!HOSTILE.has((e.name || '').toLowerCase())) continue;
    mobCells.add(`${Math.floor(e.position.x)},${Math.floor(e.position.z)}`);
  }
  const frame = [];
  for (let dx = -radius; dx <= radius; dx++) {
    const row = [];
    for (let dz = -radius; dz <= radius; dz++) {
      const x = ix + dx;
      const z = iz + dz;
      if (dx === 0 && dz === 0) row.push(AGENT_FRAME_ID);
      else if (mobCells.has(`${x},${z}`)) row.push(MOB_FRAME_ID);
      else row.push(blockToFrameCode(blockAtColumn(bot, x, z)));
    }
    frame.push(row);
  }
  return frame;
}

// time_of_day/is_night are synthesized from the tick + config so --day-length
// and --start-time behave exactly as in the sim, independent of the server clock.
function timeState(tick, config) {
  const dayLength = config.day_length || 6000;
  const startTime = config.start_time || 0;
  const timeOfDay = ((startTime + tick) % dayLength + dayLength) % dayLength;
  return { time_of_day: timeOfDay, day_length: dayLength, is_night: timeOfDay >= dayLength / 2 };
}

function buildObservation(bot, tick, config, spawn) {
  const pos = bot.entity.position;
  const t = timeState(tick, config);
  const biomeBlock = bot.blockAt(bot.entity.position);
  const biomeName = biomeBlock && biomeBlock.biome ? biomeBlock.biome.name : null;
  const dist = spawn
    ? Math.hypot(pos.x - spawn.x, pos.z - spawn.z)
    : 0;
  const data = {
    health: round(bot.health == null ? 0 : bot.health, 2),
    hunger: round(bot.food == null ? 0 : bot.food, 2),
    oxygen: round(bot.oxygenLevel == null ? 20 : bot.oxygenLevel, 2),
    position: { x: round(pos.x, 3), y: round(pos.y, 3), z: round(pos.z, 3) },
    yaw: round(yawDegrees(bot), 1),
    pitch: pitchDegrees(bot),
    time_of_day: t.time_of_day,
    day_length: t.day_length,
    is_night: t.is_night,
    biome: biomeToVocab(biomeName),
    in_water: inWater(bot),
    sheltered: isSheltered(bot),
    selected_slot: bot.quickBarSlot || 0,
    hotbar: hotbarSlots(bot),
    inventory: inventorySummary(bot),
    inventory_exact: inventoryExactSummary(bot),
    inventory_open: Boolean(bot._ccrInventoryOpen),
    nearby_blocks: nearbyBlocks(bot, 2),
    nearby_blocks_exact: nearbyBlocksExact(bot, 2),
    front_block: frontBlock(bot),
    front_block_exact: frontBlockExact(bot),
    mobs: mobs(bot, 4),
    distance_from_spawn: round(dist, 2),
    dead: (bot.health != null && bot.health <= 0),
  };
  return { tick, data, frame: renderFrame(bot, 5) };
}

module.exports = { buildObservation, mobs, isSheltered, inWater, timeState, HOSTILE };
