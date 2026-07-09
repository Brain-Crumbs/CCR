'use strict';

// One live-Minecraft episode session: owns the mineflayer bot, applies actions
// one server tick at a time, and synthesizes the SurvivalBox semantic-event
// vocabulary by diffing state across ticks.  The event strings and stats keys
// match the simulated world so the runtime's streams/reward are unchanged.

const mineflayer = require('mineflayer');
const Vec3 = require('vec3');
const {
  itemToVocab, FOOD_ITEMS, LIGHT_ITEMS, biomeToVocab, structureMarker,
} = require('./blocks');
const { applyAction, clearControls } = require('./actions');
const { buildObservation, isSheltered, timeState } = require('./observation');

function log(...args) {
  process.stderr.write('[mc-bridge] ' + args.join(' ') + '\n');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class WorldSession {
  constructor() {
    this.bot = null;
    this.config = {};
    this.connection = {};
    this.tick = 0;
    this.spawn = null;
    this.dead = false;
    this.deathReason = null;
    this._pending = [];
    this._seenItems = new Set();
    this._prevHealth = 20;
    this._prevInventory = {};
    this._prevSheltered = false;
    this._prevNight = false;
    this._lastDamageCause = 'unknown';
    this._stats = freshStats();
    // issue #40: richer event streams.
    this._prevBiome = null;
    this._prevDimension = null;
    this._seenAdvancements = new Set();
    this._discoveredStructures = new Set();
    this._advancementsWired = false;
  }

  // -- connection ----------------------------------------------------------

  async _connect() {
    if (this.bot) return;
    const opts = {
      host: this.connection.host || 'localhost',
      port: this.connection.port || 25565,
      username: this.connection.username || 'CCRAgent',
      auth: this.connection.auth || 'offline',
    };
    if (this.connection.version) opts.version = this.connection.version;
    log('connecting to', opts.host + ':' + opts.port, 'as', opts.username);
    this.bot = mineflayer.createBot(opts);
    this.bot.vec3 = Vec3;
    this._wireEvents();
    await new Promise((resolve, reject) => {
      const onSpawn = () => { cleanup(); resolve(); };
      const onError = (err) => { cleanup(); reject(err); };
      const cleanup = () => {
        this.bot.removeListener('spawn', onSpawn);
        this.bot.removeListener('error', onError);
      };
      this.bot.once('spawn', onSpawn);
      this.bot.once('error', onError);
    });
    log('spawned');
  }

  _wireEvents() {
    const bot = this.bot;
    bot.on('death', () => { this._pending.push('died'); this.dead = true; this.deathReason = this._lastDamageCause; });
    bot.on('entityHurt', (entity) => {
      if (entity === bot.entity) this._lastDamageCause = 'hit';
    });
    bot.on('kicked', (reason) => log('kicked:', typeof reason === 'string' ? reason : JSON.stringify(reason)));
    bot.on('error', (err) => log('error:', err && err.message ? err.message : String(err)));
    bot.on('end', () => log('connection ended'));

    // Dimension changes fire as a respawn packet (issue #40).
    bot.on('respawn', () => {
      const dim = bot.game ? bot.game.dimension : 'overworld';
      if (this._prevDimension !== null && dim !== this._prevDimension) {
        this._pending.push(`dimension_changed:${this._prevDimension}:${dim}`);
      }
      this._prevDimension = dim;
    });

    // Vanilla advancements (issue #40): forward newly-completed ones by their
    // vanilla id.  The raw protocol packet's shape is version-sensitive, so
    // this is best-effort and never throws -- verify per the bridge README.
    if (bot._client && !this._advancementsWired) {
      this._advancementsWired = true;
      bot._client.on('advancements', (packet) => {
        try {
          const entries = (packet && (packet.advancementMapping || packet.advancements)) || [];
          for (const entry of entries) {
            const id = entry && (entry.key || entry.id || entry.name);
            const progress = entry && (entry.value || entry.advancement || entry);
            const done = Boolean(progress && (progress.done || progress.isDone));
            if (id && done && !this._seenAdvancements.has(id)) {
              this._seenAdvancements.add(id);
              this._pending.push(`advancement:${id}`);
            }
          }
        } catch (e) { /* best-effort: packet shape varies by version */ }
      });
    }
  }

  // -- protocol handlers ---------------------------------------------------

  async reset(seed, config, connection) {
    this.config = config || {};
    this.connection = connection || {};
    await this._connect();
    // Best-effort world reset via op commands; harmless if the server refuses.
    try {
      this.bot.chat('/gamemode survival');
      this.bot.chat(`/effect clear ${this.bot.username}`);
      this.bot.chat(`/time set ${this.config.start_time || 0}`);
    } catch (e) { /* not op / commands disabled: fall through */ }
    await sleep(250);
    this.tick = 0;
    this.dead = (this.bot.health != null && this.bot.health <= 0);
    this.deathReason = null;
    this._pending = [];
    this._seenItems = new Set();
    this._prevHealth = this.bot.health == null ? 20 : this.bot.health;
    this._prevInventory = this._inventoryVocabCounts();
    this._prevSheltered = isSheltered(this.bot);
    this._prevNight = timeState(0, this.config).is_night;
    this._stats = freshStats();
    this.spawn = { x: this.bot.entity.position.x, z: this.bot.entity.position.z };
    const biomeBlock = this.bot.blockAt(this.bot.entity.position);
    this._prevBiome = biomeToVocab(biomeBlock && biomeBlock.biome ? biomeBlock.biome.name : null);
    this._prevDimension = this.bot.game ? this.bot.game.dimension : 'overworld';
    this._discoveredStructures = new Set();
    return this._status();
  }

  async step(action) {
    if (!this.bot) throw new Error('step before reset');
    const events = [];
    const beforePos = {
      x: this.bot.entity.position.x,
      z: this.bot.entity.position.z,
    };
    const hooks = {
      onBroke: (vocab, pos) => {
        this._pending.push(`broke_block:${vocab}`);
        this._pending.push(`block_broken_exact:${JSON.stringify({ block: vocab, position: posDict(pos) })}`);
        this._stats.blocks_broken += 1;
      },
      onPlaced: (vocab, exact, pos) => {
        this._pending.push('placed_block');
        this._pending.push(`block_placed_exact:${JSON.stringify({ block: exact || vocab, position: posDict(pos) })}`);
        if (LIGHT_ITEMS.has(String(exact || vocab || '').toLowerCase())) {
          this._pending.push('created_light_source');
        }
        this._stats.blocks_placed += 1;
      },
      onAte: () => { this._pending.push('ate_food'); this._stats.food_consumed += 1; },
      onKilled: () => { this._pending.push('killed_mob'); this._stats.mobs_killed += 1; },
      onContainer: (kind, pos) => {
        this._pending.push(`container_interact:${JSON.stringify({ container: kind, position: posDict(pos) })}`);
      },
      onCrafted: (recipe, inputs, outputs) => {
        this._pending.push(`crafted:${JSON.stringify({ recipe, inputs, outputs })}`);
      },
    };

    clearControls(this.bot);
    try {
      applyAction(this.bot, action, hooks);
    } catch (e) {
      log('action error:', e && e.message ? e.message : String(e));
    }
    await this._awaitTick();
    clearControls(this.bot);
    this.tick += 1;

    if (isMovementAction(action.name) && horizontalDistance(beforePos, this.bot.entity.position) < 0.01) {
      const block = blockInMoveDirection(this.bot, action.name);
      if (block && block.boundingBox === 'block') events.push('bumped');
    }

    // Diff-based events.
    const health = this.bot.health == null ? 0 : this.bot.health;
    if (health < this._prevHealth - 1e-6) {
      const drop = this._prevHealth - health;
      this._stats.damage_taken = round(this._stats.damage_taken + drop, 2);
      events.push(`damage:${this._lastDamageCause}`);
    }
    this._prevHealth = health;
    this._lastDamageCause = 'unknown';

    // Inventory: exact gain (any count increase) + first-time novelty.
    const inv = this._inventoryVocabCounts();
    for (const name of Object.keys(inv)) {
      const gained = inv[name] - (this._prevInventory[name] || 0);
      if (gained > 0) {
        events.push(`item_collected_exact:${JSON.stringify({ item: name, count: gained })}`);
      }
      if (!this._seenItems.has(name)) {
        this._seenItems.add(name);
        this._stats.unique_items_collected += 1;
        events.push(`new_item:${name}`);
        if (FOOD_ITEMS.has(name)) events.push('acquired_food');
      }
    }
    this._prevInventory = inv;

    // Shelter / night transitions.
    const sheltered = isSheltered(this.bot);
    if (sheltered && !this._prevSheltered) events.push('entered_shelter');
    this._prevSheltered = sheltered;

    const night = timeState(this.tick, this.config).is_night;
    if (this._prevNight && !night && !this.dead) {
      if (!this._stats.survived_night) { this._stats.survived_night = true; events.push('survived_night'); }
    }
    this._prevNight = night;

    // Biome underfoot (issue #40: event.biome_entered).
    const biomeBlock = this.bot.blockAt(this.bot.entity.position);
    const biome = biomeToVocab(biomeBlock && biomeBlock.biome ? biomeBlock.biome.name : null);
    if (biome !== this._prevBiome) {
      events.push(`biome_entered:${biome}`);
      this._prevBiome = biome;
    }

    // Structure discovery: best-effort marker-block heuristic (issue #40).
    // mineflayer has no direct "structure generated here" signal, so this
    // scans nearby blocks for a small curated set of structure-typical
    // blocks (see STRUCTURE_MARKERS in blocks.js) -- verify live per the
    // bridge README; false negatives are expected, false positives are rare.
    try {
      const marker = this.bot.findBlock({
        matching: (block) => Boolean(block && structureMarker(block.name)),
        maxDistance: 16,
      });
      if (marker) {
        const name = structureMarker(marker.name);
        if (name && !this._discoveredStructures.has(name)) {
          this._discoveredStructures.add(name);
          events.push(`structure_discovered:${name}`);
        }
      }
    } catch (e) { /* best-effort scan; never fail the tick over it */ }

    // Distance stat.
    const dist = this.spawn
      ? Math.hypot(this.bot.entity.position.x - this.spawn.x, this.bot.entity.position.z - this.spawn.z)
      : 0;
    if (dist > this._stats.max_distance_from_spawn) this._stats.max_distance_from_spawn = round(dist, 2);

    // Death from vitals (in case the 'death' event did not fire).
    if (health <= 0 && !this.dead) { this.dead = true; this.deathReason = this._lastDamageCause; this._pending.push('died'); }

    // Drain async completions captured during the tick.
    const drained = this._pending;
    this._pending = [];
    const all = events.concat(drained);
    return { ok: true, events: all, ...this._status() };
  }

  observe(timestamp) {
    if (!this.bot) throw new Error('observe before reset');
    const obs = buildObservation(this.bot, this.tick, this.config, this.spawn);
    return { ok: true, observation: obs };
  }

  close() {
    if (this.bot) {
      try { this.bot.quit(); } catch (e) { /* ignore */ }
      this.bot = null;
    }
  }

  // -- helpers -------------------------------------------------------------

  _awaitTick() {
    // Advance ~one server tick.  Prefer the bot's physics clock; fall back to
    // a 50 ms sleep (20 tps) if physicsTick is unavailable.
    return new Promise((resolve) => {
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      try {
        this.bot.once('physicsTick', finish);
      } catch (e) { /* older mineflayer: physicTick */ }
      setTimeout(finish, 100);
    });
  }

  _inventoryVocabCounts() {
    const counts = {};
    for (const item of this.bot.inventory.items()) {
      const name = itemToVocab(item.name);
      counts[name] = (counts[name] || 0) + item.count;
    }
    return counts;
  }

  _status() {
    return {
      ok: true,
      tick: this.tick,
      dead: this.dead,
      death_reason: this.deathReason,
      stats: { ...this._stats },
    };
  }
}

function freshStats() {
  return {
    damage_taken: 0.0,
    food_consumed: 0,
    blocks_broken: 0,
    blocks_placed: 0,
    mobs_killed: 0,
    max_distance_from_spawn: 0.0,
    unique_items_collected: 0,
    survived_night: false,
  };
}

function round(value, digits) {
  const f = Math.pow(10, digits);
  return Math.round(value * f) / f;
}

// A mineflayer/vec3 position -> the plain {x,y,z} dict the Python side
// expects in block_broken_exact/block_placed_exact/container_interact payloads.
function posDict(pos) {
  if (!pos) return { x: 0, y: 0, z: 0 };
  return { x: pos.x, y: pos.y, z: pos.z };
}

function isMovementAction(name) {
  return ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 'SPRINT', 'SNEAK'].includes(name);
}

function horizontalDistance(a, b) {
  return Math.hypot((b.x || 0) - (a.x || 0), (b.z || 0) - (a.z || 0));
}

function blockInMoveDirection(bot, actionName) {
  const yaw = bot.entity.yaw;
  const fx = -Math.sin(yaw);
  const fz = Math.cos(yaw);
  let dx = fx;
  let dz = fz;
  if (actionName === 'MOVE_BACKWARD') {
    dx = -fx; dz = -fz;
  } else if (actionName === 'MOVE_LEFT') {
    dx = fz; dz = -fx;
  } else if (actionName === 'MOVE_RIGHT') {
    dx = -fz; dz = fx;
  }
  const p = bot.entity.position;
  return bot.blockAt(new Vec3(Math.floor(p.x + dx), Math.floor(p.y), Math.floor(p.z + dz)));
}

module.exports = { WorldSession };
