'use strict';

// Vocabulary bridge: Minecraft block/item/biome names -> the SurvivalBox
// vocabulary the runtime's streams and encoders expect.  These constants MUST
// stay in sync with cognitive_runtime/programs/minecraft/world.py.

// Frame cell codes (must equal world.py BLOCK_IDS / MOB_FRAME_ID / AGENT_FRAME_ID).
const BLOCK_IDS = {
  grass: 1, dirt: 2, sand: 3, water: 4, tree: 5,
  stone: 6, coal_ore: 7, berry_bush: 8, placed_block: 9, barrier: 10,
};
const MOB_FRAME_ID = 90;
const AGENT_FRAME_ID = 99;

// The 10 block-name classes; front_block/nearby_blocks payloads use these
// strings, and the CategoryEncoder one-hots front_block against them.
const VOCAB = Object.keys(BLOCK_IDS);

// Explicit name families -> vocab.  Anything not matched falls back by
// collidability (see blockToVocab).
const NAME_MAP = new Map();
function addAll(names, vocab) {
  for (const n of names) NAME_MAP.set(n, vocab);
}
addAll(['grass_block', 'grass', 'short_grass', 'tall_grass', 'fern', 'large_fern',
  'moss_block', 'dirt_path', 'farmland', 'mycelium', 'snow', 'snow_block',
  'podzol'], 'grass');
addAll(['dirt', 'coarse_dirt', 'rooted_dirt', 'gravel', 'clay', 'mud'], 'dirt');
addAll(['sand', 'red_sand', 'sandstone', 'red_sandstone', 'smooth_sandstone',
  'soul_sand'], 'sand');
addAll(['water', 'bubble_column', 'seagrass', 'tall_seagrass', 'kelp',
  'kelp_plant'], 'water');
addAll(['sweet_berry_bush'], 'berry_bush');
addAll(['bedrock', 'obsidian', 'crying_obsidian', 'barrier', 'cobblestone_wall'],
  'barrier');

// Stone family (collidable, non-resource).
addAll(['stone', 'cobblestone', 'andesite', 'diorite', 'granite', 'tuff',
  'calcite', 'deepslate', 'cobbled_deepslate', 'dripstone_block', 'blackstone',
  'basalt', 'netherrack', 'smooth_stone', 'stone_bricks'], 'stone');

// Wood family -> "tree" (a harvestable resource in the vocab).
const WOOD_SUFFIXES = ['_log', '_wood', '_leaves', '_stem', '_hyphae', '_sapling'];

function isWood(name) {
  return WOOD_SUFFIXES.some((s) => name.endsWith(s));
}

// Any ore maps onto the single ore in the vocab ("coal_ore") so the vision
// encoder's "resource" class lights up regardless of the ore type.
function isOre(name) {
  return name.endsWith('_ore');
}

// A block that has no collision box reads as open ground for a top-down view.
function isOpen(block) {
  if (!block) return true;
  const name = block.name;
  if (name === 'air' || name === 'cave_air' || name === 'void_air') return true;
  return block.boundingBox === 'empty';
}

// Map one Minecraft block to a SurvivalBox vocab name.
function blockToVocab(block) {
  if (!block) return 'grass';
  const name = block.name;
  if (NAME_MAP.has(name)) return NAME_MAP.get(name);
  if (isWood(name)) return 'tree';
  if (isOre(name)) return 'coal_ore';
  if (isOpen(block)) return 'grass';
  // Unknown but collidable: treat as generic solid so movement/vision see a wall.
  return 'stone';
}

function blockToFrameCode(block) {
  return BLOCK_IDS[blockToVocab(block)];
}

// Biome name -> the SurvivalBox biome set {plains, forest, desert, lake}.
function biomeToVocab(name) {
  if (!name) return 'plains';
  const n = String(name).toLowerCase();
  if (n.includes('desert') || n.includes('badlands')) return 'desert';
  if (n.includes('ocean') || n.includes('river') || n.includes('lake') ||
      n.includes('beach') || n.includes('swamp')) return 'lake';
  if (n.includes('forest') || n.includes('taiga') || n.includes('jungle') ||
      n.includes('grove') || n.includes('wood')) return 'forest';
  return 'plains';
}

// Item name -> the reward's item vocabulary (food/tool names it rewards, else
// a block-vocab name, else the raw name so novelty still counts it).
const ITEM_MAP = new Map([
  ['sweet_berries', 'berries'],
  ['apple', 'apple'],
  ['bread', 'bread'],
  ['cooked_beef', 'cooked_meat'],
  ['cooked_porkchop', 'cooked_meat'],
  ['cooked_chicken', 'cooked_meat'],
  ['cooked_mutton', 'cooked_meat'],
  ['coal', 'coal'],
  ['cobblestone', 'cobblestone'],
  ['dirt', 'dirt'],
  ['sand', 'sand'],
]);

function itemToVocab(name) {
  if (!name) return name;
  if (ITEM_MAP.has(name)) return ITEM_MAP.get(name);
  if (name.endsWith('_log')) return 'log';
  return name; // keep raw so new-item novelty still fires
}

const FOOD_ITEMS = new Set(['berries', 'apple', 'bread', 'cooked_meat']);
const PLACEABLE_ITEMS = new Set(['log', 'cobblestone', 'dirt', 'sand']);

// Hostile mob entity names surfaced as vision.entities / frame mob cells.
const HOSTILE = new Set([
  'zombie', 'husk', 'drowned', 'skeleton', 'stray', 'creeper', 'spider',
  'cave_spider', 'witch', 'enderman', 'zombie_villager', 'pillager', 'vindicator',
]);

module.exports = {
  BLOCK_IDS,
  MOB_FRAME_ID,
  AGENT_FRAME_ID,
  VOCAB,
  blockToVocab,
  blockToFrameCode,
  biomeToVocab,
  itemToVocab,
  isOpen,
  FOOD_ITEMS,
  PLACEABLE_ITEMS,
  HOSTILE,
};
