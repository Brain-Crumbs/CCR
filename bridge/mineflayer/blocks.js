'use strict';

// Semantic bridge: Minecraft registry names -> the compact SurvivalBox
// vocabulary the runtime's streams and encoders expect.  These constants MUST
// stay in sync with cognitive_runtime/programs/minecraft/world.py.

const DEFAULT_VERSION = '1.19.4';

// Frame cell codes (must equal world.py BLOCK_IDS / MOB_FRAME_ID / AGENT_FRAME_ID).
const BLOCK_IDS = {
  grass: 1, dirt: 2, sand: 3, water: 4, tree: 5,
  stone: 6, coal_ore: 7, berry_bush: 8, placed_block: 9, barrier: 10,
  crafting_table: 11, furnace: 12, chest: 13, portal: 14,
};
const MOB_FRAME_ID = 90;
const AGENT_FRAME_ID = 99;

// The block-name classes; front_block/nearby_blocks payloads use these
// strings, and the CategoryEncoder one-hots front_block against them.
const VOCAB = Object.keys(BLOCK_IDS);

function loadMinecraftData(version = DEFAULT_VERSION) {
  try {
    // minecraft-data is a transitive dependency of mineflayer in normal
    // installs.  The bridge remains dependency-light in source checkouts that
    // have not run npm install yet.
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    return require('minecraft-data')(version);
  } catch (e) {
    return null;
  }
}

const MC_DATA = loadMinecraftData();

function registryNames(arrayName) {
  const values = MC_DATA && Array.isArray(MC_DATA[arrayName]) ? MC_DATA[arrayName] : [];
  return values.map((entry) => String(entry.name || '').toLowerCase()).filter(Boolean);
}

function setOf(names) {
  return new Set(names.map((n) => String(n).toLowerCase()));
}

function addAll(target, names) {
  for (const n of names) target.add(n);
}

const WOOD_TYPES = [
  'oak', 'spruce', 'birch', 'jungle', 'acacia', 'dark_oak', 'mangrove',
  'cherry', 'crimson', 'warped',
];

const DIRT_LIKE = setOf([
  'grass_block', 'grass', 'short_grass', 'tall_grass', 'fern', 'large_fern',
  'moss_block', 'moss_carpet', 'dirt_path', 'farmland', 'mycelium', 'podzol',
  'muddy_mangrove_roots', 'vine', 'glow_lichen', 'lily_pad', 'spore_blossom',
  'brown_mushroom', 'red_mushroom',
  'dandelion', 'poppy', 'blue_orchid', 'allium', 'azure_bluet',
  'red_tulip', 'orange_tulip', 'white_tulip', 'pink_tulip', 'oxeye_daisy',
  'cornflower', 'lily_of_the_valley', 'wither_rose', 'sunflower', 'lilac',
  'rose_bush', 'peony', 'torchflower', 'pitcher_plant', 'pink_petals',
  'nether_sprouts', 'warped_roots', 'crimson_roots',
  'snow', 'snow_block', 'powder_snow',
]);

const DIRT_BLOCKS = setOf([
  'dirt', 'coarse_dirt', 'rooted_dirt', 'gravel', 'clay', 'mud',
  'suspicious_gravel',
]);

const SAND_BLOCKS = setOf([
  'sand', 'red_sand', 'sandstone', 'red_sandstone', 'smooth_sandstone',
  'smooth_red_sandstone', 'cut_sandstone', 'cut_red_sandstone',
  'chiseled_sandstone', 'chiseled_red_sandstone', 'suspicious_sand',
  'soul_sand', 'soul_soil',
]);

const WATER_BLOCKS = setOf([
  'water', 'bubble_column', 'seagrass', 'tall_seagrass', 'kelp', 'kelp_plant',
  'ice', 'packed_ice', 'blue_ice', 'frosted_ice',
]);

const HAZARD_BLOCKS = setOf([
  'lava', 'fire', 'soul_fire', 'magma_block', 'cactus',
  'wither_rose', 'campfire', 'soul_campfire', 'barrier', 'bedrock',
  'end_gateway',
]);

// Container / crafting-table / furnace interaction targets (issue #40) --
// USE against one of these opens it instead of the food/placeable logic.
const CRAFTING_TABLE_BLOCKS = setOf(['crafting_table']);
const FURNACE_BLOCKS = setOf(['furnace', 'blast_furnace', 'smoker']);
const CHEST_BLOCKS = setOf([
  'chest', 'trapped_chest', 'barrel', 'ender_chest',
  'white_shulker_box', 'orange_shulker_box', 'magenta_shulker_box',
  'light_blue_shulker_box', 'yellow_shulker_box', 'lime_shulker_box',
  'pink_shulker_box', 'gray_shulker_box', 'light_gray_shulker_box',
  'cyan_shulker_box', 'purple_shulker_box', 'blue_shulker_box',
  'brown_shulker_box', 'green_shulker_box', 'red_shulker_box',
  'black_shulker_box', 'shulker_box',
]);
// Portal blocks -- crossing one flips the sim-equivalent `dimension` concept
// (see event.dimension_changed); real dimension identity comes from
// `bot.game.dimension` in world.js, this is only the frame/vocab mapping.
const PORTAL_BLOCKS = setOf(['nether_portal', 'end_portal']);

// Curated structure "marker" blocks: seeing one of these nearby is a
// best-effort, heuristic proxy for having found the structure that
// generates it -- mineflayer has no direct "structure discovered" signal.
const STRUCTURE_MARKERS = new Map([
  ['bell', 'village'], ['end_portal_frame', 'stronghold'],
  ['nether_wart_block', 'fortress'], ['nether_bricks', 'fortress'],
  ['prismarine', 'monument'], ['sea_lantern', 'monument'],
]);

const FOOD_PLANT_BLOCKS = setOf([
  'sweet_berry_bush', 'cave_vines', 'cave_vines_plant', 'wheat', 'carrots',
  'potatoes', 'beetroots', 'cocoa', 'melon', 'pumpkin', 'carved_pumpkin',
  'hay_block', 'chorus_flower', 'chorus_plant', 'attached_melon_stem',
  'attached_pumpkin_stem', 'melon_stem', 'pumpkin_stem', 'bamboo',
  'bamboo_sapling', 'sugar_cane', 'nether_wart',
]);

const STONE_BLOCKS = setOf([
  'stone', 'cobblestone', 'mossy_cobblestone', 'andesite', 'diorite',
  'granite', 'tuff', 'calcite', 'deepslate', 'cobbled_deepslate',
  'dripstone_block', 'pointed_dripstone', 'blackstone', 'polished_blackstone',
  'basalt', 'smooth_basalt', 'netherrack', 'end_stone', 'smooth_stone',
  'stone_bricks', 'mossy_stone_bricks', 'cracked_stone_bricks',
  'chiseled_stone_bricks', 'infested_stone', 'infested_cobblestone',
  'infested_deepslate', 'amethyst_block', 'budding_amethyst',
]);

const CONSTRUCTED_BLOCKS = setOf([
  'crafting_table', 'furnace', 'blast_furnace', 'smoker', 'cartography_table',
  'fletching_table', 'smithing_table', 'loom', 'stonecutter', 'grindstone',
  'anvil', 'chipped_anvil', 'damaged_anvil', 'enchanting_table', 'bookshelf',
  'chiseled_bookshelf', 'chest', 'trapped_chest', 'barrel', 'ender_chest',
  'shulker_box', 'white_shulker_box', 'orange_shulker_box', 'magenta_shulker_box',
  'light_blue_shulker_box', 'yellow_shulker_box', 'lime_shulker_box',
  'pink_shulker_box', 'gray_shulker_box', 'light_gray_shulker_box',
  'cyan_shulker_box', 'purple_shulker_box', 'blue_shulker_box',
  'brown_shulker_box', 'green_shulker_box', 'red_shulker_box',
  'black_shulker_box', 'glass', 'glass_pane', 'tinted_glass',
  'ladder', 'scaffolding', 'chain', 'iron_bars',
  'torch', 'wall_torch', 'soul_torch', 'soul_wall_torch', 'lantern',
  'soul_lantern', 'glowstone', 'sea_lantern', 'shroomlight', 'end_rod',
  'redstone_lamp', 'beacon', 'jack_o_lantern', 'ochre_froglight',
  'verdant_froglight', 'pearlescent_froglight', 'candle', 'white_candle',
  'orange_candle', 'magenta_candle', 'light_blue_candle', 'yellow_candle',
  'lime_candle', 'pink_candle', 'gray_candle', 'light_gray_candle',
  'cyan_candle', 'purple_candle', 'blue_candle', 'brown_candle',
  'green_candle', 'red_candle', 'black_candle',
]);

const ORE_ITEMS = setOf([
  'coal', 'charcoal', 'raw_iron', 'raw_gold', 'raw_copper', 'iron_ingot',
  'gold_ingot', 'copper_ingot', 'diamond', 'emerald', 'lapis_lazuli',
  'redstone', 'quartz', 'netherite_scrap', 'netherite_ingot', 'amethyst_shard',
]);

const LIGHT_ITEMS = setOf([
  'torch', 'soul_torch', 'lantern', 'soul_lantern', 'glowstone', 'sea_lantern',
  'shroomlight', 'end_rod', 'redstone_lamp', 'beacon', 'jack_o_lantern',
  'ochre_froglight', 'verdant_froglight', 'pearlescent_froglight', 'candle',
  'white_candle', 'orange_candle', 'magenta_candle', 'light_blue_candle',
  'yellow_candle', 'lime_candle', 'pink_candle', 'gray_candle',
  'light_gray_candle', 'cyan_candle', 'purple_candle', 'blue_candle',
  'brown_candle', 'green_candle', 'red_candle', 'black_candle',
]);

const FOOD_ITEM_MAP = new Map([
  ['sweet_berries', 'berries'], ['glow_berries', 'berries'],
  ['apple', 'apple'], ['golden_apple', 'apple'], ['enchanted_golden_apple', 'apple'],
  ['bread', 'bread'], ['cookie', 'bread'],
  ['cooked_beef', 'cooked_meat'], ['cooked_porkchop', 'cooked_meat'],
  ['cooked_chicken', 'cooked_meat'], ['cooked_mutton', 'cooked_meat'],
  ['cooked_rabbit', 'cooked_meat'], ['cooked_cod', 'cooked_meat'],
  ['cooked_salmon', 'cooked_meat'], ['beef', 'cooked_meat'],
  ['porkchop', 'cooked_meat'], ['chicken', 'cooked_meat'],
  ['mutton', 'cooked_meat'], ['rabbit', 'cooked_meat'],
  ['carrot', 'berries'], ['potato', 'berries'], ['baked_potato', 'berries'],
  ['beetroot', 'berries'], ['melon_slice', 'berries'], ['pumpkin_pie', 'bread'],
  ['mushroom_stew', 'cooked_meat'], ['beetroot_soup', 'berries'],
  ['rabbit_stew', 'cooked_meat'],
]);

const ITEM_MAP = new Map([
  ...FOOD_ITEM_MAP,
  ['coal', 'coal'], ['charcoal', 'coal'],
  ['cobblestone', 'cobblestone'], ['dirt', 'dirt'], ['sand', 'sand'],
]);

for (const wood of WOOD_TYPES) {
  addAll(CONSTRUCTED_BLOCKS, [
    `${wood}_planks`, `${wood}_door`, `${wood}_trapdoor`, `${wood}_fence`,
    `${wood}_fence_gate`, `${wood}_stairs`, `${wood}_slab`, `${wood}_button`,
    `${wood}_pressure_plate`, `${wood}_sign`, `${wood}_wall_sign`,
    `${wood}_hanging_sign`, `${wood}_wall_hanging_sign`,
  ]);
  addAll(CONSTRUCTED_BLOCKS, [`${wood}_bed`]);
}

for (const color of [
  'white', 'orange', 'magenta', 'light_blue', 'yellow', 'lime', 'pink', 'gray',
  'light_gray', 'cyan', 'purple', 'blue', 'brown', 'green', 'red', 'black',
]) {
  addAll(CONSTRUCTED_BLOCKS, [
    `${color}_bed`, `${color}_wool`, `${color}_carpet`, `${color}_glass`,
    `${color}_stained_glass`, `${color}_stained_glass_pane`,
    `${color}_terracotta`, `${color}_glazed_terracotta`, `${color}_concrete`,
    `${color}_concrete_powder`,
  ]);
}

for (const blockName of registryNames('blocksArray')) {
  if (blockName.endsWith('_ore') || blockName.startsWith('raw_') || blockName.endsWith('_ore_block')) {
    // registry-driven enrichment; curated rules still decide precedence below
    ORE_ITEMS.add(blockName.replace(/_ore$/, ''));
  }
}

function hasAny(name, parts) {
  return parts.some((part) => name.includes(part));
}

function hasSuffix(name, suffixes) {
  return suffixes.some((suffix) => name.endsWith(suffix));
}

function isWoodBlock(name) {
  return hasSuffix(name, ['_log', '_wood', '_leaves', '_stem', '_hyphae', '_sapling', '_roots'])
    || name === 'mangrove_roots';
}

function isOreBlock(name) {
  return name.endsWith('_ore')
    || name.endsWith('_ore_block')
    || name.startsWith('raw_')
    || [
      'coal_block', 'iron_block', 'gold_block', 'copper_block', 'diamond_block',
      'emerald_block', 'lapis_block', 'redstone_block', 'netherite_block',
      'quartz_block',
    ].includes(name);
}

function isConstructedBlock(name) {
  return CONSTRUCTED_BLOCKS.has(name)
    || hasSuffix(name, [
      '_door', '_trapdoor', '_fence', '_fence_gate', '_wall', '_wall_sign',
      '_hanging_sign', '_wall_hanging_sign', '_glass', '_glass_pane',
      '_stained_glass', '_stained_glass_pane', '_bed', '_banner',
      '_wall_banner', '_stairs', '_slab', '_button', '_pressure_plate',
    ])
    || hasAny(name, ['chest', 'shulker_box']);
}

// A block that has no collision box reads as open ground for a top-down view.
function isOpen(block) {
  if (!block) return true;
  const name = String(block.name || '').toLowerCase();
  if (name === 'air' || name === 'cave_air' || name === 'void_air') return true;
  return block.boundingBox === 'empty';
}

function nameToVocab(name, block = null) {
  const n = String(name || '').toLowerCase();
  if (!n || n === 'air' || n === 'cave_air' || n === 'void_air') return 'grass';
  if (HAZARD_BLOCKS.has(n)) return 'barrier';
  if (WATER_BLOCKS.has(n)) return 'water';
  if (FOOD_PLANT_BLOCKS.has(n)) return 'berry_bush';
  if (DIRT_LIKE.has(n)) return 'grass';
  if (DIRT_BLOCKS.has(n)) return 'dirt';
  if (SAND_BLOCKS.has(n)) return 'sand';
  if (isWoodBlock(n)) return 'tree';
  if (isOreBlock(n)) return 'coal_ore';
  if (STONE_BLOCKS.has(n)) return 'stone';
  if (CRAFTING_TABLE_BLOCKS.has(n)) return 'crafting_table';
  if (FURNACE_BLOCKS.has(n)) return 'furnace';
  if (CHEST_BLOCKS.has(n)) return 'chest';
  if (PORTAL_BLOCKS.has(n)) return 'portal';
  if (isConstructedBlock(n)) return 'placed_block';
  if (isOpen(block)) return 'grass';
  // Unknown but collidable: treat as generic solid so movement/vision see a wall.
  return 'stone';
}

// Map one Minecraft block to a SurvivalBox vocab name.
function blockToVocab(block) {
  if (!block) return 'grass';
  return nameToVocab(block.name, block);
}

// 'crafting_table' | 'furnace' | 'chest' | null -- which container/interaction
// class a block belongs to (issue #40: container / crafting-table / furnace
// interactions), independent of the coarser vision vocab above.
function containerType(name) {
  const n = String(name || '').toLowerCase();
  if (CRAFTING_TABLE_BLOCKS.has(n)) return 'crafting_table';
  if (FURNACE_BLOCKS.has(n)) return 'furnace';
  if (CHEST_BLOCKS.has(n)) return 'chest';
  return null;
}

// The structure name a marker block is best-effort evidence of, or null.
function structureMarker(name) {
  return STRUCTURE_MARKERS.get(String(name || '').toLowerCase()) || null;
}

function blockToFrameCode(block) {
  return BLOCK_IDS[blockToVocab(block)];
}

function blockExactName(block) {
  if (!block) return 'air';
  return String(block.name || 'air').toLowerCase();
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

function isToolItem(name) {
  const n = String(name || '').toLowerCase();
  return hasSuffix(n, ['_pickaxe', '_axe', '_shovel', '_hoe'])
    || ['shears', 'fishing_rod', 'flint_and_steel', 'bucket'].includes(n);
}

function isWeaponItem(name) {
  const n = String(name || '').toLowerCase();
  return hasSuffix(n, ['_sword'])
    || ['bow', 'crossbow', 'trident', 'shield'].includes(n);
}

function isArmorItem(name) {
  const n = String(name || '').toLowerCase();
  return hasSuffix(n, ['_helmet', '_chestplate', '_leggings', '_boots'])
    || ['turtle_helmet', 'elytra'].includes(n);
}

// Item name -> the reward's item vocabulary (food/tool names it rewards, else
// a block-vocab name, else the raw name so novelty still counts it).
function itemToVocab(name) {
  if (!name) return name;
  const n = String(name).toLowerCase();
  if (ITEM_MAP.has(n)) return ITEM_MAP.get(n);
  if (isToolItem(n) || isWeaponItem(n) || isArmorItem(n) || LIGHT_ITEMS.has(n)) return n;
  if (hasSuffix(n, ['_log', '_wood', '_stem', '_hyphae'])) return 'log';
  if (ORE_ITEMS.has(n)) return n;
  return n; // keep raw so new-item novelty still fires
}

const FOOD_ITEMS = new Set(['berries', 'apple', 'bread', 'cooked_meat']);
const PLACEABLE_ITEMS = new Set([
  'log', 'cobblestone', 'dirt', 'sand',
  ...LIGHT_ITEMS,
]);

const HOSTILE = setOf([
  'zombie', 'husk', 'drowned', 'skeleton', 'stray', 'creeper', 'spider',
  'cave_spider', 'witch', 'enderman', 'zombie_villager', 'pillager',
  'vindicator', 'evoker', 'illusioner', 'ravager', 'vex', 'blaze', 'ghast',
  'magma_cube', 'slime', 'guardian', 'elder_guardian', 'phantom', 'shulker',
  'silverfish', 'endermite', 'hoglin', 'zoglin', 'piglin_brute',
  'wither_skeleton', 'warden', 'wither',
]);

const PASSIVE = setOf([
  'allay', 'axolotl', 'bat', 'camel', 'cat', 'chicken', 'cod', 'cow', 'donkey',
  'fox', 'frog', 'glow_squid', 'horse', 'mooshroom', 'mule', 'ocelot', 'parrot',
  'pig', 'pufferfish', 'rabbit', 'salmon', 'sheep', 'skeleton_horse',
  'snow_golem', 'squid', 'strider', 'tadpole', 'tropical_fish', 'turtle',
  'villager', 'wandering_trader',
]);

const NEUTRAL = setOf([
  'bee', 'dolphin', 'goat', 'iron_golem', 'llama', 'panda', 'piglin',
  'polar_bear', 'trader_llama', 'wolf', 'zombified_piglin',
]);

if (MC_DATA && Array.isArray(MC_DATA.entitiesArray)) {
  for (const entity of MC_DATA.entitiesArray) {
    const name = String(entity.name || '').toLowerCase();
    const type = String(entity.type || entity.category || '').toLowerCase();
    if (!name) continue;
    if (type.includes('hostile')) HOSTILE.add(name);
    else if (type.includes('passive')) PASSIVE.add(name);
    else if (type.includes('neutral')) NEUTRAL.add(name);
  }
}

function entityToSemantic(name) {
  const n = String(name || '').toLowerCase();
  if (HOSTILE.has(n)) return 'hostile_mob';
  if (PASSIVE.has(n)) return 'passive_mob';
  if (NEUTRAL.has(n)) return 'neutral_mob';
  return 'entity';
}

module.exports = {
  BLOCK_IDS,
  MOB_FRAME_ID,
  AGENT_FRAME_ID,
  VOCAB,
  blockToVocab,
  blockToFrameCode,
  blockExactName,
  nameToVocab,
  biomeToVocab,
  itemToVocab,
  isOpen,
  isToolItem,
  isWeaponItem,
  isArmorItem,
  entityToSemantic,
  containerType,
  structureMarker,
  mcDataFor: loadMinecraftData,
  FOOD_ITEMS,
  PLACEABLE_ITEMS,
  LIGHT_ITEMS,
  HOSTILE,
  PASSIVE,
  NEUTRAL,
};
