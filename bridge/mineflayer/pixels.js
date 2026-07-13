'use strict';

// Optional first-person pixel capture for the mineflayer bridge: a real
// rendered snapshot of the bot's view via prismarine-viewer's headless
// renderer, resized to the exact `vision.frame.pixels` shape (33x33x3 uint8,
// `cognitive_runtime/programs/minecraft/streams.py:PIXEL_SHAPE`) so it is a
// drop-in replacement for the colorized semantic-grid fallback
// `RemoteMinecraftBackend.observe()` already uses.
//
// Feature-detected and best-effort throughout: prismarine-viewer pulls in a
// headless-GL native dependency (`gl`) that many hosts (containers, CI,
// sandboxes with no GPU/X server) cannot build or run. Any failure --
// missing module, headless-GL init failure, a bad frame -- disables capture
// permanently for the session and falls back to the existing grid-colorization
// path. Nothing here ever throws out of `start()`/`capture()`.

// Must match streams.py:PIXEL_SHAPE ((2*PIXEL_RADIUS+1)*PIXEL_SCALE = 33):
// the catalog declares this shape and models train against it, so a
// viewer frame of any other size would contradict the session metadata.
const PIXEL_SHAPE = [33, 33, 3];
const Vec3 = require('vec3');

function log(...args) {
  process.stderr.write('[mc-bridge:pixels] ' + args.join(' ') + '\n');
}

// Nearest-neighbor resize of an RGBA/RGB buffer to PIXEL_SHAPE, dropping
// alpha if present. Pure JS -- no extra native dependency for the resize
// itself, only for the initial render.
function resizeToPixelShape(buffer, srcWidth, srcHeight, srcChannels) {
  const [dstH, dstW, dstC] = PIXEL_SHAPE;
  const out = [];
  for (let dy = 0; dy < dstH; dy++) {
    const row = [];
    const sy = Math.min(srcHeight - 1, Math.floor((dy * srcHeight) / dstH));
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(srcWidth - 1, Math.floor((dx * srcWidth) / dstW));
      const srcIdx = (sy * srcWidth + sx) * srcChannels;
      const pixel = [];
      for (let c = 0; c < dstC; c++) pixel.push(buffer[srcIdx + c] || 0);
      row.push(pixel);
    }
    out.push(row);
  }
  return out;
}

class PixelViewer {
  constructor() {
    this._viewer = null;
    this._worldView = null;
    this._renderer = null;
    this._gl = null;
    this._bot = null;
    this._onMove = null;
    this._available = null; // null = not yet probed, true/false once known
    this._raycastFallback = false;
  }

  // Best-effort start; never throws. `available()` reflects whether capture()
  // will do anything after this returns.
  async start(bot, { width = 160, height = 120 } = {}) {
    if (this._available === false) return; // already gave up this session
    let createCanvas;
    let THREE;
    let viewerApi;
    let Worker;
    try {
      // Optional dependencies. prismarine-viewer's own headless example
      // requires node-canvas-webgl; both are optional because they are native
      // and host-sensitive.
      ({ createCanvas } = require('node-canvas-webgl/lib'));
      THREE = require('three');
      ({ Worker } = require('worker_threads'));
      global.THREE = THREE;
      global.Worker = Worker;
      viewerApi = require('prismarine-viewer').viewer;
    } catch (e) {
      if (bot.world && typeof bot.world.raycast === 'function') {
        this._bot = bot;
        this._raycastFallback = true;
        this._available = true;
        log('native first-person viewer dependencies are not installed; using '
          + 'pure-JS first-person raycast pixels from live world data.');
        return;
      }
      this._available = false;
      log('first-person viewer dependencies are not installed and raycast is '
        + 'unavailable -- falling back to colorized grid pixels. Install optional '
        + 'deps with npm install --include=optional and see bridge/mineflayer/README.md '
        + 'for headless-GL setup.');
      return;
    }
    try {
      const canvas = createCanvas(width, height);
      this._renderer = new THREE.WebGLRenderer({ canvas });
      this._viewer = new viewerApi.Viewer(this._renderer);
      if (!this._viewer.setVersion(bot.version)) {
        this._available = false;
        this._viewer = null;
        log('prismarine-viewer does not support Minecraft version ' + bot.version
          + ' -- falling back to colorized grid pixels.');
        return;
      }
      this._worldView = new viewerApi.WorldView(bot.world, 2, bot.entity.position);
      this._viewer.listen(this._worldView);
      this._worldView.init(bot.entity.position);
      this._worldView.listenToBot(bot);
      this._bot = bot;
      this._onMove = () => this._syncCamera();
      bot.on('move', this._onMove);
      this._syncCamera();
      this._gl = this._renderer.getContext();
      this._width = width;
      this._height = height;
      this._available = true;
      log('prismarine-viewer first-person capture enabled (' + width + 'x' + height + ' -> '
        + PIXEL_SHAPE[1] + 'x' + PIXEL_SHAPE[0] + ')');
    } catch (e) {
      this._available = false;
      this.close();
      log('prismarine-viewer headless init failed (' + (e && e.message ? e.message : e)
        + ') -- falling back to colorized grid pixels. This is expected on hosts with no '
        + 'GPU/X server/headless-GL support.');
    }
  }

  available() {
    return this._available === true;
  }

  // Returns a PIXEL_SHAPE-nested uint8 array, or null on any failure
  // (never throws -- the caller falls back to the grid-colorization path).
  capture() {
    if (!this.available()) return null;
    if (this._raycastFallback) return renderRaycastPixels(this._bot);
    if (!this._viewer) return null;
    try {
      this._syncCamera();
      this._viewer.update();
      this._renderer.render(this._viewer.scene, this._viewer.camera);
      const raw = Buffer.alloc(this._width * this._height * 4);
      this._gl.readPixels(0, 0, this._width, this._height, this._gl.RGBA,
        this._gl.UNSIGNED_BYTE, raw);
      return resizeToPixelShape(flipRgbaRows(raw, this._width, this._height),
        this._width, this._height, 4);
    } catch (e) {
      log('capture failed (' + (e && e.message ? e.message : e) + '); disabling for this session');
      this._available = false;
      return null;
    }
  }

  close() {
    try {
      if (this._bot && this._onMove) this._bot.removeListener('move', this._onMove);
      if (this._renderer && this._renderer.dispose) this._renderer.dispose();
    } catch (e) { /* best-effort */ }
    this._viewer = null;
    this._worldView = null;
    this._renderer = null;
    this._gl = null;
    this._bot = null;
    this._onMove = null;
    this._raycastFallback = false;
  }

  _syncCamera() {
    if (!this._viewer || !this._worldView || !this._bot) return;
    this._viewer.setFirstPersonCamera(
      this._bot.entity.position, this._bot.entity.yaw, this._bot.entity.pitch,
    );
    this._worldView.updatePosition(this._bot.entity.position);
  }
}

function flipRgbaRows(buffer, width, height) {
  const stride = width * 4;
  const out = Buffer.alloc(buffer.length);
  for (let y = 0; y < height; y++) {
    const src = y * stride;
    const dst = (height - y - 1) * stride;
    buffer.copy(out, dst, src, src + stride);
  }
  return out;
}

function renderRaycastPixels(bot) {
  if (!bot || !bot.entity || !bot.world || typeof bot.world.raycast !== 'function') return null;
  const [height, width] = PIXEL_SHAPE;
  const eyeHeight = bot.entity.eyeHeight == null ? 1.62 : bot.entity.eyeHeight;
  const eye = bot.entity.position.offset(0, eyeHeight, 0);
  const yaw = bot.entity.yaw || 0;
  const pitch = bot.entity.pitch || 0;
  const hFov = Math.PI / 2.2;
  const vFov = hFov * (height / width);
  const maxRange = 32;
  const out = [];
  for (let y = 0; y < height; y++) {
    const row = [];
    const v = ((y + 0.5) / height - 0.5) * vFov;
    for (let x = 0; x < width; x++) {
      const h = ((x + 0.5) / width - 0.5) * hFov;
      const dir = directionFromYawPitch(yaw + h, pitch + v);
      let color = skyColor(v);
      try {
        const hit = bot.world.raycast(
          eye,
          dir,
          maxRange,
          (block) => Boolean(block && block.name !== 'air' && block.boundingBox !== 'empty')
        );
        if (hit && typeof hit.then !== 'function' && hit.name) {
          const distance = hit.position ? hit.position.distanceTo(eye) : maxRange;
          color = shade(blockColor(hit.name), distance, maxRange);
        }
      } catch (e) {
        color = [0, 0, 0];
      }
      row.push(color);
    }
    out.push(row);
  }
  return out;
}

function directionFromYawPitch(yaw, pitch) {
  const cp = Math.cos(pitch);
  return new Vec3(
    -Math.sin(yaw) * cp,
    -Math.sin(pitch),
    -Math.cos(yaw) * cp
  ).normalize();
}

function skyColor(verticalAngle) {
  const t = Math.max(0, Math.min(1, 0.5 + verticalAngle));
  return [
    Math.round(95 + t * 45),
    Math.round(135 + t * 55),
    Math.round(190 + t * 45),
  ];
}

function shade(rgb, distance, maxRange) {
  const f = Math.max(0.35, 1.0 - Math.min(distance, maxRange) / maxRange * 0.65);
  return rgb.map((v) => Math.max(0, Math.min(255, Math.round(v * f))));
}

function blockColor(name) {
  const n = String(name || '').toLowerCase();
  if (n.includes('water')) return [55, 90, 205];
  if (n.includes('lava') || n.includes('fire')) return [240, 90, 30];
  if (n.includes('grass') || n.includes('leaves') || n.includes('moss')) return [75, 145, 65];
  if (n.includes('red_wool')) return [170, 45, 45];
  if (n.includes('blue_wool')) return [55, 75, 180];
  if (n.includes('lime_wool')) return [95, 180, 55];
  if (n.includes('orange_wool')) return [220, 120, 35];
  if (n.includes('wool')) return [185, 185, 185];
  if (n.includes('dirt') || n.includes('mud')) return [118, 82, 48];
  if (n.includes('sand')) return [215, 198, 130];
  if (n.includes('log') || n.includes('wood') || n.includes('planks')) return [126, 88, 48];
  if (n.includes('gold')) return [235, 190, 45];
  if (n.includes('torch') || n.includes('lantern')) return [245, 200, 90];
  if (n.includes('coal')) return [45, 45, 45];
  if (n.includes('stone') || n.includes('deepslate') || n.includes('ore')) return [120, 120, 125];
  if (n.includes('snow') || n.includes('quartz')) return [230, 235, 238];
  if (n.includes('glass') || n.includes('ice')) return [150, 205, 225];
  if (n.includes('air')) return [130, 175, 225];
  return [145, 130, 105];
}

module.exports = { PixelViewer, resizeToPixelShape, flipRgbaRows, PIXEL_SHAPE };
