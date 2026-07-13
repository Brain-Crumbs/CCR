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
      this._available = false;
      log('first-person viewer dependencies are not installed -- falling back to '
        + 'colorized grid pixels. Install optional deps with '
        + 'npm install --include=optional and see bridge/mineflayer/README.md '
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
    if (!this.available() || !this._viewer) return null;
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

module.exports = { PixelViewer, resizeToPixelShape, flipRgbaRows, PIXEL_SHAPE };
