'use strict';

// Optional higher-fidelity pixel capture for the mineflayer bridge (issue
// #32): a real rendered screenshot of the bot's view via prismarine-viewer's
// headless renderer, resized to the exact `vision.frame.pixels` shape
// (33x33x3 uint8, `cognitive_runtime/programs/minecraft/streams.py:PIXEL_SHAPE`)
// so it is a drop-in replacement for the colorized semantic-grid fallback
// `RemoteMinecraftBackend.observe()` already uses.
//
// Feature-detected and best-effort throughout: prismarine-viewer pulls in a
// headless-GL native dependency (`gl`) that many hosts (containers, CI,
// sandboxes with no GPU/X server) cannot build or run. Any failure --
// missing module, headless-GL init failure, a bad frame -- disables capture
// permanently for the session and falls back to the existing
// grid-colorization path. Nothing here ever throws out of `start()`/`capture()`.

const PIXEL_SHAPE = [33, 33, 3]; // keep in sync with streams.py:PIXEL_SHAPE

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
    this._available = null; // null = not yet probed, true/false once known
  }

  // Best-effort start; never throws. `available()` reflects whether capture()
  // will do anything after this returns.
  async start(bot, { width = 160, height = 120 } = {}) {
    if (this._available === false) return; // already gave up this session
    let headless;
    try {
      // Optional dependency (package.json `optionalDependencies`); not
      // installed by default because of its native headless-GL requirement.
      headless = require('prismarine-viewer/lib/headless');
    } catch (e) {
      this._available = false;
      log('prismarine-viewer not installed -- falling back to colorized grid pixels. '
        + 'Install it (npm install prismarine-viewer) and see bridge/mineflayer/README.md '
        + 'for headless-GL setup to enable higher-fidelity pixels.');
      return;
    }
    try {
      this._viewer = headless(bot, { width, height, viewDistance: 2, frames: -1 });
      this._width = width;
      this._height = height;
      this._available = true;
      log('prismarine-viewer headless capture enabled (' + width + 'x' + height + ' -> '
        + PIXEL_SHAPE[1] + 'x' + PIXEL_SHAPE[0] + ')');
    } catch (e) {
      this._available = false;
      this._viewer = null;
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
      const frame = this._viewer.getBufferAndResolution
        ? this._viewer.getBufferAndResolution()
        : null;
      if (!frame || !frame.buffer) return null;
      const channels = frame.width * frame.height * 4 <= frame.buffer.length ? 4 : 3;
      return resizeToPixelShape(frame.buffer, frame.width || this._width,
        frame.height || this._height, channels);
    } catch (e) {
      log('capture failed (' + (e && e.message ? e.message : e) + '); disabling for this session');
      this._available = false;
      return null;
    }
  }

  close() {
    try {
      if (this._viewer && this._viewer.close) this._viewer.close();
    } catch (e) { /* best-effort */ }
    this._viewer = null;
  }
}

module.exports = { PixelViewer, resizeToPixelShape, PIXEL_SHAPE };
