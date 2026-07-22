/**
 * <pixel-horizon-viewer> — reusable, framework-agnostic web component that
 * shows, for each prediction horizon h, a four-up strip: the SEEN frame the
 * model was given at t, the PREDICTED frame at t+h, the ACTUAL frame at t+h,
 * and an |error| heatmap, plus MSE/PSNR readouts, a scrubber/playback over
 * the episode and an MSE-over-time chart. The seen→predicted→actual triple
 * answers, for every frame, "what the model saw, what it predicted it would
 * see next, and what actually was shown."
 *
 * Prediction sources:
 *   - "copy-last"  : predicted(t+h) = actual(t)      (the harness baseline)
 *   - "mean-frame" : predicted(t+h) = episode mean   (the harness baseline)
 *   - "model"      : frames from an exported predictions_<episode>.json
 *                    (see viewer/export_predictions.py) when provided.
 *
 * Attributes:
 *   frames-src       URL returning the /frames API payload (required)
 *   predictions-src  URL returning a pixel-predictions-v1 payload (optional)
 *   horizons         comma list, default "1,10,100"
 *   scale            CSS pixels per frame pixel, default 6
 *
 * Usage (plain HTML):
 *   <script type="module" src="/pixel-horizon-viewer.js"></script>
 *   <pixel-horizon-viewer frames-src="/api/sessions/S/episodes/episode_00000/frames"
 *                         predictions-src="/api/sessions/S/episodes/episode_00000/predictions"
 *                         horizons="1,10,100"></pixel-horizon-viewer>
 *
 * Usage (React — custom elements work as-is; set attributes as strings):
 *   <pixel-horizon-viewer frames-src={framesUrl} horizons="1,10,100" />
 */
"use strict";

/* Chart palette (validated for light and dark surfaces; see dataviz notes in
 * the repo). Series follow the horizon entity, in fixed slot order. */
const SERIES_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"];
const SERIES_DARK = ["#3987e5", "#199e70", "#c98500", "#9085e9"];
/* Sequential blue ramp for the |error| heatmap (magnitude = one hue). */
const BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

const TEMPLATE = `
<style>
  :host {
    display: block;
    --surface-1: #fcfcfb;
    --surface-2: #f2f1ef;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --line: #dcdbd7;
    font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--surface-1);
  }
  @media (prefers-color-scheme: dark) {
    :host {
      --surface-1: #1a1a19;
      --surface-2: #242423;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --line: #3a3a38;
    }
  }
  .root { padding: 12px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 10px; }
  .controls label { color: var(--text-secondary); }
  input[type="range"] { flex: 1 1 200px; min-width: 140px; }
  select, button {
    background: var(--surface-2); color: var(--text-primary);
    border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; font: inherit;
  }
  button { cursor: pointer; }
  .status { color: var(--text-secondary); margin: 8px 0; }
  .horizons { display: flex; flex-wrap: wrap; gap: 16px; }
  .panel { background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
  .panel h3 { margin: 0 0 8px; font-size: 13px; font-weight: 600; }
  .strip { display: flex; gap: 10px; }
  .cell { text-align: center; }
  .cell figcaption { color: var(--text-secondary); margin-top: 4px; }
  canvas.px { image-rendering: pixelated; background: var(--surface-1); border: 1px solid var(--line); border-radius: 4px; display: block; }
  .metrics { margin-top: 6px; color: var(--text-secondary); }
  .metrics b { color: var(--text-primary); font-weight: 600; }
  .chartwrap { margin-top: 16px; overflow-x: auto; }
  .legend { display: flex; gap: 14px; margin: 4px 0 2px; color: var(--text-secondary); flex-wrap: wrap; }
  .legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
  .tooltip {
    position: absolute; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px;
    box-shadow: 0 2px 8px rgba(0,0,0,.18); display: none; white-space: nowrap; z-index: 3;
  }
  .tooltip .row b { font-weight: 600; }
  .chartbox { position: relative; }
  details.table { margin-top: 8px; }
  details.table table { border-collapse: collapse; margin-top: 6px; }
  details.table td, details.table th { border: 1px solid var(--line); padding: 2px 8px; text-align: right; }
</style>
<div class="root">
  <div class="controls">
    <button id="play" aria-label="play/pause">▶</button>
    <label>t = <span id="tval">0</span></label>
    <input id="scrub" type="range" min="0" max="0" value="0" step="1" aria-label="start frame">
    <label>prediction:
      <select id="source">
        <option value="copy-last">copy-last baseline</option>
        <option value="mean-frame">mean-frame baseline</option>
      </select>
    </label>
  </div>
  <div class="status" id="status">loading…</div>
  <div class="horizons" id="horizons"></div>
  <div class="chartwrap">
    <div class="legend" id="legend"></div>
    <div class="chartbox">
      <svg id="chart" width="720" height="180" role="img" aria-label="prediction error over start frame"></svg>
      <div class="tooltip" id="tooltip"></div>
    </div>
    <details class="table"><summary>error table (current t)</summary><div id="table"></div></details>
  </div>
</div>`;

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function hexRGB(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

function rampColor(v) {
  // v in [0,1] -> one-hue sequential blue anchored at the surface, so zero
  // error recedes into the background in both modes and high error pops.
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const surface = hexRGB(dark ? "#242423" : "#fcfcfb");
  const ramp = dark ? [...BLUE_RAMP].reverse() : BLUE_RAMP;
  const pos = Math.max(0, Math.min(1, v)) * ramp.length; // 0..len, 0 = pure surface
  const idx = Math.min(ramp.length - 1, Math.floor(pos));
  const lo = idx === 0 ? surface : hexRGB(ramp[idx - 1]);
  const hi = hexRGB(ramp[idx]);
  const f = pos - idx;
  return [0, 1, 2].map((i) => Math.round(lo[i] + (hi[i] - lo[i]) * f));
}

function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] - b[i]) / 255;
    s += d * d;
  }
  return s / a.length;
}

function psnrText(m) {
  if (m <= 0) return "∞";
  return (10 * Math.log10(1 / m)).toFixed(1);
}

class PixelHorizonViewer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = TEMPLATE;
    this._frames = null;      // [{i,t,hash,bytes|null}], native shape
    this._shape = null;       // [h,w,c]
    this._pred = null;        // decoded predictions payload or null
    this._meanFrame = null;   // Float array, native shape
    this._t = 0;
    this._timer = null;
    this._mseCache = new Map(); // source -> {h: Float64Array}
  }

  static get observedAttributes() { return ["frames-src", "predictions-src", "horizons", "scale"]; }

  connectedCallback() {
    this._$("#scrub").addEventListener("input", (e) => this.setTime(Number(e.target.value), true));
    this._$("#source").addEventListener("change", () => this._render(true));
    this._$("#play").addEventListener("click", () => this._togglePlay());
    this._load();
  }

  disconnectedCallback() { this._stop(); }

  attributeChangedCallback(name, oldV, newV) {
    if (oldV === newV || !this.isConnected) return;
    if (name !== "frames-src" && name !== "predictions-src") return;
    // Coalesce back-to-back attribute writes (a host page typically sets
    // predictions-src and frames-src together) into one load.
    if (this._loadQueued) return;
    this._loadQueued = true;
    queueMicrotask(() => { this._loadQueued = false; this._load(); });
  }

  get horizons() {
    return (this.getAttribute("horizons") || "1,10,100").split(",").map((s) => parseInt(s.trim(), 10)).filter((h) => h > 0);
  }
  get scale() { return Number(this.getAttribute("scale") || 6); }
  get time() { return this._t; }

  setTick(tick) {
    if (!this._frames?.length) return this.setTime(tick);
    const wanted = Number(tick) || 0;
    const closest = this._frames.reduce((best, frame, index) => (
      !best || Math.abs(Number(frame.tick) - wanted) < best.distance
        ? { index, distance: Math.abs(Number(frame.tick) - wanted) } : best
    ), null);
    this.setTime(closest.index);
  }

  setTime(t, emit = false) {
    const requested = Math.max(0, Math.round(Number(t) || 0));
    const max = Number(this._$("#scrub")?.max ?? 0);
    const next = this._frames ? Math.min(max, requested) : requested;
    this._t = next;
    if (this._$("#scrub")) this._$("#scrub").value = next;
    this._render();
    if (emit) this.dispatchEvent(new CustomEvent("timechange", {
      detail: { t: next, tick: this._frames?.[next]?.tick ?? next }, bubbles: true, composed: true,
    }));
  }

  _$(sel) { return this.shadowRoot.querySelector(sel); }

  async _load() {
    const src = this.getAttribute("frames-src");
    if (!src) { this._$("#status").textContent = "set frames-src"; return; }
    this._stop();
    this._$("#status").textContent = "loading frames…";
    try {
      const payload = await (await fetch(src)).json();
      if (payload.error) throw new Error(payload.error);
      this._shape = payload.shape;
      this._frames = payload.frames.map((f) => ({ ...f, bytes: f.data ? b64ToBytes(f.data) : null }));
      this._computeMean();
    } catch (err) {
      this._$("#status").textContent = `failed to load frames: ${err.message}`;
      return;
    }
    this._pred = null;
    const predSrc = this.getAttribute("predictions-src");
    if (predSrc) {
      try {
        const p = await (await fetch(predSrc)).json();
        if (!p.error && p.format === "pixel-predictions-v1") {
          p._decoded = {};
          for (const [h, entry] of Object.entries(p.predictions)) {
            p._decoded[h] = entry.frames.map(b64ToBytes);
          }
          p._targets = p.targets ? p.targets.map(b64ToBytes) : null;
          this._pred = p;
          if (p.horizons?.length) {
            this.setAttribute("horizons", p.horizons.join(","));
          }
        }
      } catch { /* predictions are optional */ }
    }
    const sourceSel = this._$("#source");
    const hasModel = !!this._pred;
    if (hasModel && !sourceSel.querySelector('option[value="model"]')) {
      const opt = document.createElement("option");
      opt.value = "model";
      opt.textContent = this._pred.source === "live-record" ? "model (live record)" : "model (exported)";
      sourceSel.prepend(opt);
      sourceSel.value = "model";
    } else if (!hasModel) {
      sourceSel.querySelector('option[value="model"]')?.remove();
      if (sourceSel.value === "model") sourceSel.value = "copy-last";
    }
    this._mseCache.clear();
    const maxH = Math.max(...this.horizons);
    const scrub = this._$("#scrub");
    scrub.max = Math.max(0, this._frames.length - 1 - maxH);
    this._t = Math.min(this._t, Number(scrub.max));
    scrub.value = this._t;
    this._$("#status").textContent =
      `${this._frames.length} frames (${this._shape.join("×")} ${this._pred ? `· ${this._pred.source === "live-record" ? "live" : "exported"} model predictions loaded` : "· no model predictions recorded — showing baselines"})`;
    this._render(true);
  }

  _computeMean() {
    const valid = this._frames.filter((f) => f.bytes);
    if (!valid.length) { this._meanFrame = null; return; }
    const n = valid[0].bytes.length;
    const acc = new Float64Array(n);
    for (const f of valid) for (let i = 0; i < n; i++) acc[i] += f.bytes[i];
    const mean = new Uint8ClampedArray(n);
    for (let i = 0; i < n; i++) mean[i] = Math.round(acc[i] / valid.length);
    this._meanFrame = mean;
  }

  /** Frame bytes for the last recorded frame at or before index i (frames can be null if elided). */
  _actual(i) {
    for (let j = i; j >= 0; j--) if (this._frames[j] && this._frames[j].bytes) return this._frames[j].bytes;
    return null;
  }

  /** {bytes, shape} predicted for target index t+h from source at t, or null. */
  _predicted(source, t, h) {
    if (source === "model") {
      const seq = this._pred?._decoded?.[String(h)];
      if (!seq || t >= seq.length) return null;
      return { bytes: seq[t], shape: this._pred.prediction_shape };
    }
    if (source === "mean-frame") {
      return this._meanFrame ? { bytes: this._meanFrame, shape: this._shape } : null;
    }
    const bytes = this._actual(t); // copy-last
    return bytes ? { bytes, shape: this._shape } : null;
  }

  /** Actual frame in the prediction's space (model targets are pooled by the exporter). */
  _target(source, i) {
    if (source === "model" && this._pred?._targets) {
      return { bytes: this._pred._targets[i], shape: this._pred.prediction_shape };
    }
    const bytes = this._actual(i);
    return bytes ? { bytes, shape: this._shape } : null;
  }

  _drawFrame(canvas, bytes, shape) {
    const [h, w] = shape;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w * this.scale}px`;
    canvas.style.height = `${h * this.scale}px`;
    const ctx = canvas.getContext("2d");
    if (!bytes) { ctx.clearRect(0, 0, w, h); return; }
    const img = ctx.createImageData(w, h);
    for (let p = 0, q = 0; p < bytes.length; p += 3, q += 4) {
      img.data[q] = bytes[p];
      img.data[q + 1] = bytes[p + 1];
      img.data[q + 2] = bytes[p + 2];
      img.data[q + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  _drawDiff(canvas, a, b, shape) {
    const [h, w] = shape;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w * this.scale}px`;
    canvas.style.height = `${h * this.scale}px`;
    const ctx = canvas.getContext("2d");
    if (!a || !b) { ctx.clearRect(0, 0, w, h); return; }
    const img = ctx.createImageData(w, h);
    for (let px = 0; px < w * h; px++) {
      const p = px * 3;
      const err = (Math.abs(a[p] - b[p]) + Math.abs(a[p + 1] - b[p + 1]) + Math.abs(a[p + 2] - b[p + 2])) / (3 * 255);
      const [r, g, bch] = rampColor(err);
      const q = px * 4;
      img.data[q] = r;
      img.data[q + 1] = g;
      img.data[q + 2] = bch;
      img.data[q + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  _mseSeries(source, h) {
    const key = `${source}:${h}`;
    if (this._mseCache.has(key)) return this._mseCache.get(key);
    const n = this._frames.length - h;
    const out = new Float64Array(Math.max(0, n)).fill(NaN);
    for (let t = 0; t < n; t++) {
      const pred = this._predicted(source, t, h);
      const target = this._target(source, t + h);
      if (pred && target && pred.bytes.length === target.bytes.length) {
        out[t] = mse(pred.bytes, target.bytes);
      }
    }
    this._mseCache.set(key, out);
    return out;
  }

  _render(full = false) {
    if (!this._frames) return;
    const source = this._$("#source").value;
    const t = this._t;
    this._$("#tval").textContent = `${t} / ${this._$("#scrub").max}`;

    const host = this._$("#horizons");
    if (full || host.children.length !== this.horizons.length) {
      host.innerHTML = "";
      for (const h of this.horizons) {
        const panel = document.createElement("div");
        panel.className = "panel";
        panel.dataset.h = h;
        panel.innerHTML = `
          <h3>horizon t+${h}</h3>
          <div class="strip">
            <figure class="cell"><canvas class="px seen"></canvas><figcaption>seen t</figcaption></figure>
            <figure class="cell"><canvas class="px pred"></canvas><figcaption>predicted t+${h}</figcaption></figure>
            <figure class="cell"><canvas class="px actual"></canvas><figcaption>actual t+${h}</figcaption></figure>
            <figure class="cell"><canvas class="px diff"></canvas><figcaption>|error|</figcaption></figure>
          </div>
          <div class="metrics"></div>`;
        host.appendChild(panel);
      }
    }

    const rows = [];
    for (const panel of host.children) {
      const h = Number(panel.dataset.h);
      const pred = this._predicted(source, t, h);
      const target = this._target(source, t + h);
      const seen = this._target(source, t); // the input frame the model saw at t
      const shape = pred ? pred.shape : this._shape;
      this._drawFrame(panel.querySelector(".seen"), seen?.bytes ?? null, seen?.shape ?? this._shape);
      this._drawFrame(panel.querySelector(".actual"), target?.bytes ?? null, target?.shape ?? this._shape);
      this._drawFrame(panel.querySelector(".pred"), pred?.bytes ?? null, shape);
      const comparable = pred && target && pred.bytes.length === target.bytes.length;
      this._drawDiff(panel.querySelector(".diff"), comparable ? pred.bytes : null, comparable ? target.bytes : null, shape);
      const m = comparable ? mse(pred.bytes, target.bytes) : NaN;
      panel.querySelector(".metrics").innerHTML = comparable
        ? `MSE <b>${m.toExponential(2)}</b> · PSNR <b>${psnrText(m)}</b> dB`
        : `no comparable frames at t=${t}`;
      rows.push({ h, mse: m });
    }
    this._renderChart(source, t);
    this._renderTable(rows, source);
  }

  _renderTable(rows, source) {
    this._$("#table").innerHTML =
      `<table><tr><th>horizon</th><th>MSE (${source})</th><th>PSNR dB</th></tr>` +
      rows.map((r) => `<tr><td>t+${r.h}</td><td>${Number.isNaN(r.mse) ? "–" : r.mse.toExponential(3)}</td><td>${Number.isNaN(r.mse) ? "–" : psnrText(r.mse)}</td></tr>`).join("") +
      `</table>`;
  }

  _renderChart(source, tCurrent) {
    const svg = this._$("#chart");
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const colors = dark ? SERIES_DARK : SERIES_LIGHT;
    const horizons = this.horizons;
    const W = svg.clientWidth || 720;
    const H = 180;
    const pad = { l: 56, r: 70, t: 10, b: 24 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const series = horizons.map((h) => this._mseSeries(source, h));
    let maxY = 1e-6;
    for (const s of series) for (const v of s) if (!Number.isNaN(v)) maxY = Math.max(maxY, v);
    const n = Math.max(...series.map((s) => s.length), 1);
    const x = (i) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r);
    const y = (v) => H - pad.b - (v / maxY) * (H - pad.t - pad.b);

    const grid = [0.25, 0.5, 0.75, 1].map((f) => {
      const gy = y(maxY * f);
      return `<line x1="${pad.l}" y1="${gy}" x2="${W - pad.r}" y2="${gy}" stroke="var(--line)" stroke-width="1"/>` +
        `<text x="${pad.l - 6}" y="${gy + 4}" text-anchor="end" fill="var(--text-secondary)" font-size="10">${(maxY * f).toExponential(1)}</text>`;
    }).join("");

    // Direct end labels, nudged apart when series end at the same value.
    const labelYs = series.map((s) => (s.length && !Number.isNaN(s[s.length - 1]) ? y(s[s.length - 1]) + 4 : null));
    const order = labelYs.map((ly, k) => [ly, k]).filter(([ly]) => ly !== null).sort((a, b) => a[0] - b[0]);
    for (let j = 1; j < order.length; j++) {
      if (order[j][0] - order[j - 1][0] < 13) order[j][0] = order[j - 1][0] + 13;
    }
    // Keep the whole label chain inside the plot (flatlined series would
    // otherwise get pushed below the x axis).
    const labelLimit = H - pad.b - 2;
    for (let j = order.length - 1; j >= 0; j--) {
      const cap = j === order.length - 1 ? labelLimit : order[j + 1][0] - 13;
      if (order[j][0] > cap) order[j][0] = cap;
    }
    for (const [ly, k] of order) labelYs[k] = ly;

    const paths = series.map((s, k) => {
      let d = "";
      for (let i = 0; i < s.length; i++) {
        if (Number.isNaN(s[i])) continue;
        d += (d ? "L" : "M") + `${x(i).toFixed(1)},${y(s[i]).toFixed(1)}`;
      }
      const label = labelYs[k] !== null
        ? `<text x="${x(s.length - 1) + 5}" y="${labelYs[k].toFixed(1)}" fill="var(--text-secondary)" font-size="11">t+${horizons[k]}</text>`
        : "";
      return `<path d="${d}" fill="none" stroke="${colors[k % colors.length]}" stroke-width="2"/>` + label;
    }).join("");

    const cx = x(Math.min(tCurrent, n - 1));
    const cursor = `<line x1="${cx}" y1="${pad.t}" x2="${cx}" y2="${H - pad.b}" stroke="var(--text-secondary)" stroke-width="1" stroke-dasharray="3,3"/>`;
    const xAxis = `<line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="var(--line)"/>` +
      `<text x="${pad.l}" y="${H - 8}" fill="var(--text-secondary)" font-size="10">t = 0</text>` +
      `<text x="${W - pad.r}" y="${H - 8}" text-anchor="end" fill="var(--text-secondary)" font-size="10">${n - 1}</text>`;

    svg.innerHTML = grid + xAxis + paths + cursor;

    this._$("#legend").innerHTML = horizons.map((h, k) =>
      `<span><span class="swatch" style="background:${colors[k % colors.length]}"></span>MSE @ t+${h}</span>`).join("");

    svg.onmousemove = (ev) => {
      const rect = svg.getBoundingClientRect();
      const i = Math.round(((ev.clientX - rect.left - pad.l) / (W - pad.l - pad.r)) * (n - 1));
      if (i < 0 || i >= n) return;
      const tip = this._$("#tooltip");
      tip.style.display = "block";
      tip.style.left = `${ev.clientX - rect.left + 14}px`;
      tip.style.top = `${ev.clientY - rect.top - 10}px`;
      tip.innerHTML = `<div class="row"><b>t = ${i}</b></div>` + horizons.map((h, k) => {
        const v = series[k][i];
        return `<div class="row"><span class="swatch" style="background:${colors[k % colors.length]};display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px"></span>t+${h}: <b>${Number.isNaN(v) || v === undefined ? "–" : v.toExponential(2)}</b></div>`;
      }).join("");
    };
    svg.onmouseleave = () => { this._$("#tooltip").style.display = "none"; };
    svg.onclick = (ev) => {
      const rect = svg.getBoundingClientRect();
      const i = Math.round(((ev.clientX - rect.left - pad.l) / (W - pad.l - pad.r)) * (n - 1));
      if (i >= 0 && i <= Number(this._$("#scrub").max)) {
        this.setTime(i, true);
      }
    };
  }

  _togglePlay() {
    if (this._timer) return this._stop();
    this._$("#play").textContent = "❚❚";
    this._timer = setInterval(() => {
      const max = Number(this._$("#scrub").max);
      this.setTime(this._t >= max ? 0 : this._t + 1, true);
    }, 100);
  }

  _stop() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    const btn = this._$("#play");
    if (btn) btn.textContent = "▶";
  }
}

customElements.define("pixel-horizon-viewer", PixelHorizonViewer);
