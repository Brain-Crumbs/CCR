/** Pixel/error math shared by the horizon strips and the MSE-over-time
 * chart. Pure functions only -- no DOM, no canvas -- so they're testable
 * without a browser and reusable between the two. */

/* Chart palette (validated for light and dark surfaces; see dataviz notes in
 * the repo). Series follow the horizon entity, in fixed slot order. */
export const SERIES_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"];
export const SERIES_DARK = ["#3987e5", "#199e70", "#c98500", "#9085e9"];
/* Sequential blue ramp for the |error| heatmap (magnitude = one hue). */
export const BLUE_RAMP = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];

export function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function hexRGB(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

/** v in [0,1] -> one-hue sequential blue anchored at the surface, so zero
 * error recedes into the background in both modes and high error pops. */
export function rampColor(v, dark = false) {
  const surface = hexRGB(dark ? "#242423" : "#fcfcfb");
  const ramp = dark ? [...BLUE_RAMP].reverse() : BLUE_RAMP;
  const pos = Math.max(0, Math.min(1, v)) * ramp.length; // 0..len, 0 = pure surface
  const idx = Math.min(ramp.length - 1, Math.floor(pos));
  const lo = idx === 0 ? surface : hexRGB(ramp[idx - 1]);
  const hi = hexRGB(ramp[idx]);
  const f = pos - idx;
  return [0, 1, 2].map((i) => Math.round(lo[i] + (hi[i] - lo[i]) * f));
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] - b[i]) / 255;
    s += d * d;
  }
  return s / a.length;
}

export function psnrText(m) {
  if (m <= 0) return "∞";
  return (10 * Math.log10(1 / m)).toFixed(1);
}
