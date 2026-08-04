/** Frame/prediction lookups shared by the horizon strips and the
 * MSE-over-time chart -- ported 1:1 from the original
 * &lt;pixel-horizon-viewer&gt; web component's instance methods, but as pure
 * functions over an explicit `ctx` bundle instead of `this`, so a React
 * component (or a test) can call them without mounting anything. */
import { b64ToBytes, mse } from "./pixelMath.js";

/** Decode a `/frames` API payload into {shape, dtype, nFrames, frames}. */
export function decodeFrames(payload) {
  const frames = (payload.frames || []).map((f) => ({ ...f, bytes: f.data ? b64ToBytes(f.data) : null }));
  return { shape: payload.shape, dtype: payload.dtype, nFrames: payload.n_frames ?? frames.length, frames };
}

/** Decode a `/predictions` API payload (pixel-predictions-v1 or v2) into its
 * raw fields plus decoded byte arrays, or null when absent/unsupported. */
export function decodePredictions(payload) {
  if (!payload || payload.error) return null;
  if (payload.format !== "pixel-predictions-v1" && payload.format !== "pixel-predictions-v2") return null;
  const decoded = {};
  for (const [h, entry] of Object.entries(payload.predictions || {})) decoded[h] = entry.frames.map(b64ToBytes);
  return { ...payload, decoded, decodedTargets: payload.targets ? payload.targets.map(b64ToBytes) : null };
}

export function computeMean(frames) {
  const valid = frames.filter((f) => f.bytes);
  if (!valid.length) return null;
  const n = valid[0].bytes.length;
  const acc = new Float64Array(n);
  for (const f of valid) for (let i = 0; i < n; i++) acc[i] += f.bytes[i];
  const meanFrame = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) meanFrame[i] = Math.round(acc[i] / valid.length);
  return meanFrame;
}

/** Bytes for the last recorded frame at or before index i (frames can be
 * null when elided from the recording). */
export function actualBytes(frames, i) {
  for (let j = i; j >= 0; j--) if (frames[j]?.bytes) return frames[j].bytes;
  return null;
}

/** "model" reads `ctx.pred` (this viewer's own run); "reference" reads
 * `ctx.referencePred` -- a second, independently picked run's predictions
 * for the same session/episode -- the same way, so any run in the catalog
 * can stand in as a configurable comparison baseline alongside the
 * client-computed copy-last/mean-frame ones. */
const PRED_FIELD_BY_SOURCE = { model: "pred", reference: "referencePred" };

/** {bytes, shape} predicted for target index t+h from source at t, or null. */
export function predictedFrame(source, t, h, { frames, meanFrame, shape, ...ctx }) {
  const predField = PRED_FIELD_BY_SOURCE[source];
  if (predField) {
    const pred = ctx[predField];
    const seq = pred?.decoded?.[String(h)];
    if (!seq || t >= seq.length) return null;
    return { bytes: seq[t], shape: pred.prediction_shape };
  }
  if (source === "mean-frame") return meanFrame ? { bytes: meanFrame, shape } : null;
  const bytes = actualBytes(frames, t); // copy-last
  return bytes ? { bytes, shape } : null;
}

/** Actual frame in the prediction's space (model/reference targets are pooled by the exporter). */
export function targetFrame(source, i, { frames, shape, ...ctx }) {
  const predField = PRED_FIELD_BY_SOURCE[source];
  const pred = predField ? ctx[predField] : null;
  if (pred?.decodedTargets) return { bytes: pred.decodedTargets[i], shape: pred.prediction_shape };
  const bytes = actualBytes(frames, i);
  return bytes ? { bytes, shape } : null;
}

/** MSE at every start frame t for one (source, horizon) pair. */
export function mseSeries(source, h, ctx) {
  const n = ctx.frames.length - h;
  const out = new Float64Array(Math.max(0, n)).fill(NaN);
  for (let t = 0; t < n; t++) {
    const predicted = predictedFrame(source, t, h, ctx);
    const target = targetFrame(source, t + h, ctx);
    if (predicted && target && predicted.bytes.length === target.bytes.length) out[t] = mse(predicted.bytes, target.bytes);
  }
  return out;
}

/** Index of the frame the "surprise"/"false_positive"/"static" event-nav
 * buttons should jump to: the worst (or, for `static`, the worst among
 * frames where the agent didn't move) model MSE at the first horizon. */
export function worstFrameIndex(kind, series, events = []) {
  const candidates = [...series.keys()].filter((i) => kind !== "static" || !events[i]?.position_changed);
  return candidates.reduce((best, i) => (Number.isNaN(series[i]) || (best >= 0 && series[i] <= series[best]) ? best : i), -1);
}
