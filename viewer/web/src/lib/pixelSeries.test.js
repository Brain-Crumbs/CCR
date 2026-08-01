import { describe, expect, it } from "vitest";
import {
  actualBytes, computeMean, decodeFrames, decodePredictions,
  mseSeries, predictedFrame, targetFrame, worstFrameIndex,
} from "./pixelSeries.js";

function b64(bytes) { return btoa(String.fromCharCode(...bytes)); }

describe("decodeFrames", () => {
  it("decodes base64 frame data and passes through elided (null-data) frames", () => {
    const payload = {
      shape: [1, 1, 3], dtype: "uint8", n_frames: 2,
      frames: [{ i: 0, t: 0, seq: 0, hash: "a", data: b64([1, 2, 3]) }, { i: 1, t: 1, seq: 1, hash: "a", data: null }],
    };
    const decoded = decodeFrames(payload);
    expect(decoded.nFrames).toBe(2);
    expect(Array.from(decoded.frames[0].bytes)).toEqual([1, 2, 3]);
    expect(decoded.frames[1].bytes).toBeNull();
  });
});

describe("decodePredictions", () => {
  it("rejects unsupported or error payloads", () => {
    expect(decodePredictions(null)).toBeNull();
    expect(decodePredictions({ error: "no predictions" })).toBeNull();
    expect(decodePredictions({ format: "unknown" })).toBeNull();
  });

  it("decodes prediction and target frames per horizon", () => {
    const payload = {
      format: "pixel-predictions-v2", horizons: [1], prediction_shape: [1, 1, 3],
      predictions: { 1: { frames: [b64([9, 9, 9])] } }, targets: [b64([0, 0, 0])],
    };
    const decoded = decodePredictions(payload);
    expect(Array.from(decoded.decoded["1"][0])).toEqual([9, 9, 9]);
    expect(Array.from(decoded.decodedTargets[0])).toEqual([0, 0, 0]);
  });
});

describe("actualBytes / computeMean", () => {
  const frames = [{ bytes: new Uint8Array([0, 0, 0]) }, { bytes: null }, { bytes: new Uint8Array([255, 255, 255]) }];

  it("walks backward past elided frames", () => {
    expect(Array.from(actualBytes(frames, 1))).toEqual([0, 0, 0]);
    expect(Array.from(actualBytes(frames, 2))).toEqual([255, 255, 255]);
  });

  it("averages only frames with recorded bytes", () => {
    expect(Array.from(computeMean(frames))).toEqual([128, 128, 128]);
  });

  it("is null with no recorded bytes", () => {
    expect(computeMean([{ bytes: null }])).toBeNull();
  });
});

describe("predictedFrame / targetFrame / mseSeries", () => {
  const frames = [0, 1, 2].map((i) => ({ bytes: new Uint8Array([i * 10, i * 10, i * 10]) }));
  const shape = [1, 1, 3];
  const meanFrame = new Uint8ClampedArray([10, 10, 10]);
  const pred = { decoded: { 1: [new Uint8Array([5, 5, 5]), new Uint8Array([15, 15, 15])] }, decodedTargets: null, prediction_shape: shape };
  const ctx = { frames, pred, meanFrame, shape };

  it("copy-last predicts the source frame itself", () => {
    expect(Array.from(predictedFrame("copy-last", 0, 1, ctx).bytes)).toEqual([0, 0, 0]);
  });

  it("mean-frame predicts the episode mean regardless of t/h", () => {
    expect(Array.from(predictedFrame("mean-frame", 1, 1, ctx).bytes)).toEqual([10, 10, 10]);
  });

  it("model predicts the decoded frame at the requested horizon", () => {
    expect(Array.from(predictedFrame("model", 1, 1, ctx).bytes)).toEqual([15, 15, 15]);
  });

  it("model prediction is null past the end of the decoded sequence", () => {
    expect(predictedFrame("model", 5, 1, ctx)).toBeNull();
  });

  it("target falls back to the actual frame when no decoded target exists", () => {
    expect(Array.from(targetFrame("model", 2, ctx).bytes)).toEqual([20, 20, 20]);
  });

  it("computes MSE across every valid start frame for a source/horizon pair", () => {
    const series = mseSeries("copy-last", 1, ctx);
    expect(series.length).toBe(2); // n_frames(3) - horizon(1)
    expect(series[0]).toBeCloseTo(mseSeries("copy-last", 1, ctx)[0], 10);
    expect(Number.isNaN(series[0])).toBe(false);
  });
});

describe("worstFrameIndex", () => {
  it("picks the frame with the highest error", () => {
    const series = [0.1, 0.9, 0.2];
    expect(worstFrameIndex("surprise", series)).toBe(1);
  });

  it("excludes frames where the agent moved when hunting a static mismatch", () => {
    const series = [0.1, 0.9, 0.5];
    const events = [{ position_changed: false }, { position_changed: true }, { position_changed: false }];
    expect(worstFrameIndex("static", series, events)).toBe(2);
  });
});
