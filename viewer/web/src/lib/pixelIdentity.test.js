import { describe, expect, it } from "vitest";
import { identityText, modelSourceLabel, statusText } from "./pixelIdentity.js";

describe("modelSourceLabel", () => {
  it("labels a live-record export distinctly", () => {
    expect(modelSourceLabel({ source: "live-record" })).toBe("model (live record)");
  });
  it("labels a legacy v1 export as actionless", () => {
    expect(modelSourceLabel({ format: "pixel-predictions-v1" })).toMatch(/Legacy/);
  });
  it("names the prediction mode for a current joint export", () => {
    expect(modelSourceLabel({ format: "pixel-predictions-v2", prediction_mode: "rollout" })).toBe("joint cortex (rollout)");
  });
  it("is null with no prediction payload", () => {
    expect(modelSourceLabel(null)).toBeNull();
  });
});

describe("identityText", () => {
  it("names the experiment, checkpoint, and model for a v2 export", () => {
    const text = identityText({
      format: "pixel-predictions-v2", prediction_mode: "rollout", evaluation_source: "s/nursery-x-holdout-3/episode_0",
      experiment: { experiment_id: "exp-1", trace_id: "trace-1" },
      model: { checkpoint_sha256: "abcdef123456789", model_type: "predictive_cortex", backbone: "transformer", training_objective: "windowed_rollout", uses_actions: true, uses_workspace: false },
    });
    expect(text).toContain("exp-1");
    expect(text).toContain("abcdef123456");
    expect(text).toContain("Split holdout");
    expect(text).toContain("actions on, workspace off");
  });

  it("flags a legacy export as not comparable", () => {
    expect(identityText({ format: "pixel-predictions-v1" })).toMatch(/not comparable/);
  });
});

describe("statusText", () => {
  it("reports frame count, shape, and prediction availability", () => {
    expect(statusText({ nFrames: 12, shape: [33, 33, 3], pred: null })).toBe("12 frames (33×33×3 · no model predictions recorded — showing baselines)");
  });
});
