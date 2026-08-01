import { describe, expect, it } from "vitest";
import { developmentStages, episodeDiagnostics } from "./diagnostics.js";

describe("episodeDiagnostics", () => {
  it("collects neuromodulator series, prediction error, and mode timeline", () => {
    const records = [
      { stream_id: "internal.dopamine", payload: { value: 0.2 }, seq: 1 },
      { stream_id: "internal.acetylcholine", payload: { value: 0.3 }, seq: 1 },
      { stream_id: "internal.adrenaline", payload: { value: 0.4 }, seq: 1 },
      { stream_id: "internal.prediction_error", payload: { value: 0.5 }, seq: 1 },
      { stream_id: "internal.arbiter.mode", payload: { mode: "afraid" }, seq: 1 },
    ];
    const model = episodeDiagnostics(records);
    expect(model.series.dopamine).toEqual([{ tick: 1, value: 0.2 }]);
    expect(model.series.acetylcholine).toEqual([{ tick: 1, value: 0.3 }]);
    expect(model.series.adrenaline).toEqual([{ tick: 1, value: 0.4 }]);
    expect(model.series.prediction_error).toEqual([{ tick: 1, value: 0.5 }]);
    expect(model.modes).toEqual([{ tick: 1, mode: "afraid" }]);
  });

  it("prefers DecisionRecord arbiter mode over a legacy stream echo for the same tick", () => {
    const streams = [{ stream_id: "internal.arbiter.mode", payload: { mode: "curious" }, seq: 0 }];
    const decisions = [{ tick_index: 0, arbiter_mode: { mode: "curious" } }];
    const model = episodeDiagnostics(streams, decisions);
    expect(model.modes).toEqual([{ tick: 0, mode: "curious" }]);
  });

  it("prefers DecisionRecord attention over a stream payload echo", () => {
    const streams = [{ stream_id: "internal.attention.weights", seq: 4, payload: {
      focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels", "body.health"],
    } }];
    const decisions = [{ tick_index: 4, attention: {
      focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels", "body.health"],
      reasons: { "vision.frame.pixels": { components: { novelty: 0.75, boredom: -0.1 } } },
    } }];
    const model = episodeDiagnostics(streams, decisions);
    expect(model.attention).toEqual([{
      tick: 4, focus: "vision.frame.pixels", selected: ["vision.frame.pixels", "body.health"],
      reasons: { "vision.frame.pixels": { components: { novelty: 0.75, boredom: -0.1 } } },
    }]);
  });
});

describe("developmentStages", () => {
  it("normalizes string and object stage entries with pass/pending state", () => {
    const stages = developmentStages({ development: { stages: [
      { name: "Gestation", passed: true, milestones: ["sensory baseline"] },
      { name: "Crawling", passed: false },
      "Walking",
    ] } });
    expect(stages).toEqual([
      { name: "Gestation", passed: true, milestones: ["sensory baseline"] },
      { name: "Crawling", passed: false, milestones: [] },
      { name: "Walking", passed: true, milestones: [] },
    ]);
  });

  it("returns an empty list when no ladder was recorded", () => {
    expect(developmentStages({})).toEqual([]);
  });
});
