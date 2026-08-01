import { describe, expect, it } from "vitest";
import { actionTimeline } from "./actions.js";

describe("actionTimeline", () => {
  it("reports the voluntary/actuated pair and marks divergence", () => {
    const decisions = [
      { tick_index: 0, motor_decision: { voluntary: "move_up", reflex: null, caregiver_override: null, actuated: "move_up" } },
      { tick_index: 1, motor_decision: {
        voluntary: "move_up",
        reflex: { name: "wall_avoidance", action: "move_left", reason: "collision imminent", priority: 5 },
        caregiver_override: null, actuated: "move_left",
      } },
    ];
    const timeline = actionTimeline(decisions);
    expect(timeline[0]).toMatchObject({ tick: 0, voluntary: "move_up", actuated: "move_up", diverged: false });
    expect(timeline[1]).toMatchObject({ tick: 1, voluntary: "move_up", actuated: "move_left", diverged: true });
    expect(timeline[1].motorReflex.name).toBe("wall_avoidance");
  });

  it("surfaces a caregiver override distinctly from a reflex override", () => {
    const decisions = [{ tick_index: 2, motor_decision: {
      voluntary: "move_down", reflex: null,
      caregiver_override: { action: "move_right", reason: "caregiver" }, actuated: "move_right",
    } }];
    const [entry] = actionTimeline(decisions);
    expect(entry.caregiverOverride).toEqual({ action: "move_right", reason: "caregiver" });
    expect(entry.motorReflex).toBeNull();
    expect(entry.diverged).toBe(true);
  });

  it("falls back to voluntary when no motor decision was recorded", () => {
    const [entry] = actionTimeline([{ tick_index: 0 }]);
    expect(entry).toMatchObject({ voluntary: null, actuated: null, diverged: false });
  });
});
