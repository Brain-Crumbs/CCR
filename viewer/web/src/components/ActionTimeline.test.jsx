import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ActionTimeline } from "./ActionTimeline.jsx";

afterEach(cleanup);

const decisions = [
  { tick_index: 0, motor_decision: { voluntary: "move_up", reflex: null, caregiver_override: null, actuated: "move_up" } },
  { tick_index: 1, motor_decision: {
    voluntary: "move_up", reflex: { name: "wall_avoidance", action: "move_left", reason: "collision imminent", priority: 5 },
    caregiver_override: null, actuated: "move_left",
  } },
  { tick_index: 2, motor_decision: { voluntary: "move_down", reflex: null, caregiver_override: { action: "move_right", reason: "caregiver" }, actuated: "move_right" } },
];

describe("ActionTimeline", () => {
  it("renders the actuated action for every tick and flags divergence with the assumed action", () => {
    render(<ActionTimeline decisions={decisions} />);
    expect(screen.getByText("move up")).toBeInTheDocument();
    expect(screen.getByText("move left")).toBeInTheDocument();
    expect(screen.getByText(/assumed move up/)).toBeInTheDocument();
    expect(screen.getByText("wall_avoidance")).toBeInTheDocument();
    expect(screen.getByText("caregiver")).toBeInTheDocument();
  });

  it("marks the current tick and reports a click back to the shared cursor", () => {
    const onTickChange = vi.fn();
    render(<ActionTimeline decisions={decisions} tick={1} onTickChange={onTickChange} />);
    const items = screen.getAllByRole("listitem");
    expect(items[1]).toHaveClass("is-current");
    fireEvent.click(screen.getByText("t2"));
    expect(onTickChange).toHaveBeenCalledWith(2);
  });

  it("renders an empty state when no motor decisions were recorded", () => {
    render(<ActionTimeline decisions={[]} />);
    expect(screen.getByText(/no motor decisions recorded/)).toBeInTheDocument();
  });
});
