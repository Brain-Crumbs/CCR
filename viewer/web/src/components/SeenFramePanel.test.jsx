import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { SeenFramePanel } from "./SeenFramePanel.jsx";

afterEach(cleanup);

describe("SeenFramePanel", () => {
  it("pairs the seen pixel frame with the action actually taken", () => {
    render(<SeenFramePanel t={2} tick={2} bytes={new Uint8Array([1, 2, 3])} shape={[1, 1, 3]}
      action={{ tick: 2, voluntary: "move_up", actuated: "move_up", diverged: false, motorReflex: null, caregiverOverride: null }} />);
    expect(screen.getByText("t = 2")).toBeInTheDocument();
    expect(screen.getByText("tick 2")).toBeInTheDocument();
    expect(screen.getByText("move up")).toBeInTheDocument();
  });

  it("flags a divergent tick with the assumed action and its override reason", () => {
    render(<SeenFramePanel t={3} tick={3} bytes={null} shape={[1, 1, 3]}
      action={{ tick: 3, voluntary: "move_up", actuated: "move_left", diverged: true, motorReflex: { name: "wall_avoidance" }, caregiverOverride: null }} />);
    expect(screen.getByText("move left")).toBeInTheDocument();
    expect(screen.getByText(/assumed/)).toHaveTextContent("assumed move up — wall avoidance");
  });

  it("shows a no-data state when no motor decision was recorded for the tick", () => {
    render(<SeenFramePanel t={0} tick={0} bytes={null} shape={[1, 1, 3]} action={null} />);
    expect(screen.getByText(/no motor decision recorded/)).toBeInTheDocument();
  });
});
