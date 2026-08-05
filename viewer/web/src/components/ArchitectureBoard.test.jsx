import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { ArchitectureBoard, architectureState, buildArchitectureBoard } from "./ArchitectureBoard.jsx";

const SELECTION_METRIC = "rollout.t+4.model_over_copy_last_mse";

function run(id, state, value) {
  return { run: id, state, selection_metric: SELECTION_METRIC, selection_metric_value: value, configuration_parents: [] };
}

// One outer generation of two architectures, each running a 2 x 1 inner
// campaign; a0 is the better architecture (lower is better on this metric).
const RUNS = [
  run("arch7-arch0-a0-p0-c0", "completed", 0.6),
  run("arch7-arch0-a0-p0-c1", "completed", 0.4),
  run("arch7-arch0-a1-p0-c0", "completed", 0.9),
  run("arch7-arch0-a1-p0-c1", "failed", null),
  // A neighbouring campaign's runs, and an unrelated hand-launched run:
  // neither belongs to this board and neither may leak into it.
  run("arch9-arch0-a0-p0-c0", "completed", 0.1),
  run("baseline-2026-01-01", "completed", 0.2),
];

describe("buildArchitectureBoard", () => {
  it("groups runs by outer generation and architecture from run ids alone", () => {
    const board = buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 1,
      selectionMetric: SELECTION_METRIC,
    });
    expect(board.trainedCount).toBe(4);
    expect(board.generations).toHaveLength(1);
    const [a0, a1] = board.generations[0].architectures;
    expect(a0.prefix).toBe("arch7-arch0-a0");
    expect(a0.runs.map((r) => r.run)).toEqual(["arch7-arch0-a0-p0-c0", "arch7-arch0-a0-p0-c1"]);
    expect(a1.runs).toHaveLength(2);
  });

  it("scores each architecture by its inner campaign's best value, in the metric's own direction", () => {
    const minimized = buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 1,
      selectionMetric: SELECTION_METRIC,
    });
    expect(minimized.mode).toBe("min");
    expect(minimized.generations[0].architectures[0].bestMetricValue).toBe(0.4);
    expect(minimized.generations[0].architectures[0].bestRunId).toBe("arch7-arch0-a0-p0-c1");
    expect(minimized.generations[0].best).toBe("arch7-arch0-a0");

    const maximized = buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 1,
      selectionMetric: "goal_navigation.success_rate",
    });
    expect(maximized.mode).toBe("max");
    expect(maximized.generations[0].architectures[0].bestMetricValue).toBe(0.6);
    expect(maximized.generations[0].best).toBe("arch7-arch0-a1");
  });

  it("renders configured generations that have nothing on disk yet, and any generation beyond them", () => {
    const configured = buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 3, outerPopulationCount: 2,
      selectionMetric: SELECTION_METRIC,
    });
    expect(configured.generations).toHaveLength(2);
    expect(configured.generations[0].architectures).toHaveLength(3);
    expect(configured.generations[0].architectures[2].state).toBe("not started");
    expect(configured.generations[1].architectures.every((a) => !a.runs.length)).toBe(true);

    const overrun = buildArchitectureBoard({
      runs: [...RUNS, run("arch7-arch1-a0-p0-c0", "running", null)],
      prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 1,
      selectionMetric: SELECTION_METRIC,
    });
    expect(overrun.generations).toHaveLength(2);
    expect(overrun.generations[1].architectures[0].state).toBe("running");
  });
});

describe("architectureState", () => {
  it("aggregates the inner campaign rather than claiming a state an architecture does not have", () => {
    expect(architectureState([])).toBe("not started");
    expect(architectureState([run("a", "completed", 1), run("b", "running", null)])).toBe("running");
    // One failed candidate out of several is an ordinary campaign, not a
    // failed architecture -- only an all-failed inner campaign is.
    expect(architectureState([run("a", "completed", 1), run("b", "failed", null)])).toBe("completed");
    expect(architectureState([run("a", "failed", null), run("b", "failed", null)])).toBe("failed");
  });
});

describe("ArchitectureBoard", () => {
  afterEach(cleanup);

  function renderBoard(props = {}) {
    return render(
      <ArchitectureBoard
        organism="Crafter" prefix="arch7" runs={RUNS}
        outerPopulationSize={2} outerPopulationCount={1}
        innerPopulationSize={2} innerPopulationCount={1}
        selectionMetric={SELECTION_METRIC}
        {...props}
      />,
    );
  }

  it("lists one row per architecture with its aggregate state and best inner metric", () => {
    renderBoard();
    expect(screen.getByText("outer generation 0")).toBeInTheDocument();
    expect(screen.getByText("arch7-arch0-a0")).toBeInTheDocument();
    expect(screen.getByText("arch7-arch0-a1")).toBeInTheDocument();
    expect(screen.getByText("arch7-arch0-a0-p0-c1")).toBeInTheDocument();
    // Nothing from the neighbouring arch9 campaign appears on this board.
    expect(screen.queryByText(/arch9/)).not.toBeInTheDocument();
  });

  it("expands one architecture into its own inner population grid, collapsed by default", () => {
    renderBoard();
    expect(screen.queryByText("Population board")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("expand arch7-arch0-a0"));
    const inner = screen.getByRole("region", { name: "Population board" });
    expect(within(inner).getByText("arch7-arch0-a0-p0-c0")).toBeInTheDocument();
    // The inner grid is scoped to its own architecture: the sibling
    // architecture's candidates are not in it.
    expect(within(inner).queryByText("arch7-arch0-a1-p0-c0")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("collapse arch7-arch0-a0"));
    expect(screen.queryByText("Population board")).not.toBeInTheDocument();
  });

  it("explains the empty board before the campaign has written anything", () => {
    renderBoard({ runs: [] });
    expect(screen.getByText(/no campaign runs on disk yet/)).toBeInTheDocument();
    expect(screen.getByText("arch7-arch0-a0-p0-c0")).toBeInTheDocument();
  });
});
