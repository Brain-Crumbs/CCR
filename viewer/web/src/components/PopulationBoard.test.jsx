import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { PopulationBoard, buildBoard, selectionMetricMode } from "./PopulationBoard.jsx";

const METRIC = "rollout.t+4.model_over_copy_last_mse";

function run(id, state, value, parents = []) {
  return {
    run: id, mode: "fresh", state, state_reason: null, updated_at: "2026-01-01T00:00:00Z", promoted: false,
    selection_metric: METRIC, selection_metric_value: value, configuration_parents: parents,
  };
}

describe("selectionMetricMode", () => {
  it("minimizes prediction errors and maximizes goal-navigation scores", () => {
    expect(selectionMetricMode(METRIC)).toBe("min");
    expect(selectionMetricMode("direct.t+1.model_mse")).toBe("min");
    expect(selectionMetricMode("goal_navigation.success_rate")).toBe("max");
    expect(selectionMetricMode("goal_navigation.collision_rate")).toBe("min");
  });
});

describe("buildBoard", () => {
  it("lays generation 0 out fully, marking not-yet-allocated candidates with their precomputable ids", () => {
    const board = buildBoard({
      runs: [run("evo7-p0-c0", "completed", 0.8), run("evo7-p0-c1", "running", null)],
      prefix: "evo7", populationSize: 4, populationCount: 2, selectionMetric: METRIC,
    });
    const [generation] = board.generations;
    expect(board.generations).toHaveLength(2);
    expect(generation.cells.map((cell) => cell.kind)).toEqual(["trained", "trained", "pending", "pending"]);
    expect(generation.cells[2].expectedRunId).toBe("evo7-p0-c2");
    expect(generation.survivorCount).toBe(0);
    expect(generation.best).toBe("evo7-p0-c0");
  });

  it("ignores runs from other campaigns and from a prefix this one is a substring of", () => {
    const board = buildBoard({
      runs: [run("evo7-p0-c0", "completed", 0.5), run("evo8-p0-c0", "completed", 0.1), run("xevo7-p0-c1", "completed", 0.2)],
      prefix: "evo7", populationSize: 2, populationCount: 1, selectionMetric: METRIC,
    });
    expect(board.trainedCount).toBe(1);
    expect(board.generations[0].cells[1].kind).toBe("pending");
  });

  it("recovers a later generation's survivor count from its lowest offspring slot and names survivors from lineage", () => {
    const board = buildBoard({
      runs: [
        run("evo7-p0-c0", "completed", 0.9),
        run("evo7-p0-c1", "completed", 0.4),
        run("evo7-p0-c2", "completed", 0.6),
        run("evo7-p0-c3", "completed", 0.7),
        // Two survivors were carried into p1, so breeding starts at slot 2.
        run("evo7-p1-c2", "completed", 0.3, ["evo7-p0-c1", "evo7-p0-c2"]),
        run("evo7-p1-c3", "running", null, ["evo7-p0-c2", "evo7-p0-c1"]),
      ],
      prefix: "evo7", populationSize: 4, populationCount: 2, selectionMetric: METRIC,
    });
    const [, second] = board.generations;
    expect(second.survivorCount).toBe(2);
    expect(second.cells.map((cell) => cell.kind)).toEqual(["carried", "carried", "trained", "trained"]);
    // Survivors are slotted the way search.py orders them: metric ascending.
    expect(second.cells[0].run.run).toBe("evo7-p0-c1");
    expect(second.cells[1].run.run).toBe("evo7-p0-c2");
    expect(second.cells[2].parents).toEqual(["evo7-p0-c1", "evo7-p0-c2"]);
    expect(second.best).toBe("evo7-p1-c2");
  });

  it("leaves a survivor slot unnamed when lineage does not account for every carried candidate", () => {
    const board = buildBoard({
      runs: [
        run("evo7-p0-c0", "completed", 0.9),
        run("evo7-p0-c1", "completed", 0.4),
        // Only one distinct parent recorded, but two survivor slots exist.
        run("evo7-p1-c2", "completed", 0.3, ["evo7-p0-c1", "evo7-p0-c1"]),
      ],
      prefix: "evo7", populationSize: 4, populationCount: 2, selectionMetric: METRIC,
    });
    const [, second] = board.generations;
    expect(second.survivorCount).toBe(2);
    expect(second.cells[0].kind).toBe("carried");
    expect(second.cells[0].run).toBeNull();
  });

  it("renders a generation the campaign wrote beyond its configured count", () => {
    const board = buildBoard({
      runs: [run("evo7-p0-c0", "completed", 0.5), run("evo7-p2-c1", "completed", 0.2, ["evo7-p0-c0", "evo7-p0-c0"])],
      prefix: "evo7", populationSize: 2, populationCount: 1, selectionMetric: METRIC,
    });
    expect(board.generations).toHaveLength(3);
  });

  it("breaks a metric tie on run id in codepoint order, the way search.py does", () => {
    const board = buildBoard({
      runs: [
        run("evo7-p0-c0", "completed", 0.5),
        run("evo7-p0-c1", "completed", 0.5),
        run("evo7-p0-c2", "completed", 0.5),
        run("evo7-p0-c3", "completed", 0.5),
        run("evo7-p1-c2", "completed", 0.4, ["evo7-p0-c0", "evo7-p0-c1"]),
      ],
      prefix: "evo7", populationSize: 4, populationCount: 2, selectionMetric: METRIC,
    });
    expect(board.generations[0].best).toBe("evo7-p0-c0");
    const [, second] = board.generations;
    expect(second.cells[0].run.run).toBe("evo7-p0-c0");
    expect(second.cells[1].run.run).toBe("evo7-p0-c1");
  });

  it("sorts an unevaluated candidate last rather than treating a missing metric as best", () => {
    const board = buildBoard({
      runs: [run("evo7-p0-c0", "running", null), run("evo7-p0-c1", "completed", 0.9)],
      prefix: "evo7", populationSize: 2, populationCount: 1, selectionMetric: METRIC,
    });
    expect(board.generations[0].best).toBe("evo7-p0-c1");
  });

  it("ranks a maximized selection metric the other way round", () => {
    const board = buildBoard({
      runs: [run("evo7-p0-c0", "completed", 0.2), run("evo7-p0-c1", "completed", 0.8)],
      prefix: "evo7", populationSize: 2, populationCount: 1, selectionMetric: "goal_navigation.success_rate",
    });
    expect(board.mode).toBe("max");
    expect(board.generations[0].best).toBe("evo7-p0-c1");
  });
});

describe("PopulationBoard", () => {
  afterEach(cleanup);

  it("renders the generation x candidate grid from supplied runs without polling", () => {
    render(
      <PopulationBoard
        organism="Crafter" prefix="evo7" populationSize={2} populationCount={2} selectionMetric={METRIC}
        runs={[
          run("evo7-p0-c0", "completed", 0.42),
          run("evo7-p0-c1", "failed", null),
          run("evo7-p1-c1", "running", null, ["evo7-p0-c0", "evo7-p0-c0"]),
        ]}
      />,
    );
    // Twice: once as p0's own trained cell, once as the survivor carried
    // into p1 under that same original run id.
    expect(screen.getAllByText("evo7-p0-c0")).toHaveLength(2);
    expect(screen.getAllByText("4.20e-1")).toHaveLength(2);
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("carried")).toBeInTheDocument();
    expect(screen.getByText("↑ evo7-p0-c0 × evo7-p0-c0")).toBeInTheDocument();
    expect(screen.getByText(/lower is better/)).toBeInTheDocument();
  });

  it("says nothing is on disk yet rather than showing an empty grid", () => {
    render(
      <PopulationBoard
        organism="Crafter" prefix="evo7" populationSize={2} populationCount={1} selectionMetric={METRIC} runs={[]}
      />,
    );
    expect(screen.getByText(/no campaign runs on disk yet/)).toBeInTheDocument();
  });
});
