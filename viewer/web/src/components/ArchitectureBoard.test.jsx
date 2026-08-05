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
    // The generation's leader is named by *slot*, not by architecture id:
    // a carried architecture occupies one slot per generation it survives
    // while keeping a single id throughout.
    expect(minimized.generations[0].best).toBe("g0-a0");

    const maximized = buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 1,
      selectionMetric: "goal_navigation.success_rate",
    });
    expect(maximized.mode).toBe("max");
    expect(maximized.generations[0].architectures[0].bestMetricValue).toBe(0.6);
    expect(maximized.generations[0].best).toBe("g0-a1");
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

// `architecture_search`'s own progress document: generation 0 evaluated both
// architectures and retained a0; generation 1 carries a0 into slot 0 (under
// the architecture_id it was first evaluated as) and breeds one child into
// slot 1, which has not started.
const PROGRESS = {
  format: "model-factory-architecture-campaign-progress-v1",
  campaign_id: "arch7",
  complete: false,
  generations: [
    {
      generation_index: 0,
      complete: true,
      survivor_architecture_ids: ["arch7-arch0-a0"],
      offspring_architecture_ids: ["arch7-arch1-a1"],
      results: [
        { candidate_index: 0, architecture_id: "arch7-arch0-a0", state: "completed", carried: false, retained: true, metric_value: 0.4, best_run_id: "arch7-arch0-a0-p0-c1", reason: "retained among the best 1 architecture(s)" },
        { candidate_index: 1, architecture_id: "arch7-arch0-a1", state: "no_inner_survivor", carried: false, retained: false, metric_value: null, best_run_id: null, reason: "eliminated: the inner hyperparameter campaign produced no surviving candidate" },
      ],
    },
    {
      generation_index: 1,
      complete: false,
      survivor_architecture_ids: [],
      offspring_architecture_ids: [],
      results: [
        { candidate_index: 0, architecture_id: "arch7-arch0-a0", state: "completed", carried: true, retained: false, metric_value: 0.4, best_run_id: "arch7-arch0-a0-p0-c1", reason: "carried forward from the preceding generation" },
      ],
    },
  ],
};

describe("buildArchitectureBoard with the campaign's progress document", () => {
  function board(progress) {
    return buildArchitectureBoard({
      runs: RUNS, prefix: "arch7", outerPopulationSize: 2, outerPopulationCount: 2,
      selectionMetric: SELECTION_METRIC, progress,
    });
  }

  it("names a carried slot and the earlier architecture whose evidence it carries", () => {
    const [, second] = board(PROGRESS).generations;
    expect(second.architectures[0].carried).toBe(true);
    expect(second.architectures[0].state).toBe("carried");
    // It keeps the id it was first evaluated under, and shows that
    // campaign's own inner runs rather than an empty grid.
    expect(second.architectures[0].prefix).toBe("arch7-arch0-a0");
    expect(second.architectures[0].runs.map((r) => r.run))
      .toEqual(["arch7-arch0-a0-p0-c0", "arch7-arch0-a0-p0-c1"]);
    expect(second.architectures[0].bestMetricValue).toBe(0.4);
  });

  it("still shows a not-reached slot as empty, so the two are distinguishable", () => {
    const [, second] = board(PROGRESS).generations;
    expect(second.architectures[1].carried).toBe(false);
    expect(second.architectures[1].state).toBe("not started");
    expect(second.architectures[1].runs).toEqual([]);
  });

  it("prefers the campaign's own recorded state and retention over what run states can say", () => {
    const [first] = board(PROGRESS).generations;
    expect(first.architectures[0].retained).toBe(true);
    // a1's two inner runs are one completed and one failed, so run states
    // alone would read "completed"; the campaign recorded that it produced
    // no surviving candidate at all.
    expect(first.architectures[1].state).toBe("no_inner_survivor");
    expect(first.architectures[1].bestMetricValue).toBe(null);
  });

  it("falls back to the run-id view when a campaign published no progress", () => {
    const [, second] = board(null).generations;
    expect(second.architectures[0].carried).toBe(false);
    expect(second.architectures[0].state).toBe("not started");
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

    fireEvent.click(screen.getByLabelText("expand arch7-arch0-a0 (outer generation 0)"));
    const inner = screen.getByRole("region", { name: "Population board" });
    expect(within(inner).getByText("arch7-arch0-a0-p0-c0")).toBeInTheDocument();
    // The inner grid is scoped to its own architecture: the sibling
    // architecture's candidates are not in it.
    expect(within(inner).queryByText("arch7-arch0-a1-p0-c0")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("collapse arch7-arch0-a0 (outer generation 0)"));
    expect(screen.queryByText("Population board")).not.toBeInTheDocument();
  });

  it("opens a carried slot onto the inner grid it already ran", () => {
    renderBoard({ progress: PROGRESS, outerPopulationCount: 2 });
    const carriedRow = screen.getByText("evidence carried from an earlier generation").closest("tr");
    expect(within(carriedRow).getByText("carried")).toBeInTheDocument();

    fireEvent.click(within(carriedRow).getByLabelText(/^expand /));
    const inner = screen.getByRole("region", { name: "Population board" });
    expect(within(inner).getByText("arch7-arch0-a0-p0-c1")).toBeInTheDocument();
  });

  it("explains the empty board before the campaign has written anything", () => {
    renderBoard({ runs: [] });
    expect(screen.getByText(/no campaign runs on disk yet/)).toBeInTheDocument();
    expect(screen.getByText("arch7-arch0-a0-p0-c0")).toBeInTheDocument();
  });
});
