import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { PairedComparisonPanel } from "./PairedComparisonPanel.jsx";

afterEach(cleanup);

describe("PairedComparisonPanel", () => {
  it("renders mean delta, win rate, and confidence intervals for an evaluable comparison", () => {
    render(<PairedComparisonPanel comparison={{
      format: "model-factory-paired-comparison-v1", status: "evaluable", primary_metric: "rollout.t+4.mse",
      episode_count: 8, mean_delta: -0.01, win_rate: 0.75, confidence: 0.95,
      intervals: { paired_bootstrap: { low: -0.02, high: -0.001 }, paired_permutation: { low: -0.019, high: -0.002 } },
    }} />);
    expect(screen.getByText("evaluable")).toBeInTheDocument();
    expect(screen.getByText(/candidate improves on baseline/)).toBeInTheDocument();
    expect(screen.getByText("75.0%")).toBeInTheDocument();
    expect(screen.getByText("Paired bootstrap CI")).toBeInTheDocument();
  });

  it("explains why a comparison could not be evaluated", () => {
    render(<PairedComparisonPanel comparison={{ status: "not_evaluable", reason: "no matching episodes", episode_count: 0, primary_metric: "rollout.t+4.mse" }} />);
    expect(screen.getByText("no matching episodes")).toBeInTheDocument();
  });

  it("shows a no-data state when the run has no comparison (e.g. a fresh trial)", () => {
    render(<PairedComparisonPanel comparison={null} />);
    expect(screen.getByText(/no candidate\/baseline comparison recorded/)).toBeInTheDocument();
  });
});
