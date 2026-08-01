import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { RunSummary } from "./RunSummary.jsx";

afterEach(cleanup);

describe("RunSummary", () => {
  it("renders training/evaluation stats and a promoted badge", () => {
    render(<RunSummary summary={{
      model: { type: "predictive_cortex", checkpoint_sha256: "abcdef123456789" },
      training: { objective: "windowed_rollout", epochs: 12, episodes: 3, samples: 48, final_total_loss: 0.2 },
      evaluation: { horizon: 4, model_mse: 0.01, model_over_copy_last_mse: 0.5, rollout_health: "healthy", split: "held_out", prediction_mode: "rollout" },
      promotion: { promoted: true, reasons: [] },
    }} />);
    expect(screen.getByText("Promoted")).toHaveClass("run-summary__status--good");
    expect(screen.getByText("predictive_cortex")).toBeInTheDocument();
    expect(screen.getByText(/checkpoint abcdef123456/)).toBeInTheDocument();
    expect(screen.getByText("t+4 MSE")).toBeInTheDocument();
    expect(screen.getByText(/below copy-last/)).toBeInTheDocument();
  });

  it("shows an unavailable state with no summary", () => {
    render(<RunSummary summary={null} />);
    expect(screen.getByText("Report unavailable")).toBeInTheDocument();
    expect(screen.getByText("No summary recorded")).toBeInTheDocument();
  });
});
