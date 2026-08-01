import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ChampionRegistryPanel } from "./ChampionRegistryPanel.jsx";

afterEach(cleanup);

const registry = {
  format: "model-factory-registry-v1",
  slots: { generic_action_effects_v1: { fast: { "rollout.t+4": {
    leading_champion: "run-1",
    population: [
      { run_id: "run-1", checkpoint_path: "checkpoints/best-validation.pt", checkpoint_sha256: "abcdef123456789", metrics: { "rollout.t+4.model_mse": 0.01, "rollout.t+1.model_mse": 0.002 }, promoted_at: "2026-01-01T00:00:00Z", test_uses: 1 },
      { run_id: "run-2", checkpoint_path: "checkpoints/last.pt", checkpoint_sha256: "0987654321fedcba", metrics: { "rollout.t+4.model_mse": 0.02 }, promoted_at: "2026-01-02T00:00:00Z", test_uses: 0 },
    ],
    history: [{ action: "promote", run_id: "run-1", at: "2026-01-01T00:00:00Z", reason: "beats champion" }],
  } } } },
};

describe("ChampionRegistryPanel", () => {
  it("renders a population table with per-horizon metric columns for cross-model comparison", () => {
    render(<ChampionRegistryPanel registry={registry} />);
    expect(screen.getByText("generic_action_effects_v1 · fast · rollout.t+4")).toBeInTheDocument();
    expect(screen.getByText("rollout.t+4.model_mse")).toBeInTheDocument();
    expect(screen.getByText("rollout.t+1.model_mse")).toBeInTheDocument();
    expect(screen.getByText("1.000e-2")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /run-1/ })).toHaveClass("is-current");
    expect(screen.getByText("history (1)")).toBeInTheDocument();
  });

  it("shows a no-data state with an empty registry", () => {
    render(<ChampionRegistryPanel registry={{ format: "model-factory-registry-v1", slots: {} }} />);
    expect(screen.getByText(/no champions promoted yet/)).toBeInTheDocument();
  });
});
