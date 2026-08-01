import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { FactoryRunsPanel } from "./FactoryRunsPanel.jsx";

afterEach(cleanup);

const factoryRuns = {
  organism: "Test",
  runs: [
    { run: "run-b", mode: "clone", state: "running", state_reason: null, updated_at: "2026-01-03T00:00:00Z", promoted: null },
    { run: "run-a", mode: "fresh", state: "completed", state_reason: "completed", updated_at: "2026-01-02T00:00:00Z", promoted: true },
    { run: "run-c", mode: "fine_tune", state: "failed", state_reason: "ValueError: bad", updated_at: "2026-01-01T00:00:00Z", promoted: false },
  ],
};

describe("FactoryRunsPanel", () => {
  it("renders one row per run with a state badge, mode, and last-updated time", () => {
    render(<FactoryRunsPanel factoryRuns={factoryRuns} />);
    expect(screen.getByText("run-b")).toBeInTheDocument();
    expect(screen.getByText("clone")).toBeInTheDocument();
    expect(screen.getByText("running")).toHaveClass("state-badge--info");
    expect(screen.getByText("completed")).toHaveClass("state-badge--green");
    expect(screen.getByText("failed")).toHaveClass("state-badge--red");
    expect(screen.getByText(/ValueError: bad/)).toBeInTheDocument();
  });

  it("shows a no-data state with no runs", () => {
    render(<FactoryRunsPanel factoryRuns={{ organism: "Test", runs: [] }} />);
    expect(screen.getByText(/no factory runs found/)).toBeInTheDocument();
  });
});
