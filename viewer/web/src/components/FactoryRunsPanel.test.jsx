import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { FactoryRunsPanel } from "./FactoryRunsPanel.jsx";

afterEach(cleanup);

const runs = [
  { run: "run-1", mode: "fresh", state: "completed", state_reason: "completed", updated_at: "2026-01-01T00:00:00Z", promoted: true },
  { run: "run-2", mode: "clone", state: "running", state_reason: null, updated_at: "2026-01-02T00:00:00Z", promoted: null },
  { run: "run-3", mode: "clone", state: "failed", state_reason: "RuntimeError: boom", updated_at: "2026-01-03T00:00:00Z", promoted: null },
];

describe("FactoryRunsPanel", () => {
  it("renders every run with its mode, state, and promotion", () => {
    render(<FactoryRunsPanel runs={runs} />);
    expect(screen.getByText("run-1")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("yes")).toBeInTheDocument();
  });

  it("highlights the currently selected run and reports a row click", () => {
    const onSelectRun = vi.fn();
    render(<FactoryRunsPanel runs={runs} selectedRun="run-2" onSelectRun={onSelectRun} />);
    expect(screen.getByRole("row", { name: /run-2/ })).toHaveClass("is-current");
    fireEvent.click(screen.getByText("run-3"));
    expect(onSelectRun).toHaveBeenCalledWith("run-3");
  });

  it("shows a no-data state with no runs", () => {
    render(<FactoryRunsPanel runs={[]} />);
    expect(screen.getByText(/no runs recorded/)).toBeInTheDocument();
  });
});
