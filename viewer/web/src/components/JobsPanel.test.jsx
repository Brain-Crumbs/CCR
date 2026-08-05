import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { JobsPanel } from "./JobsPanel.jsx";

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const JOBS = [
  {
    job_id: "job-2", kind: "search", status: "running", organism: "Crafter",
    started_at: "2026-01-02T00:00:00Z", run_ids: ["evo7-p0-c0"],
    runs: [{ run_id: "evo7-p0-c0", state: "running", state_reason: null, updated_at: null, heartbeat_seconds: null }],
  },
  {
    job_id: "job-1", kind: "baseline", status: "completed", organism: "Crafter",
    started_at: "2026-01-01T00:00:00Z", run_ids: ["job-1-run"],
    runs: [{ run_id: "job-1-run", state: "completed", state_reason: "completed", updated_at: null, heartbeat_seconds: null }],
  },
];

function makeRoute({ jobs = JOBS, log = "line one\nline two\n" } = {}) {
  const calls = [];
  const route = (url, init) => {
    const path = String(url);
    calls.push({ path, method: init?.method || "GET" });
    if (path.startsWith("/api/jobs/job-2/cancel")) {
      return jsonResponse({ ...jobs[0], status: "cancelled" });
    }
    if (path.startsWith("/api/jobs/job-2/log")) {
      const offset = Number(new URL(path, "http://localhost").searchParams.get("offset")) || 0;
      const text = offset === 0 ? log : "";
      return jsonResponse({ text, next_offset: log.length, size: log.length });
    }
    if (path.startsWith("/api/jobs")) return jsonResponse({ jobs });
    return jsonResponse({ error: "unhandled" }, false);
  };
  return { route, calls };
}

describe("JobsPanel", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows a no-data state with no jobs", async () => {
    const { route } = makeRoute({ jobs: [] });
    vi.stubGlobal("fetch", vi.fn(route));
    render(<JobsPanel organism="Crafter" />);
    await waitFor(() => expect(screen.getByText(/no jobs launched yet/)).toBeInTheDocument());
  });

  it("renders one row per job with a status badge, kind, and per-run state badges", async () => {
    const { route } = makeRoute();
    vi.stubGlobal("fetch", vi.fn(route));
    render(<JobsPanel organism="Crafter" />);
    await waitFor(() => expect(screen.getByText(/search/)).toBeInTheDocument());
    // Both the job's own status badge and its one run's state badge read
    // "running"/"completed" here (a single-run job whose subprocess status
    // and training state happen to agree) -- checking every match keeps the
    // assertion honest without depending on DOM order between the two.
    for (const el of screen.getAllByText("running")) expect(el).toHaveClass("state-badge--info");
    for (const el of screen.getAllByText("completed")) expect(el).toHaveClass("state-badge--green");
    expect(screen.getByText("evo7-p0-c0")).toBeInTheDocument();
    // A running job offers a Cancel action; a completed one does not.
    expect(screen.getAllByRole("button", { name: "Cancel" })).toHaveLength(1);
  });

  it("expanding a job tails its log", async () => {
    const { route } = makeRoute();
    vi.stubGlobal("fetch", vi.fn(route));
    render(<JobsPanel organism="Crafter" />);
    await waitFor(() => expect(screen.getByText(/search/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /search/ }));
    await waitFor(() => expect(screen.getByText(/line one/)).toBeInTheDocument());
  });

  it("cancelling a running job posts to its cancel endpoint", async () => {
    const { route, calls } = makeRoute();
    vi.stubGlobal("fetch", vi.fn(route));
    render(<JobsPanel organism="Crafter" />);
    await waitFor(() => expect(screen.getByText(/search/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path.includes("/cancel"))).toBe(true));
  });
});
