import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { App } from "./App.jsx";

function b64(bytes) { return btoa(String.fromCharCode(...bytes)); }

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const CATALOG = { organisms: ["Crafter"], runs: [{ organism: "Crafter", run: "run-1" }] };
const RUN_SUMMARY = {
  organism: "Crafter", run: "run-1", model: { type: "predictive_cortex", checkpoint_sha256: "abc123" },
  training: {}, evaluation: {}, promotion: { promoted: true, reasons: [] },
};
const EXPERIMENTS = { experiment: { experiment_id: "run-1" }, trial_spec: null, contracts: null, lineage: { mode: "fresh" }, data_manifest: null, execution: null, experiment_report: null, metrics: { validation: null, test: null, comparison: null } };
const SESSIONS = [{ id: "run/seed-1", name: "Crafter", episodes: ["episode_00000"], quality: { verdict: "green", issues: [], warnings: [] } }];
const SESSION_DETAIL = {
  session: { id: "run/seed-1", episodes: ["episode_00000"], development: { stages: [{ name: "Gestation", passed: true }] } },
  streams: { episode_00000: [{ stream_id: "internal.dopamine", payload: { value: 0.2 }, seq: 0 }] },
  decisions: { episode_00000: [{ tick_index: 0, motor_decision: { voluntary: "move_up", reflex: null, caregiver_override: null, actuated: "move_up" } }] },
  exports: [], quality: { verdict: "green", issues: [], warnings: [] },
};
const REGISTRY = { format: "model-factory-registry-v1", slots: {} };
const FACTORY_RUNS = { organism: "Crafter", runs: [{ run: "run-1", mode: "fresh", state: "completed", state_reason: "completed", updated_at: "2026-01-01T00:00:00Z", promoted: true }] };
const FRAMES = { shape: [1, 1, 3], dtype: "uint8", n_frames: 1, frames: [{ i: 0, t: 0, tick: 0, seq: 0, hash: "a", data: b64([1, 2, 3]) }] };

function route(url) {
  const path = String(url);
  if (path.startsWith("/api/catalog")) return jsonResponse(CATALOG);
  if (path.startsWith("/api/runs")) return jsonResponse(RUN_SUMMARY);
  if (path.startsWith("/api/experiments")) return jsonResponse(EXPERIMENTS);
  if (path.startsWith("/api/factory-runs")) return jsonResponse(FACTORY_RUNS);
  if (path.startsWith("/api/registry")) return jsonResponse(REGISTRY);
  if (path.startsWith("/api/sessions") && path.includes("/episodes/") && path.endsWith("/frames") || path.includes("/frames?")) return jsonResponse(FRAMES);
  if (path.includes("/predictions")) return jsonResponse({ error: "no recorded predictions" }, false);
  if (path.startsWith("/api/sessions/")) return jsonResponse(SESSION_DETAIL);
  if (path.startsWith("/api/sessions")) return jsonResponse({ sessions: SESSIONS });
  return jsonResponse({ error: "unhandled" }, false);
}

describe("App", () => {
  beforeEach(() => { location.hash = ""; vi.stubGlobal("fetch", vi.fn(route)); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); location.hash = ""; });

  it("loads the catalog, selects the first run, and renders the episode tab by default", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Crafter / run-1 / run/seed-1")).toBeInTheDocument());
    expect(screen.getByText("predictive_cortex")).toBeInTheDocument(); // RunSummary
    await waitFor(() => expect(screen.getByLabelText("prediction source")).toBeInTheDocument()); // PixelHorizonViewer
    expect(screen.getByText(/What the organism intended/)).toBeInTheDocument(); // ActionTimeline
    expect(screen.getByText("dopamine")).toBeInTheDocument(); // EEGPanel
    expect(location.hash).toBe("#Crafter/run-1/run%2Fseed-1/episode_00000");
  });

  it("switches to the Factory tab and renders run status plus the registry panel", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Crafter / run-1 / run/seed-1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: "Factory" }));
    await waitFor(() => expect(screen.getByRole("row", { name: /run-1/ })).toBeInTheDocument());
    expect(screen.getByText(/no champions promoted yet/)).toBeInTheDocument();
  });

  it("switches to the Compare tab and renders a pixel viewer per run", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Crafter / run-1 / run/seed-1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: "Compare" }));
    await waitFor(() => expect(screen.getByText("Run A — run-1")).toBeInTheDocument());
    // The fixture catalog has only one run, so Run B falls back to it too.
    expect(screen.getByText("Run B — run-1")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByLabelText("prediction source")).toHaveLength(2));
  });

  it("switches to the Development tab and renders the ladder from the session", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Crafter / run-1 / run/seed-1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: "Development" }));
    await waitFor(() => expect(screen.getByText("Gestation")).toBeInTheDocument());
  });

  it("switches to the Experiment tab and renders lineage plus the promotion verdict", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Crafter / run-1 / run/seed-1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: "Experiment" }));
    expect(screen.getByText("fresh")).toBeInTheDocument();
    expect(screen.getAllByText("Promoted").length).toBeGreaterThan(0);
  });
});
