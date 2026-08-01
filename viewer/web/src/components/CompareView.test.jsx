import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CompareView } from "./CompareView.jsx";

function b64(bytes) { return btoa(String.fromCharCode(...bytes)); }
function jsonResponse(body, ok = true) { return Promise.resolve({ ok, json: () => Promise.resolve(body) }); }

const CATALOG = { organisms: ["Crafter"], runs: [{ organism: "Crafter", run: "run-a" }, { organism: "Crafter", run: "run-b" }] };
const SESSIONS = [{ id: "run/seed-1", name: "Crafter", episodes: ["episode_00000"], quality: { verdict: "green", issues: [], warnings: [] } }];
const FRAMES = {
  shape: [1, 1, 3], dtype: "uint8", n_frames: 2,
  frames: [
    { i: 0, t: 0, tick: 0, seq: 0, hash: "a", data: b64([1, 2, 3]) },
    { i: 1, t: 1, tick: 1, seq: 1, hash: "b", data: b64([4, 5, 6]) },
  ],
};
const PREDICTIONS_A = {
  format: "pixel-predictions-v2", experiment: { experiment_id: "run-a" },
  model: { model_type: "predictive_cortex", checkpoint_sha256: "aaa", uses_actions: true, uses_workspace: false },
  prediction_mode: "rollout", horizons: [1], prediction_shape: [1, 1, 3],
  predictions: { 1: { frames: [b64([9, 9, 9])] } },
  targets: [b64([1, 2, 3]), b64([4, 5, 6])],
  events: [], evaluation_source: "seed-1-train-0",
};

function runSummary(run) {
  return { organism: "Crafter", run, model: { type: "predictive_cortex", checkpoint_sha256: run }, training: {}, evaluation: {}, promotion: { promoted: null, reasons: [] } };
}

function route(url) {
  const path = String(url);
  if (path.startsWith("/api/sessions") && path.includes("run=run-a") && !path.includes("/episodes/")) return jsonResponse({ sessions: SESSIONS });
  if (path.startsWith("/api/runs")) {
    const run = new URL(path, "http://localhost").searchParams.get("run");
    return jsonResponse(runSummary(run));
  }
  if (path.includes("/frames")) return jsonResponse(FRAMES);
  if (path.includes("/predictions")) {
    const experiment = new URL(path, "http://localhost").searchParams.get("experiment");
    if (experiment === "run-a") return jsonResponse(PREDICTIONS_A);
    return jsonResponse({ error: "no recorded predictions for this episode" }, false);
  }
  return jsonResponse({ error: "unhandled" }, false);
}

describe("CompareView", () => {
  beforeEach(() => { vi.stubGlobal("fetch", vi.fn(route)); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("defaults Run A to the currently selected run and Run B to a different run, then renders both strips", async () => {
    render(<CompareView catalog={CATALOG} organism="Crafter" defaultRunA="run-a" />);
    await waitFor(() => expect(screen.getByLabelText("Run A").value).toBe("run-a"));
    expect(screen.getByLabelText("Run B").value).toBe("run-b");
    await waitFor(() => expect(screen.getAllByLabelText("prediction source")).toHaveLength(2));
    expect(screen.getByText(/Experiment run-a/)).toBeInTheDocument();
    expect(screen.getByText("No export identity; baselines only.")).toBeInTheDocument();
  });

  it("switching Run B re-fetches its summary and prediction export", async () => {
    render(<CompareView catalog={CATALOG} organism="Crafter" defaultRunA="run-a" />);
    await waitFor(() => expect(screen.getAllByLabelText("prediction source")).toHaveLength(2));
    fireEvent.change(screen.getByLabelText("Run B"), { target: { value: "run-a" } });
    await waitFor(() => expect(screen.getByLabelText("Run B").value).toBe("run-a"));
  });
});
