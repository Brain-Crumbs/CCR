import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CompareView } from "./CompareView.jsx";

function b64(bytes) { return btoa(String.fromCharCode(...bytes)); }
function jsonResponse(body, ok = true) { return Promise.resolve({ ok, json: () => Promise.resolve(body) }); }

const SESSIONS = [{ id: "corpus/seed-1", name: "Crafter", episodes: ["episode_00000"], quality: { verdict: "green", issues: [], warnings: [] } }];
const FRAMES = { shape: [1, 1, 3], dtype: "uint8", n_frames: 1, frames: [{ i: 0, t: 0, tick: 0, seq: 0, hash: "a", data: b64([1, 2, 3]) }] };

function summaryFor(run) {
  return { organism: "Crafter", run, model: { type: "predictive_cortex", checkpoint_sha256: `${run}-sha` }, training: {}, evaluation: {}, promotion: { promoted: null, reasons: [] } };
}

function route(url) {
  const path = String(url);
  if (path.startsWith("/api/runs")) {
    const run = new URL(path, "http://localhost").searchParams.get("run");
    return jsonResponse(summaryFor(run));
  }
  if (path.includes("/decisions")) return jsonResponse({ records: [] });
  if (path.includes("/frames")) return jsonResponse(FRAMES);
  if (path.includes("/predictions")) return jsonResponse({ error: "no recorded predictions" }, false);
  if (path.startsWith("/api/sessions")) return jsonResponse({ sessions: SESSIONS });
  return jsonResponse({ error: "unhandled" }, false);
}

describe("CompareView", () => {
  beforeEach(() => { vi.stubGlobal("fetch", vi.fn(route)); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("renders both runs' summaries and pixel viewers for the same session", async () => {
    render(<CompareView organism="Crafter" runs={["run-1", "run-2"]} defaultRun="run-1" />);
    await waitFor(() => expect(screen.getByText("Run A — run-1")).toBeInTheDocument());
    expect(screen.getByText("Run B — run-2")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByLabelText("prediction source")).toHaveLength(2));
    await waitFor(() => expect(screen.getByText(/checkpoint run-1-sha/)).toBeInTheDocument());
    expect(screen.getByText(/checkpoint run-2-sha/)).toBeInTheDocument();
  });

  it("defaults Run B to a different run than Run A when one is available", async () => {
    render(<CompareView organism="Crafter" runs={["run-1", "run-2", "run-3"]} defaultRun="run-1" />);
    await waitFor(() => expect(screen.getByLabelText("Run B").value).not.toBe("run-1"));
  });

  it("re-fetches Run B's summary when the picker changes", async () => {
    render(<CompareView organism="Crafter" runs={["run-1", "run-2"]} defaultRun="run-1" />);
    await waitFor(() => expect(screen.getByText(/checkpoint run-2-sha/)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Run B"), { target: { value: "run-1" } });
    await waitFor(() => expect(screen.getAllByText(/checkpoint run-1-sha/).length).toBeGreaterThan(1));
  });

  it("shows an empty state when Run A has no recordings", async () => {
    vi.stubGlobal("fetch", vi.fn((url) => (String(url).startsWith("/api/sessions") ? jsonResponse({ sessions: [] }) : route(url))));
    render(<CompareView organism="Crafter" runs={["run-1", "run-2"]} defaultRun="run-1" />);
    await waitFor(() => expect(screen.getByText(/Run A has no recordings yet/)).toBeInTheDocument());
  });
});
