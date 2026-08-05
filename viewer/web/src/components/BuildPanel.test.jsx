import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { BuildPanel } from "./BuildPanel.jsx";

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const CATALOG = {
  organisms: ["Crafter"],
  runs: [{ organism: "Crafter", run: "run-1" }, { organism: "Crafter", run: "run-2" }],
};
const META = {
  modes: ["fresh", "clone", "resume", "fine_tune"],
  objectives: ["windowed_rollout"],
  genome_schemas: ["generic_action_effects_v1"],
  backbones: ["gru", "dilated_conv", "transformer"],
};
const CORPORA = [{ corpus_id: "crafter-corpus-1", organism: "Crafter", data_contract_hash: "abc", session_counts: { train: 3, validation: 1, test: 0 } }];
const FACTORY_RUNS = {
  organism: "Crafter",
  runs: [
    { run: "run-1", mode: "fresh", state: "completed", state_reason: "completed", updated_at: "2026-01-01T00:00:00Z", promoted: true },
    { run: "run-2", mode: "fresh", state: "completed", state_reason: "completed", updated_at: "2026-01-02T00:00:00Z", promoted: false },
  ],
};

function makeRoute(posted) {
  return function route(url, init) {
    const path = String(url);
    if (init?.method === "POST" && path.startsWith("/api/jobs/")) {
      const kind = path.replace("/api/jobs/", "");
      const body = JSON.parse(init.body);
      posted.push({ kind, body });
      return jsonResponse({
        job_id: "job-1", kind, status: "running", organism: body.organism,
        run_ids: [`${kind}-run`], started_at: "2026-01-03T00:00:00Z", exit_code: null, error: null,
      });
    }
    if (init?.method === "POST" && path === "/api/organisms") {
      const body = JSON.parse(init.body);
      posted.push({ kind: "organism", body });
      return jsonResponse({ organism: body.organism, created: true });
    }
    if (path.startsWith("/api/factory-meta")) return jsonResponse(META);
    if (path.startsWith("/api/corpora")) return jsonResponse({ corpora: CORPORA });
    if (path.startsWith("/api/corpus-specs")) return jsonResponse({ specs: [] });
    if (path.startsWith("/api/factory-runs")) return jsonResponse(FACTORY_RUNS);
    if (path.startsWith("/api/jobs")) return jsonResponse({ jobs: [] });
    return jsonResponse({ error: "unhandled" }, false);
  };
}

describe("BuildPanel", () => {
  let posted;
  beforeEach(() => { posted = []; vi.stubGlobal("fetch", vi.fn(makeRoute(posted))); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("defaults to fresh mode, sources enums from factory-meta, and launches a baseline job", async () => {
    render(<BuildPanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Corpus").value).toBe("crafter-corpus-1"));
    expect(screen.getByLabelText("Build mode").value).toBe("fresh");
    expect([...screen.getByLabelText("Backbone").options].map((o) => o.value)).toEqual(["gru", "dilated_conv", "transformer"]);
    expect([...screen.getByLabelText("Objective").options].map((o) => o.value)).toEqual(["windowed_rollout"]);

    fireEvent.click(screen.getByRole("button", { name: "Launch" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].kind).toBe("baseline");
    expect(posted[0].body.organism).toBe("Crafter");
    expect(posted[0].body.spec.mode).toBe("fresh");
    expect(posted[0].body.spec.data.corpus_id).toBe("crafter-corpus-1");
    expect(posted[0].body.spec.evaluation.selection_metric).toBe("rollout.t+4.model_over_copy_last_mse");
  });

  it("switching to clone mode shows a parent-run picker and checkpoint field, and launches a clone job with overrides", async () => {
    render(<BuildPanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Build mode").value).toBe("fresh"));

    fireEvent.change(screen.getByLabelText("Build mode"), { target: { value: "clone" } });
    await waitFor(() => expect(screen.getByLabelText("Parent run")).toBeInTheDocument());
    expect(screen.getByLabelText("Checkpoint").value).toBe("best-validation.pt");

    fireEvent.click(screen.getByRole("button", { name: "+ Add override" }));
    fireEvent.change(screen.getByLabelText("override 1 path"), { target: { value: "training.seed" } });
    fireEvent.change(screen.getByLabelText("override 1 value"), { target: { value: "5" } });

    fireEvent.click(screen.getByRole("button", { name: "Launch" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].kind).toBe("clone");
    expect(posted[0].body.run).toBe("run-1");
    expect(posted[0].body.options.mode).toBe("clone");
    expect(posted[0].body.options.set).toEqual(["training.seed=5"]);
  });

  it("switching to resume mode hides checkpoint/overrides and launches a resume job", async () => {
    render(<BuildPanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Build mode").value).toBe("fresh"));

    fireEvent.change(screen.getByLabelText("Build mode"), { target: { value: "resume" } });
    await waitFor(() => expect(screen.getByLabelText("Parent run")).toBeInTheDocument());
    expect(screen.queryByLabelText("Checkpoint")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Launch" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].kind).toBe("resume");
    expect(posted[0].body.run).toBe("run-1");
  });

  it("bootstraps a missing corpus from a repository recipe before enabling the first trial", async () => {
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      const path = String(url);
      if (init?.method === "POST" && path === "/api/jobs/corpus") {
        const body = JSON.parse(init.body);
        posted.push({ kind: "corpus", body });
        return jsonResponse({ job_id: "corpus-job", kind: "corpus", status: "running", organism: body.organism, run_ids: [] });
      }
      if (path.startsWith("/api/factory-meta")) return jsonResponse(META);
      if (path.startsWith("/api/corpus-specs")) return jsonResponse({
        specs: [{ spec_id: "generic-action-effects-v1.yaml", corpus_id: "crafter-generic-action-effects-v1", organism: "Crafter" }],
      });
      if (path.startsWith("/api/corpora")) return jsonResponse({ corpora: [] });
      if (path.startsWith("/api/factory-runs")) return jsonResponse({ organism: "Crafter", runs: [] });
      if (path.startsWith("/api/jobs")) return jsonResponse({ jobs: [] });
      return jsonResponse({ error: "unhandled" }, false);
    }));

    render(<BuildPanel catalog={{ organisms: ["Crafter"], runs: [] }} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Corpus recipe").value).toBe("generic-action-effects-v1.yaml"));
    expect(screen.getByRole("button", { name: "Launch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Build corpus" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]).toEqual({
      kind: "corpus",
      body: { organism: "Crafter", spec_id: "generic-action-effects-v1.yaml" },
    });
  });

  it("creates and selects a new organism without inheriting the Crafter name", async () => {
    const onOrganismCreated = vi.fn();
    render(<BuildPanel catalog={CATALOG} organism="Crafter" onOrganismCreated={onOrganismCreated} />);
    await waitFor(() => expect(screen.getByLabelText("Build mode")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("New organism name"), { target: { value: "Nova-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Create organism" }));

    await waitFor(() => expect(onOrganismCreated).toHaveBeenCalledWith("Nova-1"));
    expect(posted[0]).toEqual({ kind: "organism", body: { organism: "Nova-1" } });
    expect(screen.getByLabelText("Build organism").value).toBe("Nova-1");
  });
});
