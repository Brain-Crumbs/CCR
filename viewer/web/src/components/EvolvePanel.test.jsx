import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { EvolvePanel, estimateMaxTrials, varyingGenePaths } from "./EvolvePanel.jsx";

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const CATALOG = { organisms: ["Crafter"], runs: [{ organism: "Crafter", run: "run-1" }, { organism: "Crafter", run: "run-2" }] };
const META = {
  modes: ["fresh", "clone", "resume", "fine_tune"],
  objectives: ["windowed_rollout"],
  genome_schemas: ["generic_action_effects_v1"],
  backbones: ["gru", "dilated_conv", "transformer"],
};
const FACTORY_RUNS = {
  organism: "Crafter",
  runs: [
    { run: "run-2", mode: "fresh", state: "running", promoted: null, selection_metric: null, selection_metric_value: null, configuration_parents: [] },
    { run: "run-1", mode: "fresh", state: "completed", promoted: true, selection_metric: "rollout.t+4.model_over_copy_last_mse", selection_metric_value: 0.7, configuration_parents: [] },
  ],
};
const REGISTRY = {
  format: "model-factory-registry-v1",
  slots: { generic: { small: { "rollout.t+4.model_over_copy_last_mse": { leading_champion: "run-1", population: [], history: [] } } } },
};
const TRIAL_SPEC = {
  format: "model-factory-spec-v1",
  organism: "Crafter",
  mode: "fresh",
  parent: null,
  evolution: { configuration_parents: ["x", "y"], weight_donor: "x" },
  data: { corpus_id: "crafter-corpus-1", horizons_ticks: [1, 2, 3, 4] },
  model: { backbone: "transformer" },
  training: { objective: "windowed_rollout", optimizer: { lr: 0.0003 }, rollout_frames: 8 },
  evaluation: { selection_metric: "rollout.t+4.model_over_copy_last_mse", gates: [], confidence: 0.95 },
};
const PREVIEW = {
  campaign_id: "evo7",
  dry_run: true,
  workflow: "evolutionary",
  schema: "generic_action_effects_v1",
  schema_hash: "e247442c3fac00d30f90ca411cce0a7c",
  method: "lhs",
  seed: 7,
  population_size: 2,
  population_count: 3,
  mutation_rate: 0.2,
  candidates: [
    { candidate_index: 0, run_id: "evo7-p0-c0", spec: { training: { objective: "windowed_rollout", optimizer: { lr: 0.0001 }, rollout_frames: 8 } } },
    { candidate_index: 1, run_id: "evo7-p0-c1", spec: { training: { objective: "windowed_rollout", optimizer: { lr: 0.002 }, rollout_frames: 12 } } },
  ],
};

function makeRoute(posted) {
  return function route(url, init) {
    const path = String(url);
    if (init?.method === "POST" && path.startsWith("/api/preview/")) {
      posted.push({ route: path, body: JSON.parse(init.body) });
      return jsonResponse(PREVIEW);
    }
    if (init?.method === "POST" && path.startsWith("/api/jobs/")) {
      const body = JSON.parse(init.body);
      posted.push({ route: path, body });
      return jsonResponse({
        job_id: "job-1", kind: "search", status: "running", organism: body.organism,
        run_ids: ["evo7-p0-c0"], started_at: "2026-01-03T00:00:00Z", exit_code: null, error: null,
      });
    }
    if (path.startsWith("/api/factory-meta")) return jsonResponse(META);
    if (path.startsWith("/api/factory-runs")) return jsonResponse(FACTORY_RUNS);
    if (path.startsWith("/api/registry")) return jsonResponse(REGISTRY);
    if (path.startsWith("/api/experiments")) return jsonResponse({ trial_spec: TRIAL_SPEC, metrics: {} });
    if (path.startsWith("/api/jobs")) return jsonResponse({ jobs: [] });
    return jsonResponse({ error: `unhandled ${path}` }, false);
  };
}

describe("varyingGenePaths", () => {
  it("returns only the dotted training paths that actually differ across proposals", () => {
    expect(varyingGenePaths(PREVIEW.candidates)).toEqual(["optimizer.lr", "rollout_frames"]);
  });

  it("returns nothing for fewer than two proposals", () => {
    expect(varyingGenePaths(PREVIEW.candidates.slice(0, 1))).toEqual([]);
    expect(varyingGenePaths(undefined)).toEqual([]);
  });
});

describe("estimateMaxTrials", () => {
  it("counts a full first generation plus at most population_size - 1 offspring per later one", () => {
    expect(estimateMaxTrials(6, 3)).toBe(16);
    expect(estimateMaxTrials(6, 1)).toBe(6);
    expect(estimateMaxTrials(0, 3)).toBe(0);
  });
});

describe("EvolvePanel", () => {
  let posted;
  beforeEach(() => { posted = []; vi.stubGlobal("fetch", vi.fn(makeRoute(posted))); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  // Both action buttons stay disabled until the base run's trial_spec.json
  // has actually loaded -- there is no campaign to preview or launch without
  // a base spec -- so every interaction test waits for that first.
  async function renderPanel() {
    render(<EvolvePanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Base run").value).toBe("run-1"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Dry-run preview" })).toBeEnabled());
  }

  it("offers only completed runs as a base, sources schemas from factory-meta, and defaults the reference to the leading champion", async () => {
    await renderPanel();
    expect([...screen.getByLabelText("Base run").options].map((o) => o.value)).toEqual(["run-1"]);
    expect([...screen.getByLabelText("Genome schema").options].map((o) => o.value)).toEqual(["generic_action_effects_v1"]);
    await waitFor(() => expect(screen.getByLabelText("Reference run").value).toBe("run-1"));
    await waitFor(() => expect(screen.getByLabelText("Selection metric").value).toBe("rollout.t+4.model_over_copy_last_mse"));
  });

  it("shows the campaign's worst-case training count and precomputed run-id shape", async () => {
    await renderPanel();
    const estimate = screen.getByTestId("cost-estimate");
    expect(estimate.textContent).toContain("at most 16 training run(s)");
    expect(estimate.textContent).toContain("evo7-p<generation>-c<candidate>");

    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "3" } });
    await waitFor(() => expect(screen.getByTestId("cost-estimate").textContent).toContain("evo3-p<generation>"));
  });

  it("dry-run previews through /api/preview/search and tabulates only the genes that vary", async () => {
    await renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Dry-run preview" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].route).toBe("/api/preview/search");
    // The base run's own evolution block is deliberately not carried into
    // the campaign's base spec.
    expect(posted[0].body.spec.evolution).toBeUndefined();
    expect(posted[0].body.spec.mode).toBe("fresh");
    expect(posted[0].body.spec.data.corpus_id).toBe("crafter-corpus-1");
    expect(posted[0].body.options.run_id_prefix).toBe("evo7");
    expect(posted[0].body.options.populations).toBe(3);
    expect(posted[0].body.options.reference_run).toBe("run-1");
    expect(posted[0].body.options.reference_checkpoint).toBe("best-validation.pt");
    expect(posted[0].body.options.set).toBeNull();

    await waitFor(() => expect(screen.getByText("evo7-p0-c1")).toBeInTheDocument());
    expect(screen.getByRole("columnheader", { name: "optimizer.lr" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "rollout_frames" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "objective" })).not.toBeInTheDocument();
  });

  it("sends an edited selection metric as one --set override, and drops the reference checkpoint when no reference is chosen", async () => {
    await renderPanel();
    await waitFor(() => expect(screen.getByLabelText("Reference run").value).toBe("run-1"));

    fireEvent.change(screen.getByLabelText("Selection metric"), { target: { value: "rollout.t+4.model_over_reference_mse" } });
    fireEvent.change(screen.getByLabelText("Reference run"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Dry-run preview" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].body.options.set).toEqual(["evaluation.selection_metric=rollout.t+4.model_over_reference_mse"]);
    expect(posted[0].body.options.reference_run).toBeNull();
    expect(posted[0].body.options.reference_checkpoint).toBeNull();
  });

  it("launches through /api/jobs/search and opens the population board on the campaign's prefix", async () => {
    await renderPanel();
    fireEvent.change(screen.getByLabelText("Population size"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Generations"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch campaign" }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0].route).toBe("/api/jobs/search");
    expect(posted[0].body.options.population_size).toBe(4);
    expect(posted[0].body.options.populations).toBe(2);

    await waitFor(() => expect(screen.getByText("Population board")).toBeInTheDocument());
    expect(screen.getByText(/4 candidates × 2 generation\(s\)/)).toBeInTheDocument();
  });

  it("does not default the reference run to a champion the picker cannot offer", async () => {
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      if (String(url).startsWith("/api/registry")) {
        return jsonResponse({
          format: "model-factory-registry-v1",
          slots: { generic: { small: { obj: { leading_champion: "archived-run", population: [], history: [] } } } },
        });
      }
      return makeRoute([])(url, init);
    }));
    render(<EvolvePanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Base run").value).toBe("run-1"));

    const reference = screen.getByLabelText("Reference run");
    expect([...reference.options].map((o) => o.value)).toEqual(["", "run-1"]);
    expect(reference.value).toBe("");
  });

  it("surfaces a rejected preview instead of a stale candidate table", async () => {
    vi.stubGlobal("fetch", vi.fn((url, init) => {
      if (init?.method === "POST") return jsonResponse({ error: "invalid factory search: population_size must be at least 2" }, false);
      return makeRoute([])(url, init);
    }));
    render(<EvolvePanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Base run").value).toBe("run-1"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Dry-run preview" })).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: "Dry-run preview" }));
    await waitFor(() => expect(screen.getByText(/population_size must be at least 2/)).toBeInTheDocument());
    expect(screen.queryByText("Dry-run preview", { selector: "h4" })).not.toBeInTheDocument();
  });
});
