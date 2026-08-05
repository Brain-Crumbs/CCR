import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { BreedPanel, geneRows } from "./BreedPanel.jsx";

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const CATALOG = {
  organisms: ["Crafter"],
  runs: [{ organism: "Crafter", run: "run-1" }, { organism: "Crafter", run: "run-2" }, { organism: "Crafter", run: "run-3" }],
};
const META = {
  modes: ["fresh", "clone", "resume", "fine_tune"],
  objectives: ["windowed_rollout", "autoregressive"],
  genome_schemas: ["generic_action_effects_v1"],
  backbones: ["gru", "dilated_conv", "transformer"],
};
const FACTORY_RUNS = {
  organism: "Crafter",
  runs: [
    { run: "run-3", mode: "clone", state: "running", promoted: null, selection_metric: null, selection_metric_value: null, configuration_parents: [] },
    { run: "run-1", mode: "fresh", state: "completed", promoted: true, selection_metric: "rollout.t+4.model_over_copy_last_mse", selection_metric_value: 0.71, configuration_parents: [] },
    { run: "run-2", mode: "clone", state: "completed", promoted: false, selection_metric: "rollout.t+4.model_over_copy_last_mse", selection_metric_value: 0.83, configuration_parents: [] },
  ],
};
const PREVIEW = {
  dry_run: true,
  schema: "generic_action_effects_v1",
  schema_hash: "e247442c3fac00d30f90ca411cce0a7c",
  child_spec: {
    format: "model-factory-spec-v1",
    organism: "Crafter",
    mode: "clone",
    parent: { run_id: "run-2", checkpoint: "best-validation.pt", sha256: "abc123" },
    data: { corpus_id: "crafter-corpus-1" },
    training: { objective: "windowed_rollout", optimizer: { lr: 0.0005 }, rollout_frames: 12 },
    evaluation: { selection_metric: "rollout.t+4.model_over_copy_last_mse" },
  },
  breeding_lineage: {
    format: "model-factory-breeding-lineage-v1",
    generation: 2,
    parent_a_run_id: "run-1",
    parent_b_run_id: "run-2",
    weight_donor_run_id: "run-2",
    genome_operator: "uniform_crossover+bounded_mutation",
    mutation_seed: 7,
    parent_a_genome: { "optimizer.lr": 0.0003, rollout_frames: 8, warmup_frames: 4 },
    parent_b_genome: { "optimizer.lr": 0.001, rollout_frames: 12, warmup_frames: 4 },
    crossover_choices: { "optimizer.lr": "parent_a", rollout_frames: "parent_b", warmup_frames: "parent_a" },
    mutations: { "optimizer.lr": { from: 0.0003, to: 0.0005 } },
    repairs: { rollout_frames: { from: 16, to: 12 } },
    child_genome: { "optimizer.lr": 0.0005, rollout_frames: 12, warmup_frames: 4 },
  },
};

function makeRoute(posted, { previewBody = PREVIEW, previewOk = true } = {}) {
  return function route(url, init) {
    const path = String(url);
    if (init?.method === "POST" && path.startsWith("/api/preview/")) {
      posted.push({ route: path, body: JSON.parse(init.body) });
      return jsonResponse(previewBody, previewOk);
    }
    if (init?.method === "POST" && path.startsWith("/api/jobs/")) {
      const body = JSON.parse(init.body);
      posted.push({ route: path, body });
      return jsonResponse({
        job_id: "job-9", kind: "breed", status: "running", organism: body.organism,
        run_ids: ["job-9"], started_at: "2026-01-04T00:00:00Z", exit_code: null, error: null,
      });
    }
    if (path.startsWith("/api/factory-meta")) return jsonResponse(META);
    if (path.startsWith("/api/factory-runs")) return jsonResponse(FACTORY_RUNS);
    if (path.startsWith("/api/jobs")) return jsonResponse({ jobs: [] });
    return jsonResponse({ error: `unhandled ${path}` }, false);
  };
}

describe("geneRows", () => {
  it("joins each gene's parent values, crossover choice, mutation and repair", () => {
    const rows = geneRows(PREVIEW.breeding_lineage);
    expect(rows.map((row) => row.gene)).toEqual(["optimizer.lr", "rollout_frames", "warmup_frames"]);

    const [lr, rollout, warmup] = rows;
    expect(lr).toMatchObject({ parent_a: 0.0003, parent_b: 0.001, inherited: "parent_a", child: 0.0005, moved: true });
    expect(lr.mutation).toEqual({ from: 0.0003, to: 0.0005 });
    expect(rollout).toMatchObject({ inherited: "parent_b", child: 12, moved: true });
    expect(rollout.repair).toEqual({ from: 16, to: 12 });
    // Inherited untouched: neither mutated nor repaired, so nothing is flagged.
    expect(warmup).toMatchObject({ inherited: "parent_a", child: 4, mutation: null, repair: null, moved: false });
  });

  it("returns nothing without a lineage record", () => {
    expect(geneRows(null)).toEqual([]);
    expect(geneRows({})).toEqual([]);
  });
});

describe("BreedPanel", () => {
  let posted;
  beforeEach(() => { posted = []; vi.stubGlobal("fetch", vi.fn(makeRoute(posted))); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  // Waits on parent B, not parent A: a select whose form state is still the
  // empty initial string reports its first option ("run-1") as its value, so
  // parent A alone would pass before the defaults have actually landed.
  async function renderPanel() {
    render(<BreedPanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Parent B").value).toBe("run-2"));
    expect(screen.getByLabelText("Parent A").value).toBe("run-1");
  }

  it("offers only completed runs and defaults the two parents to different ones", async () => {
    await renderPanel();
    expect([...screen.getByLabelText("Parent A").options].map((o) => o.value)).toEqual(["run-1", "run-2"]);
    expect(screen.getByLabelText("Parent B").value).toBe("run-2");
    // The running run is never offered: load_parent refuses anything that
    // did not reach `completed`.
    expect([...screen.getByLabelText("Parent B").options].map((o) => o.value)).not.toContain("run-3");
  });

  it("previews compatibility through /api/preview/breed and tabulates the child's genome", async () => {
    await renderPanel();
    await waitFor(() => expect(posted).toHaveLength(1));

    expect(posted[0].route).toBe("/api/preview/breed");
    expect(posted[0].body.parent_a).toBe("run-1");
    expect(posted[0].body.parent_b).toBe("run-2");
    expect(posted[0].body.options).toMatchObject({
      organism: "Crafter", schema: "generic_action_effects_v1", seed: 7, mutation_rate: 0.2,
      checkpoint_a: "best-validation.pt", checkpoint_b: "best-validation.pt",
      tier: null, objective: null, generation: null, weight_donor: null,
    });
    // Launch-only options never ride along on a dry run.
    expect(posted[0].body.options.naming_seed).toBeUndefined();

    await waitFor(() => expect(screen.getByTestId("breed-summary").textContent).toContain("compatible"));
    expect(screen.getByTestId("breed-summary").textContent).toContain("generation 2");
    expect(screen.getByTestId("breed-summary").textContent).toContain("run-2");
    expect(screen.getByRole("rowheader", { name: "optimizer.lr" })).toBeInTheDocument();
    expect(screen.getByText("mutated")).toBeInTheDocument();
    expect(screen.getByText("repaired")).toBeInTheDocument();
  });

  it("re-previews when a genetic parameter changes, and sends the edited values", async () => {
    await renderPanel();
    await waitFor(() => expect(posted).toHaveLength(1));

    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "11" } });
    fireEvent.change(screen.getByLabelText("Mutation rate"), { target: { value: "0.5" } });
    fireEvent.change(screen.getByLabelText("Weight donor"), { target: { value: "run-1" } });

    await waitFor(() => expect(posted.length).toBeGreaterThan(1));
    const latest = posted[posted.length - 1];
    expect(latest.route).toBe("/api/preview/breed");
    expect(latest.body.options).toMatchObject({ seed: 11, mutation_rate: 0.5, weight_donor: "run-1" });
  });

  it("surfaces check_compatible's own mismatch reasons and refuses to launch", async () => {
    posted = [];
    vi.stubGlobal("fetch", vi.fn(makeRoute(posted, {
      previewOk: false,
      previewBody: {
        error: "invalid factory breed: parents 'run-1' and 'run-2' are incompatible for breeding "
          + "(epic #212 §14.1 requires an identical architecture hash, corpus, stage, budget tier and "
          + "evaluation contract): architecture_hash ('aaa' != 'bbb'); budget tier ('fast' != 'scale')",
      },
    })));
    render(<BreedPanel catalog={CATALOG} organism="Crafter" />);
    await waitFor(() => expect(screen.getByLabelText("Parent A").value).toBe("run-1"));

    await waitFor(() => expect(screen.getByTestId("breed-error")).toBeInTheDocument());
    const message = screen.getByTestId("breed-error").textContent;
    expect(message).toContain("architecture_hash ('aaa' != 'bbb')");
    expect(message).toContain("budget tier ('fast' != 'scale')");
    expect(screen.getByRole("button", { name: "Breed child" })).toBeDisabled();
  });

  it("never previews or launches a run crossed with itself", async () => {
    await renderPanel();
    await waitFor(() => expect(posted).toHaveLength(1));

    fireEvent.change(screen.getByLabelText("Parent B"), { target: { value: "run-1" } });
    await waitFor(() => expect(screen.getByText(/pick two different runs/)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Breed child" })).toBeDisabled();
    // The stale preview is dropped with it, so nothing on screen still
    // describes a child these parents would not produce.
    expect(screen.queryByTestId("breed-summary")).not.toBeInTheDocument();
    expect(posted).toHaveLength(1);
  });

  it("launches through /api/jobs/breed with the previewed options plus the launch-only ones", async () => {
    await renderPanel();
    await waitFor(() => expect(screen.getByRole("button", { name: "Breed child" })).toBeEnabled());

    fireEvent.change(screen.getByLabelText("Naming seed (optional)"), { target: { value: "kestrel" } });
    fireEvent.click(screen.getByRole("button", { name: "Breed child" }));

    await waitFor(() => expect(posted.some((entry) => entry.route === "/api/jobs/breed")).toBe(true));
    const launch = posted.find((entry) => entry.route === "/api/jobs/breed");
    expect(launch.body.parent_a).toBe("run-1");
    expect(launch.body.parent_b).toBe("run-2");
    expect(launch.body.options).toMatchObject({
      organism: "Crafter", seed: 7, checkpoint_a: "best-validation.pt",
      naming_seed: "kestrel", export_predictions_max: 3, no_export_predictions: false,
    });
  });
});
