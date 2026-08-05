import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import {
  EvolvePanel, estimateArchitectureTrials, estimateMaxTrials, varyingGenePaths, varyingGenomeGenes,
} from "./EvolvePanel.jsx";

function jsonResponse(body, ok = true) {
  return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

const CATALOG = { organisms: ["Crafter"], runs: [{ organism: "Crafter", run: "run-1" }, { organism: "Crafter", run: "run-2" }] };
const META = {
  modes: ["fresh", "clone", "resume", "fine_tune"],
  objectives: ["windowed_rollout"],
  genome_schemas: ["generic_action_effects_v1"],
  architecture_genome_schemas: ["architecture_search_v1"],
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
  training: { objective: "windowed_rollout", optimizer: { lr: 0.0003 }, rollout_frames: 8, max_training_seconds: 600 },
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

/** `ccr factory search --genome architecture --dry-run`'s own payload shape
 * (architecture_search.ArchitectureSearchPreview.to_dict): the cost estimate
 * plus generation zero's sampled architectures. */
const ARCHITECTURE_PREVIEW = {
  format: "model-factory-architecture-search-v1",
  campaign_id: "arch7",
  dry_run: true,
  workflow: "architecture",
  method: "lhs",
  seed: 7,
  architecture_schema_version: "architecture_search_v1",
  architecture_schema_hash: "74b4c7b0ed043587da360bd2990ea185",
  hyperparameter_schema_version: "generic_action_effects_v1",
  estimate: {
    outer_population_size: 2, outer_population_count: 2,
    inner_population_size: 6, inner_population_count: 3,
    estimated_max_trials: 72, max_training_seconds_per_trial: 600, estimated_max_seconds: 43200,
  },
  candidates: [
    {
      candidate_index: 0, architecture_id: "arch7-arch0-a0",
      genome: { backbone_preset: "gru", latent_width: 128, hidden_dim: 256, action_embed_dim: 8, context_length: 8 },
      spec: { model: { backbone: "gru" } },
    },
    {
      candidate_index: 1, architecture_id: "arch7-arch0-a1",
      genome: { backbone_preset: "transformer_h4_l2", latent_width: 128, hidden_dim: 512, action_embed_dim: 8, context_length: 8 },
      spec: { model: { backbone: "transformer" } },
    },
  ],
};

function makeRoute(posted) {
  return function route(url, init) {
    const path = String(url);
    if (init?.method === "POST" && path.startsWith("/api/preview/")) {
      const body = JSON.parse(init.body);
      posted.push({ route: path, body });
      return jsonResponse(body.options?.genome === "architecture" ? ARCHITECTURE_PREVIEW : PREVIEW);
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

describe("estimateArchitectureTrials", () => {
  it("multiplies both loops' budgets, as architecture_search.estimate_cost does", () => {
    expect(estimateArchitectureTrials(4, 2, 3, 1)).toBe(24);
    expect(estimateArchitectureTrials(2, 1, 2, 1)).toBe(4);
    expect(estimateArchitectureTrials(4, 2, 3, 0)).toBe(0);
  });
});

describe("varyingGenomeGenes", () => {
  it("returns only the architecture genes that actually differ across sampled architectures", () => {
    expect(varyingGenomeGenes(ARCHITECTURE_PREVIEW.candidates)).toEqual(["backbone_preset", "hidden_dim"]);
    expect(varyingGenomeGenes(ARCHITECTURE_PREVIEW.candidates.slice(0, 1))).toEqual([]);
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

  describe("architecture (NAS) mode", () => {
    async function switchToArchitecture() {
      await renderPanel();
      fireEvent.click(screen.getByRole("button", { name: "Architecture (NAS)" }));
      await waitFor(() => expect(screen.getByLabelText("Architecture schema")).toBeInTheDocument());
    }

    it("offers the architecture schema registry, kept separate from the training one", async () => {
      await switchToArchitecture();
      expect([...screen.getByLabelText("Architecture schema").options].map((o) => o.value))
        .toEqual(["architecture_search_v1"]);
      expect([...screen.getByLabelText("Inner genome schema").options].map((o) => o.value))
        .toEqual(["generic_action_effects_v1"]);
    });

    it("puts the multiplied cost bound front and center, with the base spec's own wall clock", async () => {
      await switchToArchitecture();
      fireEvent.change(screen.getByLabelText("Architectures per generation"), { target: { value: "4" } });
      fireEvent.change(screen.getByLabelText("Outer generations"), { target: { value: "2" } });
      fireEvent.change(screen.getByLabelText("Inner population size"), { target: { value: "3" } });
      fireEvent.change(screen.getByLabelText("Inner generations"), { target: { value: "1" } });

      const estimate = await waitFor(() => {
        const node = screen.getByTestId("cost-estimate");
        expect(node.textContent).toContain("up to 24 full trainings");
        return node;
      });
      // 24 trials x the base spec's 600s per-trial cap.
      expect(estimate.textContent).toContain("4.0 h");
      expect(estimate.textContent).toContain("arch7-arch<gen>-a<arch>-p<gen>-c<candidate>");
    });

    it("keeps the launch disabled until the cost is explicitly acknowledged, and un-acknowledges on any change", async () => {
      await switchToArchitecture();
      const launch = screen.getByRole("button", { name: "Launch campaign" });
      expect(launch).toBeDisabled();
      // The dry run stays available -- it is free, and seeing the candidates
      // is exactly what should inform the acknowledgement.
      expect(screen.getByRole("button", { name: "Dry-run preview" })).toBeEnabled();

      fireEvent.click(screen.getByLabelText("Acknowledge campaign cost"));
      await waitFor(() => expect(launch).toBeEnabled());

      fireEvent.change(screen.getByLabelText("Architectures per generation"), { target: { value: "8" } });
      await waitFor(() => expect(launch).toBeDisabled());
      expect(screen.getByLabelText("Acknowledge campaign cost")).not.toBeChecked();
    });

    it("previews through the same endpoint with --genome architecture's own flags", async () => {
      await switchToArchitecture();
      fireEvent.change(screen.getByLabelText("Architectures per generation"), { target: { value: "2" } });
      fireEvent.change(screen.getByLabelText("Outer generations"), { target: { value: "2" } });
      fireEvent.change(screen.getByLabelText("Outer mutation rate"), { target: { value: "0.3" } });
      fireEvent.click(screen.getByRole("button", { name: "Dry-run preview" }));

      await waitFor(() => expect(posted).toHaveLength(1));
      expect(posted[0].route).toBe("/api/preview/search");
      expect(posted[0].body.options.genome).toBe("architecture");
      expect(posted[0].body.options.outer_schema).toBe("architecture_search_v1");
      expect(posted[0].body.options.outer_population_size).toBe(2);
      expect(posted[0].body.options.outer_populations).toBe(2);
      expect(posted[0].body.options.outer_mutation_rate).toBe(0.3);
      // The inner campaign keeps --schema/--population-size/--populations,
      // exactly as the CLI declares them.
      expect(posted[0].body.options.schema).toBe("generic_action_effects_v1");
      expect(posted[0].body.options.run_id_prefix).toBe("arch7");

      // The dry run's own estimate replaces the client-side one once it lands.
      await waitFor(() => expect(screen.getByTestId("cost-estimate").textContent).toContain("up to 72 full trainings"));
      expect(screen.getByText("arch7-arch0-a1")).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "backbone_preset" })).toBeInTheDocument();
      expect(screen.getByRole("columnheader", { name: "hidden_dim" })).toBeInTheDocument();
      expect(screen.queryByRole("columnheader", { name: "latent_width" })).not.toBeInTheDocument();
    });

    it("launches through /api/jobs/search and opens the nested architecture board", async () => {
      await switchToArchitecture();
      fireEvent.change(screen.getByLabelText("Architectures per generation"), { target: { value: "2" } });
      fireEvent.change(screen.getByLabelText("Inner population size"), { target: { value: "3" } });
      fireEvent.click(screen.getByLabelText("Acknowledge campaign cost"));
      await waitFor(() => expect(screen.getByRole("button", { name: "Launch campaign" })).toBeEnabled());
      fireEvent.click(screen.getByRole("button", { name: "Launch campaign" }));

      await waitFor(() => expect(posted).toHaveLength(1));
      expect(posted[0].route).toBe("/api/jobs/search");
      expect(posted[0].body.options.genome).toBe("architecture");

      await waitFor(() => expect(screen.getByText("Architecture board")).toBeInTheDocument());
      const board = screen.getByRole("region", { name: "Architecture board" });
      expect(board.textContent).toContain("2 architecture(s) × 2 outer generation(s)");
      expect(board.textContent).toContain("3 × 3 inner campaign");
      expect(screen.queryByText("Population board")).not.toBeInTheDocument();
    });
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
