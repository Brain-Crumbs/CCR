import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { JobsPanel } from "./JobsPanel.jsx";

const FRESH_MODE = "fresh";
const NON_FRESH_MODES = ["clone", "fine_tune", "resume"];

const DEFAULT_FORM = {
  corpusId: "",
  world: "crafter",
  trainScenarios: "",
  horizonsTicks: "1,2,3,4",
  backbone: "transformer",
  latentWidth: "128",
  hiddenDim: "256",
  actionEmbedDim: "8",
  reconstructionSize: "64",
  contextLength: "8",
  nHeads: "2",
  nLayers: "1",
  objective: "windowed_rollout",
  optimizerName: "adamw",
  lr: "0.0003",
  weightDecay: "0.00001",
  batchSize: "32",
  seed: "0",
  rolloutFrames: "8",
  warmupFrames: "4",
  scheduledSamplingP: "0",
  lossPixel: "1.0",
  lossLatent: "1.0",
  lossSemantic: "1.0",
  checkpointMetric: "validation_loss",
  checkpointMode: "min",
  device: "cpu",
  precision: "fp32",
  deterministic: true,
  selectionMetric: "rollout.t+4.model_over_copy_last_mse",
  gates: "rollout_health, representation_collapse",
  confidence: "0.95",
  parentRun: "",
  checkpoint: "best-validation.pt",
  namingSeed: "",
  exportPredictionsMax: "3",
  skipExportPredictions: false,
};

const MODE_HELP = {
  fresh: "Train a brand-new run from scratch against the chosen corpus.",
  clone: "Start a new, independent run from a completed run's checkpoint -- a controlled sibling with its own run id.",
  fine_tune: "Same as clone, but marks the lineage as a fine-tune of the parent rather than an independent sibling.",
  resume: "Reopen an interrupted run's own directory and continue training from its last checkpoint -- not a new run.",
};

function splitList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function buildFreshSpec(form, organism) {
  const horizons = splitList(form.horizonsTicks).map(Number).filter((n) => Number.isFinite(n) && n > 0);
  const backboneKwargs = form.backbone === "transformer"
    ? { n_heads: Number(form.nHeads), n_layers: Number(form.nLayers) }
    : {};
  return {
    format: "model-factory-spec-v1",
    organism,
    mode: FRESH_MODE,
    data: {
      corpus_id: form.corpusId,
      world: form.world,
      train_scenarios: splitList(form.trainScenarios),
      train_sessions: [],
      validation_sessions: [],
      test_sessions: [],
      expected_pixel_source: form.world,
      horizons_ticks: horizons.length ? horizons : [1, 2, 3, 4],
    },
    model: {
      backbone: form.backbone,
      latent_width: Number(form.latentWidth),
      hidden_dim: Number(form.hiddenDim),
      action_embed_dim: Number(form.actionEmbedDim),
      reconstruction_size: Number(form.reconstructionSize),
      context_length: Number(form.contextLength),
      backbone_kwargs: backboneKwargs,
    },
    training: {
      objective: form.objective,
      optimizer: { name: form.optimizerName, lr: Number(form.lr), weight_decay: Number(form.weightDecay) },
      batch_size: Number(form.batchSize),
      seed: Number(form.seed),
      rollout_frames: Number(form.rolloutFrames),
      warmup_frames: Number(form.warmupFrames),
      scheduled_sampling_p: Number(form.scheduledSamplingP),
      loss_weights: { pixel: Number(form.lossPixel), latent: Number(form.lossLatent), semantic: Number(form.lossSemantic) },
      checkpoint_selection_policy: { metric: form.checkpointMetric, mode: form.checkpointMode },
      device: form.device,
      precision: form.precision,
      determinism_policy: { deterministic: form.deterministic, seed: Number(form.seed) },
    },
    evaluation: {
      selection_metric: form.selectionMetric,
      gates: splitList(form.gates),
      confidence: Number(form.confidence),
    },
  };
}

function launchOptions(form) {
  return {
    naming_seed: form.namingSeed || null,
    export_predictions_max: Number(form.exportPredictionsMax),
    no_export_predictions: form.skipExportPredictions,
  };
}

/**
 * The Build tab (clinic redesign, Phase 3): a form over spec.model/data/
 * training/evaluation for launching one Model Factory trial, superseding
 * the notebook's hand-edited-dict workflow for this one action. Enums
 * (modes/objectives/backbones) come from `GET /api/factory-meta` rather
 * than being duplicated here; corpora from `GET /api/corpora`. Launch hands
 * off to the matching `POST /api/jobs/<kind>` control-plane route (Phase 2)
 * and the result is tracked in `JobsPanel`, not re-derived here.
 */
export function BuildPanel({ catalog, organism: initialOrganism, onOrganismCreated = null }) {
  const catalogOrganisms = catalog?.organisms || [];
  const [organism, setOrganism] = useState(initialOrganism ?? catalogOrganisms[0] ?? null);
  const organisms = useMemo(
    () => [...new Set([...catalogOrganisms, organism].filter(Boolean))].sort(),
    [catalogOrganisms, organism],
  );
  const [mode, setMode] = useState(FRESH_MODE);
  const [newOrganism, setNewOrganism] = useState("");
  const [creatingOrganism, setCreatingOrganism] = useState(false);
  const [organismError, setOrganismError] = useState(null);
  const [meta, setMeta] = useState(null);
  const [metaError, setMetaError] = useState(null);
  const [corpora, setCorpora] = useState(null);
  const [corpusSpecs, setCorpusSpecs] = useState(null);
  const [corpusSpecId, setCorpusSpecId] = useState("");
  const [corpusLaunching, setCorpusLaunching] = useState(false);
  const [corpusError, setCorpusError] = useState(null);
  const [factoryRuns, setFactoryRuns] = useState(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [overrides, setOverrides] = useState([]);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState(null);
  const [lastJobId, setLastJobId] = useState(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    if (organism || !organisms.length) return;
    setOrganism(organisms[0]);
  }, [organism, organisms]);

  useEffect(() => {
    if (initialOrganism && initialOrganism !== organism) setOrganism(initialOrganism);
  }, [initialOrganism]);

  useEffect(() => {
    let cancelled = false;
    api.factoryMeta().then((m) => !cancelled && setMeta(m), (e) => !cancelled && setMetaError(e.message));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!organism) return;
    let cancelled = false;
    setCorpora(null);
    async function poll() {
      try {
        const list = await api.corpora(organism);
        if (cancelled) return;
        setCorpora(list);
        setForm((current) => (list.some((item) => item.corpus_id === current.corpusId)
          ? current : { ...current, corpusId: list[0]?.corpus_id ?? "" }));
      } catch {
        if (!cancelled) setCorpora([]);
      }
    }
    poll();
    const interval = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [organism, refreshToken]);

  useEffect(() => {
    if (!organism) return;
    let cancelled = false;
    setCorpusSpecs(null);
    api.corpusSpecs().then((list) => {
      if (cancelled) return;
      setCorpusSpecs(list);
      setCorpusSpecId((current) => (list.some((item) => item.spec_id === current) ? current : (list[0]?.spec_id ?? "")));
    }, () => !cancelled && setCorpusSpecs([]));
    return () => { cancelled = true; };
  }, [organism]);

  useEffect(() => {
    if (!organism) return;
    let cancelled = false;
    async function poll() {
      api.factoryRuns(organism).then((r) => !cancelled && setFactoryRuns(r), () => !cancelled && setFactoryRuns(null));
    }
    poll();
    const interval = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [organism, refreshToken]);

  const runOptions = useMemo(() => (factoryRuns?.runs || []).map((r) => r.run), [factoryRuns]);
  useEffect(() => {
    setForm((current) => (runOptions.includes(current.parentRun) ? current : { ...current, parentRun: runOptions[0] ?? "" }));
  }, [runOptions]);

  const selectedRunState = factoryRuns?.runs?.find((r) => r.run === form.parentRun) ?? null;

  function set(field) {
    return (value) => setForm((current) => ({ ...current, [field]: value }));
  }

  function addOverride() {
    setOverrides((current) => [...current, { path: "", value: "" }]);
  }
  function updateOverride(index, key, value) {
    setOverrides((current) => current.map((row, i) => (i === index ? { ...row, [key]: value } : row)));
  }
  function removeOverride(index) {
    setOverrides((current) => current.filter((_, i) => i !== index));
  }

  async function handleLaunch(event) {
    event.preventDefault();
    if (!organism) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      let entry;
      if (mode === FRESH_MODE) {
        entry = await api.launchJob("baseline", {
          organism, spec: buildFreshSpec(form, organism), options: launchOptions(form),
        });
      } else if (mode === "resume") {
        if (!form.parentRun) throw new Error("select a run to resume");
        // `ccr factory resume` has no --naming-seed flag (unlike baseline/
        // clone); naming_seed is always null here since the UI only shows
        // that field for fresh, so it's dropped by flagsFromOptions anyway.
        entry = await api.launchJob("resume", {
          organism, run: form.parentRun,
          options: { organism, ...launchOptions(form) },
        });
      } else {
        if (!form.parentRun) throw new Error("select a parent run");
        const setOverridesArg = overrides
          .filter((row) => row.path && row.value !== "")
          .map((row) => `${row.path}=${row.value}`);
        entry = await api.launchJob("clone", {
          organism, run: form.parentRun,
          options: {
            mode, organism, checkpoint: form.checkpoint,
            set: setOverridesArg.length ? setOverridesArg : null,
            ...launchOptions(form),
          },
        });
      }
      setLastJobId(entry.job_id);
      setRefreshToken((n) => n + 1);
    } catch (err) {
      setLaunchError(err.message);
    } finally {
      setLaunching(false);
    }
  }

  async function handleCorpusBuild() {
    if (!organism || !corpusSpecId) return;
    setCorpusLaunching(true);
    setCorpusError(null);
    try {
      const entry = await api.launchJob("corpus", { organism, spec_id: corpusSpecId });
      setLastJobId(entry.job_id);
      setRefreshToken((n) => n + 1);
    } catch (err) {
      setCorpusError(err.message);
    } finally {
      setCorpusLaunching(false);
    }
  }

  async function handleCreateOrganism() {
    const requested = newOrganism.trim();
    if (!requested) return;
    setCreatingOrganism(true);
    setOrganismError(null);
    try {
      const result = await api.createOrganism(requested);
      setOrganism(result.organism);
      setNewOrganism("");
      onOrganismCreated?.(result.organism);
    } catch (err) {
      setOrganismError(err.message);
    } finally {
      setCreatingOrganism(false);
    }
  }

  const organismCreator = (
    <>
      <div className="organism-create">
        <label className="field">New organism
          <input
            value={newOrganism}
            onChange={(event) => setNewOrganism(event.target.value)}
            placeholder="MyOrganism"
            aria-label="New organism name"
          />
        </label>
        <button type="button" onClick={handleCreateOrganism} disabled={creatingOrganism || !newOrganism.trim()}>
          {creatingOrganism ? "Creating…" : "Create organism"}
        </button>
      </div>
      {organismError && <p className="no-data">{organismError}</p>}
    </>
  );

  if (!organisms.length) {
    return (
      <section className="diagnostic build" aria-labelledby="build-title">
        <h3 id="build-title">Build</h3>
        <p>Create an organism namespace to begin building from the interface.</p>
        {organismCreator}
      </section>
    );
  }

  return (
    <section className="diagnostic build" aria-labelledby="build-title">
      <h3 id="build-title">Build</h3>
      <p>Configure and launch one Model Factory trial. Progress and logs appear in the Jobs panel below.</p>
      {metaError && <p className="no-data">factory metadata unavailable: {metaError}</p>}

      <form className="build-form" onSubmit={handleLaunch}>
        <div className="pickers">
          <Picker label="Organism" ariaLabel="Build organism" value={organism} options={organisms} onChange={setOrganism} />
          <Picker
            label="Mode" ariaLabel="Build mode" value={mode}
            options={meta?.modes || [FRESH_MODE, ...NON_FRESH_MODES]} onChange={setMode}
          />
        </div>
        {organismCreator}
        <p className="build-form__help">{MODE_HELP[mode]}</p>

        {mode === FRESH_MODE ? (
          <>
            <h4>Data</h4>
            {corpora?.length === 0 && (
              <div className="corpus-bootstrap">
                <strong>No frozen corpus is available for {organism} yet.</strong>
                <p>
                  Build one from a repository corpus spec here. This records and quality-gates the data needed
                  by both individual trials and evolutionary searches; progress appears in Jobs.
                </p>
                <div className="pickers">
                  <label className="picker">Corpus recipe
                    <select value={corpusSpecId} onChange={(event) => setCorpusSpecId(event.target.value)} aria-label="Corpus recipe">
                      {!corpusSpecs?.length && <option value="">{corpusSpecs ? "no corpus recipes found" : "loading…"}</option>}
                      {corpusSpecs?.map((item) => (
                        <option key={item.spec_id} value={item.spec_id}>{item.corpus_id}</option>
                      ))}
                    </select>
                  </label>
                  <button type="button" onClick={handleCorpusBuild} disabled={corpusLaunching || !corpusSpecId}>
                    {corpusLaunching ? "Starting corpus build…" : "Build corpus"}
                  </button>
                </div>
                {corpusError && <p className="no-data">{corpusError}</p>}
              </div>
            )}
            <div className="field-grid">
              <label className="field">Corpus
                <select value={form.corpusId} onChange={(e) => set("corpusId")(e.target.value)} aria-label="Corpus">
                  {!corpora?.length && <option value="">{corpora ? "no corpora found" : "loading…"}</option>}
                  {corpora?.map((c) => <option key={c.corpus_id} value={c.corpus_id}>{c.corpus_id}</option>)}
                </select>
              </label>
              <label className="field">World<input value={form.world} onChange={(e) => set("world")(e.target.value)} /></label>
              <label className="field">Train scenarios (comma-separated)
                <input value={form.trainScenarios} onChange={(e) => set("trainScenarios")(e.target.value)} placeholder="motor_babbling_open, approach_entity" />
              </label>
              <label className="field">Horizons (ticks)<input value={form.horizonsTicks} onChange={(e) => set("horizonsTicks")(e.target.value)} /></label>
            </div>

            <h4>Model</h4>
            <div className="field-grid">
              <label className="field">Backbone
                <select value={form.backbone} onChange={(e) => set("backbone")(e.target.value)} aria-label="Backbone">
                  {(meta?.backbones || ["gru", "dilated_conv", "transformer"]).map((b) => <option key={b} value={b}>{b}</option>)}
                </select>
              </label>
              {form.backbone === "transformer" && (
                <>
                  <label className="field">Heads<input type="number" min="1" value={form.nHeads} onChange={(e) => set("nHeads")(e.target.value)} /></label>
                  <label className="field">Layers<input type="number" min="1" value={form.nLayers} onChange={(e) => set("nLayers")(e.target.value)} /></label>
                </>
              )}
              <label className="field">Latent width<input type="number" value={form.latentWidth} onChange={(e) => set("latentWidth")(e.target.value)} /></label>
              <label className="field">Hidden dim<input type="number" value={form.hiddenDim} onChange={(e) => set("hiddenDim")(e.target.value)} /></label>
              <label className="field">Action embed dim<input type="number" value={form.actionEmbedDim} onChange={(e) => set("actionEmbedDim")(e.target.value)} /></label>
              <label className="field">Reconstruction size<input type="number" value={form.reconstructionSize} onChange={(e) => set("reconstructionSize")(e.target.value)} /></label>
              <label className="field">Context length<input type="number" value={form.contextLength} onChange={(e) => set("contextLength")(e.target.value)} /></label>
            </div>

            <h4>Training</h4>
            <div className="field-grid">
              <label className="field">Objective
                <select value={form.objective} onChange={(e) => set("objective")(e.target.value)} aria-label="Objective">
                  {(meta?.objectives || ["windowed_rollout"]).map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </label>
              <label className="field">Optimizer<input value={form.optimizerName} onChange={(e) => set("optimizerName")(e.target.value)} /></label>
              <label className="field">Learning rate<input type="number" step="any" value={form.lr} onChange={(e) => set("lr")(e.target.value)} /></label>
              <label className="field">Weight decay<input type="number" step="any" value={form.weightDecay} onChange={(e) => set("weightDecay")(e.target.value)} /></label>
              <label className="field">Batch size<input type="number" value={form.batchSize} onChange={(e) => set("batchSize")(e.target.value)} /></label>
              <label className="field">Seed<input type="number" value={form.seed} onChange={(e) => set("seed")(e.target.value)} /></label>
              <label className="field">Rollout frames<input type="number" value={form.rolloutFrames} onChange={(e) => set("rolloutFrames")(e.target.value)} /></label>
              <label className="field">Warmup frames<input type="number" value={form.warmupFrames} onChange={(e) => set("warmupFrames")(e.target.value)} /></label>
              <label className="field">Scheduled sampling p<input type="number" step="any" value={form.scheduledSamplingP} onChange={(e) => set("scheduledSamplingP")(e.target.value)} /></label>
              <label className="field">Loss: pixel<input type="number" step="any" value={form.lossPixel} onChange={(e) => set("lossPixel")(e.target.value)} /></label>
              <label className="field">Loss: latent<input type="number" step="any" value={form.lossLatent} onChange={(e) => set("lossLatent")(e.target.value)} /></label>
              <label className="field">Loss: semantic<input type="number" step="any" value={form.lossSemantic} onChange={(e) => set("lossSemantic")(e.target.value)} /></label>
              <label className="field">Device
                <select value={form.device} onChange={(e) => set("device")(e.target.value)} aria-label="Device">
                  <option value="cpu">cpu</option>
                  <option value="cuda">cuda</option>
                </select>
              </label>
              <label className="field">Precision
                <select value={form.precision} onChange={(e) => set("precision")(e.target.value)} aria-label="Precision">
                  <option value="fp32">fp32</option>
                  <option value="fp16">fp16</option>
                  <option value="bf16">bf16</option>
                </select>
              </label>
            </div>

            <h4>Evaluation</h4>
            <div className="field-grid">
              <label className="field">Selection metric<input value={form.selectionMetric} onChange={(e) => set("selectionMetric")(e.target.value)} /></label>
              <label className="field">Gates (comma-separated)<input value={form.gates} onChange={(e) => set("gates")(e.target.value)} /></label>
              <label className="field">Confidence<input type="number" step="any" value={form.confidence} onChange={(e) => set("confidence")(e.target.value)} /></label>
            </div>
          </>
        ) : (
          <>
            <h4>Parent run</h4>
            <div className="field-grid">
              <label className="field">Run
                <select value={form.parentRun} onChange={(e) => set("parentRun")(e.target.value)} aria-label="Parent run">
                  {!runOptions.length && <option value="">{factoryRuns ? "no runs found" : "loading…"}</option>}
                  {runOptions.map((run) => <option key={run} value={run}>{run}</option>)}
                </select>
              </label>
              {selectedRunState && (
                <p className="build-form__hint">state: {selectedRunState.state || "unknown"} · mode: {selectedRunState.mode || "–"}</p>
              )}
              {mode !== "resume" && (
                <label className="field">Checkpoint<input value={form.checkpoint} onChange={(e) => set("checkpoint")(e.target.value)} /></label>
              )}
            </div>

            {mode !== "resume" && (
              <>
                <h4>Overrides</h4>
                <div className="overrides">
                  {overrides.map((row, i) => (
                    <div className="overrides__row" key={i}>
                      <input
                        aria-label={`override ${i + 1} path`} placeholder="training.seed"
                        value={row.path} onChange={(e) => updateOverride(i, "path", e.target.value)}
                      />
                      <span>=</span>
                      <input
                        aria-label={`override ${i + 1} value`} placeholder="1"
                        value={row.value} onChange={(e) => updateOverride(i, "value", e.target.value)}
                      />
                      <button type="button" onClick={() => removeOverride(i)} aria-label={`remove override ${i + 1}`}>×</button>
                    </div>
                  ))}
                  <button type="button" onClick={addOverride}>+ Add override</button>
                </div>
              </>
            )}
          </>
        )}

        <h4>Launch options</h4>
        <div className="field-grid">
          {mode !== "resume" && <label className="field">Naming seed (optional)<input value={form.namingSeed} onChange={(e) => set("namingSeed")(e.target.value)} /></label>}
          <label className="field">Export predictions (max episodes)<input type="number" min="0" value={form.exportPredictionsMax} onChange={(e) => set("exportPredictionsMax")(e.target.value)} /></label>
          <label className="field field--checkbox">
            <input type="checkbox" checked={form.skipExportPredictions} onChange={(e) => set("skipExportPredictions")(e.target.checked)} />
            Skip prediction export
          </label>
        </div>

        {launchError && <p className="no-data">{launchError}</p>}
        <button
          type="submit"
          disabled={launching || !organism || (mode === FRESH_MODE ? !form.corpusId : !form.parentRun)}
        >
          {launching ? "Launching…" : "Launch"}
        </button>
      </form>

      <JobsPanel organism={organism} refreshToken={refreshToken} highlightJobId={lastJobId} />
    </section>
  );
}
