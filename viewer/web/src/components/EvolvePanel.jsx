import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { JobsPanel } from "./JobsPanel.jsx";
import { PopulationBoard } from "./PopulationBoard.jsx";
import { ArchitectureBoard } from "./ArchitectureBoard.jsx";

const DEFAULT_FORM = {
  baseRun: "",
  schema: "generic_action_effects_v1",
  outerSchema: "architecture_search_v1",
  outerPopulationSize: "4",
  outerPopulationCount: "2",
  outerMutationRate: "0.2",
  populationSize: "6",
  populationCount: "3",
  mutationRate: "0.2",
  seed: "7",
  method: "lhs",
  runIdPrefix: "",
  selectionMetric: "",
  champion: "",
  referenceRun: "",
  referenceCheckpoint: "best-validation.pt",
  stageBudgetSeconds: "",
  minEpisodeLength: "",
  namingSeed: "",
};

/** `run_evolutionary_search`'s own default campaign prefix, mirrored so the
 * population board's precomputed `{prefix}-p{n}-c{i}` ids match what the
 * campaign actually writes even when the field is left untouched. Always
 * sent explicitly rather than relied on, since `jobs.buildLaunch` would
 * otherwise substitute an opaque `job-<uuid>` prefix of its own. */
function defaultPrefix(seed, genome = "training") {
  return genome === "architecture" ? `arch${Number(seed) || 0}` : `evo${Number(seed) || 0}`;
}

/**
 * A base spec for the campaign, taken from a run's own persisted
 * `trial_spec.json` -- the same "read the parent's resolved spec back"
 * move `ccr factory clone` makes, and the reason this panel needs no
 * duplicate of BuildPanel's whole model/data/training form.
 *
 * `mode` and `parent` carry through deliberately: `propose()` varies only
 * `training`, holding organism/mode/parent/data/model/evaluation fixed, so
 * a clone-mode base means a fixed-parent campaign (epic §13.1) and a
 * fresh-mode base means every candidate trains from scratch.
 *
 * `evolution` is the one block deliberately dropped. It records *this run's*
 * breeding provenance; carried into a proposal it would attribute that
 * run's configuration parents to a candidate that is not its child.
 */
function baseSpecFrom(trialSpec, organism) {
  const { evolution: _evolution, ...rest } = trialSpec || {};
  return { ...rest, organism };
}

function flattenLeaves(value, prefix = "", out = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, nested] of Object.entries(value)) {
      flattenLeaves(nested, prefix ? `${prefix}.${key}` : key, out);
    }
  } else if (prefix) {
    out[prefix] = value;
  }
  return out;
}

/** The dotted `training.*` paths that actually differ across a dry run's
 * proposals -- i.e. the schema's genes, as *observed* rather than as
 * re-declared here. A gene that is inactive for the base spec's objective
 * (genome's `active_objectives`) is repaired to one shared value across
 * every candidate and so correctly never shows up as a column. */
export function varyingGenePaths(candidates) {
  const flattened = (candidates || []).map((candidate) => flattenLeaves(candidate.spec?.training || {}));
  if (flattened.length < 2) return [];
  return [...new Set(flattened.flatMap(Object.keys))]
    .sort()
    .filter((path) => new Set(flattened.map((leaves) => JSON.stringify(leaves[path]))).size > 1);
}

function geneValue(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : Number(value.toPrecision(4)).toString();
  }
  return value === null || value === undefined ? "–" : String(value);
}

/** The campaign's worst-case training count, stated as a bound rather than
 * a figure: generation 0 trains every candidate, and each later generation
 * trains only the offspring that replace non-survivors. `survivor_limit` is
 * at least 1 and at most `population_size // 2`, so a later generation
 * trains at most `population_size - 1` candidates -- and fewer, or none at
 * all, if quality filtering eliminates candidates or the stage budget runs
 * out first. */
export function estimateMaxTrials(populationSize, populationCount) {
  const size = Math.max(0, Number(populationSize) || 0);
  const count = Math.max(0, Number(populationCount) || 0);
  if (!size || !count) return 0;
  return size + (count - 1) * Math.max(0, size - 1);
}

/** The nested campaign's worst-case training count, mirroring
 * `architecture_search.estimate_cost` exactly -- the honest multiplicative
 * bound, deliberately *not* discounted for retained architectures or
 * carried inner survivors, because a bound that can be exceeded is not a
 * guardrail. Computed client-side so the number is on screen while the form
 * is still being edited; the same figure comes back authoritatively on the
 * dry run's `estimate`, which is what the confirm step quotes. */
export function estimateArchitectureTrials(
  outerPopulationSize, outerPopulationCount, innerPopulationSize, innerPopulationCount,
) {
  return [outerPopulationSize, outerPopulationCount, innerPopulationSize, innerPopulationCount]
    .reduce((product, value) => product * Math.max(0, Number(value) || 0), 1);
}

/** The architecture genes that actually differ across a dry run's sampled
 * architectures -- the outer analogue of `varyingGenePaths`. Genes are flat
 * by construction here (`architecture_genome`'s allowlist is one level
 * deep, with `backbone_preset` a single composite gene), so this needs no
 * nested flattening. */
export function varyingGenomeGenes(candidates) {
  const genomes = (candidates || []).map((candidate) => candidate.genome || {});
  if (genomes.length < 2) return [];
  return [...new Set(genomes.flatMap(Object.keys))]
    .sort()
    .filter((gene) => new Set(genomes.map((genome) => JSON.stringify(genome[gene]))).size > 1);
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return null;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  const hours = seconds / 3600;
  return hours < 48 ? `${hours.toFixed(1)} h` : `${(hours / 24).toFixed(1)} days`;
}

/** The organism's current leading champion, if any -- the natural default
 * for "compare every candidate against the model we already trust"
 * (evaluation.reference_run, Phase 1). Reads `GET /api/registry`'s own slot
 * structure the way ChampionRegistryPanel flattens it. */
function leadingChampion(registry) {
  for (const tiers of Object.values(registry?.slots || {})) {
    for (const objectives of Object.values(tiers || {})) {
      for (const slot of Object.values(objectives || {})) {
        if (slot?.leading_champion) return slot.leading_champion;
      }
    }
  }
  return null;
}

/**
 * The Evolve tab, hyperparameter mode (clinic redesign, Phase 4): configure
 * a `ccr factory search` evolutionary campaign, dry-run it to see the exact
 * proposed candidates before spending anything, launch it, and watch the
 * population evolve generation by generation.
 *
 * Nothing about proposal, selection, or breeding is reimplemented here. The
 * preview is the CLI's own `--dry-run` (`POST /api/preview/search` shells it
 * out; it is torch-free and resolves real candidate specs through
 * `propose()`), the launch is the same CLI subcommand a human would type,
 * and the population board reads back only what the campaign wrote to disk.
 */
export function EvolvePanel({ catalog, organism: initialOrganism }) {
  const organisms = catalog?.organisms || [];
  const [organism, setOrganism] = useState(initialOrganism ?? organisms[0] ?? null);
  // "training" varies the training genes against one fixed architecture;
  // "architecture" is Phase 6's nested NAS campaign, whose *inner* step is
  // an unchanged training-gene campaign per architecture. The two schema
  // registries are deliberately separate on the backend, so the mode
  // switches which one the form offers rather than merging them.
  const [genome, setGenome] = useState("training");
  const [costAcknowledged, setCostAcknowledged] = useState(false);
  const [meta, setMeta] = useState(null);
  const [factoryRuns, setFactoryRuns] = useState(null);
  const [registry, setRegistry] = useState(null);
  const [baseSpec, setBaseSpec] = useState(null);
  const [baseSpecError, setBaseSpecError] = useState(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [overrides, setOverrides] = useState([]);
  const [preview, setPreview] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState(null);
  const [launching, setLaunching] = useState(false);
  const [campaign, setCampaign] = useState(null);
  const [lastJobId, setLastJobId] = useState(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    if (organism || !organisms.length) return;
    setOrganism(organisms[0]);
  }, [organism, organisms]);

  useEffect(() => {
    let cancelled = false;
    api.factoryMeta().then((m) => !cancelled && setMeta(m), () => !cancelled && setMeta(null));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!organism) return undefined;
    let cancelled = false;
    setFactoryRuns(null);
    setRegistry(null);
    api.factoryRuns(organism).then((r) => !cancelled && setFactoryRuns(r), () => !cancelled && setFactoryRuns(null));
    api.registry(organism).then((r) => !cancelled && setRegistry(r), () => !cancelled && setRegistry(null));
    return () => { cancelled = true; };
  }, [organism]);

  // Only a run that reached a terminal, successful state has the resolved
  // spec (and, for the reference/champion pickers, the checkpoint) this
  // campaign would build on.
  const completedRuns = useMemo(
    () => (factoryRuns?.runs || []).filter((run) => run.state === "completed").map((run) => run.run),
    [factoryRuns],
  );

  useEffect(() => {
    setForm((current) => (completedRuns.includes(current.baseRun)
      ? current
      : { ...current, baseRun: completedRuns[0] ?? "" }));
  }, [completedRuns]);

  // The reference run defaults to the organism's leading champion, so the
  // out-of-the-box campaign selects against "beats the model we already
  // trust" rather than only against copy-last-frame -- but stays a plain
  // editable choice, including "none".
  //
  // Only when that champion is one of the runs the picker actually offers:
  // the registry can name a champion this organism's run catalog no longer
  // lists, and a select whose value is absent from its own options renders
  // as something other than what the form state says it will submit.
  useEffect(() => {
    const champion = leadingChampion(registry);
    if (!champion || !completedRuns.includes(champion)) return;
    setForm((current) => (current.referenceRun ? current : { ...current, referenceRun: champion }));
  }, [registry, completedRuns]);

  useEffect(() => {
    if (!organism || !form.baseRun) { setBaseSpec(null); return undefined; }
    let cancelled = false;
    setBaseSpec(null);
    setBaseSpecError(null);
    api.experimentArtifacts(organism, form.baseRun).then((artifacts) => {
      if (cancelled) return;
      const trialSpec = artifacts?.trial_spec;
      if (!trialSpec) {
        setBaseSpecError(`${form.baseRun} has no trial_spec.json to build a campaign from`);
        return;
      }
      setBaseSpec(baseSpecFrom(trialSpec, organism));
    }, (err) => !cancelled && setBaseSpecError(err.message));
    return () => { cancelled = true; };
  }, [organism, form.baseRun]);

  // The base spec's own selection metric is the campaign's unless edited;
  // editing it sends one `--set evaluation.selection_metric=...` rather
  // than a second, divergent copy of the whole evaluation block.
  const baseSelectionMetric = baseSpec?.evaluation?.selection_metric ?? "";
  useEffect(() => {
    setForm((current) => ({ ...current, selectionMetric: baseSelectionMetric }));
  }, [baseSelectionMetric]);

  const architecture = genome === "architecture";
  const prefix = form.runIdPrefix || defaultPrefix(form.seed, genome);
  const selectionMetric = form.selectionMetric || baseSelectionMetric;
  const maxTrials = architecture
    ? estimateArchitectureTrials(
      form.outerPopulationSize, form.outerPopulationCount, form.populationSize, form.populationCount,
    )
    : estimateMaxTrials(form.populationSize, form.populationCount);
  // The per-trial wall clock the campaign's own cost bound multiplies. Read
  // off the base spec rather than re-declared, exactly as `estimate_cost`
  // reads `training.max_training_seconds`; a spec that declares none has
  // genuinely no bound to report, which is itself worth showing.
  const perTrialSeconds = baseSpec?.training?.max_training_seconds ?? null;
  const maxSeconds = typeof perTrialSeconds === "number" ? perTrialSeconds * maxTrials : null;
  // Everything the quoted cost depends on. A dry run's estimate is tagged
  // with this at the moment it is requested, so it can only ever be quoted
  // back for the configuration it actually describes -- and the
  // acknowledgement below is withdrawn on the same signal.
  const costKey = JSON.stringify([
    architecture, form.outerPopulationSize, form.outerPopulationCount,
    form.populationSize, form.populationCount, perTrialSeconds,
  ]);

  function set(field) {
    return (value) => setForm((current) => ({ ...current, [field]: value }));
  }

  // Any change to what the campaign would cost withdraws the
  // acknowledgement: a confirm that survives an edit from 24 trials to 240
  // is not a confirm of anything.
  useEffect(() => { setCostAcknowledged(false); }, [costKey, form.baseRun]);

  function specOverrides() {
    const rows = overrides
      .filter((row) => row.path && row.value !== "")
      .map((row) => `${row.path}=${row.value}`);
    if (form.selectionMetric && form.selectionMetric !== baseSelectionMetric) {
      rows.push(`evaluation.selection_metric=${form.selectionMetric}`);
    }
    return rows;
  }

  function searchBody() {
    const setArgs = specOverrides();
    // Architecture mode is the same `ccr factory search` subcommand with
    // `--genome architecture` plus its own `--outer-*` flags; --schema and
    // --population-size/--populations keep their meaning as the *inner*
    // campaign's parameters, exactly as the CLI declares them. Nothing about
    // the nested engine is reimplemented here.
    const architectureOptions = architecture ? {
      genome: "architecture",
      outer_schema: form.outerSchema,
      outer_population_size: Number(form.outerPopulationSize),
      // argparse declares this as `--outer-populations` (dest
      // outer_population_count), mirroring `--populations`.
      outer_populations: Number(form.outerPopulationCount),
      outer_mutation_rate: Number(form.outerMutationRate),
    } : {};
    return {
      organism,
      spec: baseSpec,
      options: {
        ...architectureOptions,
        schema: form.schema,
        population_size: Number(form.populationSize),
        // argparse declares this as `--populations` (dest population_count).
        populations: Number(form.populationCount),
        mutation_rate: Number(form.mutationRate),
        seed: Number(form.seed),
        method: form.method,
        run_id_prefix: prefix,
        champion: form.champion || null,
        reference_run: form.referenceRun || null,
        reference_checkpoint: form.referenceRun ? form.referenceCheckpoint : null,
        stage_budget_seconds: form.stageBudgetSeconds === "" ? null : Number(form.stageBudgetSeconds),
        min_episode_length: form.minEpisodeLength === "" ? null : Number(form.minEpisodeLength),
        naming_seed: form.namingSeed || null,
        set: setArgs.length ? setArgs : null,
      },
    };
  }

  async function handlePreview() {
    if (!baseSpec) return;
    setPreviewing(true);
    setError(null);
    try {
      // Tagged with the cost configuration in force when the request was
      // sent, never the one in force when the response lands: an edit made
      // while the dry run was in flight must invalidate its estimate too.
      const requestedCostKey = costKey;
      setPreview({ ...await api.previewJob("search", searchBody()), costKey: requestedCostKey });
    } catch (err) {
      setPreview(null);
      setError(err.message);
    } finally {
      setPreviewing(false);
    }
  }

  async function handleLaunch(event) {
    event.preventDefault();
    if (!baseSpec) return;
    setLaunching(true);
    setError(null);
    try {
      const entry = await api.launchJob("search", searchBody());
      setLastJobId(entry.job_id);
      setCampaign({
        genome,
        prefix,
        populationSize: Number(form.populationSize),
        populationCount: Number(form.populationCount),
        outerPopulationSize: Number(form.outerPopulationSize),
        outerPopulationCount: Number(form.outerPopulationCount),
        selectionMetric,
      });
      setRefreshToken((token) => token + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLaunching(false);
    }
  }

  if (!organisms.length) {
    return (
      <section className="diagnostic evolve" aria-labelledby="evolve-title">
        <h3 id="evolve-title">Evolve</h3>
        <p className="no-data">no organisms found -- launch at least one run from the Build tab first</p>
      </section>
    );
  }

  const geneColumns = varyingGenePaths(preview?.candidates);
  const architectureGeneColumns = varyingGenomeGenes(preview?.candidates);
  // The dry run's own estimate wins over the client-side one -- but only
  // while the form still describes the campaign that estimate was computed
  // for. Otherwise editing the population fields after a preview would leave
  // the gate quoting the old figure while `searchBody()` launches the new
  // one, so a user could acknowledge 72 trainings and start 720 (Codex
  // review, PR #281).
  const previewEstimate = preview?.estimate && preview.costKey === costKey ? preview.estimate : null;
  const quotedTrials = previewEstimate?.estimated_max_trials ?? maxTrials;
  const quotedSeconds = previewEstimate ? previewEstimate.estimated_max_seconds : maxSeconds;
  const launchBlocked = architecture && !costAcknowledged;

  return (
    <>
      <section className="diagnostic evolve" aria-labelledby="evolve-title">
        <h3 id="evolve-title">Evolve</h3>
        <p>
          {architecture ? (
            <>
              Run a nested architecture campaign: an outer evolutionary loop over model architectures, each
              scored by a complete inner training-gene campaign of its own. An architecture-changing child
              trains from scratch — no weights cross architectures.
            </>
          ) : (
            <>
              Run a bounded evolutionary campaign: propose a population of sibling trials that vary only the
              declared genome&apos;s training genes, keep the best half, and breed the next generation from them.
            </>
          )}
        </p>

        <div className="evolve-mode" role="group" aria-label="Search axis">
          <button
            type="button" aria-pressed={!architecture}
            className={architecture ? "" : "is-active"}
            onClick={() => setGenome("training")}
          >
            Hyperparameters
          </button>
          <button
            type="button" aria-pressed={architecture}
            className={architecture ? "is-active" : ""}
            onClick={() => setGenome("architecture")}
          >
            Architecture (NAS)
          </button>
        </div>

        <form className="build-form" onSubmit={handleLaunch}>
          <div className="pickers">
            <Picker label="Organism" ariaLabel="Evolve organism" value={organism} options={organisms} onChange={setOrganism} />
          </div>

          <h4>Base spec</h4>
          <p className="build-form__help">
            {architecture ? (
              <>
                Every architecture inherits this run&apos;s data and evaluation blocks unchanged; the outer
                genome overwrites its <code>model</code> block, and each architecture&apos;s inner campaign
                then varies the training genes beneath it. Every candidate runs <code>mode=&quot;fresh&quot;</code>
                with no parent, whatever the base run&apos;s own mode.
              </>
            ) : (
              <>
                Every candidate inherits this run&apos;s data, model, and evaluation blocks unchanged; only the
                genome&apos;s training genes vary.
              </>
            )}
          </p>
          <div className="field-grid">
            <label className="field">Base run
              <select value={form.baseRun} onChange={(e) => set("baseRun")(e.target.value)} aria-label="Base run">
                {!completedRuns.length && <option value="">{factoryRuns ? "no completed runs found" : "loading…"}</option>}
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
            <label className="field">{architecture ? "Inner genome schema" : "Genome schema"}
              <select
                value={form.schema} onChange={(e) => set("schema")(e.target.value)}
                aria-label={architecture ? "Inner genome schema" : "Genome schema"}
              >
                {(meta?.genome_schemas || [DEFAULT_FORM.schema]).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            {architecture && (
              <label className="field">Architecture schema
                <select value={form.outerSchema} onChange={(e) => set("outerSchema")(e.target.value)} aria-label="Architecture schema">
                  {(meta?.architecture_genome_schemas || [DEFAULT_FORM.outerSchema])
                    .map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
            )}
            <label className="field">Selection metric
              <input value={form.selectionMetric} onChange={(e) => set("selectionMetric")(e.target.value)} aria-label="Selection metric" />
            </label>
          </div>
          {baseSpecError && <p className="no-data">{baseSpecError}</p>}
          {baseSpec && (
            <p className="build-form__hint">
              base mode: {baseSpec.mode || "–"} · corpus: {baseSpec.data?.corpus_id || "–"} · objective:{" "}
              {baseSpec.training?.objective || "–"}
              {baseSpec.parent?.run_id ? ` · fixed parent: ${baseSpec.parent.run_id}` : ""}
            </p>
          )}

          {architecture && (
            <>
              <h4>Outer loop (architectures)</h4>
              <div className="field-grid">
                <label className="field">Architectures per generation
                  <input type="number" min="2" value={form.outerPopulationSize} onChange={(e) => set("outerPopulationSize")(e.target.value)} aria-label="Architectures per generation" />
                </label>
                <label className="field">Outer generations
                  <input type="number" min="1" value={form.outerPopulationCount} onChange={(e) => set("outerPopulationCount")(e.target.value)} aria-label="Outer generations" />
                </label>
                <label className="field">Outer mutation rate
                  <input type="number" step="any" min="0" max="1" value={form.outerMutationRate} onChange={(e) => set("outerMutationRate")(e.target.value)} aria-label="Outer mutation rate" />
                </label>
              </div>
            </>
          )}

          <h4>{architecture ? "Inner loop (training genes, per architecture)" : "Campaign"}</h4>
          <div className="field-grid">
            <label className="field">{architecture ? "Inner population size" : "Population size"}
              <input
                type="number" min="2" value={form.populationSize}
                onChange={(e) => set("populationSize")(e.target.value)}
                aria-label={architecture ? "Inner population size" : "Population size"}
              />
            </label>
            <label className="field">{architecture ? "Inner generations" : "Generations"}
              <input
                type="number" min="1" value={form.populationCount}
                onChange={(e) => set("populationCount")(e.target.value)}
                aria-label={architecture ? "Inner generations" : "Generations"}
              />
            </label>
            <label className="field">{architecture ? "Inner mutation rate" : "Mutation rate"}
              <input
                type="number" step="any" min="0" max="1" value={form.mutationRate}
                onChange={(e) => set("mutationRate")(e.target.value)}
                aria-label={architecture ? "Inner mutation rate" : "Mutation rate"}
              />
            </label>
            <label className="field">Seed
              <input type="number" value={form.seed} onChange={(e) => set("seed")(e.target.value)} aria-label="Seed" />
            </label>
            <label className="field">Proposal method
              <select value={form.method} onChange={(e) => set("method")(e.target.value)} aria-label="Proposal method">
                <option value="lhs">lhs</option>
                <option value="random">random</option>
              </select>
            </label>
            <label className="field">Run id prefix
              <input value={form.runIdPrefix} onChange={(e) => set("runIdPrefix")(e.target.value)} placeholder={defaultPrefix(form.seed, genome)} aria-label="Run id prefix" />
            </label>
            <label className="field">Stage budget (seconds, optional)
              <input type="number" step="any" value={form.stageBudgetSeconds} onChange={(e) => set("stageBudgetSeconds")(e.target.value)} aria-label="Stage budget seconds" />
            </label>
            <label className="field">Min episode length (optional)
              <input type="number" value={form.minEpisodeLength} onChange={(e) => set("minEpisodeLength")(e.target.value)} aria-label="Min episode length" />
            </label>
            <label className="field">Naming seed (optional)
              <input value={form.namingSeed} onChange={(e) => set("namingSeed")(e.target.value)} aria-label="Naming seed" />
            </label>
          </div>
          {architecture ? (
            <div className="evolve-cost" data-testid="cost-estimate">
              <h4>Cost</h4>
              <p className="evolve-cost__figure">
                up to <strong>{quotedTrials}</strong> full trainings
                {quotedSeconds !== null && quotedSeconds !== undefined
                  ? <> — <strong>{formatDuration(quotedSeconds)}</strong> of training</>
                  : null}
              </p>
              <p className="evolve-cost__breakdown">
                {form.outerPopulationSize} architecture(s) × {form.outerPopulationCount} outer generation(s) ×{" "}
                {form.populationSize} candidate(s) × {form.populationCount} inner generation(s). Every
                architecture gets a complete inner campaign before it is scored, so the two budgets
                multiply. Campaign runs land under{" "}
                <code>{prefix}-arch&lt;gen&gt;-a&lt;arch&gt;-p&lt;gen&gt;-c&lt;candidate&gt;</code>.
              </p>
              {quotedSeconds === null || quotedSeconds === undefined ? (
                <p className="evolve-cost__breakdown">
                  This base spec declares no <code>training.max_training_seconds</code>, so the campaign has no
                  wall-clock bound at all — only the trial count above.
                </p>
              ) : null}
              <label className="evolve-cost__confirm">
                <input
                  type="checkbox" checked={costAcknowledged}
                  onChange={(e) => setCostAcknowledged(e.target.checked)}
                  aria-label="Acknowledge campaign cost"
                />
                I understand this launches up to {quotedTrials} trainings and cannot be resumed once cancelled.
              </label>
            </div>
          ) : (
            <p className="build-form__hint" data-testid="cost-estimate">
              at most {maxTrials} training run(s): {form.populationSize} in the first generation, then up to{" "}
              {Math.max(0, Number(form.populationSize) - 1)} bred offspring per later generation. Campaign runs land
              under <code>{prefix}-p&lt;generation&gt;-c&lt;candidate&gt;</code>.
            </p>
          )}

          <h4>Comparison baseline</h4>
          <p className="build-form__help">
            A reference run makes every candidate report <code>model_over_reference_mse</code> against that
            model&apos;s own predictions, and gates the campaign on beating it -- a configurable floor beyond
            copy-last-frame. Set the selection metric above to a <code>*.model_over_reference_mse</code> path to
            select on it directly.
          </p>
          <div className="field-grid">
            <label className="field">Reference run
              <select value={form.referenceRun} onChange={(e) => set("referenceRun")(e.target.value)} aria-label="Reference run">
                <option value="">none (copy-last-frame only)</option>
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
            {form.referenceRun && (
              <label className="field">Reference checkpoint
                <input value={form.referenceCheckpoint} onChange={(e) => set("referenceCheckpoint")(e.target.value)} aria-label="Reference checkpoint" />
              </label>
            )}
            <label className="field">Champion (final paired comparison, optional)
              <select value={form.champion} onChange={(e) => set("champion")(e.target.value)} aria-label="Champion">
                <option value="">none</option>
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
          </div>

          <h4>Base spec overrides</h4>
          <div className="overrides">
            {overrides.map((row, index) => (
              <div className="overrides__row" key={index}>
                <input
                  aria-label={`override ${index + 1} path`} placeholder="training.batch_size"
                  value={row.path}
                  onChange={(e) => setOverrides((current) => current.map((r, i) => (i === index ? { ...r, path: e.target.value } : r)))}
                />
                <span>=</span>
                <input
                  aria-label={`override ${index + 1} value`} placeholder="64"
                  value={row.value}
                  onChange={(e) => setOverrides((current) => current.map((r, i) => (i === index ? { ...r, value: e.target.value } : r)))}
                />
                <button type="button" onClick={() => setOverrides((current) => current.filter((_, i) => i !== index))} aria-label={`remove override ${index + 1}`}>×</button>
              </div>
            ))}
            <button type="button" onClick={() => setOverrides((current) => [...current, { path: "", value: "" }])}>+ Add override</button>
          </div>

          {error && <p className="no-data">{error}</p>}
          <div className="evolve-actions">
            <button type="button" onClick={handlePreview} disabled={previewing || !baseSpec}>
              {previewing ? "Previewing…" : "Dry-run preview"}
            </button>
            <button type="submit" disabled={launching || !baseSpec || launchBlocked}>
              {launching ? "Launching…" : "Launch campaign"}
            </button>
          </div>
          {launchBlocked && (
            <p className="build-form__hint">
              Acknowledge the cost above to enable the launch.
            </p>
          )}
        </form>

        {preview && architecture && (
          <div className="evolve-preview">
            <h4>Dry-run preview</h4>
            <p className="build-form__hint">
              campaign <code>{preview.campaign_id}</code> · {preview.workflow} · architecture schema{" "}
              {preview.architecture_schema_version} ({String(preview.architecture_schema_hash).slice(0, 12)}) ·
              inner schema {preview.hyperparameter_schema_version} · {preview.method} · seed {preview.seed}
            </p>
            {architectureGeneColumns.length ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>architecture</th><th>inner run prefix</th>
                      {architectureGeneColumns.map((gene) => <th key={gene}>{gene}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.candidates.map((candidate) => (
                      <tr key={candidate.architecture_id}>
                        <th scope="row">a{candidate.candidate_index}</th>
                        <td>{candidate.architecture_id}</td>
                        {architectureGeneColumns.map((gene) => (
                          <td key={gene}>{geneValue(candidate.genome?.[gene])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="no-data">
                {preview.candidates?.length
                  ? "every sampled architecture resolved to the same model block -- widen the architecture schema or outer population size"
                  : "the dry run sampled no architectures"}
              </p>
            )}
          </div>
        )}

        {preview && !architecture && (
          <div className="evolve-preview">
            <h4>Dry-run preview</h4>
            <p className="build-form__hint">
              campaign <code>{preview.campaign_id}</code> · {preview.workflow} · schema {preview.schema} (
              {String(preview.schema_hash).slice(0, 12)}) · {preview.method} · seed {preview.seed}
            </p>
            {geneColumns.length ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr><th>candidate</th><th>run id</th>{geneColumns.map((path) => <th key={path}>{path}</th>)}</tr>
                  </thead>
                  <tbody>
                    {preview.candidates.map((candidate) => {
                      const leaves = flattenLeaves(candidate.spec?.training || {});
                      return (
                        <tr key={candidate.run_id}>
                          <th scope="row">c{candidate.candidate_index}</th>
                          <td>{candidate.run_id}</td>
                          {geneColumns.map((path) => <td key={path}>{geneValue(leaves[path])}</td>)}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="no-data">
                {preview.candidates?.length
                  ? "every proposal resolved to the same training block -- widen the genome schema or population size"
                  : "the dry run proposed no candidates"}
              </p>
            )}
          </div>
        )}
      </section>

      {campaign && (campaign.genome === "architecture" ? (
        <ArchitectureBoard
          organism={organism} prefix={campaign.prefix}
          outerPopulationSize={campaign.outerPopulationSize}
          outerPopulationCount={campaign.outerPopulationCount}
          innerPopulationSize={campaign.populationSize}
          innerPopulationCount={campaign.populationCount}
          selectionMetric={campaign.selectionMetric}
        />
      ) : (
        <PopulationBoard
          organism={organism} prefix={campaign.prefix}
          populationSize={campaign.populationSize} populationCount={campaign.populationCount}
          selectionMetric={campaign.selectionMetric}
        />
      ))}

      <JobsPanel organism={organism} refreshToken={refreshToken} highlightJobId={lastJobId} />
    </>
  );
}
