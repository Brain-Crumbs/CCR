import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { JobsPanel } from "./JobsPanel.jsx";
import { PopulationBoard } from "./PopulationBoard.jsx";

const DEFAULT_FORM = {
  baseRun: "",
  schema: "generic_action_effects_v1",
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
function defaultPrefix(seed) {
  return `evo${Number(seed) || 0}`;
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

  const prefix = form.runIdPrefix || defaultPrefix(form.seed);
  const selectionMetric = form.selectionMetric || baseSelectionMetric;
  const maxTrials = estimateMaxTrials(form.populationSize, form.populationCount);

  function set(field) {
    return (value) => setForm((current) => ({ ...current, [field]: value }));
  }

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
    return {
      organism,
      spec: baseSpec,
      options: {
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
      setPreview(await api.previewJob("search", searchBody()));
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
        prefix,
        populationSize: Number(form.populationSize),
        populationCount: Number(form.populationCount),
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

  return (
    <>
      <section className="diagnostic evolve" aria-labelledby="evolve-title">
        <h3 id="evolve-title">Evolve</h3>
        <p>
          Run a bounded evolutionary campaign: propose a population of sibling trials that vary only the
          declared genome&apos;s training genes, keep the best half, and breed the next generation from them.
        </p>

        <form className="build-form" onSubmit={handleLaunch}>
          <div className="pickers">
            <Picker label="Organism" ariaLabel="Evolve organism" value={organism} options={organisms} onChange={setOrganism} />
          </div>

          <h4>Base spec</h4>
          <p className="build-form__help">
            Every candidate inherits this run&apos;s data, model, and evaluation blocks unchanged; only the
            genome&apos;s training genes vary.
          </p>
          <div className="field-grid">
            <label className="field">Base run
              <select value={form.baseRun} onChange={(e) => set("baseRun")(e.target.value)} aria-label="Base run">
                {!completedRuns.length && <option value="">{factoryRuns ? "no completed runs found" : "loading…"}</option>}
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
            <label className="field">Genome schema
              <select value={form.schema} onChange={(e) => set("schema")(e.target.value)} aria-label="Genome schema">
                {(meta?.genome_schemas || [DEFAULT_FORM.schema]).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
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

          <h4>Campaign</h4>
          <div className="field-grid">
            <label className="field">Population size
              <input type="number" min="2" value={form.populationSize} onChange={(e) => set("populationSize")(e.target.value)} aria-label="Population size" />
            </label>
            <label className="field">Generations
              <input type="number" min="1" value={form.populationCount} onChange={(e) => set("populationCount")(e.target.value)} aria-label="Generations" />
            </label>
            <label className="field">Mutation rate
              <input type="number" step="any" min="0" max="1" value={form.mutationRate} onChange={(e) => set("mutationRate")(e.target.value)} aria-label="Mutation rate" />
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
              <input value={form.runIdPrefix} onChange={(e) => set("runIdPrefix")(e.target.value)} placeholder={defaultPrefix(form.seed)} aria-label="Run id prefix" />
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
          <p className="build-form__hint" data-testid="cost-estimate">
            at most {maxTrials} training run(s): {form.populationSize} in the first generation, then up to{" "}
            {Math.max(0, Number(form.populationSize) - 1)} bred offspring per later generation. Campaign runs land
            under <code>{prefix}-p&lt;generation&gt;-c&lt;candidate&gt;</code>.
          </p>

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
            <button type="submit" disabled={launching || !baseSpec}>
              {launching ? "Launching…" : "Launch campaign"}
            </button>
          </div>
        </form>

        {preview && (
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

      {campaign && (
        <PopulationBoard
          organism={organism} prefix={campaign.prefix}
          populationSize={campaign.populationSize} populationCount={campaign.populationCount}
          selectionMetric={campaign.selectionMetric}
        />
      )}

      <JobsPanel organism={organism} refreshToken={refreshToken} highlightJobId={lastJobId} />
    </>
  );
}
