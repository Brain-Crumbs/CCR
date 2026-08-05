import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { JobsPanel } from "./JobsPanel.jsx";

/** How long the form stays quiet before the compatibility preview re-runs.
 * The preview is the CLI's own torch-free `--dry-run` (it crosses/mutates
 * genomes and reads checkpoint metadata only), but it is still a spawned
 * Python process per call, so typing a seed shouldn't launch one per
 * keystroke. */
const PREVIEW_DEBOUNCE_MS = 250;

const DEFAULT_FORM = {
  parentA: "",
  parentB: "",
  schema: "generic_action_effects_v1",
  tier: "",
  objective: "",
  generation: "",
  seed: "7",
  mutationRate: "0.2",
  weightDonor: "",
  checkpointA: "best-validation.pt",
  checkpointB: "best-validation.pt",
  stageBudgetSeconds: "",
  minEpisodeLength: "",
  namingSeed: "",
  exportPredictionsMax: "3",
  skipExportPredictions: false,
};

function formatValue(value) {
  if (value === null || value === undefined) return "–";
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : Number(value.toPrecision(4)).toString();
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/**
 * One row per gene in the dry run's `breeding_lineage`, joining what each
 * parent carried, which parent crossover chose it from, whether mutation
 * and/or repair then moved it, and what the child ends up training with.
 *
 * Every field is read straight out of the lineage record `breed()` itself
 * writes (`crossover_choices`/`mutations`/`repairs`/`child_genome`) -- no
 * genetic operator is re-derived here, so a gene the browser has never
 * heard of still renders correctly.
 */
export function geneRows(lineage) {
  if (!lineage) return [];
  const parentA = lineage.parent_a_genome || {};
  const parentB = lineage.parent_b_genome || {};
  const child = lineage.child_genome || {};
  const names = [...new Set([...Object.keys(parentA), ...Object.keys(parentB), ...Object.keys(child)])].sort();
  return names.map((gene) => {
    const mutation = (lineage.mutations || {})[gene] || null;
    const repair = (lineage.repairs || {})[gene] || null;
    const inherited = (lineage.crossover_choices || {})[gene] || null;
    return {
      gene,
      parent_a: parentA[gene],
      parent_b: parentB[gene],
      inherited,
      mutation,
      repair,
      child: child[gene],
      // A gene is "moved" when the child's value differs from whichever
      // parent crossover took it from -- i.e. mutation or repair changed it
      // after inheritance, which is what a reader scanning the table is
      // actually looking for.
      moved: Boolean(mutation) || Boolean(repair),
    };
  });
}

/** The two parents' rows out of `GET /api/factory-runs`, in the order the
 * form names them, so the picker's choice can be sanity-checked (state,
 * promotion, its own selection-metric value) without a second request. */
function parentRow(factoryRuns, runId) {
  return (factoryRuns?.runs || []).find((run) => run.run === runId) || null;
}

/**
 * The Breed tab (clinic redesign, Phase 5): pick two completed runs of one
 * organism, see live whether they are compatible parents at all, inspect the
 * exact child the genetic operators would produce, and launch it.
 *
 * Nothing about compatibility, crossover, mutation or repair is
 * reimplemented here. The preview is `ccr factory breed --dry-run` through
 * `POST /api/preview/breed` -- so `check_compatible`'s own message (which
 * names *every* mismatched dimension at once, not just the first) is what
 * the incompatible case shows -- and the launch is the same subcommand with
 * the same flags, minus `--dry-run`, through `POST /api/jobs/breed`.
 */
export function BreedPanel({ catalog, organism: initialOrganism }) {
  const organisms = catalog?.organisms || [];
  const [organism, setOrganism] = useState(initialOrganism ?? organisms[0] ?? null);
  const [meta, setMeta] = useState(null);
  const [factoryRuns, setFactoryRuns] = useState(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [preview, setPreview] = useState(null);
  const [previewError, setPreviewError] = useState(null);
  // Exactly which request body the current preview/previewError describes.
  // A preview that no longer matches the form is shown but never launched:
  // between an edit and its debounced re-check, the displayed child is not
  // the child these settings would produce.
  const [previewedBody, setPreviewedBody] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewToken, setPreviewToken] = useState(0);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState(null);
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
    api.factoryRuns(organism).then((r) => !cancelled && setFactoryRuns(r), () => !cancelled && setFactoryRuns(null));
    return () => { cancelled = true; };
  }, [organism, refreshToken]);

  // `load_parent` refuses anything that is not `completed` -- the same gate
  // promotion enforces -- so a run in any other state is never offered as a
  // parent rather than offered and then rejected by the preview.
  const completedRuns = useMemo(
    () => (factoryRuns?.runs || []).filter((run) => run.state === "completed").map((run) => run.run),
    [factoryRuns],
  );

  useEffect(() => {
    setForm((current) => {
      const parentA = completedRuns.includes(current.parentA) ? current.parentA : (completedRuns[0] ?? "");
      const parentB = completedRuns.includes(current.parentB)
        ? current.parentB
        : (completedRuns.find((run) => run !== parentA) ?? "");
      return parentA === current.parentA && parentB === current.parentB ? current : { ...current, parentA, parentB };
    });
  }, [completedRuns]);

  // The weight donor must name one of the two parents (breed() rejects
  // anything else outright), so a donor left over from a since-replaced
  // parent falls back to the seeded choice instead of guaranteeing an error.
  useEffect(() => {
    setForm((current) => (
      !current.weightDonor || current.weightDonor === current.parentA || current.weightDonor === current.parentB
        ? current
        : { ...current, weightDonor: "" }
    ));
  }, [form.parentA, form.parentB]);

  function set(field) {
    return (value) => setForm((current) => ({ ...current, [field]: value }));
  }

  const parentsChosen = Boolean(form.parentA && form.parentB);
  const parentsDistinct = form.parentA !== form.parentB;
  const canPreview = parentsChosen && parentsDistinct && Boolean(organism);

  // One JSON string, used both as the effect's dependency and as the body
  // actually posted: the preview a reader is looking at is always the one
  // for exactly these form values, never a stale closure's.
  const previewBody = useMemo(() => JSON.stringify({
    organism,
    parent_a: form.parentA,
    parent_b: form.parentB,
    options: {
      // --organism disambiguates both positional run ids under --root, the
      // same way BuildPanel's clone/resume launches pass it.
      organism,
      schema: form.schema,
      tier: form.tier || null,
      objective: form.objective || null,
      generation: form.generation === "" ? null : Number(form.generation),
      seed: Number(form.seed),
      mutation_rate: Number(form.mutationRate),
      weight_donor: form.weightDonor || null,
      checkpoint_a: form.checkpointA,
      checkpoint_b: form.checkpointB,
      stage_budget_seconds: form.stageBudgetSeconds === "" ? null : Number(form.stageBudgetSeconds),
      min_episode_length: form.minEpisodeLength === "" ? null : Number(form.minEpisodeLength),
    },
  }), [organism, form]);

  useEffect(() => {
    if (!canPreview) { setPreview(null); setPreviewError(null); setPreviewedBody(null); return undefined; }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setPreviewing(true);
      try {
        const result = await api.previewJob("breed", JSON.parse(previewBody));
        if (cancelled) return;
        setPreview(result);
        setPreviewError(null);
      } catch (err) {
        if (cancelled) return;
        setPreview(null);
        setPreviewError(err.message);
      } finally {
        if (!cancelled) { setPreviewedBody(previewBody); setPreviewing(false); }
      }
    }, PREVIEW_DEBOUNCE_MS);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [previewBody, canPreview, previewToken]);

  // What the panel is showing describes the form as it stands right now --
  // not an earlier one whose re-check has not landed yet.
  const previewIsCurrent = previewedBody === previewBody;
  const launchable = Boolean(preview) && previewIsCurrent;

  async function handleLaunch(event) {
    event.preventDefault();
    // Never launch a body the displayed lineage does not describe: an edit
    // made inside the debounce window would otherwise breed a different
    // child than the one on screen, or an incompatible pair the pending
    // re-check is about to reject.
    if (!canPreview || !launchable) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      const body = JSON.parse(previewBody);
      const entry = await api.launchJob("breed", {
        ...body,
        options: {
          ...body.options,
          naming_seed: form.namingSeed || null,
          export_predictions_max: Number(form.exportPredictionsMax),
          no_export_predictions: form.skipExportPredictions,
        },
      });
      setLastJobId(entry.job_id);
      setRefreshToken((token) => token + 1);
    } catch (err) {
      setLaunchError(err.message);
    } finally {
      setLaunching(false);
    }
  }

  if (!organisms.length) {
    return (
      <section className="diagnostic breed" aria-labelledby="breed-title">
        <h3 id="breed-title">Breed</h3>
        <p className="no-data">no organisms found -- launch at least one run from the Build tab first</p>
      </section>
    );
  }

  const rowA = parentRow(factoryRuns, form.parentA);
  const rowB = parentRow(factoryRuns, form.parentB);
  const lineage = preview?.breeding_lineage ?? null;
  const childSpec = preview?.child_spec ?? null;
  const rows = geneRows(lineage);

  return (
    <>
      <section className="diagnostic breed" aria-labelledby="breed-title">
        <h3 id="breed-title">Breed</h3>
        <p>
          Cross two completed runs of one organism into a single explicit-lineage child: uniform crossover over
          the declared genome schema, bounded per-gene mutation, then repair -- with exactly one parent&apos;s
          checkpoint carried as the child&apos;s weights.
        </p>

        <form className="build-form" onSubmit={handleLaunch}>
          <div className="pickers">
            <Picker label="Organism" ariaLabel="Breed organism" value={organism} options={organisms} onChange={setOrganism} />
          </div>

          <h4>Parents</h4>
          <p className="build-form__help">
            Only completed runs are offered: breeding reads a parent&apos;s frozen genome and checkpoint, so a run
            that never finished evaluation is refused for the same reason promotion refuses it.
          </p>
          <div className="field-grid">
            <label className="field">Parent A
              <select value={form.parentA} onChange={(e) => set("parentA")(e.target.value)} aria-label="Parent A">
                {!completedRuns.length && <option value="">{factoryRuns ? "no completed runs found" : "loading…"}</option>}
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
            <label className="field">Parent B
              <select value={form.parentB} onChange={(e) => set("parentB")(e.target.value)} aria-label="Parent B">
                {!completedRuns.length && <option value="">{factoryRuns ? "no completed runs found" : "loading…"}</option>}
                {completedRuns.map((run) => <option key={run} value={run}>{run}</option>)}
              </select>
            </label>
            <label className="field">Parent A checkpoint
              <input value={form.checkpointA} onChange={(e) => set("checkpointA")(e.target.value)} aria-label="Parent A checkpoint" />
            </label>
            <label className="field">Parent B checkpoint
              <input value={form.checkpointB} onChange={(e) => set("checkpointB")(e.target.value)} aria-label="Parent B checkpoint" />
            </label>
          </div>
          {[["A", rowA], ["B", rowB]].map(([label, row]) => row && (
            <p className="build-form__hint" key={label}>
              parent {label}: <code>{row.run}</code> · state {row.state || "unknown"} · mode {row.mode || "–"}
              {row.selection_metric ? ` · ${row.selection_metric} = ${formatValue(row.selection_metric_value)}` : ""}
              {row.promoted ? " · promoted" : ""}
            </p>
          ))}
          {parentsChosen && !parentsDistinct && (
            <p className="no-data">pick two different runs -- crossing a run with itself only reproduces it</p>
          )}
          {!completedRuns.length && factoryRuns && (
            <p className="no-data">
              this organism has no completed runs to breed from yet -- launch one from the Build tab first
            </p>
          )}

          <h4>Genetics</h4>
          <div className="field-grid">
            <label className="field">Genome schema
              <select value={form.schema} onChange={(e) => set("schema")(e.target.value)} aria-label="Genome schema">
                {(meta?.genome_schemas || [DEFAULT_FORM.schema]).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            <label className="field">Objective
              <select value={form.objective} onChange={(e) => set("objective")(e.target.value)} aria-label="Objective">
                <option value="">parent A&apos;s objective (default)</option>
                {(meta?.objectives || []).map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>
            <label className="field">Weight donor
              <select value={form.weightDonor} onChange={(e) => set("weightDonor")(e.target.value)} aria-label="Weight donor">
                <option value="">seeded choice (default)</option>
                {form.parentA && <option value={form.parentA}>{form.parentA} (A)</option>}
                {form.parentB && form.parentB !== form.parentA && <option value={form.parentB}>{form.parentB} (B)</option>}
              </select>
            </label>
            <label className="field">Seed
              <input type="number" value={form.seed} onChange={(e) => set("seed")(e.target.value)} aria-label="Seed" />
            </label>
            <label className="field">Mutation rate
              <input type="number" step="any" min="0" max="1" value={form.mutationRate} onChange={(e) => set("mutationRate")(e.target.value)} aria-label="Mutation rate" />
            </label>
            <label className="field">Budget tier (optional)
              <input value={form.tier} onChange={(e) => set("tier")(e.target.value)} placeholder="infer from budget reports" aria-label="Budget tier" />
            </label>
            <label className="field">Generation (optional)
              <input type="number" value={form.generation} onChange={(e) => set("generation")(e.target.value)} placeholder="max parent + 1" aria-label="Generation" />
            </label>
            <label className="field">Stage budget (seconds, optional)
              <input type="number" step="any" value={form.stageBudgetSeconds} onChange={(e) => set("stageBudgetSeconds")(e.target.value)} aria-label="Stage budget seconds" />
            </label>
            <label className="field">Min episode length (optional)
              <input type="number" value={form.minEpisodeLength} onChange={(e) => set("minEpisodeLength")(e.target.value)} aria-label="Min episode length" />
            </label>
          </div>
          <p className="build-form__hint">
            The same seed, parents and schema always reproduce the same child exactly -- every draw (donor
            choice, crossover, mutation) runs in one fixed order off this seed.
          </p>

          <h4>Launch options</h4>
          <div className="field-grid">
            <label className="field">Naming seed (optional)
              <input value={form.namingSeed} onChange={(e) => set("namingSeed")(e.target.value)} aria-label="Naming seed" />
            </label>
            <label className="field">Export predictions (max episodes)
              <input type="number" min="0" value={form.exportPredictionsMax} onChange={(e) => set("exportPredictionsMax")(e.target.value)} aria-label="Export predictions max" />
            </label>
            <label className="field field--checkbox">
              <input type="checkbox" checked={form.skipExportPredictions} onChange={(e) => set("skipExportPredictions")(e.target.checked)} />
              Skip prediction export
            </label>
          </div>

          {launchError && <p className="no-data">{launchError}</p>}
          <div className="evolve-actions">
            <button type="button" onClick={() => setPreviewToken((token) => token + 1)} disabled={!canPreview || previewing}>
              {previewing ? "Checking…" : "Recheck compatibility"}
            </button>
            <button type="submit" disabled={launching || !launchable}>
              {launching ? "Launching…" : "Breed child"}
            </button>
          </div>
          {canPreview && !launchable && !previewError && (
            <p className="build-form__hint">breeding stays disabled until these settings preview as compatible</p>
          )}
        </form>

        <div className="breed-preview">
          <h4>Compatibility &amp; child preview</h4>
          {/* The last completed check stays on screen while the next one
              runs -- re-blanking the panel on every keystroke would flicker
              -- but it is labelled as describing the previous settings, and
              breeding is disabled until the re-check lands. */}
          {canPreview && previewedBody && !previewIsCurrent && (
            <p className="build-form__hint" data-testid="breed-stale">
              settings changed -- re-checking; what follows describes the previous check
            </p>
          )}
          {!canPreview ? (
            <p className="no-data">select two different completed runs to check</p>
          ) : previewError ? (
            <>
              <p className="no-data">these parents cannot be bred as configured</p>
              <pre className="breed-error" data-testid="breed-error">{previewError}</pre>
            </>
          ) : !preview ? (
            <p className="empty">Checking compatibility…</p>
          ) : (
            <>
              <p className="build-form__hint" data-testid="breed-summary">
                compatible · generation {lineage?.generation} · weight donor <code>{lineage?.weight_donor_run_id}</code>{" "}
                · {lineage?.genome_operator} · mutation seed {lineage?.mutation_seed} · schema {preview.schema} (
                {String(preview.schema_hash).slice(0, 12)})
              </p>
              <p className="build-form__hint">
                child: {childSpec?.mode || "–"} of <code>{childSpec?.parent?.run_id || "–"}</code>
                {childSpec?.parent?.checkpoint ? ` / ${childSpec.parent.checkpoint}` : ""} · corpus{" "}
                {childSpec?.data?.corpus_id || "–"} · objective {childSpec?.training?.objective || "–"} · metric{" "}
                {childSpec?.evaluation?.selection_metric || "–"}
              </p>
              {rows.length ? (
                <div className="table-scroll">
                  <table className="gene-table">
                    <thead>
                      <tr><th>gene</th><th>parent A</th><th>parent B</th><th>inherited</th><th>child</th></tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr key={row.gene} className={row.moved ? "is-current" : undefined}>
                          <th scope="row">{row.gene}</th>
                          <td>{formatValue(row.parent_a)}</td>
                          <td>{formatValue(row.parent_b)}</td>
                          <td>{row.inherited === "parent_a" ? "A" : row.inherited === "parent_b" ? "B" : "–"}</td>
                          <td>
                            {formatValue(row.child)}
                            {row.mutation && <span className="state-badge state-badge--info">mutated</span>}
                            {row.repair && <span className="state-badge state-badge--amber">repaired</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="no-data">this schema declared no genes for the preview to cross</p>
              )}
            </>
          )}
        </div>
      </section>

      <JobsPanel organism={organism} refreshToken={refreshToken} highlightJobId={lastJobId} />
    </>
  );
}
