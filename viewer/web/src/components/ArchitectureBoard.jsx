import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { formatExponential } from "../lib/format.js";
import { PopulationBoard, selectionMetricMode } from "./PopulationBoard.jsx";

const RUNNING_STATES = new Set(["queued", "running", "checkpointing"]);

/** One architecture's inner-campaign prefix, mirroring
 * `architecture_search.architecture_run_id_prefix` exactly -- the outer
 * coordinates are appended to the campaign prefix, and the inner campaign
 * then allocates its own `-p{pop}-c{cand}` ids beneath that. Reproduced here
 * (rather than fetched) for the same reason Phase 4's grid reproduces the
 * flat convention: the ids are precomputable from the campaign's
 * configuration alone, so the board can lay out and start watching every
 * cell before the subprocess reaches it. */
export function architecturePrefix(campaignPrefix, generation, candidate) {
  return `${campaignPrefix}-arch${generation}-a${candidate}`;
}

/** The architecture cell's own state pill: an architecture is not a run, so
 * it has no `state.json` of its own -- its state is the aggregate of the
 * inner campaign's runs. "running" wins over everything (something is
 * training right now), then "failed" only when *every* inner run failed
 * (one failed candidate out of six is a normal campaign, not a failed
 * architecture), then "completed". */
export function architectureState(runs) {
  if (!runs.length) return "not started";
  if (runs.some((run) => RUNNING_STATES.has(run.state))) return "running";
  if (runs.every((run) => run.state === "failed")) return "failed";
  if (runs.some((run) => run.state === "completed")) return "completed";
  return runs[0].state || "unknown";
}

function stateBadgeClass(state) {
  if (state === "running") return "info";
  if (state === "failed") return "red";
  if (state === "completed") return "green";
  return "neutral";
}

/**
 * Group one nested campaign's runs into outer generation → architecture →
 * inner runs, from run ids alone.
 *
 * Only what the campaign actually wrote to disk is interpreted. In
 * particular an architecture *retained* into the next outer generation is
 * not re-run (`run_architecture_search` carries its evaluation forward
 * verbatim), so it allocates no runs under the later generation's prefix and
 * shows up there as an empty cell -- the same "carried keeps its original
 * id" property Phase 4's flat board already handles one level down. Which
 * of the empty cells are carried survivors and which are simply not reached
 * yet is not recoverable from run ids, so neither is claimed: an empty cell
 * says "no inner runs on disk", and the legend explains both ways that
 * happens.
 */
export function buildArchitectureBoard({
  runs, prefix, outerPopulationSize, outerPopulationCount, selectionMetric,
}) {
  const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const idPattern = new RegExp(`^${escaped}-arch(\\d+)-a(\\d+)-p\\d+-c\\d+$`);

  const grouped = new Map(); // generation -> Map(architecture -> run[])
  for (const run of runs || []) {
    const match = idPattern.exec(run.run);
    if (!match) continue;
    const generation = Number(match[1]), architecture = Number(match[2]);
    if (!grouped.has(generation)) grouped.set(generation, new Map());
    const byArchitecture = grouped.get(generation);
    if (!byArchitecture.has(architecture)) byArchitecture.set(architecture, []);
    byArchitecture.get(architecture).push(run);
  }

  const mode = selectionMetricMode(selectionMetric);
  // An outer candidate's fitness *is* its inner campaign's best metric value
  // (architecture_search scores it exactly that way), so the board reads the
  // same number back off the inner runs rather than inventing a summary of
  // its own.
  const bestOf = (runsForArchitecture) => {
    let best = null;
    for (const run of runsForArchitecture) {
      const value = run.selection_metric_value;
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      if (best === null) { best = run; continue; }
      const better = mode === "min"
        ? value < best.selection_metric_value
        : value > best.selection_metric_value;
      if (better) best = run;
    }
    return best;
  };

  // As in the flat board: render whatever is configured *or* on disk,
  // whichever reaches further, so a campaign that wrote a generation beyond
  // its configured count is never silently truncated.
  const highestOnDisk = grouped.size ? Math.max(...grouped.keys()) : -1;
  const generationCount = Math.max(outerPopulationCount, highestOnDisk + 1);

  const generations = [];
  for (let generation = 0; generation < generationCount; generation += 1) {
    const byArchitecture = grouped.get(generation) || new Map();
    const highestArchitecture = byArchitecture.size ? Math.max(...byArchitecture.keys()) : -1;
    const width = Math.max(outerPopulationSize, highestArchitecture + 1);
    const architectures = [];
    for (let candidate = 0; candidate < width; candidate += 1) {
      const innerRuns = byArchitecture.get(candidate) || [];
      const best = bestOf(innerRuns);
      architectures.push({
        candidate,
        prefix: architecturePrefix(prefix, generation, candidate),
        runs: innerRuns,
        state: architectureState(innerRuns),
        bestRunId: best?.run ?? null,
        bestMetricValue: best?.selection_metric_value ?? null,
      });
    }
    const ranked = architectures.filter((architecture) => architecture.bestMetricValue !== null);
    const best = ranked.length
      ? ranked.reduce((leader, architecture) => {
        const better = mode === "min"
          ? architecture.bestMetricValue < leader.bestMetricValue
          : architecture.bestMetricValue > leader.bestMetricValue;
        return better ? architecture : leader;
      })
      : null;
    generations.push({ generation, architectures, best: best?.prefix ?? null });
  }

  return {
    generations, mode,
    trainedCount: [...grouped.values()].reduce(
      (total, byArchitecture) => total + [...byArchitecture.values()].reduce((n, list) => n + list.length, 0),
      0,
    ),
  };
}

function ArchitectureRow({
  architecture, isBest, expanded, onToggle,
  organism, innerPopulationSize, innerPopulationCount, selectionMetric,
}) {
  const badge = stateBadgeClass(architecture.state);
  return (
    <>
      <tr className={`architecture-row${isBest ? " is-current" : ""}`}>
        <th scope="row">
          <button
            type="button" className="architecture-row__toggle" onClick={onToggle}
            aria-expanded={expanded}
            aria-label={`${expanded ? "collapse" : "expand"} ${architecture.prefix}`}
          >
            {expanded ? "▾" : "▸"} a{architecture.candidate}
          </button>
        </th>
        <td className="architecture-row__id">{architecture.prefix}</td>
        <td><span className={`state-badge state-badge--${badge}`}>{architecture.state}</span></td>
        <td className="architecture-row__metric">{formatExponential(architecture.bestMetricValue)}</td>
        <td className="architecture-row__id">{architecture.bestRunId || "–"}</td>
        <td className="architecture-row__metric">{architecture.runs.length}</td>
      </tr>
      {expanded && (
        <tr className="architecture-row__inner">
          <td colSpan={6}>
            {/* `runs` is supplied rather than left to poll: the outer board
                already holds one organism-wide poll of /api/factory-runs, and
                every expanded inner grid reading from it keeps the request
                count flat no matter how many architectures are open. */}
            <PopulationBoard
              organism={organism} prefix={architecture.prefix} runs={architecture.runs}
              populationSize={innerPopulationSize} populationCount={innerPopulationCount}
              selectionMetric={selectionMetric}
            />
          </td>
        </tr>
      )}
    </>
  );
}

/**
 * One nested architecture (NAS) campaign's board (clinic redesign, Phase 7):
 * outer generation → architecture candidate, each expandable into Phase 4's
 * own generation × candidate grid for that architecture's inner
 * hyperparameter campaign.
 *
 * Like the flat board it polls only `/api/factory-runs` -- the campaign
 * subprocess publishes progress to disk and nothing else -- and derives the
 * whole nested structure from the run-id convention both loops allocate by.
 */
export function ArchitectureBoard({
  organism, prefix, outerPopulationSize, outerPopulationCount,
  innerPopulationSize, innerPopulationCount, selectionMetric,
  runs: providedRuns = null, pollIntervalMs = 4000,
}) {
  const [polledRuns, setPolledRuns] = useState(null);
  const [expanded, setExpanded] = useState(() => new Set());
  const polling = providedRuns === null;

  useEffect(() => {
    if (!polling || !organism) return undefined;
    let cancelled = false;
    async function poll() {
      try {
        const document = await api.factoryRuns(organism);
        if (!cancelled) setPolledRuns(document?.runs || []);
      } catch {
        if (!cancelled) setPolledRuns([]);
      }
    }
    poll();
    const interval = setInterval(poll, pollIntervalMs);
    return () => { cancelled = true; clearInterval(interval); };
  }, [polling, organism, pollIntervalMs]);

  const runs = providedRuns ?? polledRuns;
  const board = useMemo(
    () => (runs
      ? buildArchitectureBoard({ runs, prefix, outerPopulationSize, outerPopulationCount, selectionMetric })
      : null),
    [runs, prefix, outerPopulationSize, outerPopulationCount, selectionMetric],
  );

  function toggle(architecturePrefixId) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(architecturePrefixId)) next.delete(architecturePrefixId);
      else next.add(architecturePrefixId);
      return next;
    });
  }

  return (
    <section className="diagnostic architecture-board" aria-labelledby="architecture-board-title">
      <h3 id="architecture-board-title">Architecture board</h3>
      <p>
        Campaign <code>{prefix}</code> — {outerPopulationSize} architecture(s) ×{" "}
        {outerPopulationCount} outer generation(s), each scored by a complete{" "}
        {innerPopulationSize} × {innerPopulationCount} inner campaign on{" "}
        <code>{selectionMetric}</code> ({board?.mode === "max" ? "higher is better" : "lower is better"}).
      </p>
      {!board ? (
        <p className="empty">Loading campaign runs…</p>
      ) : !board.trainedCount ? (
        <p className="no-data">
          no campaign runs on disk yet — the first inner trial appears as{" "}
          <code>{architecturePrefix(prefix, 0, 0)}-p0-c0</code> once the subprocess reaches it
        </p>
      ) : (
        board.generations.map((generation) => (
          <div className="architecture-generation" key={generation.generation}>
            <h4>outer generation {generation.generation}</h4>
            <div className="table-scroll">
              <table className="architecture-grid">
                <thead>
                  <tr>
                    <th>architecture</th><th>inner campaign</th><th>state</th>
                    <th>best {selectionMetric}</th><th>best run</th><th>inner runs</th>
                  </tr>
                </thead>
                <tbody>
                  {generation.architectures.map((architecture) => (
                    <ArchitectureRow
                      key={architecture.prefix}
                      architecture={architecture}
                      isBest={generation.best === architecture.prefix}
                      expanded={expanded.has(architecture.prefix)}
                      onToggle={() => toggle(architecture.prefix)}
                      organism={organism}
                      innerPopulationSize={innerPopulationSize}
                      innerPopulationCount={innerPopulationCount}
                      selectionMetric={selectionMetric}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))
      )}
      <p className="build-form__hint">
        An architecture retained into the next outer generation is not re-trained — its inner campaign&apos;s
        evidence is carried forward under its <em>original</em> generation&apos;s ids, so its cell in the later
        generation stays empty. An empty cell therefore means either &ldquo;carried&rdquo; or &ldquo;not reached
        yet&rdquo;; the campaign&apos;s own JSON report (printed when the job finishes) is what distinguishes them.
      </p>
    </section>
  );
}
