import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { formatExponential } from "../lib/format.js";

const RUNNING_STATES = new Set(["queued", "running", "checkpointing"]);

/** FactoryRunsPanel's badge convention, applied per grid cell. */
function stateBadgeClass(state) {
  if (RUNNING_STATES.has(state)) return "info";
  if (state === "failed") return "red";
  if (state === "budget_exceeded" || state === "cancelled") return "amber";
  if (state === "completed") return "green";
  return "neutral";
}

/** runner._selection_metric_mode: every prediction metric is an error to
 * minimize; the three goal-navigation scores are the only maximized ones.
 * Drives which cell in a generation is marked best, nothing else. */
export function selectionMetricMode(selectionMetric) {
  return [
    "goal_navigation.success_rate",
    "goal_navigation.geodesic_efficiency",
    "goal_navigation.replan_recovery",
  ].includes(selectionMetric) ? "max" : "min";
}

/**
 * Group one campaign's runs by the `{prefix}-p{population}-c{candidate}`
 * ids `search.run_evolutionary_search` allocates, and reconstruct each
 * generation's survivor slots.
 *
 * Only generation 0 is fully precomputable from the campaign's
 * configuration: `run_evolutionary_search` fills a later generation's slots
 * `0..S-1` with *carried* survivors, which keep the run ids they were first
 * trained under rather than taking fresh slot-shaped ones, and only slots
 * `S..N-1` with freshly bred offspring. So `S` is not known up front -- but
 * it is recoverable from disk the moment any offspring appears: the lowest
 * offspring candidate index in a generation *is* that generation's survivor
 * count, and each offspring's `configuration_parents` names two of those
 * survivors by run id. When the union of those parents accounts for every
 * survivor slot, the survivors are placed in their real slots (ordered the
 * way search.py orders them: by selection metric, run id breaking ties);
 * when it does not, the slots are still shown as carried, just unnamed.
 * Nothing here re-implements survivor *selection* -- it only reads back
 * decisions the campaign already wrote down.
 */
export function buildBoard({ runs, prefix, populationSize, populationCount, selectionMetric }) {
  const byRunId = new Map((runs || []).map((run) => [run.run, run]));
  const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const idPattern = new RegExp(`^${escaped}-p(\\d+)-c(\\d+)$`);

  const allocated = new Map(); // population -> Map(candidate -> run entry)
  for (const run of runs || []) {
    const match = idPattern.exec(run.run);
    if (!match) continue;
    const population = Number(match[1]), candidate = Number(match[2]);
    if (!allocated.has(population)) allocated.set(population, new Map());
    allocated.get(population).set(candidate, run);
  }

  const mode = selectionMetricMode(selectionMetric);
  const rank = (run) => {
    const value = run?.selection_metric_value;
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return mode === "min" ? value : -value;
  };
  // Codepoint order, not `localeCompare`: this reproduces search.py's own
  // survivor ordering, whose tiebreak is a plain Python string comparison.
  // Locale collation disagrees with that on punctuation ("a-b" vs "ab"), and
  // run ids are punctuated by construction.
  const byRunIdOrder = (a, b) => (String(a.run) < String(b.run) ? -1 : String(a.run) > String(b.run) ? 1 : 0);
  const bySelection = (a, b) => {
    const ra = rank(a), rb = rank(b);
    // An unevaluated candidate sorts last regardless of id -- it has no
    // score to be "best" or to have survived on.
    if (ra === null && rb === null) return byRunIdOrder(a, b);
    if (ra === null) return 1;
    if (rb === null) return -1;
    return ra - rb || byRunIdOrder(a, b);
  };

  // A campaign can stop short of its configured population_count (no
  // quality-passing survivors, or a stage budget running out), and could in
  // principle have written a generation beyond it; render whatever is
  // configured *or* on disk, whichever reaches further.
  const highestOnDisk = allocated.size ? Math.max(...allocated.keys()) : -1;
  const generationCount = Math.max(populationCount, highestOnDisk + 1);

  const generations = [];
  for (let population = 0; population < generationCount; population += 1) {
    const slots = allocated.get(population) || new Map();
    const offspringIndices = [...slots.keys()].sort((a, b) => a - b);
    // Generation 0 is entirely proposed, never bred: no survivor slots.
    const survivorCount = population === 0 || !offspringIndices.length ? 0 : offspringIndices[0];
    const parentIds = [...new Set(
      offspringIndices.flatMap((index) => slots.get(index).configuration_parents || []),
    )];
    const survivors = parentIds
      .map((runId) => byRunId.get(runId) || { run: runId })
      .sort(bySelection);
    // Naming a survivor's slot is only safe when the recorded parents
    // account for every survivor slot -- breeding samples two parents per
    // child, so a survivor that never happened to be sampled would
    // otherwise silently shift every other survivor into the wrong slot.
    const namedSurvivors = survivors.length === survivorCount ? survivors : null;

    const cells = [];
    for (let candidate = 0; candidate < Math.max(populationSize, offspringIndices.length ? offspringIndices.at(-1) + 1 : 0); candidate += 1) {
      const allocatedRun = slots.get(candidate);
      if (allocatedRun) {
        cells.push({ candidate, kind: "trained", run: allocatedRun, parents: allocatedRun.configuration_parents || [] });
      } else if (candidate < survivorCount) {
        cells.push({ candidate, kind: "carried", run: namedSurvivors?.[candidate] ?? null });
      } else {
        // Generation 0's ids are exact ahead of the subprocess reaching
        // them; a later generation's offspring id is only exact once the
        // survivor count above is known, which it is here.
        cells.push({
          candidate, kind: "pending",
          expectedRunId: population === 0 || survivorCount ? `${prefix}-p${population}-c${candidate}` : null,
        });
      }
    }

    const best = cells
      .map((cell) => cell.run)
      .filter((run) => run && rank(run) !== null)
      .sort(bySelection)[0] ?? null;
    generations.push({ population, cells, survivorCount, survivors, best: best?.run ?? null });
  }
  return { generations, mode, trainedCount: [...allocated.values()].reduce((total, slots) => total + slots.size, 0) };
}

function Cell({ cell, isBest, selectionMetric }) {
  if (cell.kind === "pending") {
    return (
      <td className="population-cell population-cell--pending">
        <span className="population-cell__id">{cell.expectedRunId || "id not yet determined"}</span>
        <span className="state-badge state-badge--neutral">not started</span>
        {!cell.expectedRunId && (
          <span className="population-cell__lineage">awaits the previous generation&apos;s survivors</span>
        )}
      </td>
    );
  }
  if (cell.kind === "carried") {
    return (
      <td className="population-cell population-cell--carried">
        <span className="population-cell__id">{cell.run?.run || "survivor"}</span>
        <span className="state-badge state-badge--green">carried</span>
        {cell.run?.selection_metric_value !== undefined && (
          <span className="population-cell__metric">{formatExponential(cell.run.selection_metric_value)}</span>
        )}
      </td>
    );
  }
  const run = cell.run;
  return (
    <td className={`population-cell${isBest ? " is-current" : ""}`}>
      <span className="population-cell__id">{run.run}</span>
      <span className={`state-badge state-badge--${stateBadgeClass(run.state)}`}>{run.state || "unknown"}</span>
      <span className="population-cell__metric" title={run.selection_metric || selectionMetric || ""}>
        {formatExponential(run.selection_metric_value)}
      </span>
      {cell.parents.length === 2 && (
        <span className="population-cell__lineage" title={`bred from ${cell.parents.join(" × ")}`}>
          ↑ {cell.parents.join(" × ")}
        </span>
      )}
    </td>
  );
}

/**
 * One evolutionary campaign's generation × candidate grid (clinic redesign,
 * Phase 4): every candidate's live state pill and selection-metric value,
 * plus the survivor/offspring lineage that connects one generation to the
 * next. Polls `/api/factory-runs` -- the same organism-wide read the
 * Factory tab already uses -- rather than any campaign-specific endpoint,
 * because the campaign subprocess publishes progress only to disk (per-run
 * `state.json`/`metrics/validation.json`), refreshed once per checkpoint
 * chunk.
 *
 * `runs` may be supplied instead of polling, which is how Phase 7's nested
 * architecture board will reuse this grid one level down without every
 * expanded inner campaign opening its own poller.
 */
export function PopulationBoard({
  organism, prefix, populationSize, populationCount, selectionMetric,
  runs: providedRuns = null, pollIntervalMs = 4000,
}) {
  const [polledRuns, setPolledRuns] = useState(null);
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
    () => (runs ? buildBoard({ runs, prefix, populationSize, populationCount, selectionMetric }) : null),
    [runs, prefix, populationSize, populationCount, selectionMetric],
  );

  const columnCount = board ? Math.max(...board.generations.map((g) => g.cells.length), 1) : populationSize;

  return (
    <section className="diagnostic population-board" aria-labelledby="population-board-title">
      <h3 id="population-board-title">Population board</h3>
      <p>
        Campaign <code>{prefix}</code> — {populationSize} candidates × {populationCount} generation(s), selected on{" "}
        <code>{selectionMetric}</code> ({board?.mode === "max" ? "higher is better" : "lower is better"}).
      </p>
      {!board ? (
        <p className="empty">Loading campaign runs…</p>
      ) : !board.trainedCount ? (
        <p className="no-data">
          no campaign runs on disk yet — the first generation appears as <code>{prefix}-p0-c0</code> once the
          subprocess reaches it
        </p>
      ) : (
        <div className="table-scroll">
          <table className="population-grid">
            <thead>
              <tr>
                <th>generation</th>
                {Array.from({ length: columnCount }, (_, index) => <th key={index}>c{index}</th>)}
              </tr>
            </thead>
            <tbody>
              {board.generations.map((generation) => (
                <tr key={generation.population}>
                  <th scope="row">
                    p{generation.population}
                    {generation.survivorCount > 0 && (
                      <small> · {generation.survivorCount} carried</small>
                    )}
                  </th>
                  {generation.cells.map((cell) => (
                    <Cell
                      key={cell.candidate} cell={cell} selectionMetric={selectionMetric}
                      isBest={Boolean(generation.best) && cell.run?.run === generation.best}
                    />
                  ))}
                  {Array.from(
                    { length: Math.max(0, columnCount - generation.cells.length) },
                    (_, index) => <td key={`pad-${index}`} className="population-cell population-cell--absent" />,
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="build-form__hint">
        A survivor carried into the next generation keeps its original run id rather than taking a
        <code> -p&lt;n&gt;-c&lt;i&gt;</code> slot id, so its later-generation cell is labelled “carried”. An
        offspring cell names the two survivors it was bred from.
      </p>
    </section>
  );
}
