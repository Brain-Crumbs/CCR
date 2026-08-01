const RUNNING_STATES = new Set(["queued", "running", "checkpointing"]);

/** Map one factory run's state (state.json's convention) plus its
 * promotion outcome onto the clinic's badge palette: running/queued reads
 * as in-progress, a completed+promoted run reads as success, a failure is
 * an error, and budget_exceeded/cancelled read as a soft warning. */
function stateBadgeClass(state, promoted) {
  if (RUNNING_STATES.has(state)) return "info";
  if (state === "failed") return "red";
  if (state === "budget_exceeded" || state === "cancelled") return "amber";
  if (state === "completed") return promoted ? "green" : "neutral";
  return "neutral";
}

function StateBadge({ state, promoted }) {
  return <span className={`state-badge state-badge--${stateBadgeClass(state, promoted)}`}>{state || "unknown"}</span>;
}

/**
 * The Factory tab's organism-wide build status (`/api/factory-runs`):
 * one row per run under the current organism, independent of which run is
 * currently selected in the pickers -- "what's queued/running/completed/
 * failed" across the whole Model Factory build, which the champion
 * registry alone cannot answer since it only ever holds *promoted* runs.
 */
export function FactoryRunsPanel({ factoryRuns }) {
  const runs = factoryRuns?.runs || [];
  return (
    <section className="diagnostic factory-runs" aria-labelledby="factory-runs-title">
      <h3 id="factory-runs-title">Factory runs</h3>
      <p>Every run under this organism, however it currently stands.</p>
      {runs.length ? (
        <div className="table-scroll">
          <table>
            <thead><tr><th>run</th><th>mode</th><th>state</th><th>updated</th></tr></thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run}>
                  <th scope="row">{run.run}</th>
                  <td>{run.mode || "–"}</td>
                  <td><StateBadge state={run.state} promoted={run.promoted} />{run.state_reason ? ` — ${run.state_reason}` : ""}</td>
                  <td>{run.updated_at || "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="no-data">no factory runs found for this organism</p>}
    </section>
  );
}
