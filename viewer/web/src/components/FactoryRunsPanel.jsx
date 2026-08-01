const STATE_CLASS = {
  queued: "state-badge--neutral",
  running: "state-badge--active",
  checkpointing: "state-badge--active",
  completed: "state-badge--good",
  budget_exceeded: "state-badge--warn",
  failed: "state-badge--bad",
  cancelled: "state-badge--warn",
};

function StateBadge({ state }) {
  if (!state) return <span className="state-badge state-badge--neutral">unknown</span>;
  return <span className={`state-badge ${STATE_CLASS[state] || "state-badge--neutral"}`}>{state.replace(/_/g, " ")}</span>;
}

/**
 * Every run under an organism and where it sits in the factory's state
 * machine (state.py's queued/running/checkpointing/completed/
 * budget_exceeded/failed/cancelled) -- the "what's in flight, what
 * finished, what failed" view the champion registry alone can't give,
 * since the registry only ever holds promoted runs.
 */
export function FactoryRunsPanel({ runs, selectedRun = null, onSelectRun }) {
  return (
    <section className="diagnostic factory-runs" aria-labelledby="factory-runs-title">
      <h3 id="factory-runs-title">Factory runs</h3>
      <p>Every trial launched for this organism, regardless of promotion.</p>
      {runs?.length ? (
        <div className="table-scroll">
          <table>
            <thead><tr><th>Run</th><th>Mode</th><th>State</th><th>Promoted</th><th>Updated</th></tr></thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run} className={run.run === selectedRun ? "is-current" : ""}
                  onClick={() => onSelectRun?.(run.run)} style={onSelectRun ? { cursor: "pointer" } : undefined}>
                  <th scope="row">{run.run}</th>
                  <td>{run.mode || "–"}</td>
                  <td><StateBadge state={run.state} /></td>
                  <td>{run.promoted === true ? "yes" : run.promoted === false ? "no" : "–"}</td>
                  <td>{run.updated_at || "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="no-data">no runs recorded for this organism yet</p>}
    </section>
  );
}
