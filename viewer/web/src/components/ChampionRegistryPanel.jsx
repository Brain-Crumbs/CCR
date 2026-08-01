import { useMemo } from "react";
import { shortHash } from "../lib/format.js";

function flattenSlots(registry) {
  const slots = registry?.slots || {};
  const rows = [];
  for (const [family, tiers] of Object.entries(slots)) {
    for (const [tier, objectives] of Object.entries(tiers)) {
      for (const [objective, payload] of Object.entries(objectives)) rows.push({ family, tier, objective, ...payload });
    }
  }
  return rows;
}

function metricValue(v) {
  return typeof v === "number" && Number.isFinite(v) ? v.toExponential(3) : String(v ?? "–");
}

function SlotCard({ slot }) {
  const population = slot.population || [];
  // Union of metric keys across the population -- these are the factory's
  // own metric names (often horizon-scoped, e.g. "rollout.t+4.model_mse"),
  // so a population table doubles as a cross-model, cross-horizon comparison.
  const metricKeys = useMemo(() => [...new Set(population.flatMap((entry) => Object.keys(entry.metrics || {})))], [population]);

  return (
    <div className="panel champion-slot">
      <h4>{slot.family} · {slot.tier} · {slot.objective}</h4>
      <p className="run-summary__detail">leading champion: <strong>{slot.leading_champion || "none promoted yet"}</strong></p>
      {population.length ? (
        <div className="table-scroll">
          <table>
            <thead><tr><th>run</th><th>checkpoint</th>{metricKeys.map((k) => <th key={k}>{k}</th>)}<th>test uses</th></tr></thead>
            <tbody>
              {population.map((entry) => (
                <tr key={entry.run_id} className={entry.run_id === slot.leading_champion ? "is-current" : ""}>
                  <th scope="row">{entry.run_id}</th>
                  <td title={entry.checkpoint_sha256}>{shortHash(entry.checkpoint_sha256)}</td>
                  {metricKeys.map((k) => <td key={k}>{metricValue(entry.metrics?.[k])}</td>)}
                  <td>{entry.test_uses ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="no-data">no retained population yet</p>}
      {slot.history?.length > 0 && (
        <details>
          <summary>history ({slot.history.length})</summary>
          <ul>
            {slot.history.map((event, i) => (
              <li key={i}>{event.at ? `${event.at} · ` : ""}{event.action} {event.run_id}{event.reason ? ` — ${event.reason}` : ""}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/**
 * The champion registry (registry.json, `model-factory-registry-v1`):
 * a portfolio of (family, tier, objective) slots, each with its leading
 * champion, retained population, and promote/hold history. This is the
 * durable cross-run answer to "which model wins" -- the paired comparison
 * panel is one A/B; this is the standing leaderboard it feeds.
 */
export function ChampionRegistryPanel({ registry }) {
  const slots = useMemo(() => flattenSlots(registry), [registry]);

  return (
    <section className="diagnostic registry" aria-labelledby="registry-title">
      <h3 id="registry-title">Champion registry</h3>
      <p>Retained population and promotion history per benchmark family, budget tier, and objective.</p>
      {slots.length ? <div className="horizons">{slots.map((slot) => <SlotCard key={`${slot.family}/${slot.tier}/${slot.objective}`} slot={slot} />)}</div>
        : <p className="no-data">no champions promoted yet for this organism</p>}
    </section>
  );
}
