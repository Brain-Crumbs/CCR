import { useMemo } from "react";
import { episodeDiagnostics } from "../lib/diagnostics.js";

function reasonText(item) {
  const reason = item.reasons[item.focus];
  if (!reason?.components) return "reason unavailable";
  return Object.entries(reason.components)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(" · ");
}

/** What the organism attended to and why, per tick (core/attention.py's
 * reasons), from DecisionRecord.attention. */
export function AttentionPanel({ records = [], decisions = [], tick = null, onTickChange }) {
  const diagnostics = useMemo(() => episodeDiagnostics(records, decisions), [records, decisions]);

  return (
    <section className="diagnostic attention" aria-labelledby="attention-title">
      <h3 id="attention-title">Attention / focus</h3>
      <p>Selected streams and the controller's reason breakdown.</p>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Tick</th><th>Focus</th><th>Attended streams</th><th>Why</th></tr></thead>
          <tbody>
            {diagnostics.attention.length ? diagnostics.attention.map((item) => (
              <tr key={item.tick} className={item.tick === tick ? "is-current" : ""}
                onClick={() => onTickChange?.(item.tick)} style={onTickChange ? { cursor: "pointer" } : undefined}>
                <td>t{item.tick}</td>
                <th scope="row">{item.focus}</th>
                <td>{item.selected.join(", ") || "none"}</td>
                <td>{reasonText(item)}</td>
              </tr>
            )) : <tr><td colSpan={4} className="no-data">attention was not recorded</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
