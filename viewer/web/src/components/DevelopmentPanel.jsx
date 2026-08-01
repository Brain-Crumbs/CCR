import { useMemo } from "react";
import { developmentStages } from "../lib/diagnostics.js";

/** Milestones passed per stage, for one organism's developmental ladder
 * (Phase 7 gates). */
export function DevelopmentPanel({ session = {} }) {
  const stages = useMemo(() => developmentStages(session), [session]);

  return (
    <section className="diagnostic development" aria-labelledby="development-title">
      <h3 id="development-title">Developmental ladder</h3>
      <p>Milestones passed by this organism.</p>
      <ol className="stage-list">
        {stages.length ? stages.map((stage) => (
          <li key={stage.name} className={`stage ${stage.passed ? "stage--passed" : "stage--pending"}`}>
            <span className="stage-mark" aria-hidden="true">{stage.passed ? "✓" : "○"}</span>
            <strong>{stage.name}</strong>
            {stage.milestones.length > 0 && (
              <small>{stage.milestones.map((m) => (typeof m === "string" ? m : m.name ?? m.gate)).join(" · ")}</small>
            )}
          </li>
        )) : <li className="no-data">no developmental gates recorded</li>}
      </ol>
    </section>
  );
}
