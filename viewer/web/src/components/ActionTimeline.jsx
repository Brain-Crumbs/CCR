import { useMemo } from "react";
import { actionTimeline } from "../lib/actions.js";

function actionLabel(action) {
  return action ? String(action).replace(/_/g, " ") : "–";
}

function overrideBadge(entry) {
  if (entry.caregiverOverride) return { text: "caregiver", title: entry.caregiverOverride.reason || "caregiver override" };
  if (entry.motorReflex) return { text: entry.motorReflex.name, title: entry.motorReflex.reason || "reflex override" };
  return null;
}

/**
 * Per-tick predicted/assumed vs. actuated motor action (DecisionRecord.motor_decision,
 * see motor/reflexes.py's MotorDecision): `voluntary` is what the organism
 * intended, `actuated` is what it actually did once a reflex or caregiver
 * override may have intervened. A divergence marker makes every override
 * visible instead of silently reclassifying it as "the" action.
 */
export function ActionTimeline({ decisions = [], tick = null, onTickChange }) {
  const timeline = useMemo(() => actionTimeline(decisions), [decisions]);

  if (!timeline.length) {
    return (
      <section className="diagnostic actions" aria-labelledby="actions-title">
        <h3 id="actions-title">Actions</h3>
        <p className="no-data">no motor decisions recorded</p>
      </section>
    );
  }

  return (
    <section className="diagnostic actions" aria-labelledby="actions-title">
      <h3 id="actions-title">Actions</h3>
      <p>What the organism intended (voluntary) versus what it actuated, tick by tick.</p>
      <ol className="action-timeline" aria-label="predicted vs actuated action timeline">
        {timeline.map((entry) => {
          const badge = overrideBadge(entry);
          const current = tick != null && entry.tick === tick;
          return (
            <li key={entry.tick}
              className={`action${entry.diverged ? " action--diverged" : ""}${current ? " is-current" : ""}`}
              data-tick={entry.tick}>
              <button type="button" onClick={() => onTickChange?.(entry.tick)} title={badge?.title}>
                <span className="action-tick">t{entry.tick}</span>
                <span className="action-actuated">{actionLabel(entry.actuated)}</span>
                {entry.diverged && <span className="action-voluntary">assumed {actionLabel(entry.voluntary)}</span>}
                {badge && <span className="action-badge">{badge.text}</span>}
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
