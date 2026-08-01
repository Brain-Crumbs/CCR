import { useMemo } from "react";
import { episodeDiagnostics } from "../lib/diagnostics.js";
import { Sparkline } from "./Sparkline.jsx";

const COLORS = { dopamine: "#7b55c7", acetylcholine: "#167c80", adrenaline: "#c34b36", prediction_error: "#b27a00" };

/** Neuromodulation, prediction error, and arbiter-mode state, tick by tick --
 * "the organism flipping bored→curious→afraid over a recorded session"
 * (phase-8 acceptance criterion). Shares the episode's time cursor with the
 * pixel viewer and the action timeline via `tick`/`onTickChange`. */
export function EEGPanel({ records = [], decisions = [], tick = null, onTickChange }) {
  const diagnostics = useMemo(() => episodeDiagnostics(records, decisions), [records, decisions]);

  const closestModeTick = useMemo(() => {
    if (tick == null || !diagnostics.modes.length) return null;
    return diagnostics.modes.reduce((best, m) => (best === null || Math.abs(m.tick - tick) < Math.abs(best - tick) ? m.tick : best), null);
  }, [diagnostics.modes, tick]);

  return (
    <section className="diagnostic eeg" aria-labelledby="eeg-title">
      <h3 id="eeg-title">EEG</h3>
      <p>Neuromodulation, error, and arbiter state tick by tick.</p>
      <div className="traces">
        {Object.entries(diagnostics.series).map(([name, points]) => (
          <div className="trace" key={name}>
            <strong>{name.replace("_", " ")}</strong>
            <Sparkline points={points} color={COLORS[name]} tick={tick} onSeek={onTickChange} />
          </div>
        ))}
      </div>
      <ol className="mode-timeline" aria-label="arbiter mode timeline">
        {diagnostics.modes.length ? diagnostics.modes.map((m) => (
          <li key={m.tick} className={`mode mode--${m.mode}${m.tick === closestModeTick ? " is-current" : ""}`}
            data-tick={m.tick} tabIndex={0}
            onClick={() => onTickChange?.(m.tick)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onTickChange?.(m.tick))}>
            <span>t{m.tick}</span>{m.mode}
          </li>
        )) : <li className="no-data">arbiter mode not recorded</li>}
      </ol>
    </section>
  );
}
