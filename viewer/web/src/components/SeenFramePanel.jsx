import { PixelCanvas } from "./PixelCanvas.jsx";

function actionLabel(action) {
  return action ? String(action).replace(/_/g, " ") : "–";
}

/**
 * The leading cell of the horizon strip: the pixel frame actually seen at
 * t, paired with what the organism did right then (DecisionRecord.motor_decision) --
 * so a reader gets "pixels + action taken" first, then reads the T+1/T+2/...
 * prediction panels that follow it as one continuous strip, instead of
 * cross-referencing a separate action list.
 */
export function SeenFramePanel({ t, tick, bytes, shape, action }) {
  return (
    <div className="panel seen-panel">
      <h3>t = {t}<span className="seen-panel__tick">tick {tick}</span></h3>
      <div className="strip">
        <figure className="cell"><PixelCanvas bytes={bytes} shape={shape} label={`seen at t=${t}`} /><figcaption>seen</figcaption></figure>
      </div>
      <div className="seen-action">
        {action ? (
          <>
            <span className="seen-action__taken">action taken: <b>{actionLabel(action.actuated)}</b></span>
            {action.diverged && (
              <span className="seen-action__assumed">
                assumed <b>{actionLabel(action.voluntary)}</b>
                {action.motorReflex ? ` — ${actionLabel(action.motorReflex.name)}` : action.caregiverOverride ? " — caregiver override" : ""}
              </span>
            )}
          </>
        ) : <span className="no-data">no motor decision recorded for this tick</span>}
      </div>
    </div>
  );
}
