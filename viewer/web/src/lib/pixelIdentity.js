/** Human-readable labels for a decoded predictions payload: what "model"
 * means in the source selector, and the identity line under the controls
 * that names the exact checkpoint/experiment a strip is showing. */

export function modelSourceLabel(pred) {
  if (!pred) return null;
  if (pred.source === "live-record") return "model (live record)";
  if (pred.format === "pixel-predictions-v1") return "Legacy per-scenario actionless predictor";
  return `joint cortex (${pred.prediction_mode || "unknown"})`;
}

export function identityText(pred) {
  if (!pred) return "No export identity; baselines only.";
  if (pred.format === "pixel-predictions-v1") {
    return "Legacy export — actionless per-scenario model; not comparable to current joint cortex experiments.";
  }
  const experiment = pred.experiment || {}, model = pred.model || {};
  const source = pred.evaluation_source || "unknown";
  const split = /-(train|holdout)-/.exec(source)?.[1] || "unknown";
  return `Experiment ${experiment.experiment_id || "unknown"} · Trace ${experiment.trace_id || "unknown"} · `
    + `Checkpoint hash ${(model.checkpoint_sha256 || "unknown").slice(0, 12)} · Model type ${model.model_type || "unknown"} · `
    + `Backbone ${model.backbone || "unknown"} · Objective ${model.training_objective || "unknown"} · `
    + `Prediction mode ${pred.prediction_mode || "unknown"} · actions ${model.uses_actions ? "on" : "off"}, `
    + `workspace ${model.uses_workspace ? "on" : "off"} · Split ${split} · current joint export`;
}

export function statusText({ nFrames, shape, pred }) {
  const modelNote = pred
    ? `· ${pred.format === "pixel-predictions-v1" ? "legacy actionless" : "joint cortex"} model predictions loaded`
    : "· no model predictions recorded — showing baselines";
  return `${nFrames} frames (${shape.join("×")} ${modelNote})`;
}
