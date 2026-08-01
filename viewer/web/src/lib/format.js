export function formatMetric(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toPrecision(digits) : "Not reported";
}

export function formatExponential(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toExponential(digits) : "–";
}

export function shortHash(hash, length = 12) {
  return typeof hash === "string" && hash ? hash.slice(0, length) : "unknown";
}

/** "12.3% below copy-last" / "4.0% above copy-last" for a model/baseline MSE ratio. */
export function ratioAgainstBaseline(ratio, baselineLabel = "copy-last") {
  if (typeof ratio !== "number" || !Number.isFinite(ratio)) return `${baselineLabel} comparison unavailable`;
  return `${Math.abs((ratio - 1) * 100).toFixed(1)}% ${ratio <= 1 ? "below" : "above"} ${baselineLabel}`;
}

export function formatPercent(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "–";
}
