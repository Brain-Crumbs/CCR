import { formatExponential, formatPercent } from "../lib/format.js";

function Interval({ label, interval }) {
  if (!interval) return null;
  return <div className="run-summary__stat"><span className="run-summary__label">{label}</span><strong className="run-summary__value">[{formatExponential(interval.low)}, {formatExponential(interval.high)}]</strong></div>;
}

/**
 * A candidate-vs-baseline paired comparison over matched validation episodes
 * (metrics/comparison.json, `model-factory-paired-comparison-v1`; epic #212
 * §10.2). `mean_delta` is candidate-minus-baseline on the primary metric, so
 * negative means the candidate wins on an error metric.
 */
export function PairedComparisonPanel({ comparison }) {
  if (!comparison) {
    return (
      <section className="diagnostic comparison" aria-labelledby="comparison-title">
        <h3 id="comparison-title">Paired comparison</h3>
        <p className="no-data">no candidate/baseline comparison recorded for this run (fresh trials have no parent to compare against)</p>
      </section>
    );
  }
  const evaluable = comparison.status === "evaluable";
  return (
    <section className="diagnostic comparison" aria-labelledby="comparison-title">
      <h3 id="comparison-title">Paired comparison <span className={`run-summary__status${evaluable ? " run-summary__status--good" : " run-summary__status--bad"}`}>{comparison.status}</span></h3>
      <p>{comparison.primary_metric} across {comparison.episode_count ?? 0} matched validation episodes.</p>
      {!evaluable && comparison.reason && <p className="no-data">{comparison.reason}</p>}
      {evaluable && (
        <div className="run-summary__grid">
          <div className="run-summary__stat">
            <span className="run-summary__label">Mean delta (candidate − baseline)</span>
            <strong className="run-summary__value">{formatExponential(comparison.mean_delta)}</strong>
            <span className="run-summary__detail">{comparison.mean_delta < 0 ? "candidate improves on baseline" : "baseline still leads"}</span>
          </div>
          <div className="run-summary__stat">
            <span className="run-summary__label">Win rate</span>
            <strong className="run-summary__value">{formatPercent(comparison.win_rate)}</strong>
            <span className="run-summary__detail">confidence {formatPercent(comparison.confidence, 0)}</span>
          </div>
          <Interval label="Paired bootstrap CI" interval={comparison.intervals?.paired_bootstrap} />
          <Interval label="Paired permutation CI" interval={comparison.intervals?.paired_permutation} />
        </div>
      )}
    </section>
  );
}
