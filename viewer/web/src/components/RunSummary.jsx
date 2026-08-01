import { formatMetric, ratioAgainstBaseline, shortHash } from "../lib/format.js";

function Stat({ label, value, detail }) {
  return (
    <div className="run-summary__stat">
      <span className="run-summary__label">{label}</span>
      <strong className="run-summary__value">{value}</strong>
      {detail && <span className="run-summary__detail">{detail}</span>}
    </div>
  );
}

/** The clinic-header slice of a run's experiment_report.json: what model,
 * how it trained, how it scored against copy-last at its earliest reported
 * horizon, and whether it was promoted. */
export function RunSummary({ summary }) {
  const promoted = summary?.promotion?.promoted;
  const statusText = promoted === true ? "Promoted" : promoted === false ? "Not promoted" : "Report unavailable";
  const statusClass = promoted === true ? " run-summary__status--good" : promoted === false ? " run-summary__status--bad" : "";

  return (
    <section className="run-summary" aria-label="Run summary">
      <h3 className="run-summary__heading">Run summary<span className={`run-summary__status${statusClass}`}>{statusText}</span></h3>
      <div className="run-summary__grid">
        {!summary ? (
          <Stat label="Run report" value="No summary recorded" detail="Training and evaluation statistics are unavailable." />
        ) : (() => {
          const training = summary.training || {}, evaluation = summary.evaluation || {}, model = summary.model || {};
          const checkpoint = model.checkpoint_sha256 ? `checkpoint ${shortHash(model.checkpoint_sha256)}` : "checkpoint not recorded";
          const scale = [training.episodes != null ? `${training.episodes} episodes` : null, training.samples != null ? `${training.samples} samples` : null]
            .filter(Boolean).join(" · ");
          const comparison = ratioAgainstBaseline(evaluation.model_over_copy_last_mse);
          return (
            <>
              <Stat label="Model" value={model.type || "Not recorded"} detail={checkpoint} />
              <Stat label="Training" value={training.epochs != null ? `${training.epochs} epochs` : "Not recorded"} detail={scale || training.objective || ""} />
              <Stat label="Final loss" value={formatMetric(training.final_total_loss)} detail={training.objective || ""} />
              <Stat label={`t+${evaluation.horizon || 1} MSE`} value={formatMetric(evaluation.model_mse)} detail={comparison} />
              <Stat label="Rollout" value={evaluation.rollout_health || "Not reported"} detail={[evaluation.split, evaluation.prediction_mode].filter(Boolean).join(" · ")} />
            </>
          );
        })()}
      </div>
    </section>
  );
}
