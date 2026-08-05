import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";

const RUNNING_RUN_STATES = new Set(["queued", "running", "checkpointing"]);

/** Mirrors FactoryRunsPanel's own state-badge convention (running/queued as
 * "info", failed as "red", cancelled/budget_exceeded as an "amber" soft
 * warning, completed as "green") -- kept as its own small local helper
 * rather than importing FactoryRunsPanel's, the same "duplicate a tiny
 * labeled-badge helper per panel" convention Stat/Field already follow. */
function jobBadgeClass(status) {
  if (status === "running") return "info";
  if (status === "completed") return "green";
  if (status === "failed") return "red";
  if (status === "unknown") return "amber";
  return "neutral";
}

function runBadgeClass(state) {
  if (RUNNING_RUN_STATES.has(state)) return "info";
  if (state === "failed") return "red";
  if (state === "budget_exceeded" || state === "cancelled") return "amber";
  if (state === "completed") return "green";
  return "neutral";
}

/** Tails one job's combined stdout+stderr log by byte offset, polling for
 * new content while mounted -- unmounted (collapsed) the moment its row
 * collapses, so only the currently expanded job pays for repeated polling. */
function JobLog({ jobId }) {
  const [text, setText] = useState("");
  const [error, setError] = useState(null);
  const offsetRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    offsetRef.current = 0;
    setText("");
    setError(null);
    async function poll() {
      try {
        const result = await api.jobLog(jobId, offsetRef.current);
        if (cancelled) return;
        if (result.text) setText((prev) => prev + result.text);
        offsetRef.current = result.next_offset;
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }
    poll();
    const interval = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [jobId]);

  if (error) return <p className="no-data">log unavailable: {error}</p>;
  return <pre className="job-log">{text || "(no output yet)"}</pre>;
}

/**
 * The control plane's job registry for one organism (`GET /api/jobs`):
 * every `ccr factory <kind>` subprocess the clinic has launched, its
 * precomputed runs' live state (reusing FactoryRunsPanel's own badge
 * convention), a cancel action for anything still running, and an
 * expandable log tail per job. Shared by Build (and, later, Evolve/Breed)
 * rather than reimplemented per tab.
 */
export function JobsPanel({ organism, refreshToken, highlightJobId }) {
  const [jobs, setJobs] = useState(null);
  const [expanded, setExpanded] = useState(highlightJobId ?? null);
  const [cancelling, setCancelling] = useState(null);

  useEffect(() => {
    if (!organism) { setJobs(null); return undefined; }
    let cancelled = false;
    async function poll() {
      try {
        const list = await api.jobs(organism);
        if (!cancelled) setJobs(list);
      } catch {
        if (!cancelled) setJobs([]);
      }
    }
    poll();
    const interval = setInterval(poll, 3000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [organism, refreshToken]);

  useEffect(() => { if (highlightJobId) setExpanded(highlightJobId); }, [highlightJobId]);

  async function handleCancel(jobId) {
    setCancelling(jobId);
    try { await api.cancelJob(jobId); } finally { setCancelling(null); }
  }

  return (
    <section className="diagnostic jobs-panel" aria-labelledby="jobs-title">
      <h3 id="jobs-title">Jobs</h3>
      <p>Every Model Factory job launched from this clinic for the selected organism, newest first.</p>
      {!jobs ? (
        <p className="empty">Loading jobs…</p>
      ) : !jobs.length ? (
        <p className="no-data">no jobs launched yet</p>
      ) : (
        <ul className="job-list">
          {jobs.map((job) => (
            <li key={job.job_id} className="job-row">
              <div className="job-row__header">
                <button
                  type="button" className="job-row__toggle"
                  aria-expanded={expanded === job.job_id}
                  onClick={() => setExpanded((current) => (current === job.job_id ? null : job.job_id))}
                >
                  {expanded === job.job_id ? "▾" : "▸"} {job.kind}
                </button>
                <span className={`state-badge state-badge--${jobBadgeClass(job.status)}`}>{job.status}</span>
                <span className="job-row__started">{job.started_at}</span>
                {job.status === "running" && (
                  <button type="button" disabled={cancelling === job.job_id} onClick={() => handleCancel(job.job_id)}>
                    {cancelling === job.job_id ? "Cancelling…" : "Cancel"}
                  </button>
                )}
              </div>
              <div className="job-row__runs">
                {(job.runs || []).map((run) => (
                  <span key={run.run_id} className="job-run">
                    <span className="job-run__id">{run.run_id}</span>
                    <span className={`state-badge state-badge--${runBadgeClass(run.state)}`}>{run.state || "unknown"}</span>
                  </span>
                ))}
              </div>
              {expanded === job.job_id && <JobLog jobId={job.job_id} />}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
