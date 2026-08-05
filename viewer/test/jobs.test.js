"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const jobs = require("../lib/jobs");

const FAKE_PYTHON = path.join(__dirname, "fixtures", "fake-python.sh");

function fixtureRunsRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "clinic-jobs-"));
}

async function waitFor(predicate, { timeoutMs = 5000, intervalMs = 20 } = {}) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() > deadline) throw new Error("waitFor timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

test("jobs", async (t) => {
  let originalPython;
  t.before(() => { originalPython = process.env.PYTHON; process.env.PYTHON = FAKE_PYTHON; });
  t.after(() => { process.env.PYTHON = originalPython; });

  await t.test("launchJob(baseline) registers a running job, then reconciles to completed", async () => {
    const runsDir = fixtureRunsRoot();
    const spec = { organism: "Crafter", mode: "fresh" };
    const entry = jobs.launchJob("baseline", { organism: "Crafter", spec }, { runsDir });

    assert.equal(entry.kind, "baseline");
    assert.equal(entry.organism, "Crafter");
    assert.equal(entry.status, "running");
    assert.equal(entry.run_ids.length, 1);
    assert.match(entry.run_ids[0], /^job-/);
    assert.ok(entry.pid);
    // The spec document was written to a temp file and named as the
    // positional argument, not inlined into argv.
    assert.match(entry.argv.join(" "), /factory baseline .*\.spec\.json --root/);
    const specFile = entry.argv[entry.argv.indexOf("baseline") + 1];
    assert.deepEqual(JSON.parse(fs.readFileSync(specFile, "utf8")), spec);

    const finished = await waitFor(() => {
      const current = jobs.getJob(runsDir, entry.job_id);
      return current.status !== "running" ? current : null;
    });
    assert.equal(finished.status, "completed");
    assert.equal(finished.exit_code, 0);
  });

  await t.test("launchJob(baseline) reconciles to failed on a non-zero exit code", async () => {
    const runsDir = fixtureRunsRoot();
    process.env.FAKE_PYTHON_EXIT_CODE = "3";
    try {
      const entry = jobs.launchJob("baseline", { organism: "Crafter", spec: { organism: "Crafter" } }, { runsDir });
      const finished = await waitFor(() => {
        const current = jobs.getJob(runsDir, entry.job_id);
        return current.status !== "running" ? current : null;
      });
      assert.equal(finished.status, "failed");
      assert.equal(finished.exit_code, 3);
    } finally {
      delete process.env.FAKE_PYTHON_EXIT_CODE;
    }
  });

  await t.test("launchJob(clone) names an explicit run id and passes the parent run positionally", () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("clone", { organism: "Crafter", run: "parent-1", run_id: "clone-1" }, { runsDir });
    assert.deepEqual(entry.run_ids, ["clone-1"]);
    assert.match(entry.argv.join(" "), /factory clone parent-1 --root .* --factory-run-id clone-1/);
  });

  await t.test("launchJob(resume) uses the existing run's own id, never a fresh one", () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("resume", { organism: "Crafter", run: "interrupted-1" }, { runsDir });
    assert.deepEqual(entry.run_ids, ["interrupted-1"]);
    assert.match(entry.argv.join(" "), /factory resume interrupted-1 --root/);
    assert.doesNotMatch(entry.argv.join(" "), /--factory-run-id/);
  });

  await t.test("launchJob(breed) passes both parents positionally and names the child", () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob(
      "breed", { organism: "Crafter", parent_a: "run-a", parent_b: "run-b", run_id: "bred-1" }, { runsDir },
    );
    assert.deepEqual(entry.run_ids, ["bred-1"]);
    assert.match(entry.argv.join(" "), /factory breed run-a run-b --root .* --factory-run-id bred-1/);
  });

  await t.test("launchJob(search) precomputes population 0's run id grid and passes options through as flags", () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("search", {
      organism: "Crafter", spec: { organism: "Crafter" },
      options: { population_size: 3, seed: 11, dry_run: true },
    }, { runsDir });
    assert.equal(entry.run_ids.length, 3);
    const prefix = entry.run_ids[0].replace(/-p0-c0$/, "");
    assert.deepEqual(entry.run_ids, [`${prefix}-p0-c0`, `${prefix}-p0-c1`, `${prefix}-p0-c2`]);
    assert.match(entry.argv.join(" "), /--population-size 3/);
    assert.match(entry.argv.join(" "), /--seed 11/);
    assert.match(entry.argv.join(" "), /--dry-run\b/);
    assert.match(entry.argv.join(" "), new RegExp(`--run-id-prefix ${prefix}`));
  });

  await t.test("launchJob(search) honors an explicit run_id_prefix and legacy halving's r0 naming", () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("search", {
      organism: "Crafter", spec: { organism: "Crafter" },
      options: { run_id_prefix: "sh-preview", candidates: 2, budgets: [1, 2] },
    }, { runsDir });
    assert.deepEqual(entry.run_ids, ["sh-preview-r0-c0", "sh-preview-r0-c1"]);
  });

  await t.test("launchJob rejects an unknown kind without spawning anything", () => {
    const runsDir = fixtureRunsRoot();
    assert.throws(() => jobs.launchJob("evolve-forever", { organism: "Crafter" }, { runsDir }), /unknown job kind/);
    assert.deepEqual(jobs.listJobs(runsDir), []);
  });

  await t.test("launchJob rejects a request missing its kind-specific identity field", () => {
    const runsDir = fixtureRunsRoot();
    assert.throws(() => jobs.launchJob("clone", { organism: "Crafter" }, { runsDir }), /requires a parent run id/);
    assert.throws(() => jobs.launchJob("resume", { organism: "Crafter" }, { runsDir }), /requires a run id/);
    assert.throws(
      () => jobs.launchJob("breed", { organism: "Crafter", parent_a: "a" }, { runsDir }),
      /requires parent_a and parent_b/,
    );
    assert.throws(() => jobs.launchJob("baseline", { organism: "Crafter" }, { runsDir }), /requires a spec document/);
  });

  await t.test("launchJob requires organism", () => {
    const runsDir = fixtureRunsRoot();
    assert.throws(() => jobs.launchJob("resume", { run: "x" }, { runsDir }), /organism is required/);
  });

  await t.test("listJobs filters by organism and sorts newest first", async () => {
    const runsDir = fixtureRunsRoot();
    const a = jobs.launchJob("resume", { organism: "Crafter", run: "a" }, { runsDir });
    await new Promise((resolve) => setTimeout(resolve, 5));
    const b = jobs.launchJob("resume", { organism: "Crafter", run: "b" }, { runsDir });
    jobs.launchJob("resume", { organism: "Other", run: "c" }, { runsDir });

    const crafterJobs = jobs.listJobs(runsDir, { organism: "Crafter" });
    assert.deepEqual(crafterJobs.map((j) => j.job_id), [b.job_id, a.job_id]);
    assert.equal(jobs.listJobs(runsDir).length, 3);
    assert.equal(jobs.listJobs(runsDir, { organism: "Nobody" }).length, 0);
  });

  await t.test("getJob reconciles a running entry whose process has actually exited (e.g. after a server restart)", () => {
    const runsDir = fixtureRunsRoot();
    // A pid this test process definitely isn't -- Number.MAX_SAFE_INTEGER
    // is never a real pid.
    const deadPid = 2 ** 31 - 1;
    const entry = {
      format: "clinic-job-v1", job_id: "stale-job", kind: "resume", argv: [], pid: deadPid,
      log_path: path.join(jobs.jobsDir(runsDir), "stale-job.log"), started_at: new Date().toISOString(),
      status: "running", exit_code: null, error: null, organism: "Crafter", run_ids: ["r1"],
    };
    fs.mkdirSync(jobs.jobsDir(runsDir), { recursive: true });
    fs.writeFileSync(path.join(jobs.jobsDir(runsDir), "stale-job.json"), JSON.stringify(entry));

    const reconciled = jobs.getJob(runsDir, "stale-job");
    assert.equal(reconciled.status, "unknown");
    // Persisted, not just returned -- a second read must not re-derive it.
    const reread = JSON.parse(fs.readFileSync(path.join(jobs.jobsDir(runsDir), "stale-job.json"), "utf8"));
    assert.equal(reread.status, "unknown");
  });

  await t.test("cancelJob asks the CLI to cancel every precomputed run and SIGTERMs a still-running subprocess", async () => {
    const runsDir = fixtureRunsRoot();
    process.env.FAKE_PYTHON_DELAY_MS = "2000";
    try {
      const entry = jobs.launchJob("resume", { organism: "Crafter", run: "slow-run" }, { runsDir });
      await waitFor(() => isProcessRunning(entry.pid) || null, { timeoutMs: 1000 });

      const startedWaitingAt = Date.now();
      const cancelled = await jobs.cancelJob(runsDir, entry.job_id);
      assert.equal(cancelled.job_id, entry.job_id);
      // cancelJob polls (non-blockingly) for the SIGTERM'd subprocess to
      // actually exit before force-reconciling its state, capped at
      // killGraceMs (default 3000ms) -- this must return in well under the
      // full 2000ms FAKE_PYTHON_DELAY_MS a graceful exit would have taken,
      // proving SIGTERM was honored promptly rather than the call having
      // simply waited out the fake job's own delay.
      assert.ok(Date.now() - startedWaitingAt < 1500);

      const finished = await waitFor(() => {
        const current = jobs.getJob(runsDir, entry.job_id);
        return current.status !== "running" ? current : null;
      });
      // SIGTERM with no handler in the fake script kills it outright --
      // never "completed", the exit code a graceful stop would report.
      assert.notEqual(finished.status, "completed");
    } finally {
      delete process.env.FAKE_PYTHON_DELAY_MS;
    }
  });

  await t.test("cancelJob returns promptly (no forced-kill reconciliation) when the subprocess already exited on its own", async () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("resume", { organism: "Crafter", run: "fast-run" }, { runsDir });
    await waitFor(() => {
      const current = jobs.getJob(runsDir, entry.job_id);
      return current.status !== "running" ? current : null;
    });

    const startedWaitingAt = Date.now();
    const cancelled = await jobs.cancelJob(runsDir, entry.job_id);
    assert.equal(cancelled.job_id, entry.job_id);
    // No live pid to SIGTERM or wait out -- must not pay any of
    // killGraceMs's grace period.
    assert.ok(Date.now() - startedWaitingAt < 500);
  });

  await t.test("cancelJob is a no-op for an unknown job id", async () => {
    const runsDir = fixtureRunsRoot();
    assert.equal(await jobs.cancelJob(runsDir, "no-such-job"), null);
  });

  await t.test("readJobLog tails new content by byte offset without re-sending what the caller already has", async () => {
    const runsDir = fixtureRunsRoot();
    const entry = jobs.launchJob("resume", { organism: "Crafter", run: "logged-run" }, { runsDir });
    await waitFor(() => {
      const current = jobs.getJob(runsDir, entry.job_id);
      return current.status !== "running" ? current : null;
    });

    const first = jobs.readJobLog(runsDir, entry.job_id, { offset: 0 });
    assert.match(first.text, /argv: -m cognitive_runtime\.cli factory resume logged-run/);
    assert.ok(first.next_offset > 0);

    const second = jobs.readJobLog(runsDir, entry.job_id, { offset: first.next_offset });
    assert.equal(second.text, "");
    assert.equal(second.next_offset, first.next_offset);
  });

  await t.test("readJobLog returns null for an unknown job id", () => {
    const runsDir = fixtureRunsRoot();
    assert.equal(jobs.readJobLog(runsDir, "no-such-job", { offset: 0 }), null);
  });
});

function isProcessRunning(pid) {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
}
