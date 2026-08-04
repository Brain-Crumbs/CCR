"use strict";
/** Model Factory job launch/registry (clinic redesign, control-plane
 * foundation): spawns `ccr factory <kind> ...` as a detached OS subprocess
 * and tracks it in a small JSON registry, so the clinic can drive builds/
 * searches/breeding without reimplementing any of that logic in Node --
 * every job kind shells out to the exact same CLI a human would type,
 * mirroring server.js's own `qualityVerdict` convention for invoking the
 * Python side (`$PYTHON -m cognitive_runtime.cli ...`, never assuming a
 * `ccr` console script is on PATH).
 *
 * Registry entries live at `<runs-root>/.clinic-jobs/<job_id>.json`, a root
 * of their own outside any organism directory so they can never collide
 * with server.js's `runCatalog` scan (which only descends into organism
 * directories).
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { spawn, spawnSync } = require("child_process");

const REPO_DIR = path.join(__dirname, "..", "..");
const JOBS_SUBDIR = ".clinic-jobs";
const JOB_FORMAT = "clinic-job-v1";

const JOB_KINDS = ["baseline", "clone", "resume", "search", "breed"];

function pythonCommand() {
  return process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function jobsDir(runsDir) { return path.join(path.resolve(runsDir), JOBS_SUBDIR); }
function jobPath(runsDir, jobId) { return path.join(jobsDir(runsDir), `${jobId}.json`); }
function jobLogPath(runsDir, jobId) { return path.join(jobsDir(runsDir), `${jobId}.log`); }
function jobSpecPath(runsDir, jobId) { return path.join(jobsDir(runsDir), `${jobId}.spec.json`); }

function readJSON(file, fallback = null) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return fallback; }
}

/** Write-to-temp-then-rename: a reader never observes a partially written
 * registry entry, the same discipline every other Model Factory manifest
 * in this codebase uses (atomic_write_json on the Python side). */
function writeJSONAtomic(file, payload) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload, null, 2));
  fs.renameSync(tmp, file);
}

function isAlive(pid) {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
}

/** Convert a flat options map into CLI flags: a snake_case key becomes a
 * --kebab-case flag; an array repeats/spaces the flag's values (nargs="+"
 * style); `true` passes the bare flag (argparse action="store_true");
 * `false`/null/undefined are omitted entirely, never passed as a literal
 * string. Every subcommand's own argparse (and, beneath that,
 * spec.resolve()/validate()) is what actually validates these values --
 * this is a thin, generic passthrough, not a reimplementation of any
 * command's argument schema. */
function flagsFromOptions(options) {
  const argv = [];
  for (const [key, value] of Object.entries(options || {})) {
    if (value === null || value === undefined || value === false) continue;
    const flag = `--${key.replace(/_/g, "-")}`;
    if (value === true) { argv.push(flag); continue; }
    if (Array.isArray(value)) { argv.push(flag, ...value.map(String)); continue; }
    argv.push(flag, String(value));
  }
  return argv;
}

function writeTempSpec(runsDir, jobId, specDocument) {
  const file = jobSpecPath(runsDir, jobId);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(specDocument));
  return file;
}

/** Build one job's CLI argv (after the `factory <kind>` prefix the caller
 * adds) plus its precomputed run_ids, or throw a descriptive Error the
 * route handler turns into a 400 -- shape validation only (a required
 * field is missing); identity validation (does this run/organism/corpus
 * actually exist) is the route handler's job, exactly like server.js's
 * existing read routes already validate organism/run against the
 * enumerated catalog before this is ever called. */
function buildLaunch(kind, body, { runsDir, jobId }) {
  const options = { ...(body.options || {}) };
  const root = ["--root", path.resolve(runsDir)];

  if (kind === "baseline" || kind === "clone" || kind === "breed") {
    // baseline/clone/breed auto-generate a run id unless one is pinned;
    // the job registry must know it up front to precompute run_ids, so
    // this always pins one explicitly via --factory-run-id (added
    // alongside this job registry specifically for that reason -- see
    // cli.py). Never the global --run-id, which only names a trace
    // directory and is unrelated.
    options.factory_run_id = body.run_id || `job-${jobId}`;
  }

  switch (kind) {
    case "baseline": {
      if (!body.spec) throw new Error("baseline requires a spec document (body.spec)");
      const specFile = writeTempSpec(runsDir, jobId, body.spec);
      return { argv: [specFile, ...root, ...flagsFromOptions(options)], runIds: [options.factory_run_id] };
    }
    case "clone": {
      if (!body.run) throw new Error("clone requires a parent run id (body.run)");
      return { argv: [body.run, ...root, ...flagsFromOptions(options)], runIds: [options.factory_run_id] };
    }
    case "resume": {
      if (!body.run) throw new Error("resume requires a run id (body.run)");
      return { argv: [body.run, ...root, ...flagsFromOptions(options)], runIds: [body.run] };
    }
    case "breed": {
      if (!body.parent_a || !body.parent_b) throw new Error("breed requires parent_a and parent_b run ids");
      return {
        argv: [body.parent_a, body.parent_b, ...root, ...flagsFromOptions(options)],
        runIds: [options.factory_run_id],
      };
    }
    case "search": {
      if (!body.spec) throw new Error("search requires a spec document (body.spec)");
      const specFile = writeTempSpec(runsDir, jobId, body.spec);
      const prefix = options.run_id_prefix || `job-${jobId}`;
      options.run_id_prefix = prefix;
      const legacyHalving = Array.isArray(options.budgets) && options.budgets.length > 0;
      const populationSize = Math.max(1, Number(options.population_size ?? options.candidates ?? 8));
      // Only the first population/rung is reliably precomputable without
      // reimplementing search.py's own survivor-selection/breeding
      // decisions in JS: a carried-forward survivor keeps its *original*
      // run_id in later populations rather than a fresh {prefix}-p{N}-c{i}
      // one (search.py's own _EvolutionMember.evaluation is never
      // reassigned a new run_id when carried), so simulating the full set
      // would mean duplicating that selection logic, not just a naming
      // convention. GET /api/jobs still surfaces every run beyond
      // population 0 once it exists on disk, via /api/factory-runs' own
      // catalog scan -- this list is an early-visibility head start, not
      // the only way the UI ever learns about a run.
      const runIds = Array.from(
        { length: populationSize },
        (_, i) => `${prefix}-${legacyHalving ? "r0" : "p0"}-c${i}`,
      );
      return { argv: [specFile, ...root, ...flagsFromOptions(options)], runIds };
    }
    default:
      throw new Error(`unknown job kind ${JSON.stringify(kind)}; expected one of ${JOB_KINDS.join(", ")}`);
  }
}

/** Launch `ccr factory <kind> ...` as a detached subprocess, register it,
 * and return the registry entry immediately (the caller never waits for
 * training to finish -- a single call can run for hours). */
function launchJob(kind, body, { runsDir }) {
  if (!JOB_KINDS.includes(kind)) {
    throw new Error(`unknown job kind ${JSON.stringify(kind)}; expected one of ${JOB_KINDS.join(", ")}`);
  }
  if (!body || typeof body.organism !== "string" || !body.organism) {
    throw new Error("organism is required");
  }
  const jobId = crypto.randomUUID();
  const { argv, runIds } = buildLaunch(kind, body, { runsDir, jobId });
  const fullArgv = ["-m", "cognitive_runtime.cli", "factory", kind, ...argv];
  const logPath = jobLogPath(runsDir, jobId);
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  const logFd = fs.openSync(logPath, "a");
  let child;
  try {
    child = spawn(pythonCommand(), fullArgv, {
      cwd: REPO_DIR, detached: true, stdio: ["ignore", logFd, logFd], env: process.env,
    });
  } finally {
    fs.closeSync(logFd);
  }

  const baseEntry = {
    format: JOB_FORMAT, job_id: jobId, kind, argv: fullArgv, pid: child.pid ?? null,
    log_path: logPath, started_at: new Date().toISOString(), status: "running", exit_code: null,
    error: null, organism: body.organism, run_ids: runIds,
  };
  writeJSONAtomic(jobPath(runsDir, jobId), baseEntry);

  child.on("exit", (code) => {
    const current = readJSON(jobPath(runsDir, jobId), baseEntry);
    writeJSONAtomic(jobPath(runsDir, jobId), {
      ...current, status: code === 0 ? "completed" : "failed", exit_code: code,
    });
  });
  child.on("error", (err) => {
    const current = readJSON(jobPath(runsDir, jobId), baseEntry);
    writeJSONAtomic(jobPath(runsDir, jobId), {
      ...current, status: "failed", error: String((err && err.message) || err),
    });
  });
  child.unref();

  return baseEntry;
}

/** One job's registry entry, reconciled against real process liveness --
 * a server restart loses the in-memory `exit`/`error` handlers above, so a
 * `status: "running"` entry whose pid is actually gone is corrected to
 * "unknown" (its subprocess is gone but no exit code was ever observed;
 * the run's own state.json is the authoritative outcome, not this field)
 * rather than left silently stale. */
function getJob(runsDir, jobId) {
  const entry = readJSON(jobPath(runsDir, jobId));
  if (!entry) return null;
  if (entry.status === "running" && !isAlive(entry.pid)) {
    const reconciled = { ...entry, status: "unknown" };
    writeJSONAtomic(jobPath(runsDir, jobId), reconciled);
    return reconciled;
  }
  return entry;
}

/** Every registered job, optionally filtered to one organism, newest first. */
function listJobs(runsDir, { organism = null } = {}) {
  const dir = jobsDir(runsDir);
  if (!fs.existsSync(dir)) return [];
  const ids = fs.readdirSync(dir).filter((name) => name.endsWith(".json")).map((name) => name.slice(0, -".json".length));
  return ids
    .map((jobId) => getJob(runsDir, jobId))
    .filter((entry) => entry && (!organism || entry.organism === organism))
    .sort((a, b) => (b.started_at || "").localeCompare(a.started_at || ""));
}

/** Ask every one of a job's precomputed runs to stop gracefully at their
 * next checkpoint boundary (state.request_cancellation via the CLI, never
 * reimplemented here), then -- since a flag file only helps a worker that
 * is still checking it -- SIGTERM the subprocess itself so a hung or
 * long-idle job still actually stops. Returns the (possibly already
 * terminal) job entry, or null if the job id is unknown. */
function cancelJob(runsDir, jobId) {
  const entry = getJob(runsDir, jobId);
  if (!entry) return null;
  for (const runId of entry.run_ids) {
    spawnSync(pythonCommand(), [
      "-m", "cognitive_runtime.cli", "factory", "cancel", runId,
      "--root", path.resolve(runsDir), "--organism", entry.organism,
    ], { cwd: REPO_DIR });
  }
  if (isAlive(entry.pid)) {
    try { process.kill(entry.pid, "SIGTERM"); } catch { /* already gone */ }
  }
  return entry;
}

/** Tail a job's combined stdout+stderr log from a byte offset -- cheap,
 * repeated polling for a live log view without ever re-sending what the
 * caller already has. */
function readJobLog(runsDir, jobId, { offset = 0 } = {}) {
  const entry = readJSON(jobPath(runsDir, jobId));
  if (!entry) return null;
  if (!fs.existsSync(entry.log_path)) return { text: "", next_offset: 0, size: 0 };
  const size = fs.statSync(entry.log_path).size;
  const start = Math.max(0, Math.min(Number(offset) || 0, size));
  if (start >= size) return { text: "", next_offset: size, size };
  const fd = fs.openSync(entry.log_path, "r");
  try {
    const buffer = Buffer.alloc(size - start);
    fs.readSync(fd, buffer, 0, buffer.length, start);
    return { text: buffer.toString("utf8"), next_offset: size, size };
  } finally {
    fs.closeSync(fd);
  }
}

module.exports = {
  JOB_KINDS,
  flagsFromOptions,
  buildLaunch,
  launchJob,
  getJob,
  listJobs,
  cancelJob,
  readJobLog,
  jobsDir,
};
