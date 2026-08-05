#!/usr/bin/env node
/** Read-only clinic service over streams-v2 Record sessions. */
"use strict";

const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const path = require("path");
const { spawnSync } = require("child_process");
const jobs = require("./lib/jobs");

const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_DIR = path.join(__dirname, "..");

function parseArgs(argv) {
  const args = {
    dataDir: null, runsDir: null, episodeCacheDir: null, corpusRoot: null, corpusSpecsDir: null, port: 8787,
    // The job-launch API (POST /api/jobs/*, /api/preview/*) is unauthenticated,
    // so the server binds to localhost only unless an operator explicitly
    // opts into wider exposure -- see viewer/README.md's Security section.
    // Node's own http.Server.listen(port, callback) (no host argument, the
    // shape this file used before --host existed) binds every interface, not
    // just localhost, despite the startup message printing a localhost URL.
    host: "127.0.0.1",
    maxConcurrentJobs: 2,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--data-dir") args.dataDir = argv[++i];
    else if (argv[i] === "--runs-dir") args.runsDir = argv[++i];
    else if (argv[i] === "--episode-cache-dir") args.episodeCacheDir = argv[++i];
    else if (argv[i] === "--corpus-root") args.corpusRoot = argv[++i];
    else if (argv[i] === "--corpus-specs-dir") args.corpusSpecsDir = argv[++i];
    else if (argv[i] === "--port") args.port = Number(argv[++i]);
    else if (argv[i] === "--host") args.host = argv[++i];
    else if (argv[i] === "--max-concurrent-jobs") args.maxConcurrentJobs = Number(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") {
      console.log("usage: node server.js [--runs-dir <runs dir>] [--episode-cache-dir <cache dir>] [--corpus-root <corpora dir>] [--corpus-specs-dir <corpus specs dir>] [--data-dir <sessions dir>] [--port 8787] [--host 127.0.0.1] [--max-concurrent-jobs 2]");
      process.exit(0);
    }
  }
  // --data-dir remains the single-session-tree compatibility mode.  The
  // default clinic mode knows where experiment runs and cached recordings
  // live, and presents them as one selected run.
  if (args.dataDir) args.dataDir = path.resolve(args.dataDir);
  else {
    args.runsDir = path.resolve(args.runsDir || path.join(REPO_DIR, "runs"));
    args.episodeCacheDir = path.resolve(args.episodeCacheDir || path.join(REPO_DIR, "notebooks", "episode_cache"));
    args.corpusRoot = path.resolve(args.corpusRoot || path.join(REPO_DIR, "corpora"));
    args.corpusSpecsDir = path.resolve(args.corpusSpecsDir || path.join(REPO_DIR, "specs", "corpora"));
  }
  return args;
}

function readJSON(file, fallback = {}) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return fallback; }
}

function isSessionDir(dir) { return fs.existsSync(path.join(dir, "session.json")); }

function sessionIdsBelow(dataDir, dir = dataDir) {
  if (isSessionDir(dir)) return [path.relative(dataDir, dir).split(path.sep).join("/")];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? sessionIdsBelow(dataDir, path.join(dir, entry.name)) : []
  );
}

function qualityVerdict(dir) {
  const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
  const result = spawnSync(python, ["-m", "cognitive_runtime.record.quality_cli", dir], {
    cwd: REPO_DIR, encoding: "utf8", env: process.env,
  });
  if (result.status === 0) return JSON.parse(result.stdout);
  return { verdict: "red", issues: ["quality check could not be evaluated"], warnings: [] };
}

function qualityStamp(dir) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .reduce((latest, entry) => Math.max(latest, fs.statSync(path.join(dir, entry.name)).mtimeMs), 0);
}

function makeStore(dataDir, { qualityCheck = qualityVerdict, sessionDirectories = null } = {}) {
  dataDir = path.resolve(dataDir);
  const qualityCache = new Map();
  const mounted = sessionDirectories && new Map(sessionDirectories.map(({ id, dir }) => [id, path.resolve(dir)]));
  function sessionDir(sid) {
    if (mounted) {
      const dir = mounted.get(String(sid));
      return dir && isSessionDir(dir) ? dir : null;
    }
    const parts = String(sid).split("/");
    if (!parts.length || parts.some((part) => !/^[\w.-]+$/.test(part))) return null;
    const dir = path.join(dataDir, ...parts);
    return dir.startsWith(dataDir + path.sep) && isSessionDir(dir) ? dir : null;
  }
  function describe(id) {
    const dir = sessionDir(id);
    if (!dir) return null;
    const meta = readJSON(path.join(dir, "session.json"));
    const episodes = fs.readdirSync(dir).filter((f) => /^episode_\d+\.streams\.jsonl$/.test(f))
      .map((f) => f.replace(".streams.jsonl", "")).sort();
    const stamp = qualityStamp(dir), cached = qualityCache.get(id);
    const quality = cached?.stamp === stamp ? cached.value : qualityCheck(dir);
    if (cached?.stamp !== stamp) qualityCache.set(id, { stamp, value: quality });
    const match = id.match(/^nursery-(.+)-(train|holdout)-(\d+)$/);
    const exports = exportsFor(dir).map((entry) => entry.data).filter((entry) => entry?.format === "pixel-predictions-v2");
    const experiments = [...new Map(exports.map((entry) => [entry.experiment?.experiment_id, {
      id: entry.experiment?.experiment_id, prediction_mode: entry.prediction_mode,
      created_at: entry.experiment?.created_at,
      has_entity_event: (entry.events || []).some((event) => event.entity_entered || event.entity_left),
      has_moving: (entry.events || []).some((event) => event.position_changed),
      has_static: (entry.events || []).some((event) => !event.position_changed),
      has_blocked: (entry.events || []).some((event) => event.blocked_forward),
    }])).values()].filter((entry) => entry.id);
    return { id, name: meta.name ?? "legacy", curriculum: meta.curriculum ?? null,
      program: meta.program ?? null, tick_rate: meta.tick_rate ?? null, episodes,
      scenario: match?.[1] ?? null, split: match?.[2] ?? null, seed: match?.[3] ?? null, experiments,
      development: meta.development ?? meta.ladder ?? meta.developmental ?? null,
      quality };
  }
  function list(name = null, filters = {}) {
    if (!mounted && !fs.existsSync(dataDir)) return [];
    const ids = mounted ? [...mounted.keys()] : sessionIdsBelow(dataDir);
    return ids.map(describe).filter((s) => !name || s.name === name)
      .filter((s) => !filters.scenario || s.scenario === filters.scenario)
      .filter((s) => !filters.seed || String(s.seed) === String(filters.seed))
      .filter((s) => !filters.split || s.split === filters.split)
      .filter((s) => !filters.prediction_mode || s.experiments.some((e) => e.prediction_mode === filters.prediction_mode))
      .filter((s) => !filters.entity_event || s.experiments.some((e) => e.has_entity_event))
      .filter((s) => !filters.motion || s.experiments.some((e) => filters.motion === "blocked" ? e.has_blocked : filters.motion === "moving" ? e.has_moving : e.has_static))
      .sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  }
  return { dataDir, sessionDir, describe, list };
}

function inside(root, candidate) {
  const resolved = path.resolve(candidate), base = path.resolve(root);
  return resolved === base || resolved.startsWith(base + path.sep);
}

/** A client-chosen identifier that will be joined into a run directory path
 * (a fresh job's run id, a search campaign's run-id prefix, a job id read
 * back out of a URL) is restricted to the same safe character class
 * makeStore's own sessionDir uses for session ids. Unlike an organism or an
 * existing run -- both checked against the already-enumerated catalog --
 * these name something that doesn't exist yet, so there is no catalog to
 * check them against; a character-class guard is the equivalent floor. */
function isPlainId(value) {
  return typeof value === "string" && /^[\w.-]+$/.test(value) && value !== "." && value !== "..";
}

/** organism/run/parent_a/parent_b/run_id/run_id_prefix validation shared by
 * every route that hands a client-supplied job body to jobs.buildLaunch --
 * a real launch and a dry-run preview carry the same path-safety
 * requirements, since both end up shelling a subprocess out with these
 * values. Returns an {status, error} pair to send on failure, or null once
 * body clears every check. */
function validateJobBody(clinic, body) {
  if (!body.organism) return { status: 400, error: "select an organism" };
  if (!clinic.organisms().includes(body.organism)) {
    return { status: 404, error: "unknown organism" };
  }
  const corpusId = body.spec?.data?.corpus_id;
  if (typeof corpusId === "string" && corpusId && !corpusCatalog(clinic.corpusRoot, body.organism)
    .some((entry) => entry.corpus_id === corpusId)) {
    return { status: 404, error: `unknown corpus: ${corpusId}` };
  }
  for (const field of ["run", "parent_a", "parent_b"]) {
    if (typeof body[field] !== "string") continue;
    if (!clinic.factoryCatalog(body.organism).some((entry) => entry.run === body[field])) {
      return { status: 404, error: `unknown ${field}: ${body[field]}` };
    }
  }
  if (body.run_id !== undefined && !isPlainId(body.run_id)) {
    return { status: 400, error: "run_id must be a plain identifier" };
  }
  if (body.options?.run_id_prefix !== undefined && !isPlainId(body.options.run_id_prefix)) {
    return { status: 400, error: "options.run_id_prefix must be a plain identifier" };
  }
  // `search --champion/--reference-run` name *existing* runs the CLI
  // subprocess resolves into directories under --root, exactly like the
  // positional run/parent_a/parent_b above, so they get the same
  // already-enumerated-catalog check rather than being passed through.
  for (const field of ["champion", "reference_run"]) {
    const value = body.options?.[field];
    if (typeof value !== "string" || !value) continue;
    if (!clinic.factoryCatalog(body.organism).some((entry) => entry.run === value)) {
      return { status: 404, error: `unknown options.${field}: ${value}` };
    }
  }
  // A checkpoint file name is joined into `<run>/checkpoints/<name>` by the
  // CLI (breed's --checkpoint-a/--checkpoint-b are two more of exactly
  // these, one per parent). There is no catalog of checkpoint names to
  // check against, so the character-class floor applies -- the same
  // reasoning isPlainId documents for a not-yet-existing run id.
  for (const field of ["checkpoint", "reference_checkpoint", "checkpoint_a", "checkpoint_b"]) {
    const value = body.options?.[field];
    if (value !== undefined && value !== null && !isPlainId(value)) {
      return { status: 400, error: `options.${field} must be a plain identifier` };
    }
  }
  return null;
}

function runCatalog(runsDir) {
  if (!fs.existsSync(runsDir)) return [];
  return fs.readdirSync(runsDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).flatMap((organism) => {
    const organismDir = path.join(runsDir, organism.name);
    return fs.readdirSync(organismDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).flatMap((run) => {
      const dir = path.join(organismDir, run.name);
      const experimentPath = path.join(dir, "experiment.json"), reportPath = path.join(dir, "experiment_report.json");
      if (!fs.existsSync(experimentPath) && !fs.existsSync(reportPath)) return [];
      const experiment = readJSON(experimentPath, readJSON(reportPath, {}).experiment || {});
      return [{ organism: experiment.organism || organism.name, run: experiment.experiment_id || run.name, dir }];
    });
  }).sort((a, b) => a.organism.localeCompare(b.organism) || b.run.localeCompare(a.run));
}

/** Model Factory runs become monitorable as soon as the runner allocates a
 * state/spec directory, well before experiment.json exists.  The presentation
 * catalog above intentionally remains limited to runs with viewable experiment
 * artifacts; this wider catalog drives Factory, jobs, resume and breeding. */
function factoryRunCatalog(runsDir, organism) {
  const organismDir = path.join(runsDir, organism);
  if (!fs.existsSync(organismDir) || !fs.statSync(organismDir).isDirectory()) return [];
  const markers = ["state.json", "trial_spec.json", "experiment.json", "experiment_report.json"];
  return fs.readdirSync(organismDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && isPlainId(entry.name))
    .flatMap((entry) => {
      const dir = path.join(organismDir, entry.name);
      if (!markers.some((name) => fs.existsSync(path.join(dir, name)))) return [];
      const experiment = readJSON(path.join(dir, "experiment.json"), {});
      const report = readJSON(path.join(dir, "experiment_report.json"), {});
      return [{ organism, run: experiment.experiment_id || report.experiment?.experiment_id || entry.name, dir }];
    })
    .sort((a, b) => b.run.localeCompare(a.run));
}

/** A deliberately small, stable slice of a run report for the clinic header. */
function runSummary(entry) {
  const experiment = readJSON(path.join(entry.dir, "experiment.json"), {});
  const report = readJSON(path.join(entry.dir, "experiment_report.json"), {});
  const training = report.training_stats || {};
  const evaluation = report.metrics?.rollout || report.metrics?.direct || {};
  const horizons = evaluation.horizons || {};
  const horizon = Object.keys(horizons).map(Number).filter(Number.isFinite).sort((a, b) => a - b)[0];
  const metric = Number.isFinite(horizon) ? horizons[String(horizon)] || {} : {};
  const checkpoint = report.checkpoint || {};
  const verdict = report.promotion_verdict || {};
  return {
    organism: entry.organism,
    run: entry.run,
    created_at: experiment.created_at || report.experiment?.created_at || null,
    model: {
      type: checkpoint.model_type || null,
      checkpoint_sha256: checkpoint.sha256 || null,
    },
    training: {
      objective: training.training_objective || null,
      device: training.device || null,
      epochs: training.epochs ?? null,
      episodes: training.episodes ?? null,
      samples: training.samples ?? null,
      final_total_loss: training.final_total_loss ?? null,
    },
    evaluation: {
      split: training.evaluation_mode || null,
      prediction_mode: evaluation.prediction_mode || null,
      horizon: Number.isFinite(horizon) ? horizon : null,
      model_mse: metric.model_mse ?? null,
      copy_last_mse: metric.copy_last_mse ?? null,
      model_over_copy_last_mse: metric.model_over_copy_last_mse ?? null,
      beats_copy_last: metric.beats_copy_last ?? null,
      samples: metric.n_samples ?? null,
      rollout_health: evaluation.rollout_health?.state || null,
    },
    promotion: {
      promoted: typeof verdict.promoted === "boolean" ? verdict.promoted : null,
      reasons: Array.isArray(verdict.reasons) ? verdict.reasons : [],
    },
  };
}

/** runner._SELECTION_METRIC_RE / runner._GOAL_NAVIGATION_METRICS, mirrored
 * for the one read below. */
const SELECTION_METRIC_RE = /^(rollout|direct)\.t\+(\d+)\.([a-zA-Z_][a-zA-Z0-9_]*)$/;
const GOAL_NAVIGATION_METRICS = new Set(["success_rate", "geodesic_efficiency", "collision_rate", "replan_recovery"]);

/** Python's ``round`` -- round-half-to-even -- not JavaScript's
 * ``Math.round``, which breaks a tie away from zero instead. Mirrored
 * exactly because the line below reproduces
 * ``action_world_model.horizons_ticks_to_frames``' own tick-to-frame key
 * derivation: at a ticks_per_frame that lands a horizon precisely on .5
 * (2 ticks/frame with a 5-tick horizon, say) the two roundings disagree,
 * and this would then look up a horizon key the run never recorded and
 * report "no metric yet" for a run that has one. */
function pythonRound(value) {
  const floor = Math.floor(value), remainder = value - floor;
  if (remainder > 0.5) return floor + 1;
  if (remainder < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Resolve one run's own ``evaluation.selection_metric`` against its
 * persisted ``metrics/validation.json`` -- the single number the
 * evolutionary campaign actually selected that candidate on, which
 * state.json alone cannot answer.
 *
 * A port of ``runner._resolve_selection_metric``'s *scalar* half only (its
 * per-episode paired-comparison series has no reader here), keyed the same
 * way: a ``direct.t+N.*`` horizon is tick-denominated as written, while a
 * ``rollout.t+N.*`` horizon is recorded in frames and needs the run's own
 * ``contracts.json`` ticks_per_frame to convert -- the same two files
 * ``registry._load_validation_for_candidate`` reads for exactly this
 * purpose. Anything unresolvable (no validation.json yet, a horizon that
 * was not evaluated, a metric shape this port does not cover) is reported
 * as ``null`` rather than guessed at: a still-running candidate simply has
 * no metric yet, and the board renders that honestly. */
function selectionMetric(dir) {
  const validation = readJSON(path.join(dir, "metrics", "validation.json"), null);
  const metric = typeof validation?.selection_metric === "string" ? validation.selection_metric : null;
  if (!metric) return { selection_metric: null, selection_metric_value: null };
  return { selection_metric: metric, selection_metric_value: resolveSelectionMetric(validation, metric, dir) };
}

function resolveSelectionMetric(validation, metric, dir) {
  const finite = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const navigationPrefix = "goal_navigation.";
  if (metric.startsWith(navigationPrefix)) {
    const name = metric.slice(navigationPrefix.length);
    if (!GOAL_NAVIGATION_METRICS.has(name)) return null;
    return finite(validation.goal_navigation?.[name]);
  }
  const match = SELECTION_METRIC_RE.exec(metric);
  if (!match) return null;
  const [, report, ticksText, stat] = match;
  let horizonKey = Number(ticksText);
  if (report !== "direct") {
    const contracts = readJSON(path.join(dir, "contracts.json"), null);
    const ticksPerFrame = Number(contracts?.data_contract?.ticks_per_frame);
    if (!Number.isFinite(ticksPerFrame) || ticksPerFrame <= 0) return null;
    horizonKey = Math.max(1, pythonRound(horizonKey / ticksPerFrame));
  }
  return finite(validation[report]?.horizons?.[String(horizonKey)]?.[stat]);
}

const ARTIFACT_FILES = [
  "experiment.json", "trial_spec.json", "contracts.json", "lineage.json",
  "data_manifest.json", "execution.json", "experiment_report.json",
];
const METRIC_FILES = ["validation.json", "test.json", "comparison.json"];

/** Every Model Factory manifest recorded for one run (epic #212 §7), read
 * as-is with no reshaping -- the clinic surfaces the factory's own contracts
 * rather than re-deriving them. Missing files are reported as null so a run
 * that predates a given manifest still renders the rest. */
function runArtifacts(dir) {
  const present = (file) => fs.existsSync(path.join(dir, file));
  const manifests = Object.fromEntries(
    ARTIFACT_FILES.map((file) => [file.replace(/\.json$/, ""), present(file) ? readJSON(path.join(dir, file), null) : null])
  );
  const metrics = Object.fromEntries(
    METRIC_FILES.map((file) => {
      const full = path.join(dir, "metrics", file);
      return [file.replace(/\.json$/, ""), fs.existsSync(full) ? readJSON(full, null) : null];
    })
  );
  return { ...manifests, metrics };
}

function registryDocument(runsDir, organism) {
  const file = path.join(runsDir, organism, "registry.json");
  return fs.existsSync(file) ? readJSON(file, null) : { format: "model-factory-registry-v1", slots: {} };
}

/** Every architecture (NAS) campaign's live progress document for one
 * organism (`architecture_search.campaign_progress_path`).
 *
 * The nested campaign is the one workflow whose shape cannot be recovered
 * from run ids: an architecture retained into the next outer generation is
 * deliberately not re-run, so it allocates no runs under that generation's
 * prefix and is indistinguishable, from run directories alone, from one the
 * campaign has not reached yet. The campaign publishes its own retention and
 * breeding decisions as it makes them; this route just hands them over. A
 * campaign that never wrote one (an older run, or a read-only root) simply
 * isn't listed, and the board falls back to what run ids do say. */
function architectureCampaignsDocument(runsDir, organism) {
  const dir = path.join(runsDir, organism, ".architecture-campaigns");
  if (!fs.existsSync(dir)) return { organism, campaigns: [] };
  const campaigns = fs.readdirSync(dir)
    .filter((name) => name.endsWith(".json"))
    .map((name) => readJSON(path.join(dir, name), null))
    .filter((document) => document?.format === "model-factory-architecture-campaign-progress-v1")
    .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
  return { organism, campaigns };
}

/** The Factory tab's organism-wide state view (state.json's convention,
 * `run_directory/state.json`) -- what's queued/running/completed/failed
 * across every run under an organism, which the champion registry alone
 * cannot show since it only ever holds *promoted* runs. */
function factoryRunsDocument(runsDir, organism) {
  const runs = factoryRunCatalog(runsDir, organism)
    .map((entry) => {
      const state = readJSON(path.join(entry.dir, "state.json"), null);
      const lineage = readJSON(path.join(entry.dir, "lineage.json"), null);
      return {
        run: entry.run,
        mode: lineage?.mode ?? null,
        state: state?.state ?? null,
        state_reason: state?.reason ?? null,
        updated_at: state?.updated_at ?? null,
        promoted: runSummary(entry).promotion.promoted,
        // The Evolve tab's population board needs two things state.json
        // cannot answer: what each candidate scored on the campaign's own
        // selection metric, and (for a bred offspring) which two survivors
        // it came from -- lineage.json's `configuration_parents`, which is
        // how the board reconstructs a generation's survivor set at all,
        // since a carried-forward survivor keeps its *original* run id
        // rather than taking a fresh slot-shaped one (see search.py).
        ...selectionMetric(entry.dir),
        configuration_parents: Array.isArray(lineage?.configuration_parents) ? lineage.configuration_parents : [],
      };
    })
    .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
  return { organism, runs };
}

/** A launched job's precomputed run_ids augmented with each run's live
 * state.json/heartbeat.json, reusing factoryRunsDocument's own read
 * pattern -- the job registry only knows what it launched (argv, pid, exit
 * code), not how far training has actually gotten. heartbeat_seconds is
 * passed through exactly as state.write_heartbeat persists it (raw epoch
 * seconds), not reshaped into an ISO string that isn't really there. */
function jobRunStates(runsDir, organism, runIds) {
  return runIds.map((runId) => {
    const dir = path.join(runsDir, organism, runId);
    const state = readJSON(path.join(dir, "state.json"), null);
    const heartbeat = readJSON(path.join(dir, "heartbeat.json"), null);
    return {
      run_id: runId,
      state: state?.state ?? null,
      state_reason: state?.reason ?? null,
      updated_at: state?.updated_at ?? null,
      heartbeat_seconds: heartbeat?.heartbeat_seconds ?? null,
    };
  });
}

const CORPUS_MANIFEST_FORMAT = "model-factory-corpus-manifest-v1";

/** server.js's own port of corpus.list_corpora -- a pure read of
 * corpus_manifest.json files the Python side already wrote, mirroring that
 * function's exact behavior (including which malformed manifests get
 * silently skipped rather than hiding every other corpus) instead of
 * shelling out to Python for something this file can already read
 * directly, the same "read Python-written JSON directly" convention
 * runArtifacts documents above. */
function corpusCatalog(corpusRoot, organism = null) {
  if (!fs.existsSync(corpusRoot)) return [];
  const organismDirs = organism !== null
    ? [{ name: organism, dir: path.join(corpusRoot, organism) }]
    : fs.readdirSync(corpusRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => ({ name: entry.name, dir: path.join(corpusRoot, entry.name) }))
        .sort((a, b) => a.name.localeCompare(b.name));
  const corpora = [];
  for (const { name, dir } of organismDirs) {
    if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) continue;
    const corpusNames = fs.readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
    for (const corpusName of corpusNames) {
      const manifestPath = path.join(dir, corpusName, "corpus_manifest.json");
      if (!fs.existsSync(manifestPath)) continue;
      const manifest = readJSON(manifestPath, null);
      if (!manifest || typeof manifest !== "object" || manifest.format !== CORPUS_MANIFEST_FORMAT) continue;
      const sessions = manifest.sessions && typeof manifest.sessions === "object" && !Array.isArray(manifest.sessions) ? manifest.sessions : {};
      corpora.push({
        corpus_id: String(manifest.corpus_id ?? corpusName),
        organism: name,
        data_contract_hash: manifest.data_contract_hash ?? null,
        session_counts: Object.fromEntries(["train", "validation", "test"].map((split) => [
          split, Array.isArray(sessions[split]) ? sessions[split].length : 0,
        ])),
      });
    }
  }
  return corpora;
}

function corpusSpecCatalog(corpusSpecsDir, organism = null) {
  if (!corpusSpecsDir || !fs.existsSync(corpusSpecsDir)) return [];
  const scalar = (text, key) => {
    const match = text.match(new RegExp(`^${key}:[ \\t]*([^#\\r\\n]+)`, "m"));
    return match ? match[1].trim().replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, "$1$2") : null;
  };
  return fs.readdirSync(corpusSpecsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.(?:json|ya?ml)$/i.test(entry.name) && isPlainId(entry.name))
    .flatMap((entry) => {
      const file = path.join(corpusSpecsDir, entry.name);
      const text = fs.readFileSync(file, "utf8");
      let document = null;
      if (entry.name.toLowerCase().endsWith(".json")) document = readJSON(file, null);
      const item = {
        spec_id: entry.name,
        corpus_id: String(document?.corpus_id ?? scalar(text, "corpus_id") ?? ""),
        organism: String(document?.organism ?? scalar(text, "organism") ?? ""),
        file,
      };
      if (!isPlainId(item.corpus_id) || !isPlainId(item.organism)) return [];
      return organism && item.organism !== organism ? [] : [item];
    })
    .sort((a, b) => a.spec_id.localeCompare(b.spec_id));
}

function directoryOrganisms(root) {
  if (!root || !fs.existsSync(root)) return [];
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && isPlainId(entry.name) && !entry.name.startsWith("."))
    .map((entry) => entry.name);
}

function runRootOrganisms(runsDir) {
  return directoryOrganisms(runsDir).filter((organism) => (
    fs.existsSync(path.join(runsDir, organism, "registry.json"))
    || factoryRunCatalog(runsDir, organism).length > 0
  ));
}

/** The factory's own declared modes/objectives/genome schemas/backbones
 * (`ccr factory meta`, torch-free) -- shelled out to rather than
 * hardcoded/duplicated here, so the clinic's Build/Evolve forms always
 * reflect whatever the backend actually accepts. */
function factoryMeta() {
  const result = spawnSync(jobs.pythonCommand(), [
    ...jobs.pythonPrefixArgs(), "-m", "cognitive_runtime.cli", "factory", "meta",
  ], {
    cwd: REPO_DIR, encoding: "utf8", env: process.env,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error((result.stderr || result.stdout || "ccr factory meta failed").trim());
  return JSON.parse(result.stdout);
}

function exportedCacheSessions(cacheDir, experimentId) {
  if (!fs.existsSync(cacheDir)) return [];
  return sessionIdsBelow(cacheDir).flatMap((id) => {
    const dir = path.join(cacheDir, ...id.split("/"));
    const hasExport = fs.readdirSync(dir).some((file) => file.startsWith(`${experimentId}-predictions_`) && file.endsWith(".json"));
    return hasExport ? [{ id: `cache/${id}`, dir }] : [];
  });
}

function manifestSessions(runDir, cacheDir, corpusRoot) {
  const index = readJSON(path.join(runDir, "clinic_sessions.json"), null);
  if (!index || index.format !== "clinic-session-index-v1" || !Array.isArray(index.sessions)) return [];
  return index.sessions.flatMap((entry) => {
    if (typeof entry.session_dir !== "string") return [];
    const dir = path.resolve(entry.session_dir);
    // A run index may only mount its own recordings, the configured
    // episode cache, or the configured corpus root -- never an arbitrary
    // path (the same discipline as the /api/registry path-traversal fix).
    if (!inside(runDir, dir) && !inside(cacheDir, dir) && !(corpusRoot && inside(corpusRoot, dir))) return [];
    const prefix = inside(cacheDir, dir) ? "cache" : corpusRoot && inside(corpusRoot, dir) ? "corpus" : "run";
    const root = prefix === "cache" ? cacheDir : prefix === "corpus" ? corpusRoot : runDir;
    return isSessionDir(dir) ? [{ id: `${prefix}/${path.relative(root, dir).split(path.sep).join("/")}`, dir }] : [];
  });
}

function makeClinicStore(runsDir, episodeCacheDir, corpusRoot, corpusSpecsDir = path.join(REPO_DIR, "specs", "corpora")) {
  runsDir = path.resolve(runsDir); episodeCacheDir = path.resolve(episodeCacheDir);
  corpusRoot = corpusRoot ? path.resolve(corpusRoot) : null;
  corpusSpecsDir = corpusSpecsDir ? path.resolve(corpusSpecsDir) : null;
  function catalog() { return runCatalog(runsDir); }
  function organisms() {
    return [...new Set([
      ...catalog().map((entry) => entry.organism),
      ...runRootOrganisms(runsDir),
      ...directoryOrganisms(corpusRoot),
      ...corpusSpecCatalog(corpusSpecsDir).map((entry) => entry.organism),
      ...jobs.listJobs(runsDir).map((entry) => entry.organism).filter(Boolean),
    ])].sort();
  }
  function selected(organism, run) {
    const entry = catalog().find((candidate) => candidate.organism === organism && candidate.run === run);
    if (!entry) return null;
    const direct = fs.existsSync(entry.dir) ? sessionIdsBelow(entry.dir).map((id) => ({ id: `run/${id}`, dir: path.join(entry.dir, ...id.split("/")) })) : [];
    const indexed = manifestSessions(entry.dir, episodeCacheDir, corpusRoot);
    const fallback = indexed.length ? [] : exportedCacheSessions(episodeCacheDir, entry.run);
    const mounts = [...new Map([...direct, ...indexed, ...fallback].map((item) => [item.id, item])).values()];
    return { entry, store: makeStore(entry.dir, { sessionDirectories: mounts }) };
  }
  return {
    runsDir, episodeCacheDir, corpusRoot, corpusSpecsDir, catalog, organisms,
    factoryCatalog: (organism) => factoryRunCatalog(runsDir, organism), selected,
  };
}

function loadFrameIndex(dir) {
  const framesDir = path.join(dir, "frames"), index = new Map();
  if (!fs.existsSync(framesDir)) return index;
  for (const name of fs.readdirSync(framesDir).sort()) {
    if (!name.endsWith(".index.jsonl")) continue;
    const bin = path.join(framesDir, name.replace(".index.jsonl", ".bin"));
    for (const line of fs.readFileSync(path.join(framesDir, name), "utf8").split("\n")) {
      if (!line.trim()) continue;
      const rec = JSON.parse(line); index.set(rec.hash, { ...rec, bin });
    }
  }
  return index;
}

function readEpisodeFrames(dir, sid, eid) {
  const records = readStreams(dir, eid); if (!records) return null;
  const decisions = readDecisions(dir, eid) || [];
  const decisionWindows = decisions.flatMap((decision) => {
    const span = decision.window_span;
    return Array.isArray(span) && span.length >= 2
      ? [{ start: Number(span[0]), end: Number(span[1]), tick: decision.tick_index }]
      : [];
  });
  let decisionIndex = 0;
  const index = loadFrameIndex(dir), bins = new Map(), frames = [];
  let shape = null, dtype = null;
  for (const rec of records) {
    if (rec.stream_id !== "vision.frame.pixels") continue;
    shape = rec.shape ?? shape; dtype = rec.dtype ?? dtype;
    while (decisionIndex < decisionWindows.length && rec.timestamp > decisionWindows[decisionIndex].end) {
      decisionIndex += 1;
    }
    const window = decisionWindows[decisionIndex];
    const matchingTick = window && rec.timestamp >= window.start && rec.timestamp <= window.end ? window.tick : null;
    const entry = { i: frames.length, t: rec.timestamp, tick: matchingTick ?? rec.seq ?? frames.length,
      seq: rec.seq, hash: rec.frame_ref ?? null, data: null };
    const loc = entry.hash ? index.get(entry.hash) : null;
    if (loc && !rec.elided) {
      if (!bins.has(loc.bin)) bins.set(loc.bin, fs.readFileSync(loc.bin));
      entry.data = bins.get(loc.bin).subarray(loc.offset, loc.offset + loc.length).toString("base64");
    }
    frames.push(entry);
  }
  return { session_id: sid, episode_id: eid, shape, dtype, n_frames: frames.length, frames };
}

function readEpisodeJSONL(dir, eid, kind) {
  if (!/^episode_\d+$/.test(eid)) return null;
  if (!new Set(["streams", "decisions"]).has(kind)) return null;
  const file = path.join(dir, `${eid}.${kind}.jsonl`); if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).flatMap((line) => {
    try { return [JSON.parse(line)]; } catch { return []; }
  });
}

function readStreams(dir, eid) { return readEpisodeJSONL(dir, eid, "streams"); }
function readDecisions(dir, eid) { return readEpisodeJSONL(dir, eid, "decisions"); }

function exportsFor(dir) {
  return fs.readdirSync(dir).filter((f) => f.endsWith(".json") && f !== "session.json" && !f.endsWith(".summary.json"))
    .sort().map((file) => ({ file, data: readJSON(path.join(dir, file), null) }));
}

function livePredictionsFromDecisions(decisions, sid, eid) {
  const live = decisions.filter((decision) => decision.live_prediction?.prediction_shape);
  if (!live.length) return null;
  const shape = live[0].live_prediction.prediction_shape;
  const horizons = [...new Set(live.flatMap((decision) => Object.keys(decision.live_prediction.frames || {}).map(Number)))]
    .filter((h) => Number.isInteger(h) && h > 0).sort((a, b) => a - b);
  if (!horizons.length) return null;
  const predictions = Object.fromEntries(horizons.map((h) => [String(h), {
    frames: live.slice(0, Math.max(0, live.length - h))
      .map((decision) => decision.live_prediction.frames?.[String(h)]).filter(Boolean),
  }]));
  return {
    format: "pixel-predictions-v1", source: "live-record", session_id: sid, episode_id: eid,
    horizons, prediction_shape: shape, n_frames: live.length,
    predictions, targets: live.map((decision) => decision.live_prediction.target).filter(Boolean),
  };
}

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8" };
function sendJSON(res, status, payload) { res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" }); res.end(JSON.stringify(payload)); }

/** Collect and JSON-parse a request body -- the only place server.js reads
 * one at all, since every route before the job-launch API is GET-only.
 * Capped well above any real spec/options document to guard against an
 * unbounded client stream; rejects (never resolves) past that point. */
function readBody(req, { limit = 2 * 1024 * 1024 } = {}) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0, settled = false;
    const fail = (err) => { if (!settled) { settled = true; reject(err); } };
    const done = (value) => { if (!settled) { settled = true; resolve(value); } };
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) { fail(new Error("request body too large")); req.destroy(); return; }
      chunks.push(chunk);
    });
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw.trim()) return done({});
      try { done(JSON.parse(raw)); } catch { fail(new Error("invalid JSON request body")); }
    });
    req.on("error", fail);
    req.on("aborted", () => fail(new Error("request aborted")));
  });
}
function serveStatic(res, urlPath) {
  const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, ""), file = path.join(PUBLIC_DIR, path.normalize(rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) return sendJSON(res, 404, { error: "not found" });
  fs.readFile(file, (err, data) => { if (err) return sendJSON(res, 404, { error: "not found" }); res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" }); res.end(data); });
}

function createServer({ dataDir = null, runsDir = null, episodeCacheDir = null, corpusRoot = null, corpusSpecsDir = null, maxConcurrentJobs = 2 }) {
  const clinic = dataDir ? null : makeClinicStore(
    runsDir || path.join(REPO_DIR, "runs"),
    episodeCacheDir || path.join(REPO_DIR, "notebooks", "episode_cache"),
    corpusRoot || path.join(REPO_DIR, "corpora"),
    corpusSpecsDir || path.join(REPO_DIR, "specs", "corpora"),
  );
  const store = clinic ? null : makeStore(dataDir);
  function selectedStore(url) {
    if (!clinic) return store;
    const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
    return organism && run ? clinic.selected(organism, run)?.store : null;
  }
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost"), p = url.pathname.split("/").filter(Boolean);
    try {
      if (p[0] !== "api") return serveStatic(res, url.pathname);
      if (p.length === 2 && p[1] === "catalog") return sendJSON(res, 200, {
        runs_dir: clinic?.runsDir ?? null, episode_cache_dir: clinic?.episodeCacheDir ?? null,
        organisms: clinic?.organisms() || [],
        runs: (clinic?.catalog() || []).map(({ organism, run }) => ({ organism, run })),
      });
      if (p.length === 2 && p[1] === "runs") {
        if (!clinic) return sendJSON(res, 400, { error: "run summaries require clinic mode" });
        const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
        const entry = clinic.catalog().find((candidate) => candidate.organism === organism && candidate.run === run);
        if (!entry) return sendJSON(res, 404, { error: "unknown organism or run" });
        return sendJSON(res, 200, runSummary(entry));
      }
      if (p.length === 2 && p[1] === "experiments") {
        if (!clinic) return sendJSON(res, 400, { error: "experiment manifests require clinic mode" });
        const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
        const entry = clinic.catalog().find((candidate) => candidate.organism === organism && candidate.run === run);
        if (!entry) return sendJSON(res, 404, { error: "unknown organism or run" });
        return sendJSON(res, 200, runArtifacts(entry.dir));
      }
      if (p.length === 2 && p[1] === "registry") {
        if (!clinic) return sendJSON(res, 400, { error: "the champion registry requires clinic mode" });
        const organism = url.searchParams.get("organism");
        if (!organism) return sendJSON(res, 400, { error: "select an organism" });
        // organism is joined into a filesystem path below; require it to
        // name a catalogued organism (as the sibling /api/runs and
        // /api/experiments routes already do) rather than trusting client
        // input, which could otherwise walk outside runsDir (e.g. "../../etc").
        if (!clinic.organisms().includes(organism)) return sendJSON(res, 404, { error: "unknown organism" });
        return sendJSON(res, 200, registryDocument(clinic.runsDir, organism));
      }
      if (p.length === 2 && p[1] === "factory-runs") {
        if (!clinic) return sendJSON(res, 400, { error: "factory run state requires clinic mode" });
        const organism = url.searchParams.get("organism");
        if (!organism) return sendJSON(res, 400, { error: "select an organism" });
        // Same containment discipline as /api/registry: organism is joined
        // into a filesystem path, so only a catalogued value is trusted.
        if (!clinic.organisms().includes(organism)) return sendJSON(res, 404, { error: "unknown organism" });
        return sendJSON(res, 200, factoryRunsDocument(clinic.runsDir, organism));
      }
      if (p.length === 2 && p[1] === "architecture-campaigns") {
        if (!clinic) return sendJSON(res, 400, { error: "architecture campaigns require clinic mode" });
        const organism = url.searchParams.get("organism");
        if (!organism) return sendJSON(res, 400, { error: "select an organism" });
        // Same containment discipline as /api/registry and /api/factory-runs:
        // organism is joined into a filesystem path, so only a catalogued
        // value is trusted.
        if (!clinic.organisms().includes(organism)) return sendJSON(res, 404, { error: "unknown organism" });
        return sendJSON(res, 200, architectureCampaignsDocument(clinic.runsDir, organism));
      }
      if (p.length === 3 && p[1] === "jobs") {
        if (req.method !== "POST") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "job launches require clinic mode" });
        const kind = p[2];
        if (!jobs.JOB_KINDS.includes(kind)) return sendJSON(res, 404, { error: "unknown job kind" });
        let body;
        try { body = await readBody(req); } catch (err) { return sendJSON(res, 400, { error: String(err.message || err) }); }
        const invalid = validateJobBody(clinic, body);
        if (invalid) return sendJSON(res, invalid.status, { error: invalid.error });
        if (kind === "corpus") {
          const selectedSpec = corpusSpecCatalog(clinic.corpusSpecsDir, body.organism)
            .find((entry) => entry.spec_id === body.spec_id);
          if (!selectedSpec) return sendJSON(res, 404, { error: "unknown corpus spec" });
          body = { ...body, spec_path: selectedSpec.file };
        }
        try {
          return sendJSON(res, 200, jobs.launchJob(kind, body, {
            runsDir: clinic.runsDir, corpusRoot: clinic.corpusRoot, maxConcurrent: maxConcurrentJobs,
          }));
        } catch (err) {
          if (err instanceof jobs.ConcurrencyLimitError) return sendJSON(res, 409, { error: String(err.message) });
          return sendJSON(res, 400, { error: String(err.message || err) });
        }
      }
      if (p.length === 3 && p[1] === "preview") {
        if (req.method !== "POST") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "job previews require clinic mode" });
        const kind = p[2];
        // Only search/breed declare a --dry-run branch in the CLI (both
        // torch-free: search's resolves candidate specs via propose(),
        // breed's crosses/mutates genomes and reads checkpoint metadata
        // only) -- baseline/clone/resume have nothing to preview, since
        // there is no proposal/crossover step before they just train.
        if (kind !== "search" && kind !== "breed") return sendJSON(res, 404, { error: "unknown preview kind" });
        let body;
        try { body = await readBody(req); } catch (err) { return sendJSON(res, 400, { error: String(err.message || err) }); }
        const invalid = validateJobBody(clinic, body);
        if (invalid) return sendJSON(res, invalid.status, { error: invalid.error });
        const previewId = `preview-${crypto.randomUUID()}`;
        let launch;
        try {
          launch = jobs.buildLaunch(kind, body, { runsDir: clinic.runsDir, jobId: previewId });
        } catch (err) {
          return sendJSON(res, 400, { error: String(err.message || err) });
        }
        try {
          const result = spawnSync(
            jobs.pythonCommand(), [
              ...jobs.pythonPrefixArgs(), "-m", "cognitive_runtime.cli", "factory", kind, ...launch.argv, "--dry-run",
            ],
            { cwd: REPO_DIR, encoding: "utf8", env: process.env },
          );
          if (result.error) return sendJSON(res, 500, { error: String(result.error.message || result.error) });
          if (result.status !== 0) return sendJSON(res, 400, { error: (result.stderr || result.stdout || "preview failed").trim() });
          try {
            return sendJSON(res, 200, JSON.parse(result.stdout));
          } catch {
            return sendJSON(res, 500, { error: "preview did not return valid JSON" });
          }
        } finally {
          // A preview has no job identity to keep -- baseline/search's own
          // buildLaunch may have written a temp spec sidecar (never breed's,
          // which needs none), and nothing else ever reads a "preview-*"
          // job id, so it is cleaned up right away rather than left in
          // .clinic-jobs/ to accumulate across repeated preview calls.
          try { fs.unlinkSync(jobs.jobSpecPath(clinic.runsDir, previewId)); } catch { /* never written, e.g. breed */ }
        }
      }
      if (p.length === 2 && p[1] === "corpora") {
        if (req.method !== "GET") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic || !clinic.corpusRoot) return sendJSON(res, 400, { error: "corpus listing requires clinic mode with a corpus root" });
        const organism = url.searchParams.get("organism");
        if (organism !== null && !isPlainId(organism)) return sendJSON(res, 400, { error: "organism must be a plain identifier" });
        return sendJSON(res, 200, { corpora: corpusCatalog(clinic.corpusRoot, organism) });
      }
      if (p.length === 2 && p[1] === "corpus-specs") {
        if (req.method !== "GET") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic || !clinic.corpusSpecsDir) return sendJSON(res, 400, { error: "corpus specs require clinic mode" });
        const organism = url.searchParams.get("organism");
        if (organism !== null && !isPlainId(organism)) return sendJSON(res, 400, { error: "organism must be a plain identifier" });
        return sendJSON(res, 200, {
          specs: corpusSpecCatalog(clinic.corpusSpecsDir, organism)
            .map(({ file, ...entry }) => entry),
        });
      }
      if (p.length === 2 && p[1] === "factory-meta") {
        if (req.method !== "GET") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "factory metadata requires clinic mode" });
        try {
          return sendJSON(res, 200, factoryMeta());
        } catch (err) {
          return sendJSON(res, 500, { error: String(err.message || err) });
        }
      }
      if (p.length === 2 && p[1] === "jobs") {
        if (req.method !== "GET") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "job status requires clinic mode" });
        const organism = url.searchParams.get("organism");
        const entries = jobs.listJobs(clinic.runsDir, { organism });
        return sendJSON(res, 200, {
          jobs: entries.map((entry) => ({ ...entry, runs: jobRunStates(clinic.runsDir, entry.organism, entry.run_ids) })),
        });
      }
      if (p.length === 4 && p[1] === "jobs" && p[3] === "log") {
        if (req.method !== "GET") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "job logs require clinic mode" });
        const jobId = decodeURIComponent(p[2]);
        if (!isPlainId(jobId)) return sendJSON(res, 404, { error: "unknown job id" });
        const result = jobs.readJobLog(clinic.runsDir, jobId, { offset: url.searchParams.get("offset") });
        if (!result) return sendJSON(res, 404, { error: "unknown job id" });
        return sendJSON(res, 200, result);
      }
      if (p.length === 4 && p[1] === "jobs" && p[3] === "cancel") {
        if (req.method !== "POST") return sendJSON(res, 405, { error: "method not allowed" });
        if (!clinic) return sendJSON(res, 400, { error: "job cancellation requires clinic mode" });
        const jobId = decodeURIComponent(p[2]);
        if (!isPlainId(jobId)) return sendJSON(res, 404, { error: "unknown job id" });
        const entry = await jobs.cancelJob(clinic.runsDir, jobId);
        if (!entry) return sendJSON(res, 404, { error: "unknown job id" });
        return sendJSON(res, 200, entry);
      }
      if (p.length === 2 && p[1] === "sessions") {
        const activeStore = selectedStore(url);
        if (!activeStore) return sendJSON(res, 400, { error: "select organism and run" });
        return sendJSON(res, 200, { data_dir: activeStore.dataDir, sessions: activeStore.list(url.searchParams.get("name"), Object.fromEntries(["scenario", "seed", "split", "prediction_mode", "entity_event", "motion"].map((key) => [key, url.searchParams.get(key)])) ) });
      }
      if (p.length >= 3 && p[1] === "sessions") {
        const sid = decodeURIComponent(p[2]);
        const activeStore = selectedStore(url);
        if (!activeStore) return sendJSON(res, 400, { error: "select organism and run" });
        const dir = activeStore.sessionDir(sid); if (!dir) return sendJSON(res, 404, { error: `unknown session ${sid}` });
        if (p.length === 3) {
          const session = activeStore.describe(sid);
          const streams = Object.fromEntries(session.episodes.map((eid) => [eid, readStreams(dir, eid)]));
          const decisions = Object.fromEntries(session.episodes.map((eid) => [eid, readDecisions(dir, eid)]));
          return sendJSON(res, 200, { session, streams, decisions, exports: exportsFor(dir), quality: session.quality });
        }
        if (p.length === 6 && p[3] === "episodes" && p[5] === "streams") return sendJSON(res, 200, { records: readStreams(dir, p[4]) });
        if (p.length === 6 && p[3] === "episodes" && p[5] === "decisions") return sendJSON(res, 200, { records: readDecisions(dir, p[4]) });
        if (p.length === 6 && p[3] === "episodes" && p[5] === "frames") return sendJSON(res, 200, readEpisodeFrames(dir, sid, p[4]));
        if (p.length === 6 && p[3] === "episodes" && p[5] === "predictions") {
          const kind = url.searchParams.get("kind") === "dream" ? "dream" : "predictions";
          const candidates = [`${readJSON(path.join(dir, "session.json")).name}-${kind}_${p[4]}.json`, `${kind}_${p[4]}.json`];
          let file = candidates.map((n) => path.join(dir, n)).find(fs.existsSync);
          // v2 files are experiment-prefixed. Never merge stale exports: an
          // explicit experiment selects one, while an ambiguous directory is
          // rejected instead of guessing which model the clinic should show.
          if (!file && kind === "predictions") {
            const experiment = url.searchParams.get("experiment");
            const v2 = fs.readdirSync(dir).filter((name) =>
              name.endsWith(`-predictions_${p[4]}.json`) &&
              (!experiment || name.startsWith(`${experiment}-`))
            );
            if (v2.length === 1) file = path.join(dir, v2[0]);
            if (v2.length > 1) return sendJSON(res, 409, {
              error: "multiple experiment exports; pass ?experiment=<experiment-id>", experiments: v2,
            });
          }
          if (file) return sendJSON(res, 200, readJSON(file));
          if (kind === "predictions") {
            const live = livePredictionsFromDecisions(readDecisions(dir, p[4]), sid, p[4]);
            if (live) return sendJSON(res, 200, live);
          }
          return sendJSON(res, 404, { error: "no recorded predictions for this episode" });
        }
      }
      return sendJSON(res, 404, { error: "unknown API route" });
    } catch (err) { return sendJSON(res, 500, { error: String(err.message || err) }); }
  });
}

if (require.main === module) {
  const args = parseArgs(process.argv);
  createServer(args).listen(args.port, args.host, () => console.log(`CCR clinic: http://${args.host}:${args.port}  (${args.dataDir ? `Record: ${args.dataDir}` : `Runs: ${args.runsDir}; cache: ${args.episodeCacheDir}`})`));
}
module.exports = { createServer, livePredictionsFromDecisions, makeStore, makeClinicStore, runSummary, runArtifacts, registryDocument, factoryRunsDocument };
