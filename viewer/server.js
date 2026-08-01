#!/usr/bin/env node
/** Read-only clinic service over streams-v2 Record sessions. */
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { spawnSync } = require("child_process");

const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_DIR = path.join(__dirname, "..");

function parseArgs(argv) {
  const args = { dataDir: null, runsDir: null, episodeCacheDir: null, corpusRoot: null, port: 8787 };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--data-dir") args.dataDir = argv[++i];
    else if (argv[i] === "--runs-dir") args.runsDir = argv[++i];
    else if (argv[i] === "--episode-cache-dir") args.episodeCacheDir = argv[++i];
    else if (argv[i] === "--corpus-root") args.corpusRoot = argv[++i];
    else if (argv[i] === "--port") args.port = Number(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") {
      console.log("usage: node server.js [--runs-dir <runs dir>] [--episode-cache-dir <cache dir>] [--corpus-root <corpora dir>] [--data-dir <sessions dir>] [--port 8787]");
      process.exit(0);
    }
  }
  // --data-dir remains the single-session-tree compatibility mode.  The
  // default clinic mode knows where experiment runs and cached recordings
  // live, and presents them as one selected run.
  if (args.dataDir) args.dataDir = path.resolve(args.dataDir);
  else {
    args.runsDir = path.resolve(args.runsDir || path.join(REPO_DIR, "notebooks", "runs"));
    args.episodeCacheDir = path.resolve(args.episodeCacheDir || path.join(REPO_DIR, "notebooks", "episode_cache"));
    // Model Factory's own default (cognitive_runtime/cli.py's
    // _FACTORY_CORPORA_ROOT_DEFAULT) -- a run's clinic_sessions.json points
    // into here, not into runsDir/episodeCacheDir, so it needs its own root.
    args.corpusRoot = path.resolve(args.corpusRoot || path.join(REPO_DIR, "corpora"));
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

/** One run's position in the factory's state machine (state.py's
 * queued/running/checkpointing/completed/budget_exceeded/failed/cancelled)
 * plus its lineage mode and promotion verdict -- the "what's in flight,
 * what finished, what failed" view the champion registry alone can't give,
 * since the registry only ever holds promoted runs. */
function factoryRunStatus(entry) {
  const state = readJSON(path.join(entry.dir, "state.json"), null);
  const lineage = readJSON(path.join(entry.dir, "lineage.json"), null);
  return {
    run: entry.run,
    mode: lineage?.mode ?? null,
    state: state?.state ?? null,
    state_reason: state?.reason ?? null,
    updated_at: state?.updated_at ?? null,
    promoted: runSummary(entry).promotion.promoted,
  };
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
    // A run index may only mount its own recordings, the configured cache,
    // or the Model Factory's frozen corpus root -- never an arbitrary path.
    if (!inside(runDir, dir) && !inside(cacheDir, dir) && !(corpusRoot && inside(corpusRoot, dir))) return [];
    const prefix = inside(cacheDir, dir) ? "cache" : (corpusRoot && inside(corpusRoot, dir)) ? "corpus" : "run";
    const root = prefix === "cache" ? cacheDir : prefix === "corpus" ? corpusRoot : runDir;
    return isSessionDir(dir) ? [{ id: `${prefix}/${path.relative(root, dir).split(path.sep).join("/")}`, dir }] : [];
  });
}

function makeClinicStore(runsDir, episodeCacheDir, corpusRoot) {
  runsDir = path.resolve(runsDir); episodeCacheDir = path.resolve(episodeCacheDir);
  corpusRoot = corpusRoot ? path.resolve(corpusRoot) : null;
  function catalog() { return runCatalog(runsDir); }
  function selected(organism, run) {
    const entry = catalog().find((candidate) => candidate.organism === organism && candidate.run === run);
    if (!entry) return null;
    const direct = fs.existsSync(entry.dir) ? sessionIdsBelow(entry.dir).map((id) => ({ id: `run/${id}`, dir: path.join(entry.dir, ...id.split("/")) })) : [];
    const indexed = manifestSessions(entry.dir, episodeCacheDir, corpusRoot);
    const fallback = indexed.length ? [] : exportedCacheSessions(episodeCacheDir, entry.run);
    const mounts = [...new Map([...direct, ...indexed, ...fallback].map((item) => [item.id, item])).values()];
    return { entry, store: makeStore(entry.dir, { sessionDirectories: mounts }) };
  }
  return { runsDir, episodeCacheDir, corpusRoot, catalog, selected };
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
function serveStatic(res, urlPath) {
  const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, ""), file = path.join(PUBLIC_DIR, path.normalize(rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) return sendJSON(res, 404, { error: "not found" });
  fs.readFile(file, (err, data) => { if (err) return sendJSON(res, 404, { error: "not found" }); res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" }); res.end(data); });
}

function createServer({ dataDir = null, runsDir = null, episodeCacheDir = null, corpusRoot = null }) {
  const clinic = dataDir ? null : makeClinicStore(
    runsDir || path.join(REPO_DIR, "notebooks", "runs"),
    episodeCacheDir || path.join(REPO_DIR, "notebooks", "episode_cache"),
    corpusRoot || path.join(REPO_DIR, "corpora"),
  );
  const store = clinic ? null : makeStore(dataDir);
  function selectedStore(url) {
    if (!clinic) return store;
    const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
    return organism && run ? clinic.selected(organism, run)?.store : null;
  }
  function selectedEntry(url) {
    if (!clinic) return null;
    const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
    return organism && run ? clinic.selected(organism, run)?.entry : null;
  }
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost"), p = url.pathname.split("/").filter(Boolean);
    try {
      if (p[0] !== "api") return serveStatic(res, url.pathname);
      if (p.length === 2 && p[1] === "catalog") return sendJSON(res, 200, {
        runs_dir: clinic?.runsDir ?? null, episode_cache_dir: clinic?.episodeCacheDir ?? null,
        organisms: [...new Set((clinic?.catalog() || []).map((entry) => entry.organism))],
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
        if (!clinic.catalog().some((entry) => entry.organism === organism)) return sendJSON(res, 404, { error: "unknown organism" });
        return sendJSON(res, 200, registryDocument(clinic.runsDir, organism));
      }
      if (p.length === 2 && p[1] === "factory-runs") {
        if (!clinic) return sendJSON(res, 400, { error: "factory run status requires clinic mode" });
        const organism = url.searchParams.get("organism");
        if (!organism) return sendJSON(res, 400, { error: "select an organism" });
        const runs = clinic.catalog().filter((entry) => entry.organism === organism).map(factoryRunStatus)
          .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
        return sendJSON(res, 200, { organism, runs });
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
          // A Model Factory run's export lives in its own run directory, not
          // the (frozen, hash-verified) session directory -- so a run
          // selected via ?run= is searched too, alongside the session itself.
          if (!file && kind === "predictions") {
            const experiment = url.searchParams.get("experiment");
            const runEntry = selectedEntry(url);
            const runPredictionsDir = runEntry ? path.join(runEntry.dir, "predictions") : null;
            const searchDirs = [dir, ...(runPredictionsDir && fs.existsSync(runPredictionsDir) ? [runPredictionsDir] : [])];
            const matches = (searchDir) => fs.readdirSync(searchDir).filter((name) =>
              name.endsWith(`-predictions_${p[4]}.json`) && (!experiment || name.startsWith(`${experiment}-`))
            ).map((name) => path.join(searchDir, name));
            const v2 = searchDirs.flatMap(matches);
            if (v2.length === 1) file = v2[0];
            if (v2.length > 1) return sendJSON(res, 409, {
              error: "multiple experiment exports; pass ?experiment=<experiment-id>", experiments: v2.map((f) => path.basename(f)),
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
  createServer(args).listen(args.port, () => console.log(`CCR clinic: http://localhost:${args.port}  (${args.dataDir ? `Record: ${args.dataDir}` : `Runs: ${args.runsDir}; cache: ${args.episodeCacheDir}; corpora: ${args.corpusRoot}`})`));
}
module.exports = { createServer, livePredictionsFromDecisions, makeStore, makeClinicStore, runSummary, runArtifacts, registryDocument, factoryRunStatus };
