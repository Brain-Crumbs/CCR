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
  const args = { dataDir: null, runsDir: null, episodeCacheDir: null, port: 8787 };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--data-dir") args.dataDir = argv[++i];
    else if (argv[i] === "--runs-dir") args.runsDir = argv[++i];
    else if (argv[i] === "--episode-cache-dir") args.episodeCacheDir = argv[++i];
    else if (argv[i] === "--port") args.port = Number(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") {
      console.log("usage: node server.js [--runs-dir <runs dir>] [--episode-cache-dir <cache dir>] [--data-dir <sessions dir>] [--port 8787]");
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

function exportedCacheSessions(cacheDir, experimentId) {
  if (!fs.existsSync(cacheDir)) return [];
  return sessionIdsBelow(cacheDir).flatMap((id) => {
    const dir = path.join(cacheDir, ...id.split("/"));
    const hasExport = fs.readdirSync(dir).some((file) => file.startsWith(`${experimentId}-predictions_`) && file.endsWith(".json"));
    return hasExport ? [{ id: `cache/${id}`, dir }] : [];
  });
}

function manifestSessions(runDir, cacheDir) {
  const index = readJSON(path.join(runDir, "clinic_sessions.json"), null);
  if (!index || index.format !== "clinic-session-index-v1" || !Array.isArray(index.sessions)) return [];
  return index.sessions.flatMap((entry) => {
    if (typeof entry.session_dir !== "string") return [];
    const dir = path.resolve(entry.session_dir);
    // A run index may only mount its own recordings or the configured cache.
    if (!inside(runDir, dir) && !inside(cacheDir, dir)) return [];
    const prefix = inside(cacheDir, dir) ? "cache" : "run";
    const root = prefix === "cache" ? cacheDir : runDir;
    return isSessionDir(dir) ? [{ id: `${prefix}/${path.relative(root, dir).split(path.sep).join("/")}`, dir }] : [];
  });
}

function makeClinicStore(runsDir, episodeCacheDir) {
  runsDir = path.resolve(runsDir); episodeCacheDir = path.resolve(episodeCacheDir);
  function catalog() { return runCatalog(runsDir); }
  function selected(organism, run) {
    const entry = catalog().find((candidate) => candidate.organism === organism && candidate.run === run);
    if (!entry) return null;
    const direct = fs.existsSync(entry.dir) ? sessionIdsBelow(entry.dir).map((id) => ({ id: `run/${id}`, dir: path.join(entry.dir, ...id.split("/")) })) : [];
    const indexed = manifestSessions(entry.dir, episodeCacheDir);
    const fallback = indexed.length ? [] : exportedCacheSessions(episodeCacheDir, entry.run);
    const mounts = [...new Map([...direct, ...indexed, ...fallback].map((item) => [item.id, item])).values()];
    return { entry, store: makeStore(entry.dir, { sessionDirectories: mounts }) };
  }
  return { runsDir, episodeCacheDir, catalog, selected };
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

function createServer({ dataDir = null, runsDir = null, episodeCacheDir = null }) {
  const clinic = dataDir ? null : makeClinicStore(
    runsDir || path.join(REPO_DIR, "notebooks", "runs"),
    episodeCacheDir || path.join(REPO_DIR, "notebooks", "episode_cache"),
  );
  const store = clinic ? null : makeStore(dataDir);
  function selectedStore(url) {
    if (!clinic) return store;
    const organism = url.searchParams.get("organism"), run = url.searchParams.get("run");
    return organism && run ? clinic.selected(organism, run)?.store : null;
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
  createServer(args).listen(args.port, () => console.log(`CCR clinic: http://localhost:${args.port}  (${args.dataDir ? `Record: ${args.dataDir}` : `Runs: ${args.runsDir}; cache: ${args.episodeCacheDir}`})`));
}
module.exports = { createServer, livePredictionsFromDecisions, makeStore, makeClinicStore };
