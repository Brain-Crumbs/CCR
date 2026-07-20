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
  const args = { dataDir: null, port: 8787 };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--data-dir") args.dataDir = argv[++i];
    else if (argv[i] === "--port") args.port = Number(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") {
      console.log("usage: node server.js [--data-dir <sessions dir>] [--port 8787]");
      process.exit(0);
    }
  }
  args.dataDir = path.resolve(args.dataDir || path.join(REPO_DIR, "shared"));
  return args;
}

function readJSON(file, fallback = {}) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return fallback; }
}

function isSessionDir(dir) { return fs.existsSync(path.join(dir, "session.json")); }

function qualityVerdict(dir) {
  const result = spawnSync(process.env.PYTHON || "python3", ["-m", "cognitive_runtime.record.quality_cli", dir], {
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

function makeStore(dataDir, { qualityCheck = qualityVerdict } = {}) {
  dataDir = path.resolve(dataDir);
  const qualityCache = new Map();
  function sessionDir(sid) {
    if (!/^[\w.-]+$/.test(sid)) return null;
    const dir = path.join(dataDir, sid);
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
    return { id, name: meta.name ?? "legacy", curriculum: meta.curriculum ?? null,
      program: meta.program ?? null, tick_rate: meta.tick_rate ?? null, episodes,
      development: meta.development ?? meta.ladder ?? meta.developmental ?? null,
      quality };
  }
  function list(name = null) {
    if (!fs.existsSync(dataDir)) return [];
    return fs.readdirSync(dataDir, { withFileTypes: true })
      .filter((e) => e.isDirectory() && isSessionDir(path.join(dataDir, e.name)))
      .map((e) => describe(e.name)).filter((s) => !name || s.name === name)
      .sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  }
  return { dataDir, sessionDir, describe, list };
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

function createServer({ dataDir }) {
  const store = makeStore(dataDir);
  return http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost"), p = url.pathname.split("/").filter(Boolean);
    try {
      if (p[0] !== "api") return serveStatic(res, url.pathname);
      if (p.length === 2 && p[1] === "sessions") return sendJSON(res, 200, { data_dir: store.dataDir, sessions: store.list(url.searchParams.get("name")) });
      if (p.length >= 3 && p[1] === "sessions") {
        const dir = store.sessionDir(p[2]); if (!dir) return sendJSON(res, 404, { error: `unknown session ${p[2]}` });
        if (p.length === 3) {
          const session = store.describe(p[2]);
          const streams = Object.fromEntries(session.episodes.map((eid) => [eid, readStreams(dir, eid)]));
          const decisions = Object.fromEntries(session.episodes.map((eid) => [eid, readDecisions(dir, eid)]));
          return sendJSON(res, 200, { session, streams, decisions, exports: exportsFor(dir), quality: session.quality });
        }
        if (p.length === 6 && p[3] === "episodes" && p[5] === "streams") return sendJSON(res, 200, { records: readStreams(dir, p[4]) });
        if (p.length === 6 && p[3] === "episodes" && p[5] === "decisions") return sendJSON(res, 200, { records: readDecisions(dir, p[4]) });
        if (p.length === 6 && p[3] === "episodes" && p[5] === "frames") return sendJSON(res, 200, readEpisodeFrames(dir, p[2], p[4]));
        if (p.length === 6 && p[3] === "episodes" && p[5] === "predictions") {
          const kind = url.searchParams.get("kind") === "dream" ? "dream" : "predictions";
          const candidates = [`${readJSON(path.join(dir, "session.json")).name}-${kind}_${p[4]}.json`, `${kind}_${p[4]}.json`];
          const file = candidates.map((n) => path.join(dir, n)).find(fs.existsSync);
          if (file) return sendJSON(res, 200, readJSON(file));
          if (kind === "predictions") {
            const live = livePredictionsFromDecisions(readDecisions(dir, p[4]), p[2], p[4]);
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
  createServer(args).listen(args.port, () => console.log(`CCR clinic: http://localhost:${args.port}  (Record: ${args.dataDir})`));
}
module.exports = { createServer, livePredictionsFromDecisions, makeStore };
