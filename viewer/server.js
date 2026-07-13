#!/usr/bin/env node
/**
 * Zero-dependency HTTP server for browsing recorded streams-v2 sessions:
 * lists sessions/episodes, decodes `vision.frame.pixels` events out of the
 * binary frame store (frames/segment_*.bin + .index.jsonl), and serves the
 * static <pixel-horizon-viewer> component from public/.
 *
 *   node viewer/server.js [--data-dir shared] [--port 8787]
 *
 * API:
 *   GET /api/sessions
 *   GET /api/sessions/:sid/episodes/:eid/frames       frames as base64 raw bytes
 *   GET /api/sessions/:sid/episodes/:eid/predictions  predictions_<eid>.json, if exported
 */
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");

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
  args.dataDir = path.resolve(args.dataDir || path.join(__dirname, "..", "shared"));
  return args;
}

const ARGS = parseArgs(process.argv);
const PUBLIC_DIR = path.join(__dirname, "public");

// ---------------------------------------------------------------- sessions

function isSessionDir(dir) {
  return fs.existsSync(path.join(dir, "session.json"));
}

function listSessions() {
  if (!fs.existsSync(ARGS.dataDir)) return [];
  return fs
    .readdirSync(ARGS.dataDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && isSessionDir(path.join(ARGS.dataDir, e.name)))
    .map((e) => {
      const dir = path.join(ARGS.dataDir, e.name);
      let meta = {};
      try {
        meta = JSON.parse(fs.readFileSync(path.join(dir, "session.json"), "utf8"));
      } catch {
        /* unreadable metadata; still list the session */
      }
      const episodes = fs
        .readdirSync(dir)
        .filter((f) => /^episode_\d+\.streams\.jsonl$/.test(f))
        .map((f) => f.replace(".streams.jsonl", ""))
        .sort();
      return {
        id: e.name,
        curriculum: meta.curriculum ?? null,
        program: meta.program ?? null,
        tick_rate: meta.tick_rate ?? null,
        episodes,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}

/** Resolve a client-supplied session id to a real directory, refusing traversal. */
function sessionDir(sid) {
  if (!/^[\w.-]+$/.test(sid)) return null;
  const dir = path.join(ARGS.dataDir, sid);
  if (!dir.startsWith(ARGS.dataDir + path.sep) || !isSessionDir(dir)) return null;
  return dir;
}

// ---------------------------------------------------------------- frame store

/** Build hash -> {bin, offset, length, shape, dtype} from every segment index. */
function loadFrameIndex(dir) {
  const framesDir = path.join(dir, "frames");
  const index = new Map();
  if (!fs.existsSync(framesDir)) return index;
  for (const name of fs.readdirSync(framesDir).sort()) {
    if (!name.endsWith(".index.jsonl")) continue;
    const bin = path.join(framesDir, name.replace(".index.jsonl", ".bin"));
    for (const line of fs.readFileSync(path.join(framesDir, name), "utf8").split("\n")) {
      if (!line.trim()) continue;
      const rec = JSON.parse(line);
      index.set(rec.hash, { bin, offset: rec.offset, length: rec.length, shape: rec.shape, dtype: rec.dtype });
    }
  }
  return index;
}

function readEpisodeFrames(dir, sid, eid) {
  if (!/^episode_\d+$/.test(eid)) return null;
  const streamsPath = path.join(dir, `${eid}.streams.jsonl`);
  if (!fs.existsSync(streamsPath)) return null;

  const index = loadFrameIndex(dir);
  const bins = new Map(); // bin path -> Buffer, read once
  const frames = [];
  let shape = null;
  let dtype = null;

  for (const line of fs.readFileSync(streamsPath, "utf8").split("\n")) {
    if (!line.trim()) continue;
    let rec;
    try {
      rec = JSON.parse(line);
    } catch {
      continue;
    }
    if (rec.stream_id !== "vision.frame.pixels") continue;
    shape = rec.shape ?? shape;
    dtype = rec.dtype ?? dtype;
    const entry = { i: frames.length, t: rec.timestamp, seq: rec.seq, hash: rec.frame_ref ?? null, data: null };
    const loc = entry.hash ? index.get(entry.hash) : undefined;
    if (loc && !rec.elided) {
      if (!bins.has(loc.bin)) bins.set(loc.bin, fs.readFileSync(loc.bin));
      entry.data = bins.get(loc.bin).subarray(loc.offset, loc.offset + loc.length).toString("base64");
    }
    frames.push(entry);
  }
  return { session_id: sid, episode_id: eid, shape, dtype, n_frames: frames.length, frames };
}

// ---------------------------------------------------------------- http

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

function sendJSON(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
  res.end(body);
}

function serveStatic(res, urlPath) {
  const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, "");
  const file = path.join(PUBLIC_DIR, path.normalize(rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep) && file !== path.join(PUBLIC_DIR, "index.html")) {
    return sendJSON(res, 404, { error: "not found" });
  }
  fs.readFile(file, (err, data) => {
    if (err) return sendJSON(res, 404, { error: "not found" });
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  const parts = url.pathname.split("/").filter(Boolean);

  try {
    if (parts[0] === "api") {
      if (parts.length === 2 && parts[1] === "sessions") {
        return sendJSON(res, 200, { data_dir: ARGS.dataDir, sessions: listSessions() });
      }
      // /api/sessions/:sid/episodes/:eid/(frames|predictions)
      if (parts.length === 6 && parts[1] === "sessions" && parts[3] === "episodes") {
        const dir = sessionDir(parts[2]);
        if (!dir) return sendJSON(res, 404, { error: `unknown session ${parts[2]}` });
        if (parts[5] === "frames") {
          const payload = readEpisodeFrames(dir, parts[2], parts[4]);
          if (!payload) return sendJSON(res, 404, { error: `unknown episode ${parts[4]}` });
          return sendJSON(res, 200, payload);
        }
        if (parts[5] === "predictions") {
          if (!/^episode_\d+$/.test(parts[4])) return sendJSON(res, 404, { error: "bad episode id" });
          const predPath = path.join(dir, `predictions_${parts[4]}.json`);
          if (!fs.existsSync(predPath)) {
            return sendJSON(res, 404, { error: "no predictions exported for this episode", hint: "see viewer/export_predictions.py" });
          }
          res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
          return fs.createReadStream(predPath).pipe(res);
        }
      }
      return sendJSON(res, 404, { error: "unknown API route" });
    }
    return serveStatic(res, url.pathname);
  } catch (err) {
    return sendJSON(res, 500, { error: String((err && err.message) || err) });
  }
});

server.listen(ARGS.port, () => {
  console.log(`pixel viewer: http://localhost:${ARGS.port}  (data dir: ${ARGS.dataDir})`);
});
