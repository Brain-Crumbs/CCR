"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { createServer } = require("../server");

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-"));
  const dir = path.join(root, "pixel-session"); fs.mkdirSync(dir);
  fs.writeFileSync(path.join(dir, "session.json"), JSON.stringify({ name: "Pixel", program: "fixture", development: { stages: [{ name: "Gestation", passed: true }, { name: "Crawling", passed: false }] } }));
  const records = [
    { stream_id: "vision.frame.pixels", frame_ref: "same", timestamp: 0, seq: 0 },
    { stream_id: "internal.dopamine", payload: { value: 0.2 }, seq: 0 },
    { stream_id: "internal.acetylcholine", payload: { value: 0.4 }, seq: 0 },
    { stream_id: "internal.adrenaline", payload: { value: 0.1 }, seq: 0 },
    { stream_id: "internal.prediction_error", payload: { value: 0.3 }, seq: 0 },
    { stream_id: "internal.arbiter.mode", payload: { mode: "curious" }, seq: 0 },
    { stream_id: "internal.attention.weights", payload: { tick_index: 0, focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels"], reasons: { "vision.frame.pixels": { components: { novelty: 0.8, boredom: -0.1 } } } }, seq: 0 },
  ];
  fs.writeFileSync(path.join(dir, "episode_00000.streams.jsonl"), records.map(JSON.stringify).join("\n") + "\n");
  fs.writeFileSync(path.join(dir, "episode_00000.decisions.jsonl"), JSON.stringify({
    tick_index: 0, prediction_error: 0.35, arbiter_mode: { mode: "curious" }, attention: {
      tick_index: 0, focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels"],
      reasons: { "vision.frame.pixels": { components: { novelty: 0.8, boredom: -0.1 } } },
    },
  }) + "\n");
  fs.writeFileSync(path.join(dir, "episode_00000.summary.json"), JSON.stringify({ duration_ticks: 1, success: true, program_stats: { pixel_sources: ["grid"] } }));
  fs.writeFileSync(path.join(dir, "Pixel-predictions_episode_00000.json"), JSON.stringify({ format: "pixel-predictions-v1" }));
  fs.writeFileSync(path.join(dir, "Pixel-dream_episode_00000.json"), JSON.stringify({ format: "pixel-predictions-v1", kind: "dream" }));
  return root;
}

function get(port, route) { return new Promise((resolve, reject) => http.get({ port, path: route }, (res) => {
  let body = ""; res.on("data", (x) => body += x); res.on("end", () => resolve({ status: res.statusCode, body: JSON.parse(body) }));
}).on("error", reject)); }

test("service lists by organism and returns streams, exports, and verdict", async (t) => {
  const server = createServer({ dataDir: fixture() }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const port = server.address().port;
  const listed = await get(port, "/api/sessions?name=Pixel");
  assert.equal(listed.body.sessions.length, 1); assert.equal(listed.body.sessions[0].quality.verdict, "red");
  assert.match(listed.body.sessions[0].quality.issues[0], /recording appears frozen/);
  assert.equal((await get(port, "/api/sessions?name=SomeoneElse")).body.sessions.length, 0);
  const detail = (await get(port, "/api/sessions/pixel-session")).body;
  assert.equal(detail.streams.episode_00000[0].stream_id, "vision.frame.pixels");
  assert.equal(detail.decisions.episode_00000[0].attention.reasons["vision.frame.pixels"].components.novelty, 0.8);
  const decisionRecords = (await get(port, "/api/sessions/pixel-session/episodes/episode_00000/decisions")).body.records;
  assert.equal(decisionRecords[0].attention.focus_stream, "vision.frame.pixels");
  assert.equal(detail.exports[0].data.format, "pixel-predictions-v1"); assert.equal(detail.quality.verdict, "red");
});

test("browser links an episode to the locally served frame and prediction APIs", async () => {
  const source = fs.readFileSync(path.join(__dirname, "../public/session-browser.js"), "utf8");
  const isolated = source.replace(/^import .*diagnostic-panels\.js.*$/m, "");
  const { episodeUrls } = await import(`data:text/javascript;base64,${Buffer.from(isolated).toString("base64")}`);
  assert.deepEqual(episodeUrls("pixel session", "episode_00000"), {
    frames: "/api/sessions/pixel%20session/episodes/episode_00000/frames",
    predictions: "/api/sessions/pixel%20session/episodes/episode_00000/predictions",
  });
  assert.match(source, /createElement\("pixel-horizon-viewer"\)/);
});

async function panels() {
  const source = fs.readFileSync(path.join(__dirname, "../public/diagnostic-panels.js"), "utf8");
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

test("dream strip endpoint serves the Phase 4 export independently per episode", async (t) => {
  const server = createServer({ dataDir: fixture() }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const result = await get(server.address().port, "/api/sessions/pixel-session/episodes/episode_00000/predictions?kind=dream");
  assert.equal(result.status, 200); assert.equal(result.body.kind, "dream");
});

test("EEG component renders neuromodulators, prediction error, and mode timeline", async () => {
  const ui = await panels();
  const model = ui.episodeDiagnostics([
    { stream_id: "internal.dopamine", payload: { value: .2 }, seq: 1 },
    { stream_id: "internal.acetylcholine", payload: { value: .3 }, seq: 1 },
    { stream_id: "internal.adrenaline", payload: { value: .4 }, seq: 1 },
    { stream_id: "internal.prediction_error", payload: { value: .5 }, seq: 1 },
    { stream_id: "internal.arbiter.mode", payload: { mode: "afraid" }, seq: 1 },
  ]);
  const html = ui.renderEEGPanel(model);
  for (const label of ["dopamine", "acetylcholine", "adrenaline", "prediction error", "afraid"]) assert.match(html, new RegExp(label));
});

test("attention component renders reasons from DecisionRecord rather than the stream payload", async () => {
  const ui = await panels();
  const streams = [{ stream_id: "internal.attention.weights", seq: 4, payload: {
    focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels", "body.health"],
  } }];
  const decisions = [{ tick_index: 4, attention: {
    focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels", "body.health"],
    reasons: { "vision.frame.pixels": { components: { novelty: .75, boredom: -.1 } } },
  } }];
  const model = ui.episodeDiagnostics(streams, decisions);
  const html = ui.renderAttentionPanel(model);
  assert.match(html, /vision\.frame\.pixels/); assert.match(html, /novelty 0\.75/); assert.match(html, /body\.health/);
  assert.doesNotMatch(html, /reason unavailable/);
});

test("developmental component renders passed and pending stage gates", async () => {
  const ui = await panels();
  const html = ui.renderDevelopmentPanel({ development: { stages: [
    { name: "Gestation", passed: true, milestones: ["sensory baseline"] }, { name: "Crawling", passed: false },
  ] } });
  assert.match(html, /stage--passed[^>]*>[\s\S]*Gestation/); assert.match(html, /sensory baseline/); assert.match(html, /stage--pending[^>]*>[\s\S]*Crawling/);
});

test("clinic landing page has no public-internet runtime dependency", () => {
  const html = fs.readFileSync(path.join(__dirname, "../public/index.html"), "utf8");
  assert.doesNotMatch(html, /https?:\/\//);
  assert.match(html, /\/pixel-horizon-viewer\.js/);
});
