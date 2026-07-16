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
  fs.writeFileSync(path.join(dir, "session.json"), JSON.stringify({ name: "Pixel", program: "fixture" }));
  fs.writeFileSync(path.join(dir, "episode_00000.streams.jsonl"), JSON.stringify({ stream_id: "vision.frame.pixels", frame_ref: "same", timestamp: 0, seq: 0 }) + "\n");
  fs.writeFileSync(path.join(dir, "episode_00000.summary.json"), JSON.stringify({ duration_ticks: 1, success: true, program_stats: { pixel_sources: ["grid"] } }));
  fs.writeFileSync(path.join(dir, "Pixel-predictions_episode_00000.json"), JSON.stringify({ format: "pixel-predictions-v1" }));
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
  assert.equal(detail.exports[0].data.format, "pixel-predictions-v1"); assert.equal(detail.quality.verdict, "red");
});

test("browser links an episode to the locally served frame and prediction APIs", async () => {
  const source = fs.readFileSync(path.join(__dirname, "../public/session-browser.js"), "utf8");
  const { episodeUrls } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
  assert.deepEqual(episodeUrls("pixel session", "episode_00000"), {
    frames: "/api/sessions/pixel%20session/episodes/episode_00000/frames",
    predictions: "/api/sessions/pixel%20session/episodes/episode_00000/predictions",
  });
  assert.match(source, /createElement\("pixel-horizon-viewer"\)/);
});

test("clinic landing page has no public-internet runtime dependency", () => {
  const html = fs.readFileSync(path.join(__dirname, "../public/index.html"), "utf8");
  assert.doesNotMatch(html, /https?:\/\//);
  assert.match(html, /\/pixel-horizon-viewer\.js/);
});
