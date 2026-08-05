"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { createServer, makeStore } = require("../server");

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

function post(port, route, payload) { return new Promise((resolve, reject) => {
  const data = Buffer.from(JSON.stringify(payload ?? {}));
  const req = http.request({
    port, path: route, method: "POST",
    headers: { "Content-Type": "application/json", "Content-Length": data.length },
  }, (res) => {
    let body = ""; res.on("data", (x) => body += x); res.on("end", () => resolve({ status: res.statusCode, body: JSON.parse(body) }));
  });
  req.on("error", reject); req.end(data);
}); }

async function waitFor(predicate, { timeoutMs = 5000, intervalMs = 20 } = {}) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await predicate();
    if (value) return value;
    if (Date.now() > deadline) throw new Error("waitFor timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

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

test("session quality verdicts are memoized until a session file changes", () => {
  const root = fixture(); let calls = 0;
  const store = makeStore(root, { qualityCheck: () => ({ verdict: `call-${++calls}`, issues: [], warnings: [] }) });
  assert.equal(store.list()[0].quality.verdict, "call-1");
  assert.equal(store.list()[0].quality.verdict, "call-1");
  assert.equal(calls, 1);
  const metadata = path.join(root, "pixel-session", "session.json");
  const future = new Date(Date.now() + 2000); fs.utimesSync(metadata, future, future);
  assert.equal(store.list()[0].quality.verdict, "call-2");
  assert.equal(calls, 2);
});

test("clinic mode joins a selected organism/run to its cached prediction sessions", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-runs-"));
  const runsDir = path.join(root, "runs"), cacheDir = path.join(root, "episode_cache");
  const runDir = path.join(runsDir, "Test", "run-1"), cached = path.join(cacheDir, "seed-20");
  fs.mkdirSync(runDir, { recursive: true }); fs.mkdirSync(cached, { recursive: true });
  fs.writeFileSync(path.join(runDir, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-1" }));
  fs.writeFileSync(path.join(runDir, "experiment_report.json"), JSON.stringify({
    checkpoint: { model_type: "predictive_cortex", sha256: "abc123" },
    training_stats: { training_objective: "windowed_rollout", epochs: 12, episodes: 3, samples: 48, final_total_loss: .2, evaluation_mode: "held_out" },
    metrics: { rollout: { prediction_mode: "rollout", rollout_health: { state: "healthy" }, horizons: { 1: { model_mse: .1, copy_last_mse: .2, model_over_copy_last_mse: .5, beats_copy_last: true, n_samples: 8 } } } },
    promotion_verdict: { promoted: true, reasons: [] },
  }));
  fs.writeFileSync(path.join(cached, "session.json"), JSON.stringify({ name: "Test" }));
  fs.writeFileSync(path.join(cached, "episode_00000.streams.jsonl"), "");
  fs.writeFileSync(path.join(cached, "run-1-predictions_episode_00000.json"), JSON.stringify({
    format: "pixel-predictions-v2", experiment: { experiment_id: "run-1" }, marker: "cached",
  }));
  const server = createServer({ runsDir, episodeCacheDir: cacheDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const port = server.address().port;
  assert.deepEqual((await get(port, "/api/catalog")).body.runs, [{ organism: "Test", run: "run-1" }]);
  const summary = await get(port, "/api/runs?organism=Test&run=run-1");
  assert.equal(summary.status, 200);
  assert.deepEqual(summary.body.model, { type: "predictive_cortex", checkpoint_sha256: "abc123" });
  assert.equal(summary.body.training.final_total_loss, .2);
  assert.equal(summary.body.evaluation.model_over_copy_last_mse, .5);
  assert.equal(summary.body.promotion.promoted, true);
  const query = "?organism=Test&run=run-1";
  const listed = await get(port, `/api/sessions${query}`);
  assert.deepEqual(listed.body.sessions.map((session) => session.id), ["cache/seed-20"]);
  const prediction = await get(port, `/api/sessions/cache%2Fseed-20/episodes/episode_00000/predictions${query}&experiment=run-1`);
  assert.equal(prediction.body.marker, "cached");

  fs.writeFileSync(path.join(runDir, "contracts.json"), JSON.stringify({ architecture_hash: "arch-1" }));
  fs.writeFileSync(path.join(runDir, "lineage.json"), JSON.stringify({ parent_run_id: null, mode: "fresh" }));
  fs.mkdirSync(path.join(runDir, "metrics"), { recursive: true });
  fs.writeFileSync(path.join(runDir, "metrics", "comparison.json"), JSON.stringify({
    format: "model-factory-paired-comparison-v1", status: "evaluable", mean_delta: -0.01, win_rate: 0.7,
  }));
  const experiments = await get(port, `/api/experiments${query}`);
  assert.equal(experiments.status, 200);
  assert.equal(experiments.body.contracts.architecture_hash, "arch-1");
  assert.equal(experiments.body.lineage.mode, "fresh");
  assert.equal(experiments.body.data_manifest, null);
  assert.equal(experiments.body.metrics.comparison.win_rate, 0.7);
  assert.equal(experiments.body.metrics.validation, null);
  assert.equal((await get(port, "/api/experiments?organism=Test&run=missing")).status, 404);

  fs.writeFileSync(path.join(runsDir, "Test", "registry.json"), JSON.stringify({
    format: "model-factory-registry-v1",
    slots: { generic_action_effects_v1: { fast: { "rollout.t+4": {
      leading_champion: "run-1",
      population: [{ run_id: "run-1", checkpoint_path: "checkpoints/best-validation.pt", checkpoint_sha256: "abc123", metrics: {}, promoted_at: "2026-01-01T00:00:00Z", test_uses: 0 }],
      history: [],
    } } } },
  }));
  const registry = await get(port, "/api/registry?organism=Test");
  assert.equal(registry.status, 200);
  assert.equal(registry.body.slots.generic_action_effects_v1.fast["rollout.t+4"].leading_champion, "run-1");
  assert.equal((await get(port, "/api/registry?organism=Nobody")).status, 404);
  // organism is joined into a filesystem path; an uncatalogued value must
  // never escape runsDir, even via traversal segments.
  assert.equal((await get(port, `/api/registry?organism=${encodeURIComponent("../../../../etc")}`)).status, 404);
});

test("clinic mode uses a run session index to include cached training recordings", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-index-"));
  const runsDir = path.join(root, "runs"), cacheDir = path.join(root, "episode_cache");
  const runDir = path.join(runsDir, "Test", "run-2"), cached = path.join(cacheDir, "seed-21");
  fs.mkdirSync(runDir, { recursive: true }); fs.mkdirSync(cached, { recursive: true });
  fs.writeFileSync(path.join(runDir, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-2" }));
  fs.writeFileSync(path.join(runDir, "clinic_sessions.json"), JSON.stringify({
    format: "clinic-session-index-v1", sessions: [{ split: "train", scenario: "approach_entity", session_dir: cached }],
  }));
  fs.writeFileSync(path.join(cached, "session.json"), JSON.stringify({ name: "Test" }));
  fs.writeFileSync(path.join(cached, "episode_00000.streams.jsonl"), "");
  const server = createServer({ runsDir, episodeCacheDir: cacheDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const listed = await get(server.address().port, "/api/sessions?organism=Test&run=run-2");
  assert.deepEqual(listed.body.sessions.map((session) => session.id), ["cache/seed-21"]);
});

test("clinic mode mounts corpus sessions named in the run index under a corpus/ prefix, refusing paths outside the configured roots", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-corpus-"));
  const runsDir = path.join(root, "runs"), cacheDir = path.join(root, "episode_cache"), corpusRoot = path.join(root, "corpora");
  const runDir = path.join(runsDir, "Test", "run-3");
  const corpusSession = path.join(corpusRoot, "Test", "corpus-1", "train", "seed-1");
  const outsideSession = path.join(root, "outside", "seed-9");
  fs.mkdirSync(runDir, { recursive: true }); fs.mkdirSync(corpusSession, { recursive: true }); fs.mkdirSync(outsideSession, { recursive: true });
  fs.writeFileSync(path.join(runDir, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-3" }));
  fs.writeFileSync(path.join(runDir, "clinic_sessions.json"), JSON.stringify({
    format: "clinic-session-index-v1",
    sessions: [
      { split: "train", scenario: "goal_navigation", session_dir: corpusSession },
      { split: "train", scenario: "goal_navigation", session_dir: outsideSession },
    ],
  }));
  fs.writeFileSync(path.join(corpusSession, "session.json"), JSON.stringify({ name: "Test" }));
  fs.writeFileSync(path.join(corpusSession, "episode_00000.streams.jsonl"), "");
  fs.writeFileSync(path.join(outsideSession, "session.json"), JSON.stringify({ name: "Test" }));
  const server = createServer({ runsDir, episodeCacheDir: cacheDir, corpusRoot }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const listed = await get(server.address().port, "/api/sessions?organism=Test&run=run-3");
  assert.deepEqual(listed.body.sessions.map((session) => session.id), ["corpus/Test/corpus-1/train/seed-1"]);
});

test("/api/factory-runs reports queued/running/completed/failed state across an organism's runs", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-factory-runs-"));
  const runsDir = path.join(root, "runs");
  const runA = path.join(runsDir, "Test", "run-a"), runB = path.join(runsDir, "Test", "run-b");
  fs.mkdirSync(runA, { recursive: true }); fs.mkdirSync(runB, { recursive: true });
  fs.writeFileSync(path.join(runA, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-a" }));
  fs.writeFileSync(path.join(runA, "lineage.json"), JSON.stringify({ mode: "fresh" }));
  fs.writeFileSync(path.join(runA, "state.json"), JSON.stringify({ state: "completed", updated_at: "2026-01-02T00:00:00Z", reason: "completed" }));
  fs.writeFileSync(path.join(runA, "experiment_report.json"), JSON.stringify({ promotion_verdict: { promoted: true, reasons: [] } }));
  fs.writeFileSync(path.join(runB, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-b" }));
  fs.writeFileSync(path.join(runB, "lineage.json"), JSON.stringify({ mode: "clone" }));
  fs.writeFileSync(path.join(runB, "state.json"), JSON.stringify({ state: "running", updated_at: "2026-01-03T00:00:00Z", reason: null }));
  const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const port = server.address().port;
  const result = await get(port, "/api/factory-runs?organism=Test");
  assert.equal(result.status, 200);
  assert.deepEqual(result.body.runs.map((r) => r.run), ["run-b", "run-a"]);
  assert.deepEqual(result.body.runs[1], {
    run: "run-a", mode: "fresh", state: "completed", state_reason: "completed",
    updated_at: "2026-01-02T00:00:00Z", promoted: true,
    // No metrics/validation.json and no lineage parents: reported as absent
    // rather than omitted, so the Evolve board can render "no metric yet".
    selection_metric: null, selection_metric_value: null, configuration_parents: [],
  });
  assert.equal(result.body.runs[0].state, "running");
  assert.equal(result.body.runs[0].promoted, null);
  assert.equal((await get(port, "/api/factory-runs?organism=Nobody")).status, 404);
  assert.equal((await get(port, "/api/factory-runs")).status, 400);
});

test("/api/factory-runs resolves each run's own selection metric and breeding parents for the population board", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-selection-metric-"));
  const runsDir = path.join(root, "runs");
  function writeRun(runId, { selectionMetric, ticksPerFrame, horizons, navigation = null, parents = [] }) {
    const dir = path.join(runsDir, "Test", runId);
    fs.mkdirSync(path.join(dir, "metrics"), { recursive: true });
    fs.writeFileSync(path.join(dir, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: runId }));
    fs.writeFileSync(path.join(dir, "lineage.json"), JSON.stringify({ mode: "fresh", configuration_parents: parents }));
    fs.writeFileSync(path.join(dir, "state.json"), JSON.stringify({ state: "completed", updated_at: `2026-01-0${runId.length % 9}T00:00:00Z` }));
    // ticks_per_frame lives in contracts.json, exactly where
    // registry._load_validation_for_candidate reads it from.
    fs.writeFileSync(path.join(dir, "contracts.json"), JSON.stringify({ data_contract: { ticks_per_frame: ticksPerFrame } }));
    const payload = { selection_metric: selectionMetric, ...horizons };
    if (navigation) payload.goal_navigation = navigation;
    fs.writeFileSync(path.join(dir, "metrics", "validation.json"), JSON.stringify(payload));
    return dir;
  }
  // ticks_per_frame 2 puts a t+4 rollout horizon on frame step 2.
  writeRun("run-rollout", {
    selectionMetric: "rollout.t+4.model_over_copy_last_mse", ticksPerFrame: 2,
    horizons: { rollout: { horizons: { 2: { model_over_copy_last_mse: 0.61, model_mse: 0.2 } } } },
  });
  // A direct horizon is tick-denominated as written -- never converted.
  writeRun("run-direct", {
    selectionMetric: "direct.t+4.model_mse", ticksPerFrame: 2,
    horizons: { direct: { horizons: { 4: { model_mse: 0.33 } } } },
  });
  // A t+5 horizon at 2 ticks/frame lands exactly on 2.5, where Python's
  // round-half-to-even picks frame 2 and JavaScript's Math.round would pick
  // 3 -- so this pins the rounding to the one the run itself recorded.
  writeRun("run-half-tick", {
    selectionMetric: "rollout.t+5.model_mse", ticksPerFrame: 2,
    horizons: { rollout: { horizons: { 2: { model_mse: 0.44 }, 3: { model_mse: 0.99 } } } },
  });
  writeRun("run-nav", {
    selectionMetric: "goal_navigation.success_rate", ticksPerFrame: 1, horizons: {},
    navigation: { success_rate: 0.75 },
  });
  // The metric's horizon was not evaluated: null, never a wrong number.
  writeRun("run-missing-horizon", {
    selectionMetric: "rollout.t+8.model_mse", ticksPerFrame: 2,
    horizons: { rollout: { horizons: { 2: { model_mse: 0.5 } } } },
    parents: ["run-rollout", "run-direct"],
  });
  const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const byRun = Object.fromEntries(
    (await get(server.address().port, "/api/factory-runs?organism=Test")).body.runs.map((run) => [run.run, run]),
  );
  assert.equal(byRun["run-rollout"].selection_metric, "rollout.t+4.model_over_copy_last_mse");
  assert.equal(byRun["run-rollout"].selection_metric_value, 0.61);
  assert.equal(byRun["run-direct"].selection_metric_value, 0.33);
  assert.equal(byRun["run-half-tick"].selection_metric_value, 0.44);
  assert.equal(byRun["run-nav"].selection_metric_value, 0.75);
  assert.equal(byRun["run-missing-horizon"].selection_metric, "rollout.t+8.model_mse");
  assert.equal(byRun["run-missing-horizon"].selection_metric_value, null);
  assert.deepEqual(byRun["run-missing-horizon"].configuration_parents, ["run-rollout", "run-direct"]);
});

test("service discovers sessions nested below the configured runs directory", async (t) => {
  const root = fixture();
  const outer = path.join(root, "Test", "run-1"); fs.mkdirSync(outer, { recursive: true });
  fs.renameSync(path.join(root, "pixel-session"), path.join(outer, "pixel-session"));
  const server = createServer({ dataDir: root }); await new Promise((resolve) => server.listen(0, resolve)); t.after(() => server.close());
  const listed = await get(server.address().port, "/api/sessions");
  assert.deepEqual(listed.body.sessions.map((session) => session.id), ["Test/run-1/pixel-session"]);
  const detail = await get(server.address().port, "/api/sessions/Test%2Frun-1%2Fpixel-session");
  assert.equal(detail.status, 200);
});

test("prediction endpoint assembles forecasts recorded by a live cortex", async (t) => {
  const root = fixture(), dir = path.join(root, "pixel-session");
  fs.unlinkSync(path.join(dir, "Pixel-predictions_episode_00000.json"));
  const decisions = [0, 1, 2].map((tick) => ({ tick_index: tick, live_prediction: {
    prediction_shape: [2, 2, 3], target: `target-${tick}`,
    frames: { "1": `prediction-${tick}` },
  } }));
  fs.writeFileSync(path.join(dir, "episode_00000.decisions.jsonl"), decisions.map(JSON.stringify).join("\n") + "\n");
  const server = createServer({ dataDir: root }); await new Promise((resolve) => server.listen(0, resolve)); t.after(() => server.close());
  const result = await get(server.address().port, "/api/sessions/pixel-session/episodes/episode_00000/predictions");
  assert.equal(result.status, 200); assert.equal(result.body.source, "live-record");
  assert.deepEqual(result.body.predictions["1"].frames, ["prediction-0", "prediction-1"]);
  assert.deepEqual(result.body.targets, ["target-0", "target-1", "target-2"]);
});

test("prediction endpoint selects the requested experiment when exports coexist", async (t) => {
  const root = fixture(), dir = path.join(root, "pixel-session");
  fs.unlinkSync(path.join(dir, "Pixel-predictions_episode_00000.json"));
  fs.writeFileSync(path.join(dir, "experiment-a-predictions_episode_00000.json"), JSON.stringify({
    format: "pixel-predictions-v2", experiment: { experiment_id: "experiment-a" }, marker: "a",
  }));
  fs.writeFileSync(path.join(dir, "experiment-b-predictions_episode_00000.json"), JSON.stringify({
    format: "pixel-predictions-v2", experiment: { experiment_id: "experiment-b" }, marker: "b",
  }));
  const server = createServer({ dataDir: root }); await new Promise((resolve) => server.listen(0, resolve)); t.after(() => server.close());
  const base = "/api/sessions/pixel-session/episodes/episode_00000/predictions";
  assert.equal((await get(server.address().port, base)).status, 409);
  const selected = await get(server.address().port, `${base}?experiment=experiment-b`);
  assert.equal(selected.status, 200);
  assert.equal(selected.body.marker, "b");
});

test("frame endpoint maps ordered decision windows in one forward pass", async (t) => {
  const root = fixture(), dir = path.join(root, "pixel-session");
  const frames = [0.2, 1.2, 2.2].map((timestamp, seq) => ({
    stream_id: "vision.frame.pixels", timestamp, seq, shape: [2, 2, 3], dtype: "uint8",
  }));
  const decisions = [0, 1, 2].map((tick) => ({ tick_index: tick + 10, window_span: [tick, tick + 0.9] }));
  fs.writeFileSync(path.join(dir, "episode_00000.streams.jsonl"), frames.map(JSON.stringify).join("\n") + "\n");
  fs.writeFileSync(path.join(dir, "episode_00000.decisions.jsonl"), decisions.map(JSON.stringify).join("\n") + "\n");
  const server = createServer({ dataDir: root }); await new Promise((resolve) => server.listen(0, resolve)); t.after(() => server.close());

  const result = await get(server.address().port, "/api/sessions/pixel-session/episodes/episode_00000/frames");

  assert.equal(result.status, 200);
  assert.deepEqual(result.body.frames.map((frame) => frame.tick), [10, 11, 12]);
});

test("/api/architecture-campaigns serves a nested campaign's own retention decisions", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-arch-campaigns-"));
  const runsDir = path.join(root, "runs");
  const run = path.join(runsDir, "Test", "arch7-arch0-a0-p0-c0");
  fs.mkdirSync(run, { recursive: true });
  fs.writeFileSync(path.join(run, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "arch7-arch0-a0-p0-c0" }));
  const campaigns = path.join(runsDir, "Test", ".architecture-campaigns");
  fs.mkdirSync(campaigns, { recursive: true });
  fs.writeFileSync(path.join(campaigns, "arch7.json"), JSON.stringify({
    format: "model-factory-architecture-campaign-progress-v1",
    campaign_id: "arch7", organism: "Test", complete: false, updated_at: "2026-01-03T00:00:00Z",
    generations: [{ generation_index: 0, complete: true, results: [], survivor_architecture_ids: ["arch7-arch0-a0"], offspring_architecture_ids: [] }],
  }));
  fs.writeFileSync(path.join(campaigns, "arch1.json"), JSON.stringify({
    format: "model-factory-architecture-campaign-progress-v1",
    campaign_id: "arch1", organism: "Test", complete: true, updated_at: "2026-01-01T00:00:00Z", generations: [],
  }));
  // Neither a progress document: a half-written file and something else's
  // manifest that happens to share the directory must not be served as one.
  fs.writeFileSync(path.join(campaigns, "broken.json"), "{ not json");
  fs.writeFileSync(path.join(campaigns, "other.json"), JSON.stringify({ format: "something-else-v1" }));

  const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const port = server.address().port;

  const result = await get(port, "/api/architecture-campaigns?organism=Test");
  assert.equal(result.status, 200);
  assert.deepEqual(result.body.campaigns.map((c) => c.campaign_id), ["arch7", "arch1"]);
  assert.deepEqual(result.body.campaigns[0].generations[0].survivor_architecture_ids, ["arch7-arch0-a0"]);

  // The dot-directory holding them is inert to the run catalog, which is
  // the whole reason it lives there rather than beside the runs.
  const catalog = await get(port, "/api/catalog");
  assert.deepEqual(catalog.body.runs.map((r) => r.run), ["arch7-arch0-a0-p0-c0"]);

  // Same containment discipline as the sibling organism-scoped reads.
  assert.equal((await get(port, "/api/architecture-campaigns?organism=../../etc")).status, 404);
  assert.equal((await get(port, "/api/architecture-campaigns")).status, 400);

  // An organism that never ran one answers with an empty list, not a 404.
  const empty = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-arch-empty-"));
  fs.mkdirSync(path.join(empty, "runs", "Test", "run-a"), { recursive: true });
  fs.writeFileSync(path.join(empty, "runs", "Test", "run-a", "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: "run-a" }));
  const bare = createServer({ runsDir: path.join(empty, "runs") });
  await new Promise((r) => bare.listen(0, r)); t.after(() => bare.close());
  const bareResult = await get(bare.address().port, "/api/architecture-campaigns?organism=Test");
  assert.equal(bareResult.status, 200);
  assert.deepEqual(bareResult.body.campaigns, []);
});

test("dream strip endpoint serves the Phase 4 export independently per episode", async (t) => {
  const server = createServer({ dataDir: fixture() }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
  const result = await get(server.address().port, "/api/sessions/pixel-session/episodes/episode_00000/predictions?kind=dream");
  assert.equal(result.status, 200); assert.equal(result.body.kind, "dream");
});

// The React front end (viewer/web/src) has its own component/unit tests
// (`npm --prefix viewer/web run test`, Vitest) for the EEG, attention,
// developmental-ladder, and pixel-horizon panels this server feeds -- this
// file covers only the server's API contract.

test("web front-end source has no public-internet runtime dependency", () => {
  const html = fs.readFileSync(path.join(__dirname, "../web/index.html"), "utf8");
  assert.doesNotMatch(html, /https?:\/\//);
});

test("empty clinic discovers corpus organisms and monitors state-only runs before experiment artifacts exist", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-empty-bootstrap-"));
  const runsDir = path.join(root, "runs");
  const corpusRoot = path.join(root, "corpora");
  fs.mkdirSync(path.join(corpusRoot, "Crafter"), { recursive: true });
  const active = path.join(runsDir, "Crafter", "first-run");
  fs.mkdirSync(active, { recursive: true });
  fs.writeFileSync(path.join(active, "trial_spec.json"), JSON.stringify({ organism: "Crafter", mode: "fresh" }));
  fs.writeFileSync(path.join(active, "state.json"), JSON.stringify({
    state: "running", reason: "training", updated_at: "2026-08-05T00:00:00Z",
  }));

  const server = createServer({ runsDir, corpusRoot, corpusSpecsDir: path.join(root, "no-specs") });
  await new Promise((resolve) => server.listen(0, resolve));
  t.after(() => server.close());
  const port = server.address().port;

  const catalog = await get(port, "/api/catalog");
  assert.deepEqual(catalog.body.organisms, ["Crafter"]);
  assert.deepEqual(catalog.body.runs, []);
  const factory = await get(port, "/api/factory-runs?organism=Crafter");
  assert.equal(factory.status, 200);
  assert.equal(factory.body.runs[0].run, "first-run");
  assert.equal(factory.body.runs[0].state, "running");

  assert.equal((await get(port, "/api/organisms")).status, 405);
  assert.equal((await post(port, "/api/organisms", { organism: "../escape" })).status, 400);
  const created = await post(port, "/api/organisms", { organism: "Nova-1" });
  assert.equal(created.status, 201);
  assert.deepEqual(created.body, { organism: "Nova-1", created: true });
  assert.deepEqual((await get(port, "/api/catalog")).body.organisms, ["Crafter", "Nova-1"]);
});

// The job-launch API (Model Factory clinic redesign, control-plane
// foundation): POST /api/jobs/*, /api/preview/*, GET /api/corpora,
// GET /api/factory-meta. Every subtest below shells out to
// test/fixtures/fake-python.sh (never a real ccr/torch invocation), scoped
// to just this outer test via t.before/t.after so the pre-existing tests
// above -- several of which rely on a real python3 for quality_cli, or on
// its absence degrading gracefully -- are never affected by which python
// these subtests see.
const FAKE_PYTHON = process.execPath;
const FAKE_PYTHON_PREFIX = JSON.stringify([path.join(__dirname, "fixtures", "fake-python.js")]);

test("job-launch API", async (t) => {
  let originalPython, originalPrefix;
  t.before(() => {
    originalPython = process.env.PYTHON; originalPrefix = process.env.CCR_PYTHON_PREFIX_ARGS;
    process.env.PYTHON = FAKE_PYTHON; process.env.CCR_PYTHON_PREFIX_ARGS = FAKE_PYTHON_PREFIX;
  });
  t.after(() => { process.env.PYTHON = originalPython; process.env.CCR_PYTHON_PREFIX_ARGS = originalPrefix; });

  function fixtureRunsDir() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-jobs-api-"));
    const runsDir = path.join(root, "runs");
    for (const run of ["run-a", "run-b"]) {
      const dir = path.join(runsDir, "Test", run);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, "experiment.json"), JSON.stringify({ organism: "Test", experiment_id: run }));
    }
    return runsDir;
  }

  await t.test("POST /api/jobs/:kind validates method, kind, organism, and referenced run ids", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    assert.equal((await get(port, "/api/jobs/baseline")).status, 405);
    assert.equal((await post(port, "/api/jobs/not-a-kind", { organism: "Test" })).status, 404);
    assert.equal((await post(port, "/api/jobs/baseline", { organism: "Nobody", spec: {} })).status, 404);
    assert.equal((await post(port, "/api/jobs/baseline", {})).status, 400);
    assert.equal((await post(port, "/api/jobs/clone", { organism: "Test", run: "no-such-run" })).status, 404);
    assert.equal(
      (await post(port, "/api/jobs/breed", { organism: "Test", parent_a: "run-a", parent_b: "no-such-run" })).status,
      404,
    );
    // run_id/run_id_prefix name something that doesn't exist yet (nothing to
    // check against a catalog), so a path-traversal-shaped value must still
    // be rejected on its character class alone.
    const traversal = await post(port, "/api/jobs/baseline", { organism: "Test", spec: {}, run_id: "../../evil" });
    assert.equal(traversal.status, 400);
    const traversalPrefix = await post(port, "/api/jobs/search", {
      organism: "Test", spec: {}, options: { run_id_prefix: "../../evil" },
    });
    assert.equal(traversalPrefix.status, 400);

    // search's --champion/--reference-run name existing runs the CLI resolves
    // into directories, so they get the positional run ids' catalog check.
    for (const field of ["champion", "reference_run"]) {
      const unknown = await post(port, "/api/jobs/search", {
        organism: "Test", spec: {}, options: { [field]: "no-such-run" },
      });
      assert.equal(unknown.status, 404, `${field} should 404 for an uncatalogued run`);
      assert.match(unknown.body.error, new RegExp(`unknown options\\.${field}`));
    }
    // A checkpoint file name has no catalog to check against, so it falls
    // back to the same character-class floor a fresh run id uses.
    for (const field of ["checkpoint", "reference_checkpoint"]) {
      const traversalCheckpoint = await post(port, "/api/jobs/clone", {
        organism: "Test", run: "run-a", options: { [field]: "../../../etc/passwd" },
      });
      assert.equal(traversalCheckpoint.status, 400, `${field} should reject a traversal-shaped name`);
    }
    // breed names one checkpoint per parent, on the same footing.
    for (const field of ["checkpoint_a", "checkpoint_b"]) {
      const traversalCheckpoint = await post(port, "/api/jobs/breed", {
        organism: "Test", parent_a: "run-a", parent_b: "run-b", options: { [field]: "../../../etc/passwd" },
      });
      assert.equal(traversalCheckpoint.status, 400, `${field} should reject a traversal-shaped name`);
    }
    // The legitimate default still clears it.
    assert.equal(
      (await post(port, "/api/jobs/clone", { organism: "Test", run: "run-a", options: { checkpoint: "best-validation.pt" } })).status,
      200,
    );
    assert.equal(
      (await post(port, "/api/jobs/breed", {
        organism: "Test", parent_a: "run-a", parent_b: "run-b",
        options: { checkpoint_a: "best-validation.pt", checkpoint_b: "last.pt" },
      })).status,
      200,
    );
  });

  await t.test("a launched job is registered, completes, and is visible via GET /api/jobs with live run state", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    const launch = await post(port, "/api/jobs/baseline", { organism: "Test", spec: { organism: "Test", mode: "fresh" } });
    assert.equal(launch.status, 200);
    assert.equal(launch.body.organism, "Test");
    assert.equal(launch.body.run_ids.length, 1);
    const [jobId, runId] = [launch.body.job_id, launch.body.run_ids[0]];
    assert.match(launch.body.argv.join(" "), /factory baseline .*\.spec\.json --root/);

    // The run itself has started producing state -- written directly here,
    // exactly like a real trial's runner.run_trial would, to confirm
    // GET /api/jobs augments the registry entry with it rather than only
    // ever reporting what it launched.
    fs.mkdirSync(path.join(runsDir, "Test", runId), { recursive: true });
    fs.writeFileSync(path.join(runsDir, "Test", runId, "state.json"), JSON.stringify({
      state: "running", reason: null, updated_at: "2026-01-01T00:00:00Z",
    }));
    fs.writeFileSync(path.join(runsDir, "Test", runId, "heartbeat.json"), JSON.stringify({
      format: "model-factory-trial-heartbeat-v1", run_id: runId, heartbeat_seconds: 12345.6,
    }));

    const listed = await get(port, "/api/jobs?organism=Test");
    assert.equal(listed.status, 200);
    assert.equal(listed.body.jobs.length, 1);
    assert.equal(listed.body.jobs[0].job_id, jobId);
    assert.deepEqual(listed.body.jobs[0].runs, [{
      run_id: runId, state: "running", state_reason: null,
      updated_at: "2026-01-01T00:00:00Z", heartbeat_seconds: 12345.6,
    }]);
    assert.equal((await get(port, "/api/jobs?organism=Nobody")).body.jobs.length, 0);

    const finished = await waitFor(async () => {
      const current = await get(port, "/api/jobs?organism=Test");
      return current.body.jobs[0].status !== "running" ? current.body.jobs[0] : null;
    });
    assert.equal(finished.status, "completed");
  });

  await t.test("baseline/search launches force the temp spec's organism to match the job's, ignoring/overriding a mismatched or missing spec.organism", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    // A client that got the two organism fields out of sync (or omitted
    // spec.organism entirely) must not end up with the CLI subprocess
    // writing the run under a different directory than the job registry
    // (and every later GET /api/jobs/POST /cancel lookup) believes it's
    // in -- Codex review, PR #277.
    const mismatched = await post(port, "/api/jobs/baseline", {
      organism: "Test", spec: { organism: "SomethingElse", mode: "fresh" },
    });
    assert.equal(mismatched.status, 200);
    const mismatchedSpecFile = mismatched.body.argv[mismatched.body.argv.indexOf("baseline") + 1];
    assert.equal(JSON.parse(fs.readFileSync(mismatchedSpecFile, "utf8")).organism, "Test");

    const missing = await post(port, "/api/jobs/search", {
      organism: "Test", spec: { mode: "fresh" }, options: { population_size: 2 },
    });
    assert.equal(missing.status, 200);
    const missingSpecFile = missing.body.argv[missing.body.argv.indexOf("search") + 1];
    assert.equal(JSON.parse(fs.readFileSync(missingSpecFile, "utf8")).organism, "Test");
  });

  await t.test("GET /api/jobs/:id/log tails output; unknown and traversal-shaped ids 404, not 500", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    const launch = await post(port, "/api/jobs/clone", { organism: "Test", run: "run-a" });
    assert.equal(launch.status, 200);
    const jobId = launch.body.job_id;

    await waitFor(async () => {
      const current = await get(port, "/api/jobs?organism=Test");
      return current.body.jobs[0].status !== "running" ? current.body.jobs[0] : null;
    });

    assert.equal((await post(port, `/api/jobs/${jobId}/log`, {})).status, 405);
    const first = await get(port, `/api/jobs/${jobId}/log?offset=0`);
    assert.equal(first.status, 200);
    assert.match(first.body.text, /factory clone run-a/);
    assert.ok(first.body.next_offset > 0);
    const second = await get(port, `/api/jobs/${jobId}/log?offset=${first.body.next_offset}`);
    assert.equal(second.body.text, "");

    assert.equal((await get(port, "/api/jobs/no-such-job/log")).status, 404);
    assert.equal((await get(port, `/api/jobs/${encodeURIComponent("../../etc/passwd")}/log`)).status, 404);
  });

  await t.test("POST /api/jobs/:id/cancel stops a running job; unknown ids 404, wrong method 405", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    assert.equal((await get(port, "/api/jobs/some-id/cancel")).status, 405);
    assert.equal((await post(port, "/api/jobs/no-such-job/cancel", {})).status, 404);

    process.env.FAKE_PYTHON_DELAY_MS = "2000";
    try {
      const launch = await post(port, "/api/jobs/clone", { organism: "Test", run: "run-a" });
      await waitFor(async () => {
        const current = await get(port, "/api/jobs?organism=Test");
        return current.body.jobs[0].pid ? current.body.jobs[0] : null;
      });
      const cancelled = await post(port, `/api/jobs/${launch.body.job_id}/cancel`, {});
      assert.equal(cancelled.status, 200);
      const finished = await waitFor(async () => {
        const current = await get(port, "/api/jobs?organism=Test");
        return current.body.jobs[0].status !== "running" ? current.body.jobs[0] : null;
      });
      // SIGTERM with no handler in the fake script kills it outright --
      // never "completed", the exit code a graceful stop would report.
      assert.notEqual(finished.status, "completed");
    } finally {
      delete process.env.FAKE_PYTHON_DELAY_MS;
    }
  });

  await t.test("POST /api/jobs/:kind refuses beyond --max-concurrent-jobs with 409", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir, maxConcurrentJobs: 1 });
    await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    process.env.FAKE_PYTHON_DELAY_MS = "2000";
    try {
      const first = await post(port, "/api/jobs/clone", { organism: "Test", run: "run-a" });
      assert.equal(first.status, 200);
      const second = await post(port, "/api/jobs/clone", { organism: "Test", run: "run-b" });
      assert.equal(second.status, 409);
    } finally {
      delete process.env.FAKE_PYTHON_DELAY_MS;
    }
  });

  await t.test("GET /api/corpora lists frozen corpora, skips a corrupt manifest, and rejects a traversal-shaped organism", async (t) => {
    const runsDir = fixtureRunsDir();
    const corpusRoot = path.join(path.dirname(runsDir), "corpora");
    const good = path.join(corpusRoot, "Test", "corpus-1"); fs.mkdirSync(good, { recursive: true });
    fs.writeFileSync(path.join(good, "corpus_manifest.json"), JSON.stringify({
      format: "model-factory-corpus-manifest-v1", corpus_id: "corpus-1", data_contract_hash: "hash-1",
      sessions: { train: [1, 2, 3], validation: [1], test: [] },
    }));
    const corrupt = path.join(corpusRoot, "Test", "corpus-2"); fs.mkdirSync(corrupt, { recursive: true });
    fs.writeFileSync(path.join(corrupt, "corpus_manifest.json"), "not json{{{");

    const server = createServer({ runsDir, corpusRoot }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    const listed = await get(port, "/api/corpora?organism=Test");
    assert.equal(listed.status, 200);
    assert.deepEqual(listed.body.corpora, [{
      corpus_id: "corpus-1", organism: "Test", data_contract_hash: "hash-1",
      requires_parent_checkpoint: false,
      session_counts: { train: 3, validation: 1, test: 0 },
    }]);
    assert.equal((await get(port, "/api/corpora")).body.corpora.length, 1);
    assert.equal((await get(port, `/api/corpora?organism=${encodeURIComponent("../../etc")}`)).status, 400);
  });

  await t.test("a fresh baseline rejects a navigation retention corpus before spawning", async (t) => {
    const runsDir = fixtureRunsDir();
    const corpusRoot = path.join(path.dirname(runsDir), "corpora");
    const navigation = path.join(corpusRoot, "Test", "navigation-corpus");
    fs.mkdirSync(navigation, { recursive: true });
    fs.writeFileSync(path.join(navigation, "corpus_manifest.json"), JSON.stringify({
      format: "model-factory-corpus-manifest-v1",
      corpus_id: "navigation-corpus",
      data_contract_hash: "navigation-hash",
      data_contract: { retention_policy: { corpus_id: "generic-corpus" } },
      sessions: { train: [1], validation: [1], test: [] },
    }));

    const server = createServer({ runsDir, corpusRoot });
    await new Promise((resolve) => server.listen(0, resolve));
    t.after(() => server.close());
    const response = await post(server.address().port, "/api/jobs/baseline", {
      organism: "Test",
      spec: { organism: "Test", mode: "fresh", data: { corpus_id: "navigation-corpus" } },
    });

    assert.equal(response.status, 400);
    assert.match(response.body.error, /fine-tune-only.*generic action-effects corpus/);
  });

  await t.test("an empty workspace can launch a corpus recipe as its first monitored job", async (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "clinic-corpus-bootstrap-"));
    const runsDir = path.join(root, "runs");
    const corpusRoot = path.join(root, "corpora");
    const corpusSpecsDir = path.join(root, "specs");
    fs.mkdirSync(corpusSpecsDir, { recursive: true });
    fs.writeFileSync(path.join(corpusSpecsDir, "starter.yaml"), [
      "corpus_id: starter-corpus", "organism: Test", "generator:", "  world: crafter",
    ].join("\n"));
    fs.writeFileSync(path.join(corpusSpecsDir, "navigation.yaml"), [
      "corpus_id: navigation-corpus", "organism: Test", "retention_policy:", "  corpus_id: starter-corpus",
    ].join("\n"));

    const server = createServer({ runsDir, corpusRoot, corpusSpecsDir });
    await new Promise((resolve) => server.listen(0, resolve));
    t.after(() => server.close());
    const port = server.address().port;

    assert.deepEqual((await get(port, "/api/catalog")).body, {
      runs_dir: path.resolve(runsDir), episode_cache_dir: path.resolve(path.join(__dirname, "../..", "notebooks", "episode_cache")),
      organisms: ["Test"], runs: [],
    });
    assert.deepEqual((await get(port, "/api/corpus-specs?organism=Test")).body.specs, [
      {
        spec_id: "navigation.yaml", corpus_id: "navigation-corpus", organism: "Test",
        requires_parent_checkpoint: true,
      },
      {
        spec_id: "starter.yaml", corpus_id: "starter-corpus", organism: "Test",
        requires_parent_checkpoint: false,
      },
    ]);
    assert.equal((await post(port, "/api/jobs/corpus", { organism: "Test", spec_id: "missing.yaml" })).status, 404);

    const launched = await post(port, "/api/jobs/corpus", { organism: "Test", spec_id: "starter.yaml" });
    assert.equal(launched.status, 200);
    assert.equal(launched.body.kind, "corpus");
    assert.deepEqual(launched.body.run_ids, []);
    assert.match(launched.body.argv.join(" "), /factory corpus build .*starter\.yaml --root/);
    assert.match(launched.body.argv.join(" "), /--organism Test/);
    const finished = await waitFor(async () => {
      const current = await get(port, "/api/jobs?organism=Test");
      return current.body.jobs[0]?.status !== "running" ? current.body.jobs[0] : null;
    });
    assert.equal(finished.status, "completed");
  });

  await t.test("POST /api/preview/{search,breed} dry-runs via the CLI, validates like a launch, and cleans up its temp spec", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    assert.equal((await post(port, "/api/preview/baseline", { organism: "Test", spec: {} })).status, 404);
    assert.equal((await get(port, "/api/preview/search")).status, 405);
    assert.equal((await post(port, "/api/preview/search", { organism: "Nobody", spec: {} })).status, 404);
    assert.equal(
      (await post(port, "/api/preview/breed", { organism: "Test", parent_a: "run-a", parent_b: "no-such-run" })).status,
      404,
    );

    const searchPreview = await post(port, "/api/preview/search", { organism: "Test", spec: { organism: "Test" } });
    assert.equal(searchPreview.status, 200);
    assert.equal(searchPreview.body.marker, "fake-dry-run");
    const breedPreview = await post(port, "/api/preview/breed", { organism: "Test", parent_a: "run-a", parent_b: "run-b" });
    assert.equal(breedPreview.status, 200);
    assert.equal(breedPreview.body.marker, "fake-dry-run");

    // Neither preview kept a job identity: no "preview-*" registry entry,
    // and search's temp spec sidecar (breed's buildLaunch writes none) was
    // removed right after the dry-run call returned.
    assert.equal((await get(port, "/api/jobs?organism=Test")).body.jobs.length, 0);
    const jobsDirEntries = fs.existsSync(path.join(runsDir, ".clinic-jobs"))
      ? fs.readdirSync(path.join(runsDir, ".clinic-jobs"))
      : [];
    assert.deepEqual(jobsDirEntries.filter((name) => name.endsWith(".spec.json")), []);
  });

  await t.test("GET /api/factory-meta returns the factory's declared modes/objectives/genome schemas/backbones", async (t) => {
    const runsDir = fixtureRunsDir();
    const server = createServer({ runsDir }); await new Promise((r) => server.listen(0, r)); t.after(() => server.close());
    const port = server.address().port;

    const result = await get(port, "/api/factory-meta");
    assert.equal(result.status, 200);
    assert.deepEqual(result.body, {
      modes: ["fresh", "clone", "resume", "fine_tune"],
      objectives: ["windowed_rollout", "autoregressive"],
      genome_schemas: ["generic_action_effects_v1"],
      architecture_genome_schemas: ["architecture_search_v1"],
      backbone_presets: {
        gru: { backbone: "gru", backbone_kwargs: {} },
        dilated_conv: { backbone: "dilated_conv", backbone_kwargs: {} },
        transformer_h2_l1: { backbone: "transformer", backbone_kwargs: { n_heads: 2, n_layers: 1 } },
        transformer_h4_l2: { backbone: "transformer", backbone_kwargs: { n_heads: 4, n_layers: 2 } },
      },
      backbones: ["gru", "dilated_conv", "transformer"],
    });
  });
});
