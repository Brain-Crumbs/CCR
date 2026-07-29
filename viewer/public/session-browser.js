/** Local, dependency-free browser for one recorded pixel-prediction run. */

function selectionQuery({ organism, run }) {
  return `?organism=${encodeURIComponent(organism)}&run=${encodeURIComponent(run)}`;
}

function runSummaryQuery({ organism, run }) {
  return `/api/runs?organism=${encodeURIComponent(organism)}&run=${encodeURIComponent(run)}`;
}

export function episodeUrls(sessionId, episodeId, experimentId = null, selection = null) {
  const base = `/api/sessions/${encodeURIComponent(sessionId)}/episodes/${encodeURIComponent(episodeId)}`;
  const params = new URLSearchParams(selection || {});
  if (experimentId) params.set("experiment", experimentId);
  const suffix = params.size ? `?${[...params].map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`).join("&")}` : "";
  return { frames: `${base}/frames${suffix}`, predictions: `${base}/predictions${suffix}` };
}

function picker(label, ariaLabel, values, selected, onChange) {
  const control = document.createElement("label"); control.className = "run-picker"; control.textContent = `${label} `;
  const select = document.createElement("select"); select.setAttribute("aria-label", ariaLabel);
  for (const value of values) {
    const option = document.createElement("option"); option.value = value; option.textContent = value;
    option.selected = value === selected; select.append(option);
  }
  select.addEventListener("change", () => onChange(select.value)); control.append(select);
  return control;
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toPrecision(3) : "Not reported";
}

function summaryStat(label, value, detail = "") {
  const stat = document.createElement("div"); stat.className = "run-summary__stat";
  const labelEl = document.createElement("span"); labelEl.className = "run-summary__label"; labelEl.textContent = label;
  const valueEl = document.createElement("strong"); valueEl.className = "run-summary__value"; valueEl.textContent = value;
  stat.append(labelEl, valueEl);
  if (detail) {
    const detailEl = document.createElement("span"); detailEl.className = "run-summary__detail"; detailEl.textContent = detail;
    stat.append(detailEl);
  }
  return stat;
}

function renderRunSummary(summary) {
  const panel = document.createElement("section"); panel.className = "run-summary"; panel.setAttribute("aria-label", "Run summary");
  const heading = document.createElement("h3"); heading.className = "run-summary__heading"; heading.textContent = "Run summary";
  const status = document.createElement("span"); status.className = "run-summary__status";
  const promoted = summary?.promotion?.promoted;
  status.textContent = promoted === true ? "Promoted" : promoted === false ? "Not promoted" : "Report unavailable";
  if (promoted === true) status.classList.add("run-summary__status--good");
  if (promoted === false) status.classList.add("run-summary__status--bad");
  heading.append(status);

  const grid = document.createElement("div"); grid.className = "run-summary__grid";
  if (!summary) {
    grid.append(summaryStat("Run report", "No summary recorded", "Training and evaluation statistics are unavailable."));
  } else {
    const training = summary.training || {}, evaluation = summary.evaluation || {}, model = summary.model || {};
    const checkpoint = model.checkpoint_sha256 ? `checkpoint ${model.checkpoint_sha256.slice(0, 12)}` : "checkpoint not recorded";
    const scale = [training.episodes != null ? `${training.episodes} episodes` : null, training.samples != null ? `${training.samples} samples` : null].filter(Boolean).join(" · ");
    const ratio = evaluation.model_over_copy_last_mse;
    const comparison = typeof ratio === "number" && Number.isFinite(ratio)
      ? `${Math.abs((ratio - 1) * 100).toFixed(1)}% ${ratio <= 1 ? "below" : "above"} copy-last`
      : "copy-last comparison unavailable";
    grid.append(
      summaryStat("Model", model.type || "Not recorded", checkpoint),
      summaryStat("Training", training.epochs != null ? `${training.epochs} epochs` : "Not recorded", scale || training.objective || ""),
      summaryStat("Final loss", formatMetric(training.final_total_loss), training.objective || ""),
      summaryStat(`t+${evaluation.horizon || 1} MSE`, formatMetric(evaluation.model_mse), comparison),
      summaryStat("Rollout", evaluation.rollout_health || "Not reported", [evaluation.split, evaluation.prediction_mode].filter(Boolean).join(" · ")),
    );
  }
  panel.append(heading, grid);
  return panel;
}

export function mountSessionBrowser(root, { loadCatalog = () => fetch("/api/catalog").then((r) => {
  if (!r.ok) throw new Error(`Unable to load runs (${r.status})`);
  return r.json();
}), loadSessions = (organism, run) => fetch(`/api/sessions${selectionQuery({ organism, run })}`).then((r) => {
  if (!r.ok) throw new Error(`Unable to load sessions (${r.status})`);
  return r.json();
}).then((x) => x.sessions) } = {}) {
  let catalog = null;

  function renderEmpty(message) {
    const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = message; root.replaceChildren(empty);
  }

  function renderRun(organism, run, summary, sessions, session, episode) {
    root.replaceChildren();
    const organisms = catalog.organisms;
    const organismControl = picker("Organism", "Organism", organisms, organism, (nextOrganism) => {
      const nextRun = catalog.runs.find((entry) => entry.organism === nextOrganism)?.run;
      if (nextRun) loadAndRender(nextOrganism, nextRun);
    });
    const runs = catalog.runs.filter((entry) => entry.organism === organism).map((entry) => entry.run);
    const runControl = picker("Run", "Run", runs, run, (nextRun) => loadAndRender(organism, nextRun));

    let recordingControl = null;
    if (sessions.length > 1) {
      recordingControl = picker("Recording", "Recording", sessions.map((item) => item.id), session.id, (id) => {
        const next = sessions.find((item) => item.id === id);
        if (next) renderRun(organism, run, summary, sessions, next, next.episodes[0]);
      });
    }
    let episodeControl = null;
    if (session.episodes.length > 1) {
      episodeControl = picker("Episode", "Episode", session.episodes, episode, (next) => renderRun(organism, run, summary, sessions, session, next));
    }

    const title = document.createElement("h2"); title.textContent = "Horizons pixel prediction";
    const subtitle = document.createElement("p"); subtitle.className = "run-name"; subtitle.textContent = `${organism} / ${run} / ${session.id}`;
    const summaryPanel = renderRunSummary(summary);
    const viewer = document.createElement("pixel-horizon-viewer");
    const urls = episodeUrls(session.id, episode, run, { organism, run });
    viewer.setAttribute("frames-src", urls.frames); viewer.setAttribute("predictions-src", urls.predictions);
    root.append(...[organismControl, runControl, recordingControl, episodeControl, summaryPanel, title, subtitle, viewer].filter(Boolean));
    location.hash = `${encodeURIComponent(organism)}/${encodeURIComponent(run)}/${encodeURIComponent(session.id)}/${encodeURIComponent(episode)}`;
  }

  function loadAndRender(organism, run) {
    Promise.all([
      loadSessions(organism, run),
      fetch(runSummaryQuery({ organism, run })).then((r) => r.ok ? r.json() : null),
    ]).then(([sessions, summary]) => {
      const match = location.hash.slice(1).split("/").map(decodeURIComponent);
      const selected = sessions.find((session) => session.id === match[2] && session.episodes.includes(match[3])) || sessions[0];
      if (selected) renderRun(organism, run, summary, sessions, selected, match[3] || selected.episodes[0]);
      else renderEmpty("No recordings are associated with this run yet.");
    }, (error) => renderEmpty(String(error)));
  }

  loadCatalog().then((loaded) => {
    catalog = loaded;
    const match = location.hash.slice(1).split("/").map(decodeURIComponent);
    const initial = loaded.runs.find((entry) => entry.organism === match[0] && entry.run === match[1]) || loaded.runs[0];
    if (initial) loadAndRender(initial.organism, initial.run);
    else renderEmpty("No experiment runs found.");
  }, (error) => renderEmpty(String(error)));
}
