/** Local, dependency-free browser for one recorded pixel-prediction run. */

function selectionQuery({ organism, run }) {
  return `?organism=${encodeURIComponent(organism)}&run=${encodeURIComponent(run)}`;
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

  function renderRun(organism, run, sessions, session, episode) {
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
        if (next) renderRun(organism, run, sessions, next, next.episodes[0]);
      });
    }
    let episodeControl = null;
    if (session.episodes.length > 1) {
      episodeControl = picker("Episode", "Episode", session.episodes, episode, (next) => renderRun(organism, run, sessions, session, next));
    }

    const title = document.createElement("h2"); title.textContent = "Horizons pixel prediction";
    const subtitle = document.createElement("p"); subtitle.className = "run-name"; subtitle.textContent = `${organism} / ${run} / ${session.id}`;
    const viewer = document.createElement("pixel-horizon-viewer");
    const urls = episodeUrls(session.id, episode, run, { organism, run });
    viewer.setAttribute("frames-src", urls.frames); viewer.setAttribute("predictions-src", urls.predictions);
    root.append(organismControl, runControl, recordingControl, episodeControl, title, subtitle, viewer);
    location.hash = `${encodeURIComponent(organism)}/${encodeURIComponent(run)}/${encodeURIComponent(session.id)}/${encodeURIComponent(episode)}`;
  }

  function loadAndRender(organism, run) {
    loadSessions(organism, run).then((sessions) => {
      const match = location.hash.slice(1).split("/").map(decodeURIComponent);
      const selected = sessions.find((session) => session.id === match[2] && session.episodes.includes(match[3])) || sessions[0];
      if (selected) renderRun(organism, run, sessions, selected, match[3] || selected.episodes[0]);
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
