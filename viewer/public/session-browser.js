/** Local, dependency-free session browser for the read-only clinic. */

import { mountDiagnostics } from "./diagnostic-panels.js";

function qualityNode(quality) {
  const wrap = document.createElement("div");
  wrap.className = `quality quality--${quality.verdict}`;
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.setAttribute("aria-label", `${quality.verdict} data quality`);
  badge.textContent = quality.verdict;
  wrap.append(badge);
  const details = [...(quality.issues || []), ...(quality.warnings || [])];
  if (details.length) {
    const list = document.createElement("ul"); list.className = "checks";
    for (const detail of details) { const li = document.createElement("li"); li.textContent = detail; list.append(li); }
    wrap.append(list);
  } else {
    const ok = document.createElement("span"); ok.className = "checks-ok"; ok.textContent = "All checks passed"; wrap.append(ok);
  }
  return wrap;
}

export function episodeUrls(sessionId, episodeId) {
  const base = `/api/sessions/${encodeURIComponent(sessionId)}/episodes/${encodeURIComponent(episodeId)}`;
  return { frames: `${base}/frames`, predictions: `${base}/predictions` };
}

export function mountSessionBrowser(root, { loadSessions = () => fetch("/api/sessions").then((r) => {
  if (!r.ok) throw new Error(`Unable to load sessions (${r.status})`);
  return r.json();
}).then((x) => x.sessions) } = {}) {
  async function showEpisode(session, episode) {
    root.replaceChildren();
    const back = document.createElement("button"); back.className = "back"; back.textContent = "← All sessions";
    back.addEventListener("click", () => { location.hash = ""; renderSessions(sessions); });
    const title = document.createElement("h2"); title.textContent = `${session.id} / ${episode}`;
    const stripTitle = document.createElement("h3"); stripTitle.textContent = "Predicted vs actual";
    const viewer = document.createElement("pixel-horizon-viewer");
    const urls = episodeUrls(session.id, episode);
    viewer.setAttribute("frames-src", urls.frames); viewer.setAttribute("predictions-src", urls.predictions);
    const dreamTitle = document.createElement("h3"); dreamTitle.textContent = "Dreamed vs actual";
    const dream = document.createElement("pixel-horizon-viewer");
    dream.setAttribute("frames-src", urls.frames); dream.setAttribute("predictions-src", `${urls.predictions}?kind=dream`);
    const diagnostics = document.createElement("div"); diagnostics.className = "diagnostics"; diagnostics.textContent = "Loading diagnostic streams…";
    root.append(back, title, stripTitle, viewer, dreamTitle, dream, diagnostics);
    location.hash = `${encodeURIComponent(session.id)}/${encodeURIComponent(episode)}`;
    try {
      const response = await fetch(`/api/sessions/${encodeURIComponent(session.id)}`);
      if (!response.ok) throw new Error(`Unable to load diagnostics (${response.status})`);
      const detail = await response.json();
      mountDiagnostics(diagnostics, detail.streams?.[episode] || [], detail.decisions?.[episode] || [], detail.session || session);
    } catch (error) { diagnostics.textContent = String(error); diagnostics.setAttribute("role", "alert"); }
  }

  function renderSessions(items) {
    root.replaceChildren();
    if (!items.length) { const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = "No recorded sessions found."; root.append(empty); return; }
    const groups = Object.groupBy ? Object.groupBy(items, (s) => s.name || "legacy") : items.reduce((all, s) => { (all[s.name || "legacy"] ||= []).push(s); return all; }, {});
    const organisms = document.createElement("div"); organisms.className = "organisms";
    for (const [name, grouped] of Object.entries(groups)) {
      const section = document.createElement("section"); section.className = "organism";
      const heading = document.createElement("h2"); heading.textContent = name;
      const grid = document.createElement("div"); grid.className = "session-grid";
      for (const session of grouped) {
        const card = document.createElement("article"); card.className = "session";
        const title = document.createElement("div"); title.className = "session-title";
        const strong = document.createElement("strong"); strong.textContent = session.id;
        const count = document.createElement("span"); count.textContent = `${session.episodes.length} episode${session.episodes.length === 1 ? "" : "s"}`;
        title.append(strong, count); card.append(title, qualityNode(session.quality));
        const episodes = document.createElement("div"); episodes.className = "episode-links";
        for (const episode of session.episodes) { const open = document.createElement("button"); open.textContent = `View ${episode}`; open.addEventListener("click", () => showEpisode(session, episode)); episodes.append(open); }
        card.append(episodes); grid.append(card);
      }
      section.append(heading, grid); organisms.append(section);
    }
    root.append(organisms);
  }

  let sessions = [];
  loadSessions().then((loaded) => {
    sessions = loaded; const match = location.hash.slice(1).split("/").map(decodeURIComponent);
    const selected = sessions.find((s) => s.id === match[0] && s.episodes.includes(match[1]));
    if (selected) showEpisode(selected, match[1]); else renderSessions(sessions);
  }, (error) => { const alert = document.createElement("p"); alert.setAttribute("role", "alert"); alert.textContent = String(error); root.replaceChildren(alert); });
}
