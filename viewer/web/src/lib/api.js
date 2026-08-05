/** Thin fetch client over viewer/server.js's read-only JSON API. Every
 * function here mirrors exactly one server route; no reshaping happens here
 * so a server contract change is a one-line diff, not a UI hunt. */

async function getJSON(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error || `request failed (${res.status})`);
  return body;
}

async function postJSON(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error || `request failed (${res.status})`);
  return body;
}

function qs(params = {}) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") search.set(key, value);
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export const api = {
  catalog: () => getJSON("/api/catalog"),
  createOrganism: (organism) => postJSON("/api/organisms", { organism }),
  runSummary: (organism, run) => getJSON(`/api/runs${qs({ organism, run })}`),
  experimentArtifacts: (organism, run) => getJSON(`/api/experiments${qs({ organism, run })}`),
  registry: (organism) => getJSON(`/api/registry${qs({ organism })}`),
  factoryRuns: (organism) => getJSON(`/api/factory-runs${qs({ organism })}`),
  architectureCampaigns: (organism) => getJSON(`/api/architecture-campaigns${qs({ organism })}`),
  sessions: (organism, run, filters = {}) => getJSON(`/api/sessions${qs({ organism, run, ...filters })}`).then((x) => x.sessions),
  session: (sessionId, organism, run) => getJSON(`/api/sessions/${encodeURIComponent(sessionId)}${qs({ organism, run })}`),
  streams: (sessionId, episodeId, organism, run) =>
    getJSON(`/api/sessions/${encodeURIComponent(sessionId)}/episodes/${encodeURIComponent(episodeId)}/streams${qs({ organism, run })}`).then((x) => x.records),
  decisions: (sessionId, episodeId, organism, run) =>
    getJSON(`/api/sessions/${encodeURIComponent(sessionId)}/episodes/${encodeURIComponent(episodeId)}/decisions${qs({ organism, run })}`).then((x) => x.records),

  // Control-plane (clinic redesign, Build tab and beyond): factory-meta and
  // corpora are read-only GETs; jobs/preview are the write surface, mirrored
  // one-for-one onto viewer/server.js's job-launch routes with no reshaping.
  factoryMeta: () => getJSON("/api/factory-meta"),
  corpora: (organism) => getJSON(`/api/corpora${qs({ organism })}`).then((x) => x.corpora),
  corpusSpecs: () => getJSON("/api/corpus-specs").then((x) => x.specs),
  jobs: (organism) => getJSON(`/api/jobs${qs({ organism })}`).then((x) => x.jobs),
  jobLog: (jobId, offset = 0) => getJSON(`/api/jobs/${encodeURIComponent(jobId)}/log${qs({ offset })}`),
  cancelJob: (jobId) => postJSON(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {}),
  launchJob: (kind, body) => postJSON(`/api/jobs/${kind}`, body),
  previewJob: (kind, body) => postJSON(`/api/preview/${kind}`, body),
};

/** URLs (not fetched here) for the pixel-frame/prediction endpoints the
 * PixelHorizonViewer streams itself, since it needs the base64 frame payload
 * as-is rather than a parsed object. */
export function episodeUrls(sessionId, episodeId, { organism = null, run = null, experiment = null } = {}) {
  const base = `/api/sessions/${encodeURIComponent(sessionId)}/episodes/${encodeURIComponent(episodeId)}`;
  const params = qs({ organism, run, experiment });
  return { frames: `${base}/frames${params}`, predictions: `${base}/predictions${params}` };
}

export { getJSON, postJSON, qs };
