/** React session browser. React is injected so the component is easy to test. */
export function createSessionBrowser(React) {
  const { createElement: h, useEffect, useMemo, useState } = React;
  function Quality({ quality }) {
    const details = [...(quality.issues || []), ...(quality.warnings || [])];
    return h("div", { className: `quality quality--${quality.verdict}` },
      h("span", { className: "badge", "aria-label": `${quality.verdict} data quality` }, quality.verdict),
      details.length ? h("ul", { className: "checks" }, details.map((item, i) => h("li", { key: i }, item))) : h("span", { className: "checks-ok" }, "All checks passed"));
  }
  return function SessionBrowser({ loadSessions, initialSessions = [] }) {
    const [sessions, setSessions] = useState(initialSessions), [error, setError] = useState(null);
    useEffect(() => { if (loadSessions) loadSessions().then(setSessions, (e) => setError(String(e))); }, [loadSessions]);
    const groups = useMemo(() => sessions.reduce((all, session) => {
      (all[session.name || "legacy"] ||= []).push(session); return all;
    }, {}), [sessions]);
    if (error) return h("p", { role: "alert" }, error);
    if (!sessions.length) return h("p", { className: "empty" }, "No recorded sessions found.");
    return h("div", { className: "organisms" }, Object.entries(groups).map(([name, items]) =>
      h("section", { className: "organism", key: name }, h("h2", null, name),
        h("div", { className: "session-grid" }, items.map((session) => h("article", { className: "session", key: session.id },
          h("div", { className: "session-title" }, h("strong", null, session.id), h("span", null, `${session.episodes.length} episode${session.episodes.length === 1 ? "" : "s"}`)),
          h(Quality, { quality: session.quality })))))
    ));
  };
}
