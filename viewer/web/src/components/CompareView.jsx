import { useEffect, useMemo, useState } from "react";
import { api, episodeUrls } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { RunSummary } from "./RunSummary.jsx";
import { PixelHorizonViewer } from "./PixelHorizonViewer.jsx";

/**
 * Two runs' predictions, side by side, on the same session/episode -- "view
 * the state of the Factory builds and inspect models against each other."
 * Needs no new prediction-fetching plumbing: a Model Factory run exports
 * its predictions into its own run directory (see runner.py's
 * `_export_clinic_predictions`), so fetching Run A's and Run B's exports is
 * just two ordinary `episodeUrls(...)` calls with a different `run`/
 * `experiment`. Works fully when both runs were evaluated against the same
 * frozen corpus (the common case for a clone/fine_tune lineage); if Run B
 * has no export for the chosen session, that column just falls back to the
 * baselines exactly like a missing prediction file does anywhere else in
 * the clinic.
 */
export function CompareView({ organism, runs, defaultRun }) {
  const [runA, setRunA] = useState(defaultRun);
  const [runB, setRunB] = useState(() => runs.find((r) => r !== defaultRun) ?? defaultRun);
  const [sessions, setSessions] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [episode, setEpisode] = useState(null);
  const [decisions, setDecisions] = useState([]);
  const [summaryA, setSummaryA] = useState(null);
  const [summaryB, setSummaryB] = useState(null);
  const [tick, setTick] = useState(null);

  // Track the app's currently selected run until the user explicitly picks
  // a different Run A here.
  useEffect(() => { setRunA(defaultRun); }, [defaultRun]);

  useEffect(() => {
    if (!organism || !runA) return undefined;
    let cancelled = false;
    setSessions(null);
    setSessionId(null);
    setEpisode(null);
    api.sessions(organism, runA).then((list) => {
      if (cancelled) return;
      setSessions(list);
      setSessionId(list[0]?.id ?? null);
      setEpisode(list[0]?.episodes?.[0] ?? null);
    }, () => !cancelled && setSessions([]));
    return () => { cancelled = true; };
  }, [organism, runA]);

  useEffect(() => {
    if (!organism || !runA || !runB) return undefined;
    let cancelled = false;
    api.runSummary(organism, runA).then((s) => !cancelled && setSummaryA(s), () => !cancelled && setSummaryA(null));
    api.runSummary(organism, runB).then((s) => !cancelled && setSummaryB(s), () => !cancelled && setSummaryB(null));
    return () => { cancelled = true; };
  }, [organism, runA, runB]);

  useEffect(() => {
    if (!sessionId || !episode) { setDecisions([]); return undefined; }
    let cancelled = false;
    api.decisions(sessionId, episode, organism, runA).then((d) => !cancelled && setDecisions(d), () => !cancelled && setDecisions([]));
    return () => { cancelled = true; };
  }, [sessionId, episode, organism, runA]);

  useEffect(() => { setTick(null); }, [sessionId, episode]);

  const urlsA = useMemo(
    () => (sessionId && episode ? episodeUrls(sessionId, episode, { organism, run: runA, experiment: runA }) : null),
    [sessionId, episode, organism, runA],
  );
  const urlsB = useMemo(
    () => (sessionId && episode ? episodeUrls(sessionId, episode, { organism, run: runB, experiment: runB }) : null),
    [sessionId, episode, organism, runB],
  );

  const selectedSession = sessions?.find((s) => s.id === sessionId) ?? null;

  return (
    <section className="compare-view" aria-labelledby="compare-title">
      <h3 id="compare-title">Compare</h3>
      <p>Two runs' predictions on the same session, side by side. Full comparison needs both runs evaluated against the same frozen corpus.</p>
      <div className="pickers">
        <Picker label="Run A" ariaLabel="Run A" value={runA} options={runs} onChange={setRunA} />
        <Picker label="Run B" ariaLabel="Run B" value={runB} options={runs} onChange={setRunB} />
        {sessions?.length > 0 && (
          <Picker label="Session" ariaLabel="Session" value={sessionId} options={sessions.map((s) => s.id)}
            onChange={(id) => { setSessionId(id); setEpisode(sessions.find((s) => s.id === id)?.episodes?.[0] ?? null); }} />
        )}
        {selectedSession?.episodes?.length > 1 && (
          <Picker label="Episode" ariaLabel="Episode" value={episode} options={selectedSession.episodes} onChange={setEpisode} />
        )}
      </div>

      {!sessions ? (
        <p className="empty">Loading sessions…</p>
      ) : !sessions.length ? (
        <p className="empty">Run A has no recordings yet.</p>
      ) : urlsA && urlsB ? (
        <div className="compare-grid">
          <div className="compare-column">
            <h4>Run A — {runA}</h4>
            <RunSummary summary={summaryA} />
            <PixelHorizonViewer framesSrc={urlsA.frames} predictionsSrc={urlsA.predictions} decisions={decisions} tick={tick} onTickChange={setTick} />
          </div>
          <div className="compare-column">
            <h4>Run B — {runB}</h4>
            <RunSummary summary={summaryB} />
            <PixelHorizonViewer framesSrc={urlsB.frames} predictionsSrc={urlsB.predictions} decisions={decisions} tick={tick} onTickChange={setTick} />
          </div>
        </div>
      ) : null}
    </section>
  );
}
