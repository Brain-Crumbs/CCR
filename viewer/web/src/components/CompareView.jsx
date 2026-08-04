import { useEffect, useMemo, useState } from "react";
import { api, episodeUrls } from "../lib/api.js";
import { Picker } from "./Picker.jsx";
import { SessionList } from "./SessionList.jsx";
import { RunSummary } from "./RunSummary.jsx";
import { PixelHorizonViewer } from "./PixelHorizonViewer.jsx";

//: The reference picker's "none selected" sentinel -- Picker renders each
//: option's value as its own display text, so this doubles as the label.
const NO_REFERENCE = "(none)";

/**
 * Two trained checkpoints' predictions, side by side, for the same
 * session/episode.
 *
 * Two runs trained against the same frozen corpus (the common
 * clone/fine_tune lineage case under one organism) write their prediction
 * exports into the *same* session directory (prediction_export.py), so
 * `run` only needs to resolve that session set -- either run works -- while
 * `experiment` selects *whose* prediction file PixelHorizonViewer reads.
 * v1 scope: comparison is exact when both runs share a corpus; if Run B has
 * no export for the chosen session, that side just falls back to baselines
 * exactly like a missing prediction file does today (usePixelHorizonData
 * already tolerates the 404 -- no extra handling needed here).
 */
export function CompareView({ catalog, organism, defaultRunA }) {
  const runsForOrganism = useMemo(
    () => (catalog?.runs || []).filter((r) => r.organism === organism).map((r) => r.run),
    [catalog, organism],
  );
  const [runA, setRunA] = useState(defaultRunA ?? null);
  const [runB, setRunB] = useState(null);
  const [referenceRun, setReferenceRun] = useState(NO_REFERENCE);
  const [summaryA, setSummaryA] = useState(null);
  const [summaryB, setSummaryB] = useState(null);
  const [sessions, setSessions] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [episode, setEpisode] = useState(null);
  const [tick, setTick] = useState(null);

  // Run A defaults to the currently selected run; if the organism changes
  // under this view (rather than remounting it) and the current pick no
  // longer belongs to it, fall back to the first available run.
  useEffect(() => {
    if (!runsForOrganism.length) return;
    setRunA((current) => (current && runsForOrganism.includes(current) ? current : (defaultRunA && runsForOrganism.includes(defaultRunA) ? defaultRunA : runsForOrganism[0])));
  }, [runsForOrganism, defaultRunA]);

  // Run B defaults to a different run than A, so the picker opens on two
  // distinct checkpoints whenever the organism has more than one.
  useEffect(() => {
    if (!runsForOrganism.length) return;
    setRunB((current) => (current && runsForOrganism.includes(current) ? current : runsForOrganism.find((r) => r !== runA) || runsForOrganism[0]));
  }, [runsForOrganism, runA]);

  // The reference baseline defaults to none selected; if the organism
  // changes and the current pick no longer belongs to it, fall back to
  // none rather than silently pointing at an unrelated run.
  useEffect(() => {
    setReferenceRun((current) => (current === NO_REFERENCE || runsForOrganism.includes(current) ? current : NO_REFERENCE));
  }, [runsForOrganism]);

  // Run A's session list is the shared session/episode picker for both sides.
  useEffect(() => {
    if (!organism || !runA) return;
    let cancelled = false;
    setSessions(null);
    api.sessions(organism, runA).then((list) => {
      if (cancelled) return;
      setSessions(list);
      setSessionId((current) => (current && list.some((s) => s.id === current) ? current : list[0]?.id ?? null));
    }, () => !cancelled && setSessions([]));
    return () => { cancelled = true; };
  }, [organism, runA]);

  useEffect(() => {
    const session = sessions?.find((s) => s.id === sessionId);
    setEpisode((current) => (session?.episodes?.includes(current) ? current : session?.episodes?.[0] ?? null));
  }, [sessions, sessionId]);

  useEffect(() => { setTick(null); }, [sessionId, episode]);

  useEffect(() => {
    if (!organism || !runA) return;
    let cancelled = false;
    api.runSummary(organism, runA).then((s) => !cancelled && setSummaryA(s), () => !cancelled && setSummaryA(null));
    return () => { cancelled = true; };
  }, [organism, runA]);

  useEffect(() => {
    if (!organism || !runB) return;
    let cancelled = false;
    api.runSummary(organism, runB).then((s) => !cancelled && setSummaryB(s), () => !cancelled && setSummaryB(null));
    return () => { cancelled = true; };
  }, [organism, runB]);

  const selectedSession = sessions?.find((s) => s.id === sessionId);
  const urlsA = useMemo(
    () => (sessionId && episode && runA ? episodeUrls(sessionId, episode, { organism, run: runA, experiment: runA }) : null),
    [sessionId, episode, organism, runA],
  );
  const urlsB = useMemo(
    () => (sessionId && episode && runA && runB ? episodeUrls(sessionId, episode, { organism, run: runA, experiment: runB }) : null),
    [sessionId, episode, organism, runA, runB],
  );
  // A configurable comparison baseline (clinic redesign): any run in the
  // catalog, not just the client-computed copy-last/mean-frame ones. Its
  // session resolves through runA exactly like runB's does above (either
  // run works when both share a corpus); `experiment` is what actually
  // selects whose predictions get read.
  const hasReference = referenceRun !== NO_REFERENCE;
  const referenceUrls = useMemo(
    () => (hasReference && sessionId && episode && runA
      ? episodeUrls(sessionId, episode, { organism, run: runA, experiment: referenceRun })
      : null),
    [hasReference, sessionId, episode, organism, runA, referenceRun],
  );

  if (!runsForOrganism.length) return <p className="empty">No runs to compare for this organism.</p>;

  return (
    <section className="diagnostic compare" aria-labelledby="compare-title">
      <h3 id="compare-title">Compare</h3>
      <p>
        Two runs&apos; predictions for the same session and episode, side by side. Comparison is exact when both
        runs share a corpus (the common case for clone/fine_tune lineages under one organism); otherwise the side
        without a matching export falls back to baselines. Optionally pick a third run as a reference baseline --
        available in each side&apos;s own prediction-source dropdown alongside copy-last/mean-frame.
      </p>
      <div className="pickers">
        <Picker label="Run A" ariaLabel="Run A" value={runA} options={runsForOrganism} onChange={setRunA} />
        <Picker label="Run B" ariaLabel="Run B" value={runB} options={runsForOrganism} onChange={setRunB} />
        <Picker
          label="Reference (baseline)" ariaLabel="Reference baseline run" value={referenceRun}
          options={[NO_REFERENCE, ...runsForOrganism]} onChange={setReferenceRun}
        />
      </div>

      {!sessions ? (
        <p className="empty">Loading sessions…</p>
      ) : !sessions.length ? (
        <p className="empty">No recordings are associated with Run A yet.</p>
      ) : (
        <>
          <div className="session-row">
            <SessionList sessions={sessions} selectedId={sessionId} onSelect={setSessionId} />
            {selectedSession?.episodes?.length > 1 && (
              <Picker label="Episode" ariaLabel="Episode" value={episode} options={selectedSession.episodes} onChange={setEpisode} />
            )}
          </div>

          <h4>Run A — {runA}</h4>
          <RunSummary summary={summaryA} />
          {urlsA && (
            <PixelHorizonViewer
              framesSrc={urlsA.frames} predictionsSrc={urlsA.predictions}
              referencePredictionsSrc={referenceUrls?.predictions} referenceLabel={hasReference ? referenceRun : null}
              tick={tick} onTickChange={setTick}
            />
          )}

          <h4>Run B — {runB}</h4>
          <RunSummary summary={summaryB} />
          {urlsB && (
            <PixelHorizonViewer
              framesSrc={urlsB.frames} predictionsSrc={urlsB.predictions}
              referencePredictionsSrc={referenceUrls?.predictions} referenceLabel={hasReference ? referenceRun : null}
              tick={tick} onTickChange={setTick}
            />
          )}
        </>
      )}
    </section>
  );
}
