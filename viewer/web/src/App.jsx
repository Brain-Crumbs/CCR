import { useEffect, useMemo, useState } from "react";
import { api, episodeUrls } from "./lib/api.js";
import { buildHash, parseHash } from "./lib/hashRoute.js";
import { Picker } from "./components/Picker.jsx";
import { SessionList } from "./components/SessionList.jsx";
import { Tabs } from "./components/Tabs.jsx";
import { RunSummary } from "./components/RunSummary.jsx";
import { PixelHorizonViewer } from "./components/PixelHorizonViewer.jsx";
import { ActionTimeline } from "./components/ActionTimeline.jsx";
import { EEGPanel } from "./components/EEGPanel.jsx";
import { AttentionPanel } from "./components/AttentionPanel.jsx";
import { DevelopmentPanel } from "./components/DevelopmentPanel.jsx";
import { ExperimentDetail } from "./components/ExperimentDetail.jsx";
import { PairedComparisonPanel } from "./components/PairedComparisonPanel.jsx";
import { ChampionRegistryPanel } from "./components/ChampionRegistryPanel.jsx";
import { FactoryRunsPanel } from "./components/FactoryRunsPanel.jsx";
import { CompareView } from "./components/CompareView.jsx";
import { BuildPanel } from "./components/BuildPanel.jsx";

const TABS = [
  ["episode", "Episode"],
  ["development", "Development"],
  ["experiment", "Experiment"],
  ["factory", "Factory"],
  ["compare", "Compare"],
  ["build", "Build"],
];

/**
 * The clinic: a read-only presentation layer over one selected Model Factory
 * run (epic #212's organism/run catalog) -- pixel-horizon predictions,
 * predicted/assumed-vs-actuated actions, EEG/attention, the developmental
 * ladder, and the factory's own contracts/lineage/champion evidence, all
 * sharing one tick cursor.
 */
export function App() {
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  const [organism, setOrganism] = useState(null);
  const [run, setRun] = useState(null);
  const [sessions, setSessions] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [episode, setEpisode] = useState(null);
  const [summary, setSummary] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [registry, setRegistry] = useState(null);
  const [factoryRuns, setFactoryRuns] = useState(null);
  const [sessionDetail, setSessionDetail] = useState(null);
  const [tick, setTick] = useState(null);
  const [tab, setTab] = useState("episode");

  useEffect(() => { api.catalog().then(setCatalog, (e) => setError(e.message)); }, []);

  // Pick an organism/run from the URL hash, falling back to the first entry.
  useEffect(() => {
    if (!catalog?.runs?.length) return;
    const parsed = parseHash(location.hash);
    const initial = catalog.runs.find((r) => r.organism === parsed.organism && r.run === parsed.run) || catalog.runs[0];
    setOrganism(initial.organism);
    setRun(initial.run);
  }, [catalog]);

  // A selected organism/run pulls its sessions, header summary, and Model
  // Factory manifests together; the champion registry is keyed by organism alone.
  useEffect(() => {
    if (!organism || !run) return;
    let cancelled = false;
    setSessions(null);
    setSummary(null);
    setArtifacts(null);
    Promise.all([
      api.sessions(organism, run),
      api.runSummary(organism, run).catch(() => null),
      api.experimentArtifacts(organism, run).catch(() => null),
    ]).then(([sessionList, summaryData, artifactsData]) => {
      if (cancelled) return;
      setSessions(sessionList);
      setSummary(summaryData);
      setArtifacts(artifactsData);
      const parsed = parseHash(location.hash);
      const selected = sessionList.find((s) => s.id === parsed.session && s.episodes.includes(parsed.episode)) || sessionList[0];
      setSessionId(selected?.id ?? null);
      setEpisode(selected ? (parsed.episode && selected.episodes.includes(parsed.episode) ? parsed.episode : selected.episodes[0]) : null);
    }, (e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [organism, run]);

  useEffect(() => {
    if (!organism) return;
    let cancelled = false;
    api.registry(organism).then((r) => !cancelled && setRegistry(r), () => !cancelled && setRegistry(null));
    api.factoryRuns(organism).then((r) => !cancelled && setFactoryRuns(r), () => !cancelled && setFactoryRuns(null));
    return () => { cancelled = true; };
  }, [organism]);

  useEffect(() => {
    if (!sessionId) { setSessionDetail(null); return undefined; }
    let cancelled = false;
    api.session(sessionId, organism, run).then((detail) => !cancelled && setSessionDetail(detail));
    return () => { cancelled = true; };
  }, [sessionId, organism, run]);

  useEffect(() => { setTick(null); }, [sessionId, episode]);

  useEffect(() => {
    if (organism && run && sessionId && episode) location.hash = buildHash({ organism, run, session: sessionId, episode });
  }, [organism, run, sessionId, episode]);

  const records = sessionDetail?.streams?.[episode] || [];
  const decisions = sessionDetail?.decisions?.[episode] || [];
  const urls = useMemo(
    () => (sessionId && episode ? episodeUrls(sessionId, episode, { organism, run, experiment: run }) : null),
    [sessionId, episode, organism, run],
  );

  if (error) return <main><h1>CCR Clinic</h1><p className="empty">{error}</p></main>;
  if (!catalog) return <main><h1>CCR Clinic</h1><p className="empty">Loading runs…</p></main>;
  if (!catalog.runs.length) return <main><h1>CCR Clinic</h1><p className="empty">No experiment runs found.</p></main>;
  if (!organism || !run) return null;

  const runsForOrganism = catalog.runs.filter((r) => r.organism === organism).map((r) => r.run);

  return (
    <main>
      <h1>CCR Clinic</h1>
      <p className="lede">Model Factory presentation layer — pixels, actions, and experiment evidence, tick by tick.</p>
      <div className="pickers">
        <Picker label="Organism" ariaLabel="Organism" value={organism} options={catalog.organisms}
          onChange={(next) => { setOrganism(next); setRun(catalog.runs.find((r) => r.organism === next)?.run ?? null); }} />
        <Picker label="Run" ariaLabel="Run" value={run} options={runsForOrganism} onChange={setRun} />
      </div>
      <p className="run-name">{organism} / {run}{sessionId ? ` / ${sessionId}` : ""}</p>
      <RunSummary summary={summary} />

      {!sessions ? (
        <p className="empty">Loading sessions…</p>
      ) : sessions.length ? (
        <div className="session-row">
          <SessionList sessions={sessions} selectedId={sessionId} onSelect={(id) => {
            const next = sessions.find((s) => s.id === id);
            setSessionId(id);
            setEpisode(next?.episodes[0] ?? null);
          }} />
          {sessionDetail?.session?.episodes?.length > 1 && (
            <Picker label="Episode" ariaLabel="Episode" value={episode} options={sessionDetail.session.episodes} onChange={setEpisode} />
          )}
        </div>
      ) : (
        <p className="empty">No recordings are associated with this run yet.</p>
      )}

      {/* Tabs beyond Episode/Development don't depend on this run having any
          recorded sessions -- Factory/Compare/Build all operate at the
          organism level (or launch a brand-new run), so they stay reachable
          even for a run with none yet. */}
      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === "episode" && sessions?.length > 0 && urls && (
        <>
          <PixelHorizonViewer framesSrc={urls.frames} predictionsSrc={urls.predictions} decisions={decisions} tick={tick} onTickChange={setTick} />
          <ActionTimeline decisions={decisions} tick={tick} onTickChange={setTick} />
          <EEGPanel records={records} decisions={decisions} tick={tick} onTickChange={setTick} />
          <AttentionPanel records={records} decisions={decisions} tick={tick} onTickChange={setTick} />
        </>
      )}
      {tab === "development" && sessions?.length > 0 && <DevelopmentPanel session={sessionDetail?.session || {}} />}
      {tab === "experiment" && (
        <>
          <ExperimentDetail artifacts={artifacts} summary={summary} />
          <PairedComparisonPanel comparison={artifacts?.metrics?.comparison} />
        </>
      )}
      {tab === "factory" && (
        <>
          <FactoryRunsPanel factoryRuns={factoryRuns} />
          <ChampionRegistryPanel registry={registry} />
        </>
      )}
      {tab === "compare" && (
        <CompareView catalog={catalog} organism={organism} defaultRunA={run} />
      )}
      {tab === "build" && (
        <BuildPanel catalog={catalog} organism={organism} />
      )}
    </main>
  );
}
