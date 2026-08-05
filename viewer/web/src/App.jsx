import { useEffect, useMemo, useRef, useState } from "react";
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
import { EvolvePanel } from "./components/EvolvePanel.jsx";
import { BreedPanel } from "./components/BreedPanel.jsx";

const TABS = [
  ["episode", "Episode"],
  ["development", "Development"],
  ["experiment", "Experiment"],
  ["factory", "Factory"],
  ["compare", "Compare"],
  ["build", "Build"],
  ["evolve", "Evolve"],
  ["breed", "Breed"],
];

/**
 * The clinic: the Model Factory control and presentation layer. It launches
 * corpora, trials, searches, and breeding jobs, while one selected run drives
 * the pixel-horizon predictions,
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
  const bootstrapped = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const next = await api.catalog();
        if (!cancelled) setCatalog((current) => (current?.runs?.length ? current : next));
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }
    poll();
    const interval = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  // Pick an organism/run from the URL hash, falling back to the first entry.
  // An organism may be known from its corpus before its first run exists;
  // that bootstrap state opens directly on Build instead of dead-ending.
  useEffect(() => {
    if (!catalog) return;
    const parsed = parseHash(location.hash);
    const initial = catalog.runs.find((r) => r.organism === parsed.organism && r.run === parsed.run)
      || catalog.runs[0] || null;
    const initialOrganism = initial?.organism
      || (catalog.organisms.includes(parsed.organism) ? parsed.organism : catalog.organisms[0])
      || null;
    if (!bootstrapped.current) {
      bootstrapped.current = true;
      setOrganism(initialOrganism);
      setRun(initial?.run ?? null);
      if (!initial) setTab("build");
      return;
    }

    // Catalog refreshes must not undo an explicit selection, notably just
    // after the user creates an organism that has no runs yet.
    setOrganism((current) => (
      current && catalog.organisms.includes(current) ? current : initialOrganism
    ));
    setRun((current) => (
      current && catalog.runs.some((item) => item.run === current) ? current : null
    ));
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
    async function poll() {
      api.registry(organism).then((r) => !cancelled && setRegistry(r), () => !cancelled && setRegistry(null));
      api.factoryRuns(organism).then((r) => !cancelled && setFactoryRuns(r), () => !cancelled && setFactoryRuns(null));
    }
    poll();
    const interval = setInterval(poll, 4000);
    return () => { cancelled = true; clearInterval(interval); };
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

  function handleOrganismCreated(next) {
    setCatalog((current) => ({
      ...current,
      organisms: [...new Set([...(current?.organisms || []), next])].sort(),
    }));
    setOrganism(next);
    setRun(null);
    setSessions(null);
    setSessionId(null);
    setEpisode(null);
    setTab("build");
  }

  if (error) return <main><h1>CCR Clinic</h1><p className="empty">{error}</p></main>;
  if (!catalog) return <main><h1>CCR Clinic</h1><p className="empty">Loading runs…</p></main>;
  const runsForOrganism = organism
    ? catalog.runs.filter((r) => r.organism === organism).map((r) => r.run)
    : [];

  return (
    <main>
      <h1>CCR Clinic</h1>
      <p className="lede">Model Factory control room — build, evolve, breed, and inspect experiment evidence.</p>
      {catalog.organisms.length > 0 && (
        <div className="pickers">
          <Picker label="Organism" ariaLabel="Organism" value={organism} options={catalog.organisms}
            onChange={(next) => { setOrganism(next); setRun(catalog.runs.find((r) => r.organism === next)?.run ?? null); }} />
          {runsForOrganism.length > 0 && (
            <Picker label="Run" ariaLabel="Run" value={run} options={runsForOrganism} onChange={setRun} />
          )}
        </div>
      )}
      {run ? (
        <>
          <p className="run-name">{organism} / {run}{sessionId ? ` / ${sessionId}` : ""}</p>
          <RunSummary summary={summary} />
        </>
      ) : (
        <p className="empty empty--onboarding">
          No experiment runs yet. Start in Build to launch the first model; jobs and live run state remain
          available here while training is in progress.
        </p>
      )}

      {run && (!sessions ? (
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
      ))}

      {/* Tabs beyond Episode/Development don't depend on this run having any
          recorded sessions -- Factory/Compare/Build all operate at the
          organism level (or launch a brand-new run), so they stay reachable
          even for a run with none yet. */}
      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === "episode" && !run && <p className="no-data">Launch or select a run to inspect recorded episodes.</p>}
      {tab === "episode" && sessions?.length > 0 && urls && (
        <>
          <PixelHorizonViewer framesSrc={urls.frames} predictionsSrc={urls.predictions} decisions={decisions} tick={tick} onTickChange={setTick} />
          <ActionTimeline decisions={decisions} tick={tick} onTickChange={setTick} />
          <EEGPanel records={records} decisions={decisions} tick={tick} onTickChange={setTick} />
          <AttentionPanel records={records} decisions={decisions} tick={tick} onTickChange={setTick} />
        </>
      )}
      {tab === "development" && !run && <p className="no-data">Launch or select a run to inspect its developmental ladder.</p>}
      {tab === "development" && sessions?.length > 0 && <DevelopmentPanel session={sessionDetail?.session || {}} />}
      {tab === "experiment" && (
        run ? <>
          <ExperimentDetail artifacts={artifacts} summary={summary} />
          <PairedComparisonPanel comparison={artifacts?.metrics?.comparison} />
        </> : <p className="no-data">Launch or select a run to inspect experiment artifacts.</p>
      )}
      {tab === "factory" && (
        <>
          <FactoryRunsPanel factoryRuns={factoryRuns} />
          <ChampionRegistryPanel registry={registry} />
        </>
      )}
      {tab === "compare" && (
        run ? <CompareView catalog={catalog} organism={organism} defaultRunA={run} />
          : <p className="no-data">At least one completed run is needed for comparison.</p>
      )}
      {tab === "build" && (
        <BuildPanel catalog={catalog} organism={organism} onOrganismCreated={handleOrganismCreated} />
      )}
      {tab === "evolve" && (
        <EvolvePanel catalog={catalog} organism={organism} />
      )}
      {tab === "breed" && (
        <BreedPanel catalog={catalog} organism={organism} />
      )}
    </main>
  );
}
