import { useEffect, useMemo, useRef, useState } from "react";
import { usePixelHorizonData } from "../hooks/usePixelHorizonData.js";
import { mseSeries, targetFrame, worstFrameIndex } from "../lib/pixelSeries.js";
import { identityText, modelSourceLabel, statusText } from "../lib/pixelIdentity.js";
import { actionTimeline } from "../lib/actions.js";
import { HorizonPanel } from "./HorizonPanel.jsx";
import { SeenFramePanel } from "./SeenFramePanel.jsx";
import { MseChart } from "./MseChart.jsx";

const EVENT_BUTTONS = [
  ["surprise", "Previous surprise"],
  ["entity_entered", "Next entity entry"],
  ["entity_left", "Next entity exit"],
  ["blocked_forward", "Next blocked transition"],
  ["false_positive", "Worst false positive"],
  ["static", "Worst static mismatch"],
];

const DEFAULT_HORIZONS = [1, 10, 100];

/**
 * One strip per tick: the SEEN frame at t paired with the action taken
 * then, followed by one panel per prediction horizon h with the PREDICTED
 * frame at t+h, the ACTUAL frame at t+h, and an |error| heatmap, plus
 * MSE/PSNR and an MSE-over-time chart across the whole episode. Read
 * left-to-right it answers, for every tick, "what it saw and did, what it
 * predicted would happen next, and what actually was shown" -- and the
 * chart answers how that holds up across the episode, not just one tick.
 *
 * `tick`/`onTickChange` let a parent share one time cursor across this,
 * the EEG/attention panels, and the action timeline. `decisions` pairs the
 * leading seen-frame cell with the action taken at that same tick, so the
 * strip reads as one sequence: pixels + action at t, then each model's
 * prediction at t+h.
 */
export function PixelHorizonViewer({ framesSrc, predictionsSrc, decisions = [], horizons: horizonsProp, scale = 6, tick = null, onTickChange }) {
  const data = usePixelHorizonData(framesSrc, predictionsSrc);
  const [t, setT] = useState(0);
  const [source, setSource] = useState("copy-last");
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  const horizons = useMemo(
    () => (data.pred?.horizons?.length ? data.pred.horizons : (horizonsProp || DEFAULT_HORIZONS)),
    [data.pred, horizonsProp],
  );
  const ctx = useMemo(() => ({ frames: data.frames || [], pred: data.pred, meanFrame: data.meanFrame, shape: data.shape }), [data]);
  const maxT = useMemo(() => {
    if (!data.frames) return 0;
    return Math.max(0, data.frames.length - 1 - Math.max(...horizons));
  }, [data.frames, horizons]);
  const actionByTick = useMemo(() => new Map(actionTimeline(decisions).map((entry) => [entry.tick, entry])), [decisions]);

  // publishTick is the single funnel that tells the parent this component's
  // tick moved. It is called only from actions this component itself
  // initiates (seek, playback, the initial event-driven jump) -- never as a
  // blind reaction to `t` changing -- so following an externally driven
  // `tick` prop (below) can never loop back and re-publish what it just
  // received, even when clamping makes the two disagree.
  function publishTick(index) { onTickChange?.(data.frames[index]?.tick ?? index); }

  // A freshly loaded episode resets local playback state and opens on its
  // first notable event (entity entry/exit, a block, a scene change) rather
  // than always frame zero.
  useEffect(() => {
    if (data.status !== "ready") return;
    // A running interval from the previous episode keeps ticking on its old
    // closure (stale data/maxT) unless explicitly stopped here -- setPlaying
    // alone only changes the button's label, not the timer.
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setSource(data.pred ? "model" : "copy-last");
    const eventful = data.pred?.events?.findIndex((event) => event.entity_entered || event.entity_left || event.blocked_forward || event.semantic_scene_changed);
    const initial = eventful > 0 ? Math.min(eventful, maxT) : 0;
    setT(initial);
    setPlaying(false);
    publishTick(initial);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.frames]);

  // Follow an externally driven tick (e.g. a click in the EEG/attention/action panels).
  useEffect(() => {
    if (tick == null || !data.frames?.length) return;
    const closest = data.frames.reduce((best, frame, index) => (
      !best || Math.abs(Number(frame.tick) - tick) < best.distance ? { index, distance: Math.abs(Number(frame.tick) - tick) } : best
    ), null);
    if (closest) setT(Math.min(maxT, closest.index));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  function seek(next) {
    const clamped = Math.max(0, Math.min(maxT, Math.round(next)));
    setT(clamped);
    publishTick(clamped);
  }

  function togglePlay() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; setPlaying(false); return; }
    setPlaying(true);
    timerRef.current = setInterval(() => setT((prev) => {
      const next = prev >= maxT ? 0 : prev + 1;
      publishTick(next);
      return next;
    }), 100);
  }

  function jumpEvent(kind) {
    if (!data.frames?.length) return;
    const events = data.pred?.events || [];
    let index = -1;
    if (kind === "surprise" || kind === "false_positive" || kind === "static") {
      index = worstFrameIndex(kind, mseSeries("model", horizons[0], ctx), events);
    } else {
      index = events.findIndex((event, i) => i > t && event[kind]);
      if (index < 0) index = events.findIndex((event) => event[kind]);
    }
    if (index >= 0) seek(index);
  }

  if (data.status === "loading") return <div className="status">loading frames…</div>;
  if (data.status === "error") return <div className="status">failed to load frames: {data.error}</div>;

  const hasModel = !!data.pred;
  const sourceOptions = [
    ...(hasModel ? [["model", modelSourceLabel(data.pred)]] : []),
    ["copy-last", "copy-last baseline"],
    ["mean-frame", "mean-frame baseline"],
  ];
  // Model targets are pooled to the export's reconstruction shape (e.g.
  // 16x16 for a native 33x33 episode); a mismatched shape lays the smaller
  // byte buffer across a larger canvas and corrupts the seen frame.
  const seen = targetFrame(source, t, ctx);

  return (
    <div className="pixel-horizon-viewer">
      <div className="controls">
        <button aria-label="play/pause" onClick={togglePlay}>{playing ? "❚❚" : "▶"}</button>
        <label>t = <span>{t} / {maxT}</span></label>
        <input type="range" aria-label="start frame" min={0} max={maxT} step={1} value={t} onChange={(e) => seek(Number(e.target.value))} />
        <label>prediction:
          <select aria-label="prediction source" value={source} onChange={(e) => setSource(e.target.value)}>
            {sourceOptions.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
          </select>
        </label>
      </div>
      <div className="status">{statusText({ nFrames: data.frames.length, shape: data.shape, pred: data.pred })}</div>
      <div className="identity">{identityText(data.pred)}</div>
      <div className="event-nav">
        {EVENT_BUTTONS.map(([kind, label]) => <button key={kind} onClick={() => jumpEvent(kind)}>{label}</button>)}
      </div>
      <div className="horizons">
        <SeenFramePanel t={t} tick={data.frames[t]?.tick ?? t}
          bytes={seen?.bytes ?? null} shape={seen?.shape ?? data.shape}
          action={actionByTick.get(data.frames[t]?.tick ?? t) ?? null} />
        {horizons.map((h) => <HorizonPanel key={h} h={h} t={t} source={source} ctx={ctx} />)}
      </div>
      <MseChart horizons={horizons} source={source} ctx={ctx} tCurrent={t} onSeek={seek} />
    </div>
  );
}
