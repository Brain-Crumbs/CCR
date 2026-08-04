import { useEffect, useMemo, useState } from "react";
import { computeMean, decodeFrames, decodePredictions } from "../lib/pixelSeries.js";

const IDLE = { status: "loading", frames: null, shape: null, pred: null, meanFrame: null, error: null };

/** Fetch+decode one predictions payload, or null on any failure -- a
 * missing/failing predictions export is never fatal to the episode load,
 * whether it's this viewer's own run or a picked reference baseline. */
async function loadPredictions(src) {
  if (!src) return null;
  try {
    const res = await fetch(src);
    const payload = await res.json();
    return res.ok ? decodePredictions(payload) : null;
  } catch {
    return null;
  }
}

/** Loads and decodes one episode's frames + (optional) predictions, plus
 * (independently) an optional second run's predictions for the same
 * session/episode -- a configurable comparison baseline alongside the
 * client-computed copy-last/mean-frame ones (clinic redesign). Re-fetches
 * whenever a src changes; a stale in-flight request is ignored if the URLs
 * change again before it resolves. */
export function usePixelHorizonData(framesSrc, predictionsSrc, referencePredictionsSrc = null) {
  const [state, setState] = useState(IDLE);
  const [referencePred, setReferencePred] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setState(IDLE);
    if (!framesSrc) { setState({ ...IDLE, status: "error", error: "no frames source configured" }); return undefined; }

    (async () => {
      let framesPayload;
      try {
        const res = await fetch(framesSrc);
        framesPayload = await res.json();
        if (!res.ok || framesPayload.error) throw new Error(framesPayload.error || `failed to load frames (${res.status})`);
      } catch (err) {
        if (!cancelled) setState({ ...IDLE, status: "error", error: err.message });
        return;
      }
      const { shape, frames } = decodeFrames(framesPayload);
      const meanFrame = computeMean(frames);
      const pred = await loadPredictions(predictionsSrc);
      if (!cancelled) setState({ status: "ready", frames, shape, pred, meanFrame, error: null });
    })();

    return () => { cancelled = true; };
  }, [framesSrc, predictionsSrc]);

  // Independent of the episode's own frames/predictions load above: picking
  // or clearing a reference baseline must never re-fetch this viewer's own
  // frames or flash the whole strip back to "loading" -- it only ever
  // adds/removes one extra source option.
  useEffect(() => {
    let cancelled = false;
    setReferencePred(null);
    loadPredictions(referencePredictionsSrc).then((pred) => { if (!cancelled) setReferencePred(pred); });
    return () => { cancelled = true; };
  }, [referencePredictionsSrc]);

  // Merged, but only into a *new* object when state or referencePred
  // actually changed -- callers (PixelHorizonViewer's own ctx) memoize on
  // this return value's identity, which must stay stable across unrelated
  // re-renders exactly like the plain useState value it replaces.
  return useMemo(() => ({ ...state, referencePred }), [state, referencePred]);
}
