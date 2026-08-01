import { useEffect, useState } from "react";
import { computeMean, decodeFrames, decodePredictions } from "../lib/pixelSeries.js";

const IDLE = { status: "loading", frames: null, shape: null, pred: null, meanFrame: null, error: null };

/** Loads and decodes one episode's frames + (optional) predictions.
 * Re-fetches whenever framesSrc/predictionsSrc change; a stale in-flight
 * request is ignored if the URLs change again before it resolves. */
export function usePixelHorizonData(framesSrc, predictionsSrc) {
  const [state, setState] = useState(IDLE);

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

      let pred = null;
      if (predictionsSrc) {
        try {
          const res = await fetch(predictionsSrc);
          const payload = await res.json();
          if (res.ok) pred = decodePredictions(payload);
        } catch { /* predictions are optional */ }
      }
      if (!cancelled) setState({ status: "ready", frames, shape, pred, meanFrame, error: null });
    })();

    return () => { cancelled = true; };
  }, [framesSrc, predictionsSrc]);

  return state;
}
