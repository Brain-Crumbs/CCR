import { useMemo } from "react";
import { mse, psnrText } from "../lib/pixelMath.js";
import { predictedFrame, targetFrame } from "../lib/pixelSeries.js";
import { formatExponential } from "../lib/format.js";
import { PixelCanvas } from "./PixelCanvas.jsx";
import { DiffCanvas } from "./DiffCanvas.jsx";

/** One horizon's predicted→actual strip, plus its error heatmaps and
 * MSE/PSNR readouts against both the selected source and the copy-last
 * baseline -- the four-up comparison the epic asks the clinic to make
 * legible at every tick. The frame the model saw at t is shown once, in the
 * leading SeenFramePanel paired with the action taken then, rather than
 * repeated in every horizon's strip. */
export function HorizonPanel({ h, t, source, ctx }) {
  const { pred, shape: episodeShape } = ctx;
  const { target, predicted, copy, shape, sourceLabel } = useMemo(() => {
    const predictedFrameForSource = predictedFrame(source, t, h, ctx);
    const targetFrameForSource = targetFrame(source, t + h, ctx);
    return {
      target: targetFrameForSource,
      predicted: predictedFrameForSource,
      copy: predictedFrame("copy-last", t, h, ctx),
      shape: predictedFrameForSource ? predictedFrameForSource.shape : episodeShape,
      sourceLabel: source === "model" ? "model" : source,
    };
  }, [ctx, source, t, h, episodeShape]);

  const comparable = predicted && target && predicted.bytes.length === target.bytes.length;
  const copyComparable = copy && target && copy.bytes.length === target.bytes.length;
  const m = comparable ? mse(predicted.bytes, target.bytes) : NaN;
  const cm = copyComparable ? mse(copy.bytes, target.bytes) : NaN;

  return (
    <div className="panel" data-h={h}>
      <h3>horizon t+{h}</h3>
      <div className="strip">
        <figure className="cell"><PixelCanvas bytes={predicted?.bytes ?? null} shape={shape} label={`${sourceLabel} prediction`} /><figcaption>{sourceLabel} prediction t+{h}</figcaption></figure>
        <figure className="cell"><PixelCanvas bytes={copy?.bytes ?? null} shape={copy?.shape ?? episodeShape} label="copy-last prediction" /><figcaption>copy-last t</figcaption></figure>
        <figure className="cell"><PixelCanvas bytes={target?.bytes ?? null} shape={target?.shape ?? episodeShape} label="actual frame" /><figcaption>actual t+{h}</figcaption></figure>
        <figure className="cell"><DiffCanvas a={comparable ? predicted.bytes : null} b={comparable ? target.bytes : null} shape={shape} label="prediction error" /><figcaption>{sourceLabel} error</figcaption></figure>
        <figure className="cell"><DiffCanvas a={copyComparable ? copy.bytes : null} b={copyComparable ? target.bytes : null} shape={shape} label="copy-last error" /><figcaption>copy-last error</figcaption></figure>
      </div>
      <div className="metrics">
        {source === "model" && comparable && !Number.isNaN(cm)
          ? <>model MSE <b>{formatExponential(m)}</b> · copy-last <b>{formatExponential(cm)}</b> · <b>{m < cm ? "model beats copy-last" : "copy-last wins"}</b></>
          : comparable
            ? <>{sourceLabel} MSE <b>{formatExponential(m)}</b> · PSNR <b>{psnrText(m)}</b> dB</>
            : <>no comparable frames at t={t}</>}
      </div>
    </div>
  );
}
