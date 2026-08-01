import { useEffect, useRef } from "react";
import { rampColor } from "../lib/pixelMath.js";
import { useDarkMode } from "../hooks/useDarkMode.js";

/** Draws an |error| heatmap between two same-shape RGB frames: one hue
 * (sequential blue), magnitude = |a-b| averaged across channels. */
export function DiffCanvas({ a, b, shape, scale = 6, label }) {
  const ref = useRef(null);
  const dark = useDarkMode();
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !shape) return;
    const [h, w] = shape;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w * scale}px`;
    canvas.style.height = `${h * scale}px`;
    const ctx = canvas.getContext("2d");
    if (!a || !b) { ctx.clearRect(0, 0, w, h); return; }
    const img = ctx.createImageData(w, h);
    for (let px = 0; px < w * h; px++) {
      const p = px * 3;
      const err = (Math.abs(a[p] - b[p]) + Math.abs(a[p + 1] - b[p + 1]) + Math.abs(a[p + 2] - b[p + 2])) / (3 * 255);
      const [r, g, bch] = rampColor(err, dark);
      const q = px * 4;
      img.data[q] = r;
      img.data[q + 1] = g;
      img.data[q + 2] = bch;
      img.data[q + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [a, b, shape, scale, dark]);
  return <canvas ref={ref} className="px" role="img" aria-label={label} />;
}
