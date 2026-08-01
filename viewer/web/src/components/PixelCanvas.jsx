import { useEffect, useRef } from "react";

/** Draws one raw HWC uint8 RGB frame at `scale` CSS px per source pixel. */
export function PixelCanvas({ bytes, shape, scale = 6, label }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !shape) return;
    const [h, w] = shape;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w * scale}px`;
    canvas.style.height = `${h * scale}px`;
    const ctx = canvas.getContext("2d");
    if (!bytes) { ctx.clearRect(0, 0, w, h); return; }
    const img = ctx.createImageData(w, h);
    for (let p = 0, q = 0; p < bytes.length; p += 3, q += 4) {
      img.data[q] = bytes[p];
      img.data[q + 1] = bytes[p + 1];
      img.data[q + 2] = bytes[p + 2];
      img.data[q + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [bytes, shape, scale]);
  return <canvas ref={ref} className="px" role="img" aria-label={label} />;
}
