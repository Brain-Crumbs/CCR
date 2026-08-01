import "@testing-library/jest-dom/vitest";

// jsdom has no real canvas 2D backend; component tests only assert on
// behavior (props/attributes/handlers), never pixel output, so a minimal
// stub is enough to let PixelCanvas/DiffCanvas mount without a native
// `canvas` dependency.
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = () => ({
    clearRect: () => {},
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {},
  });
}
