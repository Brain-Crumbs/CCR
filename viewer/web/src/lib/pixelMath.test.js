import { describe, expect, it } from "vitest";
import { b64ToBytes, mse, psnrText, rampColor } from "./pixelMath.js";

describe("mse/psnr", () => {
  it("is zero for identical frames and infinite PSNR", () => {
    const a = new Uint8Array([10, 20, 30]);
    expect(mse(a, a)).toBe(0);
    expect(psnrText(0)).toBe("∞");
  });

  it("grows with pixel distance", () => {
    const a = new Uint8Array([0, 0, 0]);
    const b = new Uint8Array([255, 255, 255]);
    expect(mse(a, b)).toBeCloseTo(1, 5);
    expect(Number(psnrText(mse(a, b)))).toBeLessThan(1);
  });
});

describe("rampColor", () => {
  it("anchors zero error at the surface color", () => {
    expect(rampColor(0, false)).toEqual([252, 252, 251]);
  });

  it("clamps out-of-range input", () => {
    expect(rampColor(-1, false)).toEqual(rampColor(0, false));
    expect(rampColor(2, false)).toEqual(rampColor(1, false));
  });

  it("uses a reversed ramp in dark mode so high error still pops", () => {
    expect(rampColor(1, true)).not.toEqual(rampColor(1, false));
  });
});

describe("b64ToBytes", () => {
  it("round-trips base64 into raw bytes", () => {
    expect(Array.from(b64ToBytes(btoa("\x01\x02\x03")))).toEqual([1, 2, 3]);
  });
});
