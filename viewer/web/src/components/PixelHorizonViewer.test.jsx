import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { PixelHorizonViewer } from "./PixelHorizonViewer.jsx";

function b64(bytes) { return btoa(String.fromCharCode(...bytes)); }

function framesPayload(n) {
  return {
    shape: [1, 1, 3], dtype: "uint8", n_frames: n,
    frames: Array.from({ length: n }, (_, i) => ({ i, t: i, tick: i, seq: i, hash: `h${i}`, data: b64([i, i, i]) })),
  };
}

function predictionsPayload() {
  return {
    format: "pixel-predictions-v2", source: "export", prediction_mode: "rollout", horizons: [1],
    prediction_shape: [1, 1, 3], evaluation_source: "s/nursery-x-holdout-1/episode_0",
    experiment: { experiment_id: "exp-1" }, model: { checkpoint_sha256: "abc123456789", uses_actions: true, uses_workspace: true },
    predictions: { 1: { frames: [b64([1, 1, 1]), b64([2, 2, 2])] } },
    targets: [b64([0, 0, 0]), b64([1, 1, 1]), b64([2, 2, 2])],
    events: [{}, { entity_entered: true }, {}],
  };
}

describe("PixelHorizonViewer", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((url) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(String(url).includes("predictions") ? predictionsPayload() : framesPayload(3)),
    })));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("loads frames+predictions and renders a panel per horizon with the model source selected", async () => {
    render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" />);
    // The post-load effect (default source, initial event-driven tick) settles
    // asynchronously after the horizon panels first paint, so every
    // assertion that depends on it belongs inside the same waitFor.
    await waitFor(() => {
      expect(screen.getByText(/horizon t\+1\b/)).toBeInTheDocument();
      expect(screen.getByLabelText("prediction source").value).toBe("model");
      // The episode has an entity_entered event at frame 1; the viewer opens there.
      expect(screen.getByLabelText("start frame").value).toBe("1");
    });
    expect(screen.getByText(/Experiment exp-1/)).toBeInTheDocument();
  });

  it("falls back to copy-last with no predictions endpoint and reports it in the status line", async () => {
    render(<PixelHorizonViewer framesSrc="/frames" />);
    await waitFor(() => {
      expect(screen.getByText(/horizon t\+1\b/)).toBeInTheDocument();
      expect(screen.getByLabelText("prediction source").value).toBe("copy-last");
    });
    expect(screen.getByText(/no model predictions recorded/)).toBeInTheDocument();
  });

  it("reports its tick back to the parent as the scrubber moves", async () => {
    const onTickChange = vi.fn();
    render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" onTickChange={onTickChange} />);
    await waitFor(() => expect(screen.getByLabelText("start frame").value).toBe("1"));
    onTickChange.mockClear();
    fireEvent.change(screen.getByLabelText("start frame"), { target: { value: "0" } });
    expect(onTickChange).toHaveBeenCalledWith(0);
  });

  it("follows an externally driven tick prop", async () => {
    const { rerender } = render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" tick={null} />);
    await waitFor(() => expect(screen.getByText(/horizon t\+1\b/)).toBeInTheDocument());
    rerender(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" tick={0} />);
    await waitFor(() => expect(screen.getByLabelText("start frame").value).toBe("0"));
  });

  it("pairs the leading seen-frame cell with the action taken at the current tick, and updates it as the tick moves", async () => {
    const decisions = [
      { tick_index: 0, motor_decision: { voluntary: "move_up", reflex: null, caregiver_override: null, actuated: "move_up" } },
      { tick_index: 1, motor_decision: { voluntary: "move_up", reflex: { name: "wall_avoidance" }, caregiver_override: null, actuated: "move_left" } },
    ];
    render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" decisions={decisions} />);
    // The fixture's entity_entered event opens the viewer at t=1, whose action diverged.
    await waitFor(() => expect(screen.getByRole("heading", { name: /^t = 1/ })).toBeInTheDocument());
    expect(screen.getByText("move left")).toBeInTheDocument();
    expect(screen.getByText(/assumed/)).toHaveTextContent("assumed move up — wall avoidance");

    fireEvent.change(screen.getByLabelText("start frame"), { target: { value: "0" } });
    await waitFor(() => expect(screen.getByRole("heading", { name: /^t = 0/ })).toBeInTheDocument());
    expect(screen.getByText("move up")).toBeInTheDocument();
    expect(screen.queryByText(/assumed/)).not.toBeInTheDocument();
  });
});
