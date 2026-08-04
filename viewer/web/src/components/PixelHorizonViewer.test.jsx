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

  it("renders the seen frame in the model's pooled reconstruction shape, not the native episode shape", async () => {
    // Native frames are 2x2 (4px); the export's reconstruction/target shape
    // is pooled down to 1x1, as a real joint-cortex export would be.
    vi.stubGlobal("fetch", vi.fn((url) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(String(url).includes("predictions") ? {
        format: "pixel-predictions-v2", horizons: [1], prediction_shape: [1, 1, 3],
        predictions: { 1: { frames: [b64([9, 9, 9]), b64([8, 8, 8])] } },
        targets: [b64([0, 0, 0]), b64([1, 1, 1]), b64([2, 2, 2])],
        events: [],
      } : {
        shape: [2, 2, 3], dtype: "uint8", n_frames: 3,
        frames: Array.from({ length: 3 }, (_, i) => ({ i, t: i, tick: i, seq: i, hash: `h${i}`, data: b64(Array(4).fill([i, i, i]).flat()) })),
      }),
    })));
    render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" />);
    await waitFor(() => expect(screen.getByLabelText("prediction source").value).toBe("model"));
    const seenCanvas = document.querySelector(".seen-panel canvas");
    expect(seenCanvas.width).toBe(1);
    expect(seenCanvas.height).toBe(1);
  });

  it("offers a reference baseline option that reads a second run's own predictions, independent of the model's", async () => {
    function referencePredictionsPayload() {
      return {
        format: "pixel-predictions-v2", source: "export", prediction_mode: "rollout", horizons: [1],
        prediction_shape: [1, 1, 3], evaluation_source: "s/nursery-x-holdout-1/episode_0",
        experiment: { experiment_id: "exp-reference" }, model: { checkpoint_sha256: "reference12345", uses_actions: true, uses_workspace: true },
        // Distinct bytes from predictionsPayload()'s own [1,1,1]/[2,2,2] so a
        // wrong wire-up (e.g. reference silently reading the model's own
        // pred) is visible as the wrong pixel values, not just a missing option.
        predictions: { 1: { frames: [b64([9, 9, 9]), b64([7, 7, 7])] } },
        targets: [b64([0, 0, 0]), b64([1, 1, 1]), b64([2, 2, 2])],
        events: [{}, { entity_entered: true }, {}],
      };
    }
    vi.stubGlobal("fetch", vi.fn((url) => {
      const href = String(url);
      const payload = href.includes("reference-predictions") ? referencePredictionsPayload()
        : href.includes("predictions") ? predictionsPayload()
        : framesPayload(3);
      return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
    }));

    render(
      <PixelHorizonViewer
        framesSrc="/frames" predictionsSrc="/predictions"
        referencePredictionsSrc="/reference-predictions" referenceLabel="champion-run-7"
      />,
    );
    await waitFor(() => expect(screen.getByLabelText("prediction source").value).toBe("model"));

    const options = [...screen.getByLabelText("prediction source").options].map((o) => o.value);
    expect(options).toContain("reference");
    expect(screen.getByText(/reference: champion-run-7/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("prediction source"), { target: { value: "reference" } });
    await waitFor(() => expect(screen.getByText(/reference prediction t\+1/)).toBeInTheDocument());
  });

  it("omits the reference option entirely when no referencePredictionsSrc is given", async () => {
    render(<PixelHorizonViewer framesSrc="/frames" predictionsSrc="/predictions" />);
    await waitFor(() => expect(screen.getByLabelText("prediction source").value).toBe("model"));
    const options = [...screen.getByLabelText("prediction source").options].map((o) => o.value);
    expect(options).not.toContain("reference");
  });

  it("stops the previous episode's playback interval once a new episode loads", async () => {
    const onTickChange = vi.fn();
    const seriesFor = (n) => framesPayload(n);
    vi.stubGlobal("fetch", vi.fn((url) => {
      const isB = String(url).includes("-b");
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(String(url).includes("predictions") ? predictionsPayload() : seriesFor(isB ? 4 : 3)),
      });
    }));
    const { rerender } = render(<PixelHorizonViewer framesSrc="/frames-a" predictionsSrc="/predictions-a" onTickChange={onTickChange} />);
    await waitFor(() => expect(screen.getByLabelText("start frame").value).toBe("1"));

    fireEvent.click(screen.getByLabelText("play/pause"));
    await new Promise((resolve) => setTimeout(resolve, 250)); // let the interval fire a couple of times

    rerender(<PixelHorizonViewer framesSrc="/frames-b" predictionsSrc="/predictions-b" onTickChange={onTickChange} />);
    await waitFor(() => expect(screen.getByLabelText("play/pause")).toHaveTextContent("▶"));
    const callsAfterReload = onTickChange.mock.calls.length;

    // A leaked interval from episode A would keep firing on its stale
    // closure and publish more ticks here even though playback looks stopped.
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(onTickChange.mock.calls.length).toBe(callsAfterReload);
  });
});
