import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { EEGPanel } from "./EEGPanel.jsx";

afterEach(cleanup);

const records = [
  { stream_id: "internal.dopamine", payload: { value: 0.2 }, seq: 1 },
  { stream_id: "internal.acetylcholine", payload: { value: 0.3 }, seq: 1 },
  { stream_id: "internal.adrenaline", payload: { value: 0.4 }, seq: 1 },
  { stream_id: "internal.prediction_error", payload: { value: 0.5 }, seq: 1 },
  { stream_id: "internal.arbiter.mode", payload: { mode: "afraid" }, seq: 1 },
];

describe("EEGPanel", () => {
  it("renders neuromodulator traces and the arbiter mode timeline", () => {
    render(<EEGPanel records={records} />);
    for (const label of ["dopamine", "acetylcholine", "adrenaline", "prediction error", "afraid"]) {
      expect(screen.getByText(new RegExp(label))).toBeInTheDocument();
    }
  });

  it("reports a mode click back to the shared time cursor", () => {
    const onTickChange = vi.fn();
    render(<EEGPanel records={records} onTickChange={onTickChange} />);
    fireEvent.click(screen.getByText("afraid"));
    expect(onTickChange).toHaveBeenCalledWith(1);
  });

  it("highlights the mode closest to the current tick", () => {
    render(<EEGPanel records={records} tick={1} />);
    expect(screen.getByText("afraid").closest("li")).toHaveClass("is-current");
  });
});
