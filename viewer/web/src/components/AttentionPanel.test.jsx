import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { AttentionPanel } from "./AttentionPanel.jsx";

afterEach(cleanup);

const decisions = [{ tick_index: 4, attention: {
  focus_stream: "vision.frame.pixels", selected_streams: ["vision.frame.pixels", "body.health"],
  reasons: { "vision.frame.pixels": { components: { novelty: 0.75, boredom: -0.1 } } },
} }];

describe("AttentionPanel", () => {
  it("renders the focus, attended streams, and reason breakdown from DecisionRecord", () => {
    render(<AttentionPanel decisions={decisions} />);
    expect(screen.getByText("vision.frame.pixels")).toBeInTheDocument();
    expect(screen.getByText("novelty 0.75 · boredom -0.10")).toBeInTheDocument();
    expect(screen.getByText(/body\.health/)).toBeInTheDocument();
  });

  it("seeks the shared cursor when a row is clicked", () => {
    const onTickChange = vi.fn();
    render(<AttentionPanel decisions={decisions} onTickChange={onTickChange} />);
    fireEvent.click(screen.getByText("t4"));
    expect(onTickChange).toHaveBeenCalledWith(4);
  });

  it("shows an empty state when attention was not recorded", () => {
    render(<AttentionPanel decisions={[]} />);
    expect(screen.getByText(/attention was not recorded/)).toBeInTheDocument();
  });
});
