import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { DevelopmentPanel } from "./DevelopmentPanel.jsx";

afterEach(cleanup);

describe("DevelopmentPanel", () => {
  it("renders passed and pending stage gates with their milestones", () => {
    render(<DevelopmentPanel session={{ development: { stages: [
      { name: "Gestation", passed: true, milestones: ["sensory baseline"] },
      { name: "Crawling", passed: false },
    ] } }} />);
    expect(screen.getByText("Gestation").closest("li")).toHaveClass("stage--passed");
    expect(screen.getByText("sensory baseline")).toBeInTheDocument();
    expect(screen.getByText("Crawling").closest("li")).toHaveClass("stage--pending");
  });

  it("shows an empty state when no developmental gates were recorded", () => {
    render(<DevelopmentPanel session={{}} />);
    expect(screen.getByText(/no developmental gates recorded/)).toBeInTheDocument();
  });
});
