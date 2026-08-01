import { describe, expect, it } from "vitest";
import { episodeUrls } from "./api.js";

describe("episodeUrls", () => {
  it("encodes the session and episode id into the frames/predictions routes", () => {
    expect(episodeUrls("pixel session", "episode_00000")).toEqual({
      frames: "/api/sessions/pixel%20session/episodes/episode_00000/frames",
      predictions: "/api/sessions/pixel%20session/episodes/episode_00000/predictions",
    });
  });

  it("adds organism/run/experiment as query params when selecting a clinic run", () => {
    const urls = episodeUrls("pixel session", "episode_00000", { organism: "Crafter", run: "joint cortex/42", experiment: "joint cortex/42" });
    expect(urls.predictions).toBe(
      "/api/sessions/pixel%20session/episodes/episode_00000/predictions?organism=Crafter&run=joint+cortex%2F42&experiment=joint+cortex%2F42",
    );
  });
});
