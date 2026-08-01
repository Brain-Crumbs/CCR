import { describe, expect, it } from "vitest";
import { buildHash, parseHash } from "./hashRoute.js";

describe("parseHash", () => {
  it("decodes each path segment", () => {
    expect(parseHash("#Test/run-1/pixel%20session/episode_00000")).toEqual({
      organism: "Test", run: "run-1", session: "pixel session", episode: "episode_00000",
    });
  });

  it("tolerates a missing leading #", () => {
    expect(parseHash("Test/run-1")).toMatchObject({ organism: "Test", run: "run-1", session: null, episode: null });
  });

  it("returns all-null fields for an empty hash", () => {
    expect(parseHash("")).toEqual({ organism: null, run: null, session: null, episode: null });
  });
});

describe("buildHash", () => {
  it("round-trips through parseHash", () => {
    const route = { organism: "Test", run: "run-1", session: "pixel session", episode: "episode_00000" };
    expect(parseHash(`#${buildHash(route)}`)).toEqual(route);
  });
});
