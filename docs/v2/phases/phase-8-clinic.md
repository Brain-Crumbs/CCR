# Phase 8 — The Clinic (front-end)

> Master plan: [Phase 8](../02-implementation-plan.md#phase-8--the-clinic-front-end).
> **Goal:** operate and inspect the organism from a Node/React front-end — read-only
> first (8a), control later (8b) — never reaching into brain internals.

## Dependencies

- **8a (read-only):** needs **Phase 2** data to be interesting (dream strips, EEG),
  but read-only panels can start landing as soon as Phase 1 produces recorded
  sessions. Runs partly in parallel from Phase 1.
- **8b (control):** needs **Phase 7** (a ladder to launch/promote) and Phase 5/6
  (sleep to trigger, motor to override).

## Builds on (existing code)

- `viewer/` — `server.js` (Node/HTTP), `public/index.html`,
  `public/pixel-horizon-viewer.js` (the strip renderer to reuse), `package.json`,
  `export_predictions.py`. **The clinic extends this, not a rewrite.**
- `cognitive_runtime/record/` (the Record) + `runtime/recorder.py` — the data source;
  sessions grouped by organism name (Phase 0).
- `cognitive_runtime/record/quality.py` (Phase 1) — green/amber/red verdicts.
- `cognitive_runtime/training/prediction_export.py` — dream/prediction export format.
- The recorded streams from Phases 3–6: arbiter mode, neuromodulators, attention,
  predicted-vs-actuated motor, reflex-activation rate.
- `cognitive_runtime/tools/metrics_dashboard.py` — existing grouping logic to reuse.

## Tasks — 8a (read-only; target: usable by end of Phase 2)

1. **Node/HTTP service over the Record.**
   - A small read-only API over recorded sessions (list by organism name, fetch a
     session's streams, exports, quality verdict). Extend `viewer/server.js`.
   - *Acceptance:* the service lists sessions by organism name and serves one
     session's streams + exports.

2. **Session browser by organism name.**
   - React view listing sessions grouped by organism (Phase 0 name), with the
     data-quality verdict (green/amber/red) shown per session before you ever train
     on it.
   - *Acceptance:* selecting an organism lists its sessions with quality badges.

3. **Dream strips (reuse `pixel-horizon-viewer.js`).**
   - Render predicted-vs-actual per horizon, and dreamed-vs-actual (Phase 4), from the
     export format.
   - *Acceptance:* a session's dream strip renders per horizon.

4. **EEG panel.**
   - Timelines of neuromodulators (dopamine / acetylcholine / adrenaline),
     prediction error, and the **arbiter mode** tick-by-tick (Phase 3).
   - *Acceptance:* the panel shows the organism flipping bored→curious→afraid over a
     recorded session.

5. **Attention/focus timeline.**
   - What it attended to and why, per stream (from `core/attention.py`'s reasons).
   - *Acceptance:* per-tick attention with per-stream reasons renders.

6. **Developmental chart.**
   - Milestones passed per stage per organism (Phase 7 gates).
   - *Acceptance:* the ladder progress for an organism renders.

7. **Data-quality gate results.**
   - Surface `record/quality.py`'s green/amber/red per session (pixel provenance,
     motion floor, completed-episode, frozen-rollout).
   - *Acceptance:* each session shows its verdict and the failing checks.

## Tasks — 8b (control; after Phase 7)

8. **Thin control API (never reaches into brain internals).**
   - Launch/stop runs, pick World + scenario, over a small control plane distinct
     from the Record.
   - *Acceptance:* a run starts/stops from the UI via the control API only.

9. **Motor override during nursery.**
   - Drive the Phase 6 caregiver override from the UI during nursery stages.
   - *Acceptance:* a caregiver command issued from the UI actuates in a nursery run.

10. **Trigger and watch a sleep/consolidation phase.**
    - Kick a Phase 5 micro-sleep / long consolidation and watch its dream strips,
      loss curves, and forgetting metric.
    - *Acceptance:* a consolidation triggered from the UI runs and streams its metrics
      back.

11. **Promote stages.**
    - Advance the Phase 7 ladder from the UI when a stage's gate passes.
    - *Acceptance:* a passing stage can be promoted from the UI.

## Deliverables

- **8a:** read-only Node/HTTP service + React app: session browser by name, dream
  strips, EEG panel, attention timeline, developmental chart, data-quality verdicts.
- **8b:** thin control API for launch/stop, world/scenario pick, motor override,
  sleep trigger, stage promotion.

## Tests

- Service-level: API returns sessions by name, streams, exports, quality verdicts
  (contract tests against recorded fixtures).
- Front-end: component render tests for dream strip, EEG panel, quality badges
  against fixture sessions.
- 8b: control-API tests that a launch/stop/override/trigger/promote call reaches the
  runtime through the control plane only (no brain-internal access).

## Milestone 8 (exit gate)

You can **run a full nursery session, watch its EEG and dream strips live-ish, see
its data-quality verdict, and (8b) trigger its consolidation — without touching the
CLI.** 8a is judged usable by end of Phase 2 (read-only over real cortex data); 8b
by end of Phase 7.

## Risks / notes

- **World-agnostic, brain-agnostic.** The clinic talks to the organism only through
  the **Record** and a **thin control API** — never into brain internals — so it
  stays as World-agnostic as the brain. A panel that imports a brain module is a
  layering violation.
- **Read-only first is deliberate.** Ship 8a and get value from inspection before
  building control; control surfaces (motor override, stage promotion) are powerful
  and should follow a stable read-only base.
- Reuse over rewrite: the pixel-horizon viewer and the metrics dashboard already
  exist — extend them.
