# Phase 1 — The Nursery World (Crafter/Craftax)

> Master plan: [Phase 1](../02-implementation-plan.md#phase-1--the-nursery-world-craftercraftax).
> **Goal:** a fast, deterministic, pixel-native World to raise the infant in —
> behind the exact same World seam Minecraft uses, so the brain is unchanged.

## Why now

Most of the recording-quality pain documented in
[`nursery-turn-in-place-analysis.md`](../../history/nursery-turn-in-place-analysis.md) traces
to using live survival Minecraft as a nursery (headless-GL artifacts,
non-determinism, tick jitter). A clean deterministic pixel world removes that
variable *before* Phase 2 tunes the hard part (the cortex). Determinism, pixel
provenance, and speed are the wins here — **not** perspective (Crafter is 2-D
top-down, so ego-motion / optical flow stays out of scope until the first-person
graduation world).

## Dependencies

- **Phase 0** (organism name) — sessions recorded in Crafter should already carry
  the name.

## Builds on (existing code)

- `cognitive_runtime/core/program.py` — the `Program` (World) ABC: the streams-v2
  contract (`stream_catalog()`, `attach_buses()`, `step()`) plus the legacy
  `observe()/act()/reward()` shim. **This is the seam a Crafter world implements.**
- `cognitive_runtime/programs/minecraft/` — the reference implementation of the
  seam: `world.py`, `stream_registry.py`, `streams.py`, `observations.py`,
  `actions.py`, `action_registry.py`, `backend.py`, `config.py`. Mirror this layout.
- `cognitive_runtime/core/streams/` — bus/registry/encoder primitives the world
  publishes onto (`bus.py`, `registry.py`, `encoders/grid_vision.py`,
  `encoders/scalar.py`).
- `cognitive_runtime/core/action_registry.py` — world-changing vs
  information-gathering action classification (carried over unchanged).
- `cognitive_runtime/training/nursery.py` — `NURSERY_SCENARIOS` registry,
  `NurseryScenario`, `_walk_forward`, `_turn_in_place`, `_object_permanence`,
  `_approach_entity`; and `measure_recording_quality` / `EpisodeRecordingQuality`
  (the quality gates today, Minecraft-specific).
- `cognitive_runtime/cli.py` — `--backend` selector (`BACKENDS`) is the pattern the
  `--world` selector follows.

## Tasks

1. **Add the `worlds/crafter/` World implementing the seam.**
   - New package (e.g. `cognitive_runtime/programs/crafter/` today, target
     `worlds/crafter/`) with a `CrafterWorld(Program)` that wraps
     [Crafter/Craftax](https://arxiv.org/abs/2109.06780).
   - Publish streams on the sensory bus: `vision.frame.pixels` (the RGB frame),
     `body.*` (health/inventory/interoception scalars Crafter exposes), reward, and
     the efference-copy `motor.command`. Drain efferents from the motor bus in
     `step()`.
   - Implement `stream_catalog()`, `attach_buses()`, `step()`, `reset(seed)`, and
     `snapshot()/restore()` (Crafter state is cheap to snapshot — enables the
     byte-exact replay smoke test).
   - *Acceptance:* a `CrafterWorld` run publishes a pixel stream and consumes a
     motor command through the unmodified loop; `stream_catalog()` validates against
     the registry contract.

2. **Map Crafter's action space through the action registry.**
   - Declare Crafter's ~17 discrete actions in a `crafter/action_registry.py`,
     classifying each as world-changing or information-gathering
     (`core/action_registry.py` contract). The brain stays opaque to what the
     actions *are*.
   - *Acceptance:* `ActionRegistry.assert_complete(crafter_action_space)` passes.

3. **Wire the `--world crafter|minecraft` selector.**
   - Add a `--world` argument to `cli.py` run/nursery/train commands, defaulting to
     `minecraft` for back-compat. Route world construction through a small factory.
   - *Acceptance:* `... run --world crafter` runs end-to-end; `--world minecraft` is
     unchanged.

4. **Port the nursery scenarios to Crafter.**
   - Re-implement `walk_forward`, `object_permanence`, `approach_entity`, and a
     **discrete-facing `turn`** (Crafter's facing is a discrete flip, not the
     continuous rotation `turn_in_place` assumed — re-scope, do not port the
     optical-flow premise) as Crafter scenarios, registered alongside the Minecraft
     `NURSERY_SCENARIOS` (parameterise the registry by world, or add a parallel
     `CRAFTER_SCENARIOS`).
   - *Acceptance:* `nursery run --world crafter walk_forward` records a deterministic
     episode with genuine frame-to-frame motion.

5. **Bring the data-quality gates forward as a reusable `record/quality.py`.**
   - Generalise `measure_recording_quality` / `EpisodeRecordingQuality`
     (`training/nursery.py`) into a world-agnostic `record/quality.py`: pixel
     provenance, motion floor, completed-episode, and a discrete yaw/facing-sweep
     check (replacing the continuous yaw-sweep).
   - Emit a green/amber/red verdict per session (the shape the clinic will consume).
   - *Acceptance:* the gates run on a Crafter session and on a Minecraft session
     through the same API; a deliberately-frozen recording is flagged red.

6. **Re-enable the byte-exact replay smoke test on Crafter.**
   - Because Crafter is deterministic and snapshot-able, restore the
     record→replay→compare smoke test as a cheap plumbing check (publish-order /
     recorder regression catcher), *not* a learning gate.
   - *Acceptance:* a recorded Crafter episode replays byte-identically.

## Deliverables

- `worlds/crafter/` World + action registry behind the unchanged seam.
- `--world` selector.
- Crafter ports of `walk_forward`, discrete `turn`, `object_permanence`,
  `approach_entity`.
- Reusable `record/quality.py` with green/amber/red verdicts.
- Crafter byte-exact replay smoke test.

## Tests

- `tests/test_crafter_world.py`: seam conformance (catalog, attach, step, reset,
  snapshot/restore); pixel stream shape/provenance; efference-copy round-trip.
- `tests/test_crafter_scenarios.py`: each ported scenario records a
  quality-gate-passing, deterministic episode with motion above the floor.
- `tests/test_record_quality.py`: gates flag a frozen/zero-motion recording red and
  pass a clean one; run against both worlds.
- Extend the replay smoke test to a Crafter episode.

## Milestone 1 (exit gate)

`walk_forward` (and a discrete-facing `turn`) recorded in Crafter **pass the
data-quality gates deterministically, with genuine frame-to-frame motion.** The win
is determinism, pixel provenance, speed, and clean translational motion — **not**
perspective (ego-motion / optical flow is explicitly out of scope; Crafter is 2-D
top-down). Verified by `tests/test_crafter_scenarios.py` + the replay smoke test in
CI.

## Risks / notes

- **Do not smuggle ego-motion back in.** Any scenario or gate that assumes optical
  flow, parallax, or view rotation is wrong for Crafter and must wait for the
  first-person Minecraft graduation world.
- **The brain must not change.** If implementing Crafter requires touching anything
  under `core/`, `neural/`, or `policies/` beyond the world/action-registry seam,
  that is a seam leak — stop and fix the seam instead.
- **Learning does not transfer across worlds yet.** Crafter keeps the *code*
  portable, not the *weights*; graduating to Minecraft means starting the cortex
  over on first-person input. Do not build anything that assumes weight transfer.
- Craftax (JAX) vs Crafter (NumPy) is an implementation choice — pick the one whose
  install and determinism story is cleanest in this environment; the seam hides it.
