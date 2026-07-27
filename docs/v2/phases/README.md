# V2 Phase Plans — Task-Level Breakdown

This folder decomposes the master [implementation plan](../02-implementation-plan.md)
into one **actionable plan per phase**. Each phase document is self-contained:
it states the goal, the exact existing files it builds on, a numbered task list
with acceptance criteria, the tests to add, the milestone gate that closes the
phase, and the risks to watch.

These plans are grounded in the code that exists today under `cognitive_runtime/`
(surveyed July 2026). File paths in **Builds on** and **Task** sections are real
modules; paths in **Target** sections are the V2 names the architecture proposes
(landed behind shims, per Phase 0's deferral rule).

## The phases

| Phase | Plan | Milestone gate | Depends on |
|---|---|---|---|
| 0 | [Identity & scaffolding](phase-0-identity-scaffolding.md) | `Pixel-<session>` run, checkpoint carries `name` | — |
| 1 | [The nursery World (Crafter)](phase-1-nursery-world.md) | `walk_forward` + discrete `turn` pass quality gates in Crafter | 0 |
| 2 | [The Predictive Cortex](phase-2-predictive-cortex.md) | cortex beats copy-last at every horizon; ablating actions hurts `turn` | 1 |
| 3 | [Neuromodulators, Amygdala & Arbiter](phase-3-neuromodulators-arbiter.md) | 3-mode switch visibly changes behaviour | 2 |
| 4 | [Hippocampus & Dreams](phase-4-hippocampus-dreams.md) | dream from a seed regenerates the episode; clinic renders the strip | 2 |
| 5 | [Sleep as continuous consolidation](phase-5-sleep-consolidation.md) | learns new scenario while retaining old; no missed-tick regression | 4 |
| 6 | [The motor system](phase-6-motor-system.md) | reflex overrides voluntary; reflex-activation rate charted | 5 |
| 7 | [Development (ontogeny)](phase-7-development-ladder.md) | one organism walks Gestation→Crawling unattended | 5, 6 |
| 8 | [The Clinic (front-end)](phase-8-clinic.md) | run a nursery session, watch EEG + dream strips, trigger consolidation | 2 (8a), 7 (8b) |

## Dependency order

```
0 → 1 → 2 ─┬─→ 3 ─┐
           ├─→ 4 ─┴─→ 5 → 6 → 7 → 8b
           └─→ 8a
```

Phases 3, 4, and 8a can run in parallel once Phase 2 lands. Phase 5 needs Phase 4
(dreams to replay). Phase 7 needs 5 and 6 (one checkpoint carried across stages
with the motor stack in place). See each plan's **Dependencies** section for the
precise coupling.

## How to use a phase plan

1. Read the phase's **Goal** and **Milestone gate** first — that is the definition
   of done. No phase is complete because code exists; it is complete when its
   milestone metric passes on held-out data and is visible in the clinic (or CLI
   until 8a lands).
2. Work the **Tasks** in order; each has an acceptance line.
3. Land every rename behind a re-export shim (Phase 0 rule) so nothing breaks
   mid-migration; do not physically move ~100 files before Milestone 2.
4. Add the phase's **Tests** as you go — the milestone gate should be a CI check
   where feasible.
