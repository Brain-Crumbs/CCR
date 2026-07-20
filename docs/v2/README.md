# V2: The Biological Redesign

A proposal to re-centre this project on the biology it already half-grew: a
**predictive organism** that watches a world, forecasts its own future senses,
is surprised when wrong, dreams those surprises into a world model while it
sleeps, and is **raised developmentally** from newborn to forager.

Read in order:

1. [**00-overview.md**](00-overview.md) — the vision, the core predict→surprise→
   act loop, the three modes (reward-seeking / info-gathering / fight-or-flight),
   dreams, the developmental ladder, and the named organism.
2. [**01-architecture.md**](01-architecture.md) — the anatomy: the full old→new
   naming map, the Predictive Cortex, the three memory timescales
   (working / hippocampus / cortex), neuromodulators & the Arbiter, Sleep &
   Dreams, the two motor paths, the World seam, and the Clinic front-end.
3. [**02-implementation-plan.md**](02-implementation-plan.md) — the phased,
   milestone-gated path from today's `cognitive_runtime/` to the V2 organism.
4. [**phases/**](phases/README.md) — the task-level breakdown: one actionable
   implementation plan per phase (0–8), grounded in the current
   `cognitive_runtime/` code, each with numbered tasks, tests, and its milestone
   gate.

For onboarding and presentation:

5. [**03-onboarding-guide.md**](03-onboarding-guide.md) — a from-scratch mental
   model of the system, repository tour, setup, workflows, current assembly
   boundary, and explicit deferrals.
6. [**04-contracts-and-data-flow.md**](04-contracts-and-data-flow.md) — the exact
   Python, tensor, stream, disk, HTTP, checkpoint, development, and mineflayer
   contracts, plus one complete action-to-training-target trace.
7. [**05-presentation-runbook.md**](05-presentation-runbook.md) — a 75–90 minute
   teaching sequence, live-demo script, slide outline, and presenter cautions.

**Design commitments** (the choices these docs are built on): re-architect +
rename rather than rewrite; predict in latent space but always decode to the
same-shaped input; action-conditioned recurrent world model; a fast Crafter
nursery world plus Minecraft graduation; one voluntary motor path that **plans
over the world model** (one-step MPC by default; active-inference decoding and an
imagination actor kept as experiments) with a hardcoded **reflex stack** overriding
it and a nursery caregiver override on top, every tick recording
predicted-vs-actuated action; diagnostics instead of enforced determinism; and a
Node/React clinic (read-only first) as the primary interface.
