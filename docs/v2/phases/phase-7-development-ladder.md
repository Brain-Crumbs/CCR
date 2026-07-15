# Phase 7 — Development (ontogeny) end to end

> Master plan: [Phase 7](../02-implementation-plan.md#phase-7--development-ontogeny-end-to-end).
> **Goal:** raise one named organism from Gestation to Foraging with gated
> promotion — one checkpoint carried across every stage; only its world and its
> freedoms change.

## Dependencies

- **Phase 5** (sleep/consolidation) and **Phase 6** (motor stack) — the ladder must
  carry one checkpoint across stages with consolidation and the motor stack in place.
- Uses milestone metrics from **Phases 2–6** as promotion gates.

## Builds on (existing code)

- `cognitive_runtime/training/curriculum_runner.py` — `CurriculumDefinition`,
  `CurriculumStageSpec`, `PromotionCriteria` (`value_of`, `evaluate`), `CurriculumState`,
  `curriculum_definition_from_dict`, `load_curriculum_definition`,
  `_validate_shared_layout`. **Generalise this into `development/`.**
- `cognitive_runtime/training/nursery.py` — the scenario registry the stages drive.
- `cognitive_runtime/neural/checkpoint.py` — the one checkpoint carried across
  stages (name from Phase 0).
- `cognitive_runtime/cli.py` — `cmd_curriculum_run` (line ~1599) is the entry point
  to generalise.
- `cognitive_runtime/tests/test_curriculum.py`, `test_curriculum_runner.py`.

## The ladder

| Stage | Nickname | Learns | Motor |
|---|---|---|---|
| 0 | Gestation | see & hear, habituate (sensory regularities, calm baseline) | frozen |
| 1 | Babbling | its own body: action→sensory change (forward/inverse) | random, overridden |
| 2 | Crawling | moving changes the view predictably (`walk_forward`, discrete `turn`) | scripted / overridden |
| 3 | Objects | permanence, affordances, approach & scale | scripted / learned |
| 4 | Foraging | goal-directed reward-seeking (survival curriculum) | learned |
| (5) | Speaking | communication / language | learned — **deferred** |

## Tasks

1. **Generalise `curriculum_runner.py` into `development/`.**
   - Each stage declares: World + scenario, which senses are active, motor freedom
     (`frozen | overridden | learned`), which losses are on, and its **milestone
     gates**. Extend `CurriculumStageSpec` with these fields; keep
     `PromotionCriteria` as the gate evaluator. Land behind a shim.
   - *Acceptance:* a stage spec expresses world/senses/motor-freedom/losses/gates and
     validates; old curriculum defs still load through the shim.

2. **Encode the ladder Gestation → Foraging.**
   - Author the five-stage definition (Speaking deferred), each stage pointing at its
     Crafter scenario(s), sense set, and motor freedom per the table.
   - *Acceptance:* the ladder loads and validates; each stage names its World,
     scenario, senses, motor freedom, losses, and milestone gate.

3. **One checkpoint across all stages.**
   - The organism carries a single checkpoint (by name) through every stage —
     resumable mid-ladder. Only the world/scenario, active senses, and motor freedom
     change between stages; the brain grows up in place.
   - *Acceptance:* promoting from stage N to N+1 resumes the *same* checkpoint (no
     re-init); a run is resumable from its checkpoint at any stage boundary.

4. **Promotion uses milestone metrics, not a single scalar.**
   - Wire each stage's gate to the relevant Phase 2–6 milestone metric (e.g. Crawling
     gates on the cortex beating copy-last on `walk_forward` + action-ablation; later
     stages on the forgetting metric, reflex-override behaviour, etc.). Promotion
     fires only when the stage's milestones pass on held-out data.
   - *Acceptance:* a stage does not promote until its milestone metric passes; a
     failing metric holds the organism at the stage.

5. **Motor-freedom transitions per stage.**
   - Gestation freezes motor; Babbling/Crawling use caregiver override / scripted;
     Objects/Foraging hand control to the voluntary path (Phase 6). Drive these
     through the stage's motor-freedom field.
   - *Acceptance:* each stage runs with the declared motor freedom; the caregiver
     override is active exactly in the stages that declare it.

## Deliverables

- `development/` staged ontogeny generalised from the curriculum runner, behind a
  shim.
- The Gestation→Foraging ladder definition (Speaking deferred).
- One checkpoint carried + resumable across stages.
- Milestone-metric-gated promotion; per-stage motor freedom.

## Tests

- Extend `test_curriculum_runner.py`: stage spec carries world/senses/motor-freedom/
  losses/gates; promotion fires only on passing milestone metrics; a failing metric
  blocks promotion.
- `tests/test_development_ladder.py`: the same checkpoint resumes across a stage
  boundary (no re-init); motor freedom matches the stage; resume mid-ladder works.

## Milestone 7 (exit gate)

A single **named** organism walks the ladder unattended through **at least
Crawling**, passing each stage's milestone, **resumable from its checkpoint**, its
whole life inspectable by name in the clinic. CI-runnable: an unattended run
promotes Gestation→Babbling→Crawling on held-out gates and resumes cleanly.

## Risks / notes

- **Learning does not transfer across worlds.** The ladder runs in Crafter; the
  first-person Minecraft graduation is a *later* world with a fresh cortex. Do not
  encode cross-world weight transfer into the ladder.
- **Speaking is deferred** until Foraging works
  ([master plan deferrals](../02-implementation-plan.md#what-this-plan-deliberately-defers)).
- The promotion gate is the real test of the whole stack — if a stage's milestone
  metric isn't trustworthy (e.g. uncalibrated surprise from Phase 3), the ladder
  will promote on a lie. Gate quality is inherited from Phases 2–6; fix it there.
