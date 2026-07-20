# V2 Presentation and Onboarding Runbook

This runbook turns the V2 design and codebase into a teachable session. It is
designed for a new contributor who needs a durable mental model, not a feature
tour.

## Suggested format

Use 75–90 minutes:

| Time | Topic | Outcome |
|---:|---|---|
| 0–10 min | Thesis and measured claim | Understand why the organism exists |
| 10–25 min | Waking tick | Trace causal data flow and one-tick latency |
| 25–40 min | Architecture and repository | Know where each responsibility lives |
| 40–55 min | Cortex, memory, dreams, sleep | Understand the learning story |
| 55–65 min | Motor modes and development | Understand behavior and staged freedom |
| 65–80 min | Record and Clinic live walkthrough | Inspect evidence instead of trusting labels |
| 80–90 min | Current assembly boundary and next work | Separate implemented organs from the unified target |

## The narrative

### 1. Start with the problem

“Most agents wait for a request. CCR continuously inhabits a World. The World
grades every prediction by revealing what actually happened next.”

Then state the measured claim: development plus generative replay should reduce
forgetting versus flat training.

Avoid beginning with Minecraft. Minecraft and Crafter are habitats; they are not
the architecture.

### 2. Draw only this loop first

```text
sense → attend → bind → predict → feel → choose mode → act → remember
  ▲                                                           │
  └────────────────────── World answers ──────────────────────┘
                               │
                            Record
                               │
                         sleep / dreams
```

Explain that all arrows are streams and that internal state is published back
as interoception.

### 3. Demonstrate the causal timing

Use two consecutive ticks:

```text
tick t:   collect consequences of previous action → choose MOVE_UP → queue it
tick t+1: World applies MOVE_UP → publishes changed pixels/reward → organism sees result
```

This prevents the most common misunderstanding about reward attribution,
recording, and dataset alignment.

### 4. Introduce the three memory timescales

Use one example: the organism encounters a dangerous object.

- working memory retains the last few seconds;
- Hippocampus keeps a high-priority seed because surprise/threat is high;
- during sleep, the cortex rehearses the seed and changes lifetime weights.

Then explain the real/dream replay guardrail. A dream is generated evidence,
not ground truth.

### 5. Introduce behavior as a precedence stack

```text
caregiver override
       > reflex by priority
              > voluntary action (MPC default in the design)
```

Show that the organism records intention and actuation separately. This is both
debugging information and a possible developmental teaching signal.

### 6. Show the code by following data

Open these files in order:

1. `cognitive_runtime/core/program.py` — World boundary.
2. `cognitive_runtime/programs/crafter/adapter.py` — one concrete World.
3. `cognitive_runtime/runtime/loop.py` — one live tick.
4. `cognitive_runtime/runtime/recorder.py` — evidence written to disk.
5. `brain/cortex/predictive.py` — learned dynamics interface.
6. `brain/hippocampus.py` and `sleep/dream.py` — episodic seed to dream.
7. `motor/reflexes.py` — intention/override contract.
8. `development/ladder.py` — staged capabilities and gates.
9. `viewer/server.js` — Record-to-Clinic boundary.

Do not walk the directory tree alphabetically.

## Live-demo script

### Before the presentation

```powershell
.venv\Scripts\Activate.ps1
python -m cognitive_runtime --help
python -m pytest tests\test_core.py tests\test_runtime.py tests\test_streams.py `
  tests\test_hippocampus.py tests\test_dream.py `
  -k "not test_dream_never_reads_live_senses" --basetemp=.pytest-demo -q
```

Do not silently omit why: the excluded dream test monkeypatches the removed
`StreamBus.read_since` API and currently fails during test setup. Put that stale
test/API mismatch on the known-debt slide and explain that this preflight is a
demo smoke test, not the repository's full acceptance result.

Prepare one short frame-recorded deterministic session. If using the Clinic on
Windows, set:

```powershell
$env:PYTHON = (Resolve-Path .venv\Scripts\python.exe)
```

### Demo A — prove the substrate

```powershell
python -m cognitive_runtime run `
  --world minecraft --backend simulated --policy null `
  --episodes 1 --episode-ticks 20 --name Demo `
  --record-dir sessions
```

Show:

- the organism-prefixed session directory;
- `session.json` and its World/action/stream schemas;
- a decision with `motor_emitted: []`;
- the episode summary proving the World advanced.

### Demo B — show the Record

Use a frame-recorded session and display:

- one sensory stream line;
- one motor stream line;
- the adjacent decision record;
- one frame-store index entry;
- the content hash linking the JSONL record to frame bytes.

### Demo C — show the Clinic

```powershell
node viewer\server.js --data-dir sessions --port 8787
```

Open one episode and narrate:

1. quality verdict;
2. frame at `t` that the model saw;
3. prediction for `t+h`;
4. actual frame at `t+h`;
5. absolute error and MSE timeline;
6. dopamine/acetylcholine/adrenaline and arbiter mode;
7. attention focus and reasons;
8. developmental gate status.

If the checked-out viewer has not yet incorporated the revision's “seen” panel,
say so explicitly and show the target four-column sequence on a slide.

### Demo D — show the learning gate, not only loss

Use an existing nursery report or run a small cortex job ahead of time. Present:

- model/copy-last ratio at every horizon;
- model/oracle ratio where applicable;
- action-ablation difference;
- frozen-rollout verdict;
- representation probe;
- dream reconstruction and forgetting comparison if available.

Never present declining training loss as the Milestone-2 proof.

## Slide outline

1. **A continuously inhabiting organism** — problem and one-sentence thesis.
2. **The falsifiable result** — development + dreams versus flat training.
3. **Two habitats, one World contract** — Crafter nursery and Minecraft graduation.
4. **One waking tick** — causal flow and one-tick action latency.
5. **Streams are the nervous system** — schema, cadence, determinism.
6. **Workspace and prediction** — attention, fusion, cortex horizons.
7. **Feeling and behavioral mode** — neuromodulators, Amygdala, Arbiter.
8. **Intention versus actuation** — MPC, reflexes, caregiver.
9. **Three memory timescales** — working memory, Hippocampus, cortex.
10. **Sleep and forgetting** — real/dream mixture and guardrails.
11. **Development** — stage freedoms and metric-gated promotion.
12. **The Record and Clinic** — observability as the trust mechanism.
13. **What runs today** — integrated substrate, specialized V2 pipelines.
14. **Explicit deferrals** — retrieval, language, cross-world transfer, control UI.
15. **Next assembly step** — make the recurrent cortex the live predictor,
    training target, and MPC planning model.

## Questions a new contributor should be able to answer

- Why is `[]` different from a missing decision?
- Why is an action's consequence recorded on the next tick?
- Why do `StreamSpec` and `StreamDeclaration` both exist?
- Why does fusion have a layout hash?
- Why must the brain not know Crafter action semantics?
- Why is copy-last a more important baseline than raw MSE?
- What does action ablation prove?
- Why can dreams harm learning when the cortex is weak?
- What is the difference between a replay buffer transition and a hippocampal
  seed?
- Why does concurrent publication use a separate EMA checkpoint?
- Which parts of the target organism are in the default `run` path today?
- Which deferred capability would turn the Hippocampus into online recall?

## Presenter cautions

- Do not claim the Arbiter emerges; it is an authored lookup with calibrated
  inputs and hysteresis.
- Do not claim Crafter teaches first-person ego-motion; it is top-down.
- Do not claim weights transfer from Crafter to Minecraft.
- Do not equate the earlier MLP world model with the recurrent Predictive
  Cortex.
- Do not imply every cortex output head is fully trained because the tensor is
  present.
- Do not claim the current Clinic has write/control capabilities.
- Do not hide the current assembly boundary. It is the clearest way to explain
  both the value of the implemented organs and the next architectural step.

## Handout set

Give the learner these files in order:

1. [00-overview.md](00-overview.md)
2. [03-onboarding-guide.md](03-onboarding-guide.md)
3. [01-architecture.md](01-architecture.md)
4. [04-contracts-and-data-flow.md](04-contracts-and-data-flow.md)
5. [02-implementation-plan.md](02-implementation-plan.md)
6. [phases/README.md](phases/README.md)

The onboarding guide explains the whole; the architecture provides the intended
anatomy; the contracts reference answers exact interface questions; the plan and
phase docs explain sequencing and exclusions.
