# V2 decision log — responses to the external critique

This is a short record of the decisions taken in reply to the external critique of
the V2 design, and *where each one now lives* in the main docs. The substance has
been folded into [00-overview](00-overview.md), [01-architecture](01-architecture.md),
and [02-implementation-plan](02-implementation-plan.md); this file is the index and
the rationale, not a competing spec.

| # | Critique / question | Decision | Where it landed |
|---|---|---|---|
| 1 | "Biology becomes load-bearing / self-deception" | This is a simulation; the biological names are ergonomic labels. We **keep the vocabulary** but never let a label grant behaviour that isn't in code + a metric. The one outright overreach — the Arbiter "arises/emerges" — is fixed: it's a hand-authored 2×2 **mode selector**. | 00 success criteria; 01 Arbiter ("hand-authored 2×2 lookup, not an emergent process"); 02 Phase 3 goal |
| 2 | Motor-from-prediction / active inference is the least-proven piece, yet it was the default | **MPC-first.** Default voluntary motor = **one-step planning over the world model** (nothing in the motor learns; the cortex does). Active-inference decoding, a DreamerV3 imagination actor, and the policy head are **alternative controllers for A/B**, not the spine. | 01 commitment 5 + motor section; 02 Phase 6; README |
| 3 | Crafter can't teach ego-motion, but the docs still claimed it | Crafter is **2-D top-down**; it teaches object dynamics / action→effect / consequence, **not** ego-motion, optical flow, or parallax. That language is dropped from the Crafter stages; true ego-motion waits for the first-person Minecraft world. **Learning does not transfer across worlds yet.** | 00 ladder + "Where it lives"; 02 Phase 1 + Milestone 1 |
| 4 | Rename is highest-churn, lowest-value; don't front-load it | Phase 0 keeps only `OrganismConfig.name` (cheap provenance). The **namespace rename tree is deferred until after Milestone 2** — prove the cortex first, rename behind shims or never. | 02 Phase 0 + "First concrete step" |
| 5 | Async actor-learner instability | The **world model** is the stable spine (self-supervised regression); the motor only plans over it, so there's no bootstrapped-policy instability. **Phasic wake/sleep first**, concurrent later with **EMA-averaged weights + a staleness version stamp**. | 01 Sleep & Dreams; 02 Phase 5 |
| 6 | Generative-replay bootstrap paradox / forgetting | **Reservoir of real transitions** (never dream-only) + **dream fraction gated on measured quality** (0% until the cortex beats copy-last; ramp; cap ≈0.5). Forgetting is the headline measured claim. | 00 "The one measured claim"; 01 Sleep & Dreams; 02 Phase 5 |
| 7 | Uncertainty calibration (the mode selector depends on it) | Produce cortex uncertainty (ensemble / predicted-error head), **calibrate and report it**, and give the mode switch **hysteresis** (k-tick persistence) so it can't flap. | 01 Arbiter; 02 Phase 3 |
| 8 | Processing speed / data handling | Crafter removes the headless-GL/xvfb pain and tick jitter; keep the one-canonical-shape + provenance gate and tick-denominated horizons already paid for. Determinism (and the bit-exact replay smoke test) comes back with Crafter. | 00 "Where it lives"; 01 diagnostics section; 02 Phase 1 |
| 9 | "Parallelize to make it linear?" (your Q) | A **dilated temporal-conv or small transformer over a frame window** is a parallel-friendly alternative to the GRU: processes the window in one pass, reads multiple timescales at once, and pairs with a **context-length curriculum** (1 frame → 2 → k, your Q3). Benchmarked A/B against the GRU. | 01 prediction core; 02 Phase 2 |
| 10 | Pick one falsifiable claim | **Developmental staging + generative replay ⇒ measurably less forgetting than flat training on the same data.** | 00 "The one measured claim"; Milestone 5 |

## The one thing to build first

Unchanged from the plan: **Phase 2 is make-or-break.** An action-conditioned,
recurrent (or temporal-conv), decoded, multi-horizon world model that beats
copy-last on held-out Crafter seeds and *uses the action stream* (withholding it
measurably hurts `turn`). Everything above it — modes, dreams, sleep, motor,
the ladder, the clinic — is only worth building once that clears its bar.
