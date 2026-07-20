# Proposal — The Cortex as an Autoregressive Latent World Model ("GPT for frames")

> **Status:** design proposal. Extends the [Predictive Cortex](phases/phase-2-predictive-cortex.md)
> with an LLM-style training objective. Companion to
> [`REVIEW-2026-07-organism-audit.md`](REVIEW-2026-07-organism-audit.md);
> resolves that review's "cortex is pixels-only" gap as a side effect.

## One sentence

Train the cortex the way an LLM is trained — **one causal forward pass over an
episode, predicting the next latent at every position in parallel** — instead of
sampling fixed windows and rolling out one at a time.

## The idea, precisely

The intuition is: *"take 1 frame, predict the future; take the last 2 frames,
predict the future; take the last 3 frames, predict the future — like an LLM but
over latent-space world-model frames (visual, action, and all latent data)."*

The important realization: **those growing prefixes are not N separate passes.**
A causal transformer computes them all in **one** pass. With a causal mask,
position `i` attends only to frames `0..i` — so position `i` *is* "the last i+1
frames," and attaching a predictor at every position trains every prefix length
simultaneously. This is exactly why LLM pre-training is efficient (loss at every
token position, one forward pass), and it transfers to latent frames unchanged.

```
episode latents:   z0  z1  z2  z3  z4  ...  z_{T-1}
causal pass →      h0  h1  h2  h3  h4  ...   (h_i has seen z0..z_i)
predict at each i:  ẑ_{i+1}, ẑ_{i+4}, ẑ_{i+8}   (multi-horizon heads)
loss:              mean over all (position i × horizon h) of  ‖ẑ − z‖  (+ pixel aux)
```

## The token: the full fused workspace latent

To predict *"actions and visual and all latent data,"* the unit of prediction
should be the **bound workspace latent** `z_t` — vision **plus** body /
interoception **plus** the efference copy (its own next action) **plus**
neuromodulators — not a pixels-only latent. Then next-latent prediction predicts
everything at once, including what the organism is about to do (active-inference
flavored), and the pixels-only limitation noted in the audit disappears. Each
per-stream decoder still recovers a viewable frame / scalar for the clinic.

## Multi-horizon = multi-token prediction (and it fights collapse)

"Predict *all* future frames from each prefix" is **multi-token prediction**
(cf. LLM MTP / Medusa): from `h_i`, *direct* heads emit `ẑ_{i+1}, ẑ_{i+4},
ẑ_{i+8}` in parallel. This is strictly better than the current short-rollout
composition for one reason the existing docs stress repeatedly: deep
backprop-through-composition of a single transition "selects for the identity"
(the frozen-rollout failure). **Direct per-horizon heads don't compose**, so they
avoid that attractor by construction, while still yielding every horizon.

## The window: configurable, rolling, bounded

The context window is **configurable and rolling**, and both are already true in
the code — with one caveat worth stating so we don't trip over it later:

- **Configurable.** `context_length` is a persisted `PredictiveCortexConfig`
  field (per-organism, saved with the checkpoint).
- **Rolling.** The windowed backbones keep a ring buffer (`_slide_window`,
  `brain/cortex/backbones.py:126`); at the live tick the window slides over the
  stream — the last `k` frames, oldest dropped. Correct by construction.
- **Two distinct knobs.** The **buffer capacity** (`context_length_max`) is fixed
  at build time; the **curriculum** ramps the *effective* width 1→k during
  training (`set_context_length`). "Configurable" is build-time; "rolling" is
  runtime, up to that max.
- **Caveat — length extrapolation.** Positions use a learned
  `nn.Embedding(context_length)` (`backbones.py:201`), so the window cannot roll
  *beyond* the trained max at inference without swapping to a relative /
  extrapolating position encoding (RoPE/ALiBi). A rolling window is therefore a
  **bounded** memory horizon — which is exactly what motivates content-addressed
  recall below.

## Memory recall: loading stored tokens (retrieval-augmented cortex)

A rolling window forgets everything older than `k`. The organism already has a
longer store — the **hippocampus** — but today it is read only *offline*, as
dreams during sleep. This section proposes reading it *online*: when the present
resembles a stored episode, **load the matched stored tokens into the cortex's
context** so it predicts using remembered dynamics. This is the "hippocampal
retrieval" the [implementation plan](02-implementation-plan.md) explicitly
defers, and mechanically it is a **memory-augmented transformer** (cf.
Memorizing Transformers' kNN-over-past-keys, RETRO's chunked cross-attention,
kNN-LM). It completes the Complementary-Learning-Systems triangle *at inference*,
not just at sleep:

| Timescale | Organ | Reaches the cortex how |
|---|---|---|
| seconds | rolling window (working memory) | local self-attention (this proposal) |
| a session / a day | **hippocampus** (stored tokens) | **retrieval → context injection (new)** |
| the life | cortex weights | consolidated by sleep |

Design shape:

- **Key/value = the workspace latent `z`.** The hippocampus already prioritizes
  seeds by surprise / reward / novelty (`brain/hippocampus.py`) — good retrieval
  keys. Query by cosine similarity to the current `z_t`.
- **Injection, two options.** *Prepend* the retrieved tokens to the context
  window (simplest — reuses the causal attention, retrieved segment un-masked
  among itself), or a separate **cross-attention block** (RETRO-style) that keeps
  the local window fixed-cost and scales better to many recalled tokens.
- **Who consumes it.** The **cortex** (sharper prediction), and — powerfully —
  the **amygdala / arbiter** downstream: recalling a token from "the place I got
  hurt" raises predicted pain *before* the threat re-arrives (hippocampus →
  amygdala pattern completion — on-biology).
- **Guardrails.** Retrieval must be **gated** (recalling irrelevant tokens is a
  new hallucination surface — gate on similarity + the same calibrated-surprise
  signal the arbiter uses). And stored tokens carry the **provenance/version**
  problem from the dream-bootstrap paradox: a token encoded by a stale cortex
  must not silently poison a newer one — stamp seeds with the cortex version and
  prefer recent/consolidated keys.

This also reframes **dreams and recall as one mechanism at two duty cycles**:
sleep replays retrieved tokens generatively (consolidation); wake injects them
into context (recall). Same store, same tokens.

## What changes in the code

The architecture is already ~80% here — the gap is that the parallelism is
computed and then discarded.

1. **Backbone: expose a parallel path.** Today `TransformerBackbone.step`
   (`brain/cortex/backbones.py:210`) runs causal attention over the window but
   returns only `encoded[:, -1]` — one position. Add
   `forward_sequence(inputs) -> Tensor[B, T, hidden]` returning **every**
   position's hidden. Transformer and dilated-conv do this truly in parallel;
   the GRU implements it as a scan (same result, sequential). `step`/`readout`
   stay for closed-loop rollout and the live tick.
2. **Cortex: per-position multi-horizon heads.** Apply `latent_head` (and the
   reward/terminal/risk/uncertainty heads — which the audit found *untrained*;
   this objective is where they finally get a loss) at every position, for every
   configured horizon.
3. **Training loop: one causal pass, all-positions loss.** Replace the
   windowed warmup+rollout in `train_action_world_model` with: encode the whole
   episode once, `forward_sequence`, gather targets `z_{i+h}` (already on
   disk — the free self-supervised signal), and average latent+pixel loss over
   all `(i, h)`. Keep a **small** closed-loop scheduled-sampling term to fight
   exposure bias.
4. **Curriculum unchanged.** `set_context_length` still ramps the causal mask
   width 1→k, so the model learns one-step dynamics before exploiting long
   context.

## Risks / subtleties

- **Exposure bias.** Teacher-forced parallel training ≠ closed-loop inference.
  Mitigation: retain short scheduled-sampling / closed-loop fine-tune, and keep
  the **frozen-rollout detector** as the honesty gate.
- **Compute / memory.** O(T²) attention per episode — chunk long episodes; the
  context-length curriculum bounds the mask early.
- **Positional extrapolation.** `position_embedding` is `nn.Embedding(context_length)`
  — fine within the window; use RoPE/ALiBi if you want inference beyond the
  trained window.
- **Representation collapse** persists as a risk (shared-encoder latent target);
  pair with the **EMA target-encoder** the audit recommends, and promote
  `linear_probe_yaw` to a gate.

## Milestone / A/B

Same interface, different objective ⇒ an **A/B, not a fork** (matches the
backbone A/B already sanctioned in Phase 2). Success = the all-positions parallel
objective matches or beats the windowed-rollout loop on the Milestone-2 gate
(beats copy-last at every horizon; ablating actions hurts) **per unit compute**,
with no frozen-rollout flag. Add it to `notebooks/build_and_diagnose_organism.ipynb`
as a training-objective toggle beside the backbone selector.
