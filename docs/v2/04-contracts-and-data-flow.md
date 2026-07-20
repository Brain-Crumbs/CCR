# V2 Contracts and Data-Flow Reference

This is the precise companion to [03-onboarding-guide.md](03-onboarding-guide.md).
It describes the boundaries that let Worlds, the live runtime, neural training,
sleep, development, recording, and the Clinic evolve independently.

## Contract map

| Boundary | Producer | Consumer | Stable unit |
|---|---|---|---|
| World → runtime | `Program.step()` | sensory bus / synchronizer | `StreamEvent` |
| Runtime → World | policy/motor | `Program.step()` | `motor.command` |
| Window → workspace | synchronizer/memory | attention/fusion | `TickWindow`, `LatentState` |
| Workspace → prediction | live world-model seam | policy/modulation | `Prediction` |
| Pixels/actions → cortex | training or dream caller | `PredictiveCortex` | tensors + backbone state |
| Prediction → behavior | voluntary controller/reflex stack | motor bus | `Action`, `MotorDecision` |
| Wake → episodic memory | live loop | Hippocampus | `Seed` |
| Hippocampus → sleep | dream/replay | cortex/exports | latent rollout |
| Runtime → disk | Recorder | replay/training/Clinic | `streams-v2` |
| Trainer → actor | weight publisher | subscriber | versioned checkpoint |
| Record → Clinic | Node HTTP service | browser panels | JSON endpoints |
| Python → mineflayer | remote backend | Node bridge | JSONL request/response |

## 1. World contract (`Program`)

The primary streams-first surface is:

```python
class Program:
    def stream_catalog(self) -> list[StreamSpec]: ...
    def attach_buses(self, sensory, motor) -> None: ...
    def reset(self, seed: int | None = None) -> None: ...
    def step(self) -> None: ...
    def is_complete(self) -> bool: ...
    def metadata(self) -> ProgramMetadata: ...
```

`initialize`, `observe`, `act`, `reward`, `snapshot`, and `restore` remain for
compatibility and deterministic tooling. The live loop does not call
`observe()`; it reconstructs a policy-facing Observation from latest stream
values.

World invariants:

- `attach_buses()` registers the advertised catalog and publishes an initial
  snapshot;
- `reset(seed)` resets both buses and republishes initial state;
- `step()` drains pending motor events, advances exactly one World tick, and
  publishes post-step events;
- no motor event means NULL, but the World still advances;
- invalid motor input becomes `event.action_rejected`, not an exception;
- action and observation vocabulary belong to the World;
- `ProgramMetadata.deterministic` decides whether re-simulation is promised.

## 2. Action contract

An action is an immutable name plus sorted parameters:

```python
Action.make("SELECT_HOTBAR_SLOT", slot=3).key()
# "SELECT_HOTBAR_SLOT:slot=3"
```

The string key is the disk, checkpoint, and motor-wire identity. `NULL_ACTION`
is `Action("NULL")`.

The motor stream is:

```json
{
  "stream_id": "motor.command",
  "modality": "motor",
  "payload": {"action": "MOVE_FORWARD"}
}
```

`ActionRegistry` separately classifies action names as world-changing,
information-gathering, or both. This lets generic attention/reflex code reason
about action kind without knowing Minecraft or Crafter semantics.

## 3. Stream schema contract

`StreamSpec` advertises a stream:

```json
{
  "stream_id": "body.health",
  "modality": "body",
  "nominal_rate_hz": 1.0,
  "payload_schema": "float 0..9",
  "range": [0.0, 9.0],
  "neutral": 9.0,
  "overflow": null,
  "shape": null
}
```

`StreamEvent` is one sample:

```json
{
  "stream_id": "body.health",
  "modality": "body",
  "timestamp": 0.05,
  "sequence_number": 1,
  "payload": 8.0,
  "confidence": 1.0,
  "source": "crafter",
  "arrived_at": null
}
```

Rules:

- ids are lowercase dotted paths;
- when the first segment is a recognized modality, it matches `modality`;
- `timestamp` is simulated time and participates in hashing;
- `arrived_at` is optional realtime metadata and never participates in hashing;
- sequence numbers are monotonic per stream and reset per episode;
- bus drain order is deterministic by `(timestamp, stream_id, sequence_number)`;
- ndarray payloads hash raw contiguous bytes rather than JSON lists.

Runtime-produced ids use the `internal.*` prefix while declaring the generic
`event` modality. This preserves the existing modality taxonomy while making
interoceptive streams visibly distinct by id.

## 4. Queueing and cadence contract

Streams publish at their native cadence:

- vision: per tick or paced to a target realtime rate;
- body: on change plus heartbeat;
- reward: every World tick;
- events: only when they occur;
- motor: at cognitive cadence.

Realtime queues use `coalesce`, `drop_oldest`, or `block` overflow policies.
Every drop is counted. Fast-forward is the deterministic, lock-free path.

`TickSynchronizer.collect()` produces a `TickWindow` with `tick_index`, a
simulated-time span, ordered `events`, and a `by_stream` grouping. If
`program_ticks_per_cognitive_tick == N`, the World steps `N` times and all
arriving events are collected into one decision window.

## 5. Stream registry and fusion contract

`StreamSpec` describes what a World publishes. `StreamDeclaration` describes
how the organism treats it:

- encoder binding or explicit raw stub;
- latent width;
- fixed/trainable behavior;
- checkpoint key;
- classification: `agent_input`, `aux_debug`, or `privileged`;
- attention metadata.

`StreamRegistry.assert_complete()` is the schema gate. `TemporalFusion` creates
a stable stream-id-ordered vector with named slices:

```python
LatentState(
    vector=[...],
    slices={"body.health": (start, end), ...},
    layout_hash="...",
)
```

Silent streams receive neutral values. The `layout_hash` is recorded and stored
with checkpoints so incompatible feature layouts fail loudly.

## 6. Attention contract

`AttentionController.compute(tick_index, buffer)` returns an `AttentionState`
containing per-stream weights, selected streams, current focus, budget use,
reason breakdowns, and bottom-up capture state.

`off` mode returns uniform weights for agent inputs. `budgeted` mode scores
novelty, trend/error, reward relevance, global risk, recency, boredom, and
declared compute cost. Arbiter mode scales the next tick's total budget:
information-gathering widens it; fight-or-flight narrows it.

## 7. Live world-model contract

The loop consumes the small environment-independent interface:

```python
class WorldModel:
    def predict(self, state: State, memory: Memory) -> Prediction: ...

@dataclass
class Prediction:
    expected_features: dict[str, float]
    risk: float
    p_death: float | None
    predicted_reward: float | None
    next_latent: list[float] | None
    prediction_error: float | None
```

`TrendWorldModel` implements it with vital-stream slopes. The earlier learned
MLP world model has an adapter to this surface. A missing integration seam is a
stateful adapter from `PredictiveCortex` to this exact per-tick `Prediction`
contract.

## 8. Predictive Cortex tensor contract

Construction:

```python
cortex = PredictiveCortex(
    pixel_shape=(H, W, 3),
    action_keys=["NULL", "MOVE_UP", ...],
    config=PredictiveCortexConfig(
        latent_width=32,
        hidden_dim=64,
        horizons_ticks=(1, 4, 8),
        backbone="gru",  # or dilated_conv / transformer
        context_length=8,
    ),
)
```

One transition:

```python
next_latent, next_state = cortex.step(latent, action_index, backbone_state)
```

Closed-loop rollout:

```python
latents, final_state = cortex.rollout(start_latent, actions, initial_state)
# start_latent: [B, L]
# actions:      [B, R]
# latents:      [B, R, L]
```

Multi-horizon output:

```python
out = cortex.forward_horizons(start_latent, actions, initial_state, [1, 4, 8])
prediction = out[4]
```

Each `CortexHorizonPrediction` carries a predicted latent, decoded frame,
predicted reward, terminal logit, non-negative risk, and non-negative
uncertainty. Configured horizons are denominated in ticks and persisted.
Dataset callers convert ticks to recorded frame offsets before
`forward_horizons()`.

## 9. Policy, voluntary motor, and reflex contracts

The live policy seam is:

```python
Policy.emit(state, memory, prediction) -> list[Action]
```

An empty list is NULL. `SingleActionPolicy` maps a returned `NULL_ACTION` to an
empty emission.

The V2 voluntary seam is:

```python
VoluntaryController.choose(state, actions, goal=None) -> Action
```

`MPCController` calls a predictor and scorer once per candidate action under
`torch.no_grad()`, preserves action-space order for ties, and takes no gradient
step.

The full precedence record is:

```json
{
  "voluntary": "MOVE_UP",
  "reflex": {
    "name": "withdraw",
    "action": "MOVE_DOWN",
    "reason": "amygdala:threat>=0.5",
    "priority": 10
  },
  "caregiver_override": null,
  "actuated": "MOVE_DOWN"
}
```

Precedence is `caregiver > highest-priority eligible reflex > voluntary`.
Stimuli belong to the World-facing seam; reflex names, thresholds, priorities,
and actions belong to organism configuration.

## 10. Neuromodulator and Arbiter contracts

Internal scalar streams use the uniform payload `{"value": 0.42}`. Important
ids include:

- `internal.prediction_error`;
- `internal.reward_prediction_error` and `internal.dopamine`;
- `internal.learning_progress`;
- `internal.risk`, `internal.risk_gate`, `internal.safe_novelty`;
- `internal.acetylcholine` and `internal.adrenaline`;
- `internal.arbiter_mode`.

The Arbiter output includes its mode, surprise/pain readings, and calibration
error. A mode changes only after a threshold crossing persists for the
configured hysteresis duration.

## 11. Hippocampus and dream contracts

A seed contains a latent `z`, replayable action keys, `SeedTags`, its computed
priority, tick index, and source session. Tags include reward, terminal/damage,
novelty, surprise, dopamine, and threat.

The store is a top-K min-heap by salience priority. Context-cued similarity
retrieval is not part of the current API.

`dream_latents(seed, length, cortex)` returns `[length, latent_width]` without
consulting the sensory bus. `dream(...)` decodes those latents to a frame
iterator. A seed must contain at least `length` actions present in the cortex's
action vocabulary.

## 12. Generative replay contract

`ReplaySample` contains one starting latent, a fixed-length action sequence,
target latents, and `source = real|dream`.

`GenerativeReplayMixer.mix_batch()` guarantees:

- an empty real reservoir is an error;
- at least one real sample appears in every batch;
- requested dream fraction is a function of measured copy-last quality;
- weak cortex quality produces zero dreams;
- all samples share the same action-sequence length;
- dreams come from a frozen cortex snapshot.

## 13. Checkpoint contract

`NeuralAgentCheckpoint` owns tensor state plus a JSON sidecar. Compatibility is
based on the fusion layout hash and action-space hash/key list. It can permit
strict action-space growth for curriculum migration while rejecting unrelated
layout changes.

Common metadata includes organism name, module architecture, layout/action
hashes, training ticks/statistics, RNG state, and cortex horizons/backbone/
context length when supplied by that module.

Trainer publication is atomic. Monotonic `training_ticks` doubles as the
snapshot version. Concurrent actors subscribe to a separate EMA snapshot; the
raw checkpoint remains the true resume trajectory.

## 14. Record contract (`streams-v2`)

Session layout:

```text
sessions/<organism>-<timestamp>-<policy>/
  session.json
  episode_00000.streams.jsonl
  episode_00000.decisions.jsonl
  episode_00000.summary.json
  frames/
    segment_00000.bin
    segment_00000.index.jsonl
    pinned_segments.json
  <organism>-predictions_episode_00000.json
  <organism>-dream_episode_00000.json
```

`session.json` includes identity, World metadata, action space/hash, full stream
catalog, stream-registry interpretation, rates, seeds, curriculum, and relevant
configuration.

A stream-log line is:

```json
{
  "dir": "sensory",
  "stream_id": "body.health",
  "modality": "body",
  "timestamp": 0.05,
  "seq": 1,
  "confidence": 1.0,
  "source": "crafter",
  "hash": "...",
  "payload": 8.0
}
```

Excluded payloads retain the hash and set `elided: true`. Recorded ndarray
frames use `frame_ref`, `shape`, and `dtype`; bytes live in the bounded binary
frame store.

A decision line records one cognitive tick:

```json
{
  "tick_index": 7,
  "window_span": [0.35, 0.40],
  "n_events_by_stream": {"vision.frame.pixels": 1},
  "motor_emitted": ["<motor-event-hash>"],
  "policy_name": "constant",
  "latency_ms": 0.31,
  "reward_window_total": 0.0,
  "risk": 0.1,
  "prediction_error": 0.2,
  "attention": {},
  "reflex": null,
  "arbiter_mode": {}
}
```

NULL is represented by `motor_emitted: []`, so absence of a motor event never
makes the decision itself invisible.

## 15. Replay contract

For a deterministic World, replay restores the recorded seed/configuration,
injects motor events with the same one-tick alignment, steps the World, and
compares regenerated sensory hashes in order.

Remote Minecraft declares itself nondeterministic and cannot snapshot. Its
sessions remain inspectable and trainable, but re-simulation is skipped.
Learning quality is evaluated statistically rather than through byte identity.

## 16. Clinic HTTP contract

Read-only routes in `viewer/server.js`:

| Method and path | Response |
|---|---|
| `GET /api/sessions?name=Pixel` | session summaries/filter by organism |
| `GET /api/sessions/:id` | metadata, episode streams/decisions, exports, quality |
| `GET /api/sessions/:id/episodes/:episode/streams` | `{records: [...]}` |
| `GET /api/sessions/:id/episodes/:episode/decisions` | `{records: [...]}` |
| `GET /api/sessions/:id/episodes/:episode/frames` | frame index plus base64 bytes |
| `GET /api/sessions/:id/episodes/:episode/predictions` | `pixel-predictions-v1` |
| `GET .../predictions?kind=dream` | dream export |

Frame response:

```json
{
  "session_id": "...",
  "episode_id": "episode_00000",
  "shape": [64, 64, 3],
  "dtype": "uint8",
  "n_frames": 41,
  "frames": [
    {"i": 0, "t": 0.0, "seq": 0, "hash": "...", "data": "<base64>"}
  ]
}
```

Prediction and dream files use `pixel-predictions-v1`, with horizon-keyed
predicted frames and an aligned target-frame list. There are no write/control
endpoints yet.

## 17. Mineflayer bridge contract

The remote backend and Node bridge exchange one JSON object per line over
stdin/stdout. Logging goes to stderr so it cannot corrupt the protocol.

```text
→ {"cmd":"reset","seed":0,"config":{...},"connection":{...}}
← {"ok":true,"tick":0,"dead":false,"death_reason":null,"stats":{}}

→ {"cmd":"step","action":{"name":"MOVE_FORWARD","params":{}}}
← {"ok":true,"events":[],"tick":1,"dead":false,"stats":{...}}

→ {"cmd":"observe","timestamp":0.05}
← {"ok":true,"observation":{"tick":1,"data":{...},"frame":[...],"pixels":[...]}}

→ {"cmd":"close"}
← {"ok":true}
```

Requests are serialized. `ok: false` includes `error`. A crashed or invalid
bridge raises a recoverable episode error: the current episode ends and the next
reset can respawn/reconnect.

## 18. Development-stage contract

`CurriculumStageSpec` declares its World/config, scenario, active senses,
`motor_freedom = frozen|overridden|learned`, active losses, episode counts,
promotion gates, and maximum attempts.

All stage gates must pass. Checkpoint state contains the current stage, attempts,
hold/completion state, and promotion history. Stages carrying one checkpoint
must preserve a compatible stream/fusion layout.

## End-to-end trace: from action string to training target

```text
Policy chooses Action("MOVE_UP")
  → publish motor.command {action: "MOVE_UP"} at cognitive tick t
  → Recorder logs the motor event hash
  → CrafterWorld.step() at t+1 drains and decodes Action.from_key("MOVE_UP")
  → Crafter advances and publishes new pixels/reward/body streams
  → synchronizer groups them in window t+1
  → recorder stores the pixel frame by content hash in FrameStore
  → action-world-model dataset aligns frame_t, action_t, frame_t+1
  → encoder maps frame_t to z_t
  → cortex predicts z_t+1 conditioned on action_t
  → encoded frame_t+1 is the self-supervised target
  → decoder makes the prediction viewable
  → evaluation compares model error with copy-last/oracle
  → prediction export lets the Clinic show seen → predicted → actual → error
```

That chain is the project. Every subsystem exists to keep one part of it
generic, learnable, replayable, or inspectable.
