/** Pure transforms from recorded streams/decisions into the EEG, attention,
 * and developmental-ladder panel models. No rendering here -- see the
 * components under src/components for that -- so the shape of a tick's
 * evidence is testable independent of markup. */

const MODULATORS = ["dopamine", "acetylcholine", "adrenaline"];

const valueOf = (record) => {
  const payload = record?.payload ?? record?.value;
  if (typeof payload === "number") return payload;
  if (payload && typeof payload.value === "number") return payload.value;
  return null;
};

const tickOf = (record, fallback) => Number(record?.tick_index ?? record?.tick ?? record?.seq ?? fallback);

function namedSeries(records, names) {
  return records.flatMap((record, index) => {
    const id = String(record.stream_id || "");
    const name = names.find((candidate) => id === candidate || id.endsWith(`.${candidate}`));
    const value = valueOf(record);
    return name && value !== null ? [{ tick: tickOf(record, index), value }] : [];
  });
}

/** {series: {dopamine,acetylcholine,adrenaline,prediction_error: [{tick,value}]},
 *   modes: [{tick,mode}], attention: [{tick,focus,selected,reasons}]} */
export function episodeDiagnostics(records = [], decisions = []) {
  const series = Object.fromEntries(MODULATORS.map((name) => [name, namedSeries(records, [name])]));
  const recordedErrors = namedSeries(records, ["prediction_error", "reward_prediction_error"]);
  series.prediction_error = decisions.flatMap((decision, index) => typeof decision.prediction_error === "number"
    ? [{ tick: tickOf(decision, index), value: decision.prediction_error }] : []);
  if (!series.prediction_error.length) series.prediction_error = recordedErrors;

  const modes = [], attention = [];
  const hasDecisionAttention = decisions.some((decision) => decision.attention);
  const hasDecisionMode = decisions.some((decision) => decision.arbiter_mode?.mode ?? decision.arbiter_mode?.value);
  records.forEach((record, index) => {
    const payload = record.payload ?? record.value ?? record;
    const id = String(record.stream_id || "");
    // Older/imported recordings may include the arbiter mode/attention
    // reasons directly on the stream payload. Native streams-v2 keeps the
    // complete state on DecisionRecord instead -- prefer that source so a
    // recording carrying both never double-counts one tick.
    const mode = !hasDecisionMode && id.includes("arbiter") ? (payload?.mode ?? payload?.value ?? (typeof payload === "string" ? payload : null)) : null;
    if (mode) modes.push({ tick: tickOf(record, index), mode: String(mode) });
    const state = !hasDecisionAttention && id.includes("attention") ? payload : null;
    if (state && (state.focus_stream || state.selected_streams || state.reasons)) {
      attention.push({ tick: Number(state.tick_index ?? tickOf(record, index)), focus: state.focus_stream ?? "none",
        selected: state.selected_streams ?? [], reasons: state.reasons ?? {} });
    }
  });
  decisions.forEach((decision, index) => {
    const tick = tickOf(decision, index);
    const mode = decision.arbiter_mode?.mode ?? decision.arbiter_mode?.value;
    if (mode) modes.push({ tick, mode: String(mode) });
    const state = decision.attention;
    if (state && (state.focus_stream || state.selected_streams || state.reasons)) {
      attention.push({ tick: Number(state.tick_index ?? tick), focus: state.focus_stream ?? "none",
        selected: state.selected_streams ?? [], reasons: state.reasons ?? {} });
    }
  });
  modes.sort((a, b) => a.tick - b.tick);
  attention.sort((a, b) => a.tick - b.tick);
  return { series, modes, attention };
}

/** [{name, passed, milestones}] from a session's recorded developmental ladder. */
export function developmentStages(session = {}) {
  const raw = session.development ?? session.ladder ?? session.developmental ?? [];
  const stages = Array.isArray(raw) ? raw : (raw.stages ?? raw.milestones ?? []);
  return stages.map((stage, index) => typeof stage === "string" ? { name: stage, passed: true, milestones: [] } : {
    name: stage.name ?? stage.stage ?? `Stage ${index + 1}`,
    passed: Boolean(stage.passed ?? stage.complete ?? stage.status === "passed"),
    milestones: stage.milestones ?? stage.gates ?? [],
  });
}

export { tickOf, valueOf };
