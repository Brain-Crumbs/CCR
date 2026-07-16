/** Read-only diagnostic panels for a recorded streams-v2 episode. */
"use strict";

const MODULATORS = ["dopamine", "acetylcholine", "adrenaline"];
const valueOf = (record) => {
  const payload = record?.payload ?? record?.value;
  if (typeof payload === "number") return payload;
  if (payload && typeof payload.value === "number") return payload.value;
  return null;
};
const tickOf = (record, fallback) => Number(record?.tick_index ?? record?.tick ?? record?.seq ?? fallback);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function namedSeries(records, names) {
  return records.flatMap((record, index) => {
    const id = String(record.stream_id || "");
    const name = names.find((candidate) => id === candidate || id.endsWith(`.${candidate}`));
    const value = valueOf(record);
    return name && value !== null ? [{ tick: tickOf(record, index), value }] : [];
  });
}

export function episodeDiagnostics(records = [], decisions = []) {
  const series = Object.fromEntries(MODULATORS.map((name) => [name, namedSeries(records, [name])]));
  const recordedErrors = namedSeries(records, ["prediction_error", "reward_prediction_error"]);
  series.prediction_error = decisions.flatMap((decision, index) => typeof decision.prediction_error === "number"
    ? [{ tick: tickOf(decision, index), value: decision.prediction_error }] : []);
  if (!series.prediction_error.length) series.prediction_error = recordedErrors;
  const modes = [], attention = [];
  const hasDecisionAttention = decisions.some((decision) => decision.attention);
  records.forEach((record, index) => {
    const payload = record.payload ?? record.value ?? record;
    const id = String(record.stream_id || "");
    const mode = id.includes("arbiter") ? (payload?.mode ?? payload?.value ?? (typeof payload === "string" ? payload : null)) : null;
    if (mode) modes.push({ tick: tickOf(record, index), mode: String(mode) });
    // Older/imported recordings may include reasons in the stream payload.
    // Native streams-v2 keeps the complete state on DecisionRecord instead.
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

function sparkline(points, color) {
  if (!points.length) return '<div class="no-data">not recorded</div>';
  const values = points.map((p) => p.value), min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const coords = points.map((p, i) => `${points.length === 1 ? 50 : i * 100 / (points.length - 1)},${34 - ((p.value - min) / span) * 30}`).join(" ");
  return `<svg class="spark" viewBox="0 0 100 38" preserveAspectRatio="none" role="img" aria-label="${points.length} tick timeline"><polyline points="${coords}" fill="none" stroke="${color}" vector-effect="non-scaling-stroke"/></svg><span class="range">${esc(min.toFixed(3))}–${esc(max.toFixed(3))}</span>`;
}

export function renderEEGPanel(diagnostics) {
  const colors = { dopamine: "#7b55c7", acetylcholine: "#167c80", adrenaline: "#c34b36", prediction_error: "#b27a00" };
  const traces = Object.entries(diagnostics.series).map(([name, points]) => `<div class="trace"><strong>${esc(name.replace("_", " "))}</strong>${sparkline(points, colors[name])}</div>`).join("");
  const modes = diagnostics.modes.length ? diagnostics.modes.map((x) => `<li class="mode mode--${esc(x.mode)}"><span>t${esc(x.tick)}</span>${esc(x.mode)}</li>`).join("") : '<li class="no-data">arbiter mode not recorded</li>';
  return `<section class="diagnostic eeg" aria-labelledby="eeg-title"><h3 id="eeg-title">EEG</h3><p>Neuromodulation, error, and arbiter state tick by tick.</p><div class="traces">${traces}</div><ol class="mode-timeline" aria-label="arbiter mode timeline">${modes}</ol></section>`;
}

export function renderAttentionPanel(diagnostics) {
  const rows = diagnostics.attention.map((item) => {
    const reason = item.reasons[item.focus];
    const components = reason?.components ? Object.entries(reason.components).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(" · ") : "reason unavailable";
    return `<tr><td>t${esc(item.tick)}</td><th scope="row">${esc(item.focus)}</th><td>${esc(item.selected.join(", ") || "none")}</td><td>${esc(components)}</td></tr>`;
  }).join("");
  return `<section class="diagnostic attention" aria-labelledby="attention-title"><h3 id="attention-title">Attention / focus</h3><p>Selected streams and the controller's reason breakdown.</p><div class="table-scroll"><table><thead><tr><th>Tick</th><th>Focus</th><th>Attended streams</th><th>Why</th></tr></thead><tbody>${rows || '<tr><td colspan="4" class="no-data">attention was not recorded</td></tr>'}</tbody></table></div></section>`;
}

export function developmentStages(session = {}) {
  const raw = session.development ?? session.ladder ?? session.developmental ?? [];
  const stages = Array.isArray(raw) ? raw : (raw.stages ?? raw.milestones ?? []);
  return stages.map((stage, index) => typeof stage === "string" ? { name: stage, passed: true } : {
    name: stage.name ?? stage.stage ?? `Stage ${index + 1}`, passed: Boolean(stage.passed ?? stage.complete ?? stage.status === "passed"),
    milestones: stage.milestones ?? stage.gates ?? [],
  });
}

export function renderDevelopmentPanel(session) {
  const stages = developmentStages(session);
  const items = stages.map((stage) => `<li class="stage ${stage.passed ? "stage--passed" : "stage--pending"}"><span class="stage-mark" aria-hidden="true">${stage.passed ? "✓" : "○"}</span><strong>${esc(stage.name)}</strong>${stage.milestones?.length ? `<small>${esc(stage.milestones.map((x) => typeof x === "string" ? x : x.name ?? x.gate).join(" · "))}</small>` : ""}</li>`).join("");
  return `<section class="diagnostic development" aria-labelledby="development-title"><h3 id="development-title">Developmental ladder</h3><p>Milestones passed by this organism.</p><ol class="stage-list">${items || '<li class="no-data">no developmental gates recorded</li>'}</ol></section>`;
}

export function mountDiagnostics(root, records, decisions, session) {
  const diagnostics = episodeDiagnostics(records, decisions);
  root.innerHTML = renderEEGPanel(diagnostics) + renderAttentionPanel(diagnostics) + renderDevelopmentPanel(session);
}
