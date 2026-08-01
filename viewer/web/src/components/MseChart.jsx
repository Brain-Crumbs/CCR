import { useMemo, useRef, useState } from "react";
import { SERIES_DARK, SERIES_LIGHT } from "../lib/pixelMath.js";
import { mseSeries } from "../lib/pixelSeries.js";
import { useDarkMode } from "../hooks/useDarkMode.js";

const W = 720, H = 180, PAD = { l: 56, r: 70, t: 10, b: 24 };

/** MSE-over-time chart: one line per horizon, hover tooltip, click-to-seek.
 * This is the "compare model outputs at each horizon" view at a glance --
 * the horizon panels above answer "what does it look like at this tick,"
 * this answers "how does that generalize across the whole episode." */
export function MseChart({ horizons, source, ctx, tCurrent, onSeek }) {
  const dark = useDarkMode();
  const colors = dark ? SERIES_DARK : SERIES_LIGHT;
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const series = useMemo(() => horizons.map((h) => mseSeries(source, h, ctx)), [horizons, source, ctx]);

  const { maxY, n, x, y } = useMemo(() => {
    let maxYValue = 1e-6;
    for (const s of series) for (const v of s) if (!Number.isNaN(v)) maxYValue = Math.max(maxYValue, v);
    const nValue = Math.max(...series.map((s) => s.length), 1);
    const xFn = (i) => PAD.l + (i / Math.max(1, nValue - 1)) * (W - PAD.l - PAD.r);
    const yFn = (v) => H - PAD.b - (v / maxYValue) * (H - PAD.t - PAD.b);
    return { maxY: maxYValue, n: nValue, x: xFn, y: yFn };
  }, [series]);

  const grid = [0.25, 0.5, 0.75, 1].map((f) => {
    const gy = y(maxY * f);
    return (
      <g key={f}>
        <line x1={PAD.l} y1={gy} x2={W - PAD.r} y2={gy} stroke="var(--line)" strokeWidth={1} />
        <text x={PAD.l - 6} y={gy + 4} textAnchor="end" fill="var(--text-secondary)" fontSize={10}>{(maxY * f).toExponential(1)}</text>
      </g>
    );
  });

  const labelYs = series.map((s) => (s.length && !Number.isNaN(s[s.length - 1]) ? y(s[s.length - 1]) + 4 : null));
  const order = labelYs.map((ly, k) => [ly, k]).filter(([ly]) => ly !== null).sort((a, b) => a[0] - b[0]);
  for (let j = 1; j < order.length; j++) if (order[j][0] - order[j - 1][0] < 13) order[j][0] = order[j - 1][0] + 13;
  const labelLimit = H - PAD.b - 2;
  for (let j = order.length - 1; j >= 0; j--) {
    const cap = j === order.length - 1 ? labelLimit : order[j + 1][0] - 13;
    if (order[j][0] > cap) order[j][0] = cap;
  }
  for (const [ly, k] of order) labelYs[k] = ly;

  const paths = series.map((s, k) => {
    let d = "";
    for (let i = 0; i < s.length; i++) { if (!Number.isNaN(s[i])) d += (d ? "L" : "M") + `${x(i).toFixed(1)},${y(s[i]).toFixed(1)}`; }
    return (
      <g key={horizons[k]}>
        <path d={d} fill="none" stroke={colors[k % colors.length]} strokeWidth={2} />
        {labelYs[k] !== null && <text x={x(s.length - 1) + 5} y={labelYs[k].toFixed(1)} fill="var(--text-secondary)" fontSize={11}>t+{horizons[k]}</text>}
      </g>
    );
  });

  const cx = x(Math.min(tCurrent, n - 1));

  function pointerIndex(event) {
    const rect = svgRef.current.getBoundingClientRect();
    return Math.round(((event.clientX - rect.left - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1));
  }

  function handleMove(event) {
    const i = pointerIndex(event);
    if (i < 0 || i >= n) { setHover(null); return; }
    const rect = svgRef.current.getBoundingClientRect();
    setHover({ i, left: event.clientX - rect.left + 14, top: event.clientY - rect.top - 10 });
  }

  function handleClick(event) {
    const i = pointerIndex(event);
    if (i >= 0 && i < n) onSeek(i);
  }

  return (
    <div className="chartwrap">
      <div className="legend">
        {horizons.map((h, k) => (
          <span key={h}><span className="swatch" style={{ background: colors[k % colors.length] }} />MSE @ t+{h}</span>
        ))}
      </div>
      <div className="chartbox">
        <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="prediction error over start frame"
          onMouseMove={handleMove} onMouseLeave={() => setHover(null)} onClick={handleClick}>
          {grid}
          <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="var(--line)" />
          <text x={PAD.l} y={H - 8} fill="var(--text-secondary)" fontSize={10}>t = 0</text>
          <text x={W - PAD.r} y={H - 8} textAnchor="end" fill="var(--text-secondary)" fontSize={10}>{n - 1}</text>
          {paths}
          <line x1={cx} y1={PAD.t} x2={cx} y2={H - PAD.b} stroke="var(--text-secondary)" strokeWidth={1} strokeDasharray="3,3" />
        </svg>
        {hover && (
          <div className="tooltip" style={{ display: "block", left: hover.left, top: hover.top }}>
            <div className="row"><b>t = {hover.i}</b></div>
            {horizons.map((h, k) => {
              const v = series[k][hover.i];
              return (
                <div className="row" key={h}>
                  <span className="swatch" style={{ background: colors[k % colors.length], display: "inline-block", width: 8, height: 8, borderRadius: 2, marginRight: 4 }} />
                  t+{h}: <b>{Number.isNaN(v) || v === undefined ? "–" : v.toExponential(2)}</b>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <details className="table">
        <summary>error table (current t)</summary>
        <table>
          <thead><tr><th>horizon</th><th>MSE ({source})</th><th>PSNR dB</th></tr></thead>
          <tbody>
            {horizons.map((h, k) => {
              const idx = Math.min(tCurrent, series[k].length - 1);
              const v = idx >= 0 ? series[k][idx] : NaN;
              const known = typeof v === "number" && !Number.isNaN(v);
              return <tr key={h}><td>t+{h}</td><td>{known ? v.toExponential(3) : "–"}</td><td>{known ? (v <= 0 ? "∞" : (10 * Math.log10(1 / v)).toFixed(1)) : "–"}</td></tr>;
            })}
          </tbody>
        </table>
      </details>
    </div>
  );
}
