import { useRef } from "react";

/** A minimal single-series SVG line chart over (tick, value) points, with a
 * cursor line at the shared time position and click-to-seek. */
export function Sparkline({ points, color, tick, onSeek }) {
  const ref = useRef(null);
  if (!points.length) return <div className="no-data">not recorded</div>;

  const values = points.map((p) => p.value);
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const minTick = Math.min(...points.map((p) => p.tick)), maxTick = Math.max(...points.map((p) => p.tick));
  const tickSpan = maxTick - minTick || 1;
  const coords = points.map((p) => `${points.length === 1 ? 50 : (p.tick - minTick) * 100 / tickSpan},${34 - ((p.value - min) / span) * 30}`).join(" ");
  const cursorX = tick != null ? Math.max(0, Math.min(100, (tick - minTick) * 100 / tickSpan)) : null;

  function handleClick(event) {
    if (!onSeek) return;
    const rect = ref.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    onSeek(Math.round(minTick + ratio * (maxTick - minTick)));
  }

  return (
    <>
      <svg ref={ref} className="spark" viewBox="0 0 100 38" preserveAspectRatio="none" role="img"
        aria-label={`${points.length} tick timeline`} onClick={handleClick}>
        <polyline points={coords} fill="none" stroke={color} vectorEffect="non-scaling-stroke" />
        {cursorX !== null && <line className="time-cursor" x1={cursorX} x2={cursorX} y1={1} y2={37} vectorEffect="non-scaling-stroke" />}
      </svg>
      <span className="range">{min.toFixed(3)}–{max.toFixed(3)}</span>
    </>
  );
}
