import { QualityBadge } from "./QualityBadge.jsx";

function meta(session) {
  return [session.scenario, session.split, session.seed != null ? `seed ${session.seed}` : null].filter(Boolean).join(" · ");
}

/** Sessions grouped under one organism/run, each with its data-quality
 * verdict shown before you ever train on it (phase-8 acceptance criterion). */
export function SessionList({ sessions, selectedId, onSelect }) {
  return (
    <ul className="session-list" aria-label="Recordings">
      {sessions.map((session) => (
        <li key={session.id}>
          <button type="button" className={session.id === selectedId ? "is-current" : ""} onClick={() => onSelect(session.id)}>
            <span className="session-id">{session.id}</span>
            {meta(session) && <span className="session-meta">{meta(session)}</span>}
            <QualityBadge quality={session.quality} />
          </button>
        </li>
      ))}
    </ul>
  );
}
