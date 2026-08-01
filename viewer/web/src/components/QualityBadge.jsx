/** record/quality.py's green/amber/red verdict for a session -- surfaced
 * before you ever train on it (phase-8 acceptance criterion). */
export function QualityBadge({ quality }) {
  if (!quality) return null;
  const issues = [...(quality.issues || []), ...(quality.warnings || [])];
  return (
    <span className={`quality-badge quality-badge--${quality.verdict}`} title={issues.join("; ") || undefined}>
      {quality.verdict}
    </span>
  );
}
