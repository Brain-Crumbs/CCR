/** `#organism/run/session/episode` -- the same shareable-link shape the
 * original vanilla clinic used, so an existing bookmark keeps working. */
export function parseHash(hash) {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const parts = raw.split("/").map((part) => { try { return decodeURIComponent(part); } catch { return part; } });
  return { organism: parts[0] || null, run: parts[1] || null, session: parts[2] || null, episode: parts[3] || null };
}

export function buildHash({ organism, run, session, episode }) {
  return [organism, run, session, episode].map((part) => encodeURIComponent(part)).join("/");
}
