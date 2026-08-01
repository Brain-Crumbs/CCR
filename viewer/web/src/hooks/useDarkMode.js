import { useEffect, useState } from "react";

/** Tracks `prefers-color-scheme` so canvas/SVG drawing (which can't rely on
 * CSS custom properties) can pick the matching palette. */
export function useDarkMode() {
  const query = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  const [dark, setDark] = useState(query?.matches ?? false);
  useEffect(() => {
    if (!query) return undefined;
    const onChange = (event) => setDark(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [query]);
  return dark;
}
