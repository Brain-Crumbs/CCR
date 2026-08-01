import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The clinic stays a single deployable: `npm run build` here writes the
// production bundle into ../public, which viewer/server.js already serves
// as static files alongside its read-only JSON API. `npm run dev` instead
// proxies /api to that same Node server so the API and its data-loading
// contract are never duplicated in the front end.
export default defineConfig({
  plugins: [react()],
  root: import.meta.dirname,
  build: {
    outDir: "../public",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8787",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
    globals: false,
  },
});
