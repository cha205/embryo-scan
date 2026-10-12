// Base URL of the grading server. Empty in development, where Vite proxies /api;
// production builds set VITE_API_URL to the hosted server.
export const API = import.meta.env.VITE_API_URL ?? ''

// The hosted server sleeps when idle and takes about a minute to start, so every
// page pings it on load.
export function wakeGrader() {
  if (API) fetch(`${API}/api/health`, { mode: 'no-cors' }).catch(() => {})
}
