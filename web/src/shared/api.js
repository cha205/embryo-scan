// Where the photo grader (server/main.py) lives. In development it's the same
// origin (Vite forwards /api to it); the public site sets VITE_API_URL at build
// time to a Hugging Face Space, e.g. https://name-embryo-scan.hf.space.
export const API = import.meta.env.VITE_API_URL ?? ''

// A free Space sleeps when idle and takes about a minute to wake, so every page
// pings it on load: by the time a visitor reaches Analyze it's usually awake.
export function wakeGrader() {
  if (API) fetch(`${API}/api/health`, { mode: 'no-cors' }).catch(() => {})
}
