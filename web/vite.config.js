import { resolve } from 'node:path'
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    // grading server in development: uvicorn server.main:app --port 8000
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: {
    target: 'es2022', // top-level await
    rollupOptions: {
      // The design studies in prototypes/ show placeholder values, so they are
      // served by the development server only and are not deployed.
      input: {
        app: resolve(__dirname, 'index.html'),
        analyze: resolve(__dirname, 'analyze.html'),
        compare: resolve(__dirname, 'compare.html'),
      },
    },
  },
})
