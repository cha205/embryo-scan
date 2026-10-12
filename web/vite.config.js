import { resolve } from 'node:path'
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    // Grading API: uvicorn server.main:app --port 8000 (in production it serves this app itself)
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: {
    target: 'es2022', // top-level await
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'index.html'),
        analyze: resolve(__dirname, 'analyze.html'),
        compare: resolve(__dirname, 'compare.html'),
        index: resolve(__dirname, 'prototypes/index.html'),
        specimen: resolve(__dirname, 'prototypes/specimen.html'),
        exhibit: resolve(__dirname, 'prototypes/exhibit.html'),
        relief: resolve(__dirname, 'prototypes/relief.html'),
      },
    },
  },
})
