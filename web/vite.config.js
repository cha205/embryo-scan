import { resolve } from 'node:path'
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    target: 'es2022', // top-level await
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'index.html'),
        analyze: resolve(__dirname, 'analyze.html'),
        index: resolve(__dirname, 'prototypes/index.html'),
        specimen: resolve(__dirname, 'prototypes/specimen.html'),
        exhibit: resolve(__dirname, 'prototypes/exhibit.html'),
        relief: resolve(__dirname, 'prototypes/relief.html'),
      },
    },
  },
})
