// @ts-check
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Local dev: proxy API calls to the Flask server (no CORS needed)
    proxy: {
      '/remove-background': 'http://localhost:5010',
    },
  },
})
