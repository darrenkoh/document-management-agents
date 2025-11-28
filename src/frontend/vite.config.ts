import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/refresh': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      // Use regex to match /document/ followed by a number (document detail API)
      // This prevents /documents (frontend route) from being proxied
      '^/document/\\d+': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
    },
  },
})
