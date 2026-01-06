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
  optimizeDeps: {
    exclude: ['crypto'],
  },
  define: {
    // Polyfill crypto for Node.js environment
    'crypto.getRandomValues': `function(arr) {
      try {
        return require('crypto').randomFillSync(arr)
      } catch (e) {
        // Fallback for non-Node environments
        for (let i = 0; i < arr.length; i++) {
          arr[i] = Math.floor(Math.random() * 256)
        }
      }
    }`,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
    },
  },
})
