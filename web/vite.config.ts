import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxying /api to the FastAPI service keeps both halves same-origin in development,
// so there is no CORS middleware to configure now and unwind later.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
