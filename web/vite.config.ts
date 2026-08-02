import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxying /api to the FastAPI service keeps both halves same-origin in development,
// so there is no CORS middleware to configure now and unwind later.
export default defineConfig({
  plugins: [react()],
  server: {
    // Pinned to the loopback literal, not localhost. Spotify does not permit `localhost`
    // as a redirect URI, and because the two are different browser origins with separate
    // storage, browsing to localhost would put the PKCE verifier somewhere the callback
    // cannot read it. Setting `host` makes the URL Vite prints the one that works.
    // See docs/decisions/0003-spotify-auth.md.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
