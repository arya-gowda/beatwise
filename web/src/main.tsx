import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import SpotifyHeader from './spotify/SpotifyHeader.tsx'

// The account corner is mounted beside App rather than inside it: auth has nothing to do
// with the map, and it should still be usable on the screens where the map failed to
// load. Keeping it out of App.tsx also keeps this ticket clear of the map work.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SpotifyHeader />
    <App />
  </StrictMode>,
)
