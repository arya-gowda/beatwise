import { useEffect, useState } from 'react'
import './App.css'

type Health = { status: string; service: string }
type Probe =
  | { state: 'checking' }
  | { state: 'ok'; health: Health }
  | { state: 'down'; detail: string }

export default function App() {
  // The scaffold's job is to prove the Vite <-> FastAPI seam works, not merely that
  // both processes start. A page that renders without reaching the API is a page that
  // hides a broken proxy until the first real feature needs it.
  const [probe, setProbe] = useState<Probe>({ state: 'checking' })

  useEffect(() => {
    let cancelled = false
    fetch('/api/health')
      .then(async (res) => {
        if (!res.ok) throw new Error(`API returned ${res.status}`)
        return (await res.json()) as Health
      })
      .then((health) => !cancelled && setProbe({ state: 'ok', health }))
      .catch((err) => !cancelled && setProbe({ state: 'down', detail: String(err) }))
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main className="shell">
      <h1>Beatwise</h1>
      <p className="tagline">A map of your music, not a list.</p>
      <div className={`probe probe--${probe.state}`}>
        {probe.state === 'checking' && <span>contacting api…</span>}
        {probe.state === 'ok' && (
          <span>
            api reachable — {probe.health.service} reports {probe.health.status}
          </span>
        )}
        {probe.state === 'down' && (
          <span>api unreachable — {probe.detail}. Is uvicorn running on port 8000?</span>
        )}
      </div>
    </main>
  )
}
