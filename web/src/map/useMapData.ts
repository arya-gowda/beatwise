/**
 * The one place the artifact is read.
 *
 * No component parses the payload directly. That boundary is what keeps the renderer
 * swappable — if deck.gl ever needs to be replaced, the coords -> render seam is the
 * only thing that has to be understood.
 */
import { useEffect, useState } from 'react'

export type Point = {
  uri: string
  name: string
  artists: string
  album: string
  x: number
  y: number
  popularity: number | null
  release_date: string
  added_at: string
  tempo: number
  explicit: boolean
  genres: string[]
}

export type Manifest = {
  embedding_version: string
  n_tracks: number
  n_dropped: number
  key_encoding: string
  features: string[]
  umap: Record<string, unknown>
}

export type MapData = { version: string; manifest: Manifest; points: Point[] }

export type MapState =
  | { state: 'loading' }
  | { state: 'ready'; data: MapData }
  | { state: 'error'; detail: string }

export function useMapData(): MapState {
  const [state, setState] = useState<MapState>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    fetch('/api/map')
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }))
          throw new Error(body.detail ?? `API returned ${res.status}`)
        }
        return (await res.json()) as MapData
      })
      .then((data) => {
        if (cancelled) return
        // The artifact promises how many tracks it holds. If the payload disagrees,
        // something truncated it, and a silently short map is worse than an error.
        if (data.points.length !== data.manifest.n_tracks) {
          throw new Error(
            `artifact says ${data.manifest.n_tracks} tracks but ${data.points.length} arrived`,
          )
        }
        setState({ state: 'ready', data })
      })
      .catch((err) => !cancelled && setState({ state: 'error', detail: String(err) }))
    return () => {
      cancelled = true
    }
  }, [])

  return state
}

/** Bounding box of the cloud, used to frame the initial view. */
export function bounds(points: Point[]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of points) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY }
}
