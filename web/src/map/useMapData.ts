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

// --- genre -----------------------------------------------------------------------------
//
// A SECOND artifact on a SECOND version line, joined to the map on `track_uri` and nothing
// else (docs/decisions/0004-genre-artifact.md). It is loaded here rather than in a
// component for the same reason the map is: one place parses a payload.
//
// It is loaded SEPARATELY rather than folded into `/map` because the two can legitimately
// disagree about which version is current, and because an unreachable genre artifact must
// cost colour and not the map. A genre rebuild happens whenever the taxonomy is edited; an
// embedding rebuild moves every point. Making the second wait on the first would tie them
// together in exactly the way 0004 exists to prevent.

export type GenreRollup = {
  track_uri: string
  genre_micro_primary: string | null
  genre_micro_all: string[]
  genre_label_count: number
  genre_status: string
  /** The family that colours the point. Null means unlabelled — never "other". */
  genre_macro_primary: string | null
  /** Every family the track's labels touch, primary first. 177 tracks carry more than
   *  one, and that is information the source gave us, not noise to collapse. */
  genre_macro_set: string[]
  genre_macro_count: number
}

export type GenreFamily = { id: string; name: string }

export type GenreManifest = {
  genre_version: string
  n_tracks: number
  n_labelled_tracks: number
  n_unlabelled_tracks: number
  n_tracks_multi_macro: number
  coverage: number
  macro_families: GenreFamily[]
  macro_primary_counts: Record<string, number>
  macro_set_counts: Record<string, number>
  source_sha256: string
  user_id: string
}

/** The shape the renderer wants: a URI -> rollup lookup, built once. A `find` over 2,389
 *  rows per point per frame is fine today and is not fine at the discovery corpus, and the
 *  cheap structure costs nothing now. */
export type GenreIndex = {
  version: string
  manifest: GenreManifest
  byUri: Map<string, GenreRollup>
}

export type GenreState =
  | { state: 'loading' }
  | { state: 'ready'; index: GenreIndex }
  /** Not an error state for the app: the map renders, the genre mode is simply not
   *  offered. `detail` is kept so a future diagnostic can say why. */
  | { state: 'unavailable'; detail: string }

export function useGenreData(): GenreState {
  const [state, setState] = useState<GenreState>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    fetch('/api/genres')
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }))
          throw new Error(body.detail ?? `API returned ${res.status}`)
        }
        return (await res.json()) as {
          version: string
          manifest: GenreManifest
          tracks: GenreRollup[]
        }
      })
      .then((data) => {
        if (cancelled) return
        if (data.tracks.length !== data.manifest.n_tracks) {
          throw new Error(
            `genre artifact says ${data.manifest.n_tracks} tracks but ` +
              `${data.tracks.length} arrived`,
          )
        }
        const byUri = new Map<string, GenreRollup>()
        for (const t of data.tracks) byUri.set(t.track_uri, t)
        setState({
          state: 'ready',
          index: { version: data.version, manifest: data.manifest, byUri },
        })
      })
      .catch(
        (err) => !cancelled && setState({ state: 'unavailable', detail: String(err) }),
      )
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
