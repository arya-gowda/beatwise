/**
 * The base of the selection panel: name it, create it, open it.
 *
 * THE TRAP THIS COMPONENT EXISTS TO AVOID: `SelectionPanel` renders at most `MAX_ROWS`
 * rows. `tracks` here is the full, untruncated `chosen` array from `MapCanvas` -- never
 * the panel's `shown` slice. Wiring the export to what is on screen would cap every large
 * selection at 300 tracks while looking like it worked.
 *
 * Order is `orderByCentroid`, already applied by `MapCanvas`. This component reads the
 * array as given and never re-sorts: the panel and the playlist must not disagree about
 * what "first" means. Real sequencing is Phase 2/3.
 *
 * Every state -- in-flight, created, failed -- renders inline, here, below the list. No
 * modal: the map and the selection stay visible while the thing that acts on them works.
 */
import { useEffect, useState } from 'react'
import type { Point } from './useMapData'
import { useSpotifyAuth } from '../spotify/useSpotifyAuth'
import { usePlaylistExport } from '../spotify/usePlaylistExport'

type Props = {
  /** The whole selection, centroid-ordered. Not what is rendered above it. */
  tracks: Point[]
  /** Which map produced this route. Recorded in the playlist description. */
  embeddingVersion: string
}

/** `Beatwise — <n> tracks — <date>`, the PM-specified default. */
export function defaultPlaylistName(count: number, when: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  const date = `${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())}`
  return `Beatwise — ${count.toLocaleString()} track${count === 1 ? '' : 's'} — ${date}`
}

export default function ExportBar({ tracks, embeddingVersion }: Props) {
  const auth = useSpotifyAuth()
  const { state, run, reset } = usePlaylistExport()

  // null means "still the default", so the name keeps tracking the count until the moment
  // the user takes it over -- and keeps their words afterwards.
  const [typed, setTyped] = useState<string | null>(null)
  const name = typed ?? defaultPlaylistName(tracks.length)

  // `tracks` is memoised in MapCanvas on [points, selected], so a new identity means a new
  // lasso. The result below describes the previous selection and must not linger under
  // this one.
  useEffect(() => reset(), [tracks, reset])

  const working = state.phase === 'working'

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (working || tracks.length === 0) return
    run({
      name: name.trim() || defaultPlaylistName(tracks.length),
      description: `Lassoed in Beatwise · nearest to centroid first · embedding ${embeddingVersion}`,
      uris: tracks.map((t) => t.uri),
      isPublic: false,
    })
  }

  if (auth.status === 'unconfigured' || auth.status === 'error') {
    return (
      <div className="panel__export">
        <p className="panel__status panel__status--bad">
          spotify unavailable — {auth.detail}
        </p>
      </div>
    )
  }

  return (
    <div className="panel__export">
      {auth.status === 'connected' ? (
        <form className="panel__form" onSubmit={submit}>
          <input
            className="panel__name"
            type="text"
            value={name}
            aria-label="playlist name"
            spellCheck={false}
            disabled={working}
            onChange={(e) => setTyped(e.target.value)}
          />
          <button
            type="submit"
            className="panel__cta"
            // Disabled ONLY in flight. Never latched off after a success: exporting twice
            // is meant to give two playlists, and that has to stay one click away.
            disabled={working}
            title="Create a private Spotify playlist from this selection"
          >
            {working ? progressLabel(state.added, state.total) : 'create playlist'}
          </button>
        </form>
      ) : (
        <button
          type="button"
          className="panel__cta"
          disabled={auth.status === 'connecting'}
          onClick={auth.connect}
        >
          {auth.status === 'connecting' ? 'connecting…' : 'connect spotify to export'}
        </button>
      )}

      {auth.status === 'disconnected' && (
        // Authorising leaves the page and comes back to a fresh load, which does not
        // restore a lasso. Better said than discovered.
        <p className="panel__status">connecting reloads the app — the lasso is not kept</p>
      )}

      <Result state={state} onReconnect={auth.connect} />
    </div>
  )
}

function progressLabel(added: number, total: number): string {
  return added === 0 ? 'creating…' : `adding ${added.toLocaleString()}/${total.toLocaleString()}…`
}

function Result({
  state,
  onReconnect,
}: {
  state: ReturnType<typeof usePlaylistExport>['state']
  onReconnect: () => void
}) {
  if (state.phase === 'done') {
    const { playlist } = state
    return (
      <p className="panel__status panel__status--good">
        {playlist.added.toLocaleString()} track{playlist.added === 1 ? '' : 's'} added
        {playlist.url && (
          <>
            {' · '}
            <a className="panel__link" href={playlist.url} target="_blank" rel="noreferrer">
              open in spotify
            </a>
          </>
        )}
      </p>
    )
  }

  if (state.phase === 'partial') {
    const { playlist } = state
    return (
      <p className="panel__status panel__status--bad">
        the playlist was created but holds {playlist.added.toLocaleString()} of{' '}
        {playlist.requested.toLocaleString()} tracks — {state.detail}
        {playlist.url && (
          <>
            {' · '}
            <a className="panel__link" href={playlist.url} target="_blank" rel="noreferrer">
              open in spotify
            </a>
          </>
        )}
      </p>
    )
  }

  if (state.phase === 'error') {
    return (
      <p className="panel__status panel__status--bad">
        {state.detail}
        {state.needsReconnect && (
          <>
            {' · '}
            <button type="button" className="panel__link" onClick={onReconnect}>
              reconnect
            </button>
          </>
        )}
      </p>
    )
  }

  return null
}
