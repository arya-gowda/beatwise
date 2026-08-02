import { useState } from 'react'
import type { Point } from './useMapData'
import { formatArtists } from './Tooltip'
import ExportBar from './ExportBar'

/**
 * Rows rendered at once.
 *
 * A lasso can legitimately catch the whole library, and 2,389 rows of DOM turns the
 * clear button sluggish for a list nobody scrolls to the bottom of. The cap keeps the
 * panel responsive; the count above it always reports the true total, so the number the
 * user acts on is never the truncated one.
 *
 * IT CAPS RENDERING AND NOTHING ELSE. `tracks` is the whole selection and is what gets
 * exported; `shown` exists only to build DOM. Anything that acts on the selection takes
 * `tracks`.
 */
const MAX_ROWS = 300

type Props = {
  tracks: Point[]
  onClear: () => void
  /** Passed through to the export, which records it on the playlist. */
  embeddingVersion: string
}

export default function SelectionPanel({ tracks, onClear, embeddingVersion }: Props) {
  // Separate from clearing on purpose: a lasso caught near the top-right corner of the
  // map puts the panel right on top of the thing you just selected, and "make it go
  // away" should not mean "throw away my selection". Collapsing is a session preference,
  // not tied to any one selection -- it stays collapsed until you reopen it, same as a
  // minimised window.
  const [collapsed, setCollapsed] = useState(false)

  if (tracks.length === 0) return null

  const count = (
    <span className="panel__count">
      {tracks.length.toLocaleString()} track{tracks.length === 1 ? '' : 's'}
    </span>
  )

  if (collapsed) {
    return (
      <button
        className="panel panel--collapsed"
        onClick={() => setCollapsed(false)}
        title="Show the selection"
      >
        {count}
        <span className="panel__expand">show</span>
      </button>
    )
  }

  const shown = tracks.slice(0, MAX_ROWS)
  const hidden = tracks.length - shown.length

  return (
    <aside className="panel">
      <header className="panel__head">
        {count}
        <div className="panel__actions">
          <button
            className="panel__hide"
            onClick={() => setCollapsed(true)}
            title="Hide the panel without losing the selection"
          >
            hide
          </button>
          <button className="panel__clear" onClick={onClear} title="Clear the selection (Esc)">
            clear
          </button>
        </div>
      </header>

      <ol className="panel__list">
        {shown.map((t) => (
          <li key={t.uri} className="panel__row">
            <span className="panel__track">{t.name}</span>
            <span className="panel__artist">{formatArtists(t.artists)}</span>
          </li>
        ))}
      </ol>

      {hidden > 0 && (
        // Wording is load-bearing now that a Create playlist button sits directly beneath
        // it. "+ N more, not listed" was true of the list and would be read as a claim
        // about the playlist -- "and those are left out too" -- which is false. Say which
        // number is the list's and which is the selection's, in that order.
        <p className="panel__more">
          listing {shown.length.toLocaleString()} of {tracks.length.toLocaleString()} · the
          playlist gets all {tracks.length.toLocaleString()}
        </p>
      )}

      <ExportBar tracks={tracks} embeddingVersion={embeddingVersion} />
    </aside>
  )
}
