import { useState } from 'react'
import type { Point } from './useMapData'
import { formatArtists } from './Tooltip'

/**
 * Rows rendered at once.
 *
 * A lasso can legitimately catch the whole library, and 2,389 rows of DOM turns the
 * clear button sluggish for a list nobody scrolls to the bottom of. The cap keeps the
 * panel responsive; the count above it always reports the true total, so the number the
 * user acts on is never the truncated one.
 */
const MAX_ROWS = 300

type Props = {
  tracks: Point[]
  onClear: () => void
}

export default function SelectionPanel({ tracks, onClear }: Props) {
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
        <footer className="panel__more">
          + {hidden.toLocaleString()} more, not listed
        </footer>
      )}
    </aside>
  )
}
