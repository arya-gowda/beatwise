import type { PickingInfo } from '@deck.gl/core'
import type { Rgb } from './colour'
import { toCss } from './colour'
import type { Point } from './useMapData'

const EDGE = 16 // keep the card clear of the viewport edge

/** What the hovered point's genre says, already resolved to names and colours by the
 *  caller — the tooltip renders, it does not look anything up. Primary first, exactly as
 *  `genre_macro_set` orders it. */
export type GenreBadge = { key: string; label: string; colour: Rgb }

/**
 * Spotify exports multi-artist credits semicolon-separated ("Consequence;Kanye West"),
 * which reads as a typo on screen. 484 of 2,389 tracks are affected.
 */
export function formatArtists(raw: string) {
  return raw
    .split(';')
    .map((s) => s.trim())
    .filter(Boolean)
    .join(', ')
}

/**
 * Hover identification.
 *
 * Built against the real library, which has 200-character track titles, multi-artist
 * credits running to a dozen names, and remaster suffixes. The card clamps its own width
 * and flips side near the right edge rather than letting a long title push it off-screen.
 */
export default function Tooltip({
  info,
  genres,
}: {
  info: PickingInfo<Point>
  /** Null in every mode but genre. The card that identifies a track under a popularity
   *  ramp should not start listing families; under the genre mode it is the only place the
   *  177 multi-family tracks can say so, because the dot itself is one colour. */
  genres?: GenreBadge[] | null
}) {
  const track = info.object
  if (!track) return null

  const flipX = info.x > window.innerWidth - 280
  const flipY = info.y > window.innerHeight - 120

  return (
    <div
      className="tooltip"
      style={{
        left: flipX ? undefined : info.x + EDGE,
        right: flipX ? window.innerWidth - info.x + EDGE : undefined,
        top: flipY ? undefined : info.y + EDGE,
        bottom: flipY ? window.innerHeight - info.y + EDGE : undefined,
      }}
    >
      <div className="tooltip__name">{track.name}</div>
      <div className="tooltip__artists">{formatArtists(track.artists)}</div>
      {genres &&
        (genres.length > 0 ? (
          <div className="tooltip__genres">
            {genres.map((g, i) => (
              <span key={g.key} className="tooltip__genre">
                <span
                  className="tooltip__dot"
                  style={{ background: toCss(g.colour) }}
                  aria-hidden="true"
                />
                {g.label}
                {/* The first entry is the one the dot is coloured by. Saying so is what
                    keeps "a point is one colour" from looking like a contradiction on a
                    track that lists three families. */}
                {i === 0 && genres.length > 1 && (
                  <span className="tooltip__primary"> · shown</span>
                )}
              </span>
            ))}
          </div>
        ) : (
          <div className="tooltip__genres tooltip__genres--none">no genre labels</div>
        ))}
    </div>
  )
}
