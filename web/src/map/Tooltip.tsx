import type { PickingInfo } from '@deck.gl/core'
import type { Point } from './useMapData'

const EDGE = 16 // keep the card clear of the viewport edge

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
export default function Tooltip({ info }: { info: PickingInfo<Point> }) {
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
    </div>
  )
}
