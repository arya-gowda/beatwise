/**
 * Lasso geometry.
 *
 * Everything here works in WORLD coordinates. The caller unprojects the drawn screen
 * path once, on release, and never again. That is what makes a selection survive pan and
 * zoom: the result is a set of track URIs, not a region of the screen. Storing a screen
 * rectangle would be simpler and would silently detach from the tracks the moment the
 * user zoomed in to inspect what they caught -- which is the first thing anyone does.
 */
import type { Point } from './useMapData'

export type Vec2 = [number, number]

/**
 * Minimum screen distance between captured vertices.
 *
 * Pointer events fire far denser than the shape needs. Dropping vertices closer than
 * this keeps a full-canvas drag near a few hundred points instead of a few thousand,
 * which matters because point-in-polygon is O(vertices) per track over 2,389 tracks.
 */
export const MIN_VERTEX_PX = 3

/** A drag shorter than this is a click that wandered, not an attempt to lasso. */
export const MIN_DRAG_PX = 12

/** Append a vertex unless it is close enough to the last one to be redundant. */
export function appendVertex(path: Vec2[], v: Vec2, minDist = MIN_VERTEX_PX): Vec2[] {
  const last = path[path.length - 1]
  if (last && Math.hypot(v[0] - last[0], v[1] - last[1]) < minDist) return path
  return [...path, v]
}

/** Largest span of a screen path, used to tell a real lasso from a stray click. */
export function pathExtent(path: Vec2[]): number {
  if (path.length < 2) return 0
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of path) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  return Math.max(maxX - minX, maxY - minY)
}

export function polygonBounds(poly: Vec2[]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of poly) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  return { minX, minY, maxX, maxY }
}

/**
 * Ray casting, counting crossings of a horizontal ray to the left.
 *
 * The polygon is treated as implicitly closed, so the freehand path does not have to
 * meet its own start -- nobody lands the cursor back on the pixel they began at.
 */
export function pointInPolygon(x: number, y: number, poly: Vec2[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    // The (yi > y) !== (yj > y) test counts each edge once and handles a vertex landing
    // exactly on the ray without double-counting it.
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}

/**
 * Track URIs inside the polygon.
 *
 * The bounding-box prefilter is what keeps this cheap: a lasso around one corner of the
 * map rejects most of the library on two comparisons, so only the plausible candidates
 * pay for the full edge walk.
 */
export function selectWithin(points: Point[], poly: Vec2[]): Set<string> {
  const selected = new Set<string>()
  if (poly.length < 3) return selected

  const b = polygonBounds(poly)
  for (const p of points) {
    if (p.x < b.minX || p.x > b.maxX || p.y < b.minY || p.y > b.maxY) continue
    if (pointInPolygon(p.x, p.y, poly)) selected.add(p.uri)
  }
  return selected
}

/**
 * Selected tracks, nearest to the selection's centroid first.
 *
 * Arbitrary but not random, and deliberately the same order P1-07 exports the playlist
 * in -- the panel should not show one order and the playlist another. Real sequencing is
 * Phase 2/3 work and must not be quietly invented here.
 */
export function orderByCentroid(points: Point[], selected: Set<string>): Point[] {
  const chosen = points.filter((p) => selected.has(p.uri))
  if (chosen.length === 0) return chosen

  let cx = 0, cy = 0
  for (const p of chosen) {
    cx += p.x
    cy += p.y
  }
  cx /= chosen.length
  cy /= chosen.length

  return chosen.sort((a, b) => {
    const da = (a.x - cx) ** 2 + (a.y - cy) ** 2
    const db = (b.x - cx) ** 2 + (b.y - cy) ** 2
    // Tie-break on URI so the order is stable across runs rather than depending on the
    // sort implementation for coincident points -- of which an overplotted map has many.
    return da - db || (a.uri < b.uri ? -1 : a.uri > b.uri ? 1 : 0)
  })
}
