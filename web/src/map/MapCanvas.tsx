import { useCallback, useMemo, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { OrthographicView, type OrthographicViewState, type PickingInfo } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'

import { bounds, type Point } from './useMapData'
import Tooltip from './Tooltip'

const PADDING = 0.9 // leave a margin so edge clusters are not flush against the frame

/**
 * Frame the whole cloud.
 *
 * Orthographic zoom is log2 scale: 2^zoom pixels per world unit. Fitting means solving
 * for the axis that constrains, so a wide-but-short cloud is not cropped vertically.
 */
export function fitView(points: Point[], width: number, height: number): OrthographicViewState {
  const b = bounds(points)
  const zoom = Math.log2(
    Math.min(width / Math.max(b.width, 1e-6), height / Math.max(b.height, 1e-6)) * PADDING,
  )
  return {
    target: [b.minX + b.width / 2, b.minY + b.height / 2, 0],
    zoom,
  }
}

/** Has the view left home, by pan OR by zoom? Either one strands the user. */
export function hasMoved(view: OrthographicViewState, home: OrthographicViewState) {
  const z = (v: OrthographicViewState) => (Array.isArray(v.zoom) ? v.zoom[0] : v.zoom) ?? 0
  if (Math.abs(z(view) - z(home)) > 0.01) return true

  const [ax = 0, ay = 0] = (view.target ?? []) as number[]
  const [bx = 0, by = 0] = (home.target ?? []) as number[]
  // Tolerance in world units, scaled by zoom so it means roughly "one pixel" at any
  // scale rather than being generous when zoomed in and twitchy when zoomed out.
  const tol = 1 / Math.pow(2, z(home))
  return Math.abs(ax - bx) > tol || Math.abs(ay - by) > tol
}

type Props = {
  points: Point[]
  width: number
  height: number
}

export default function MapCanvas({ points, width, height }: Props) {
  const home = useMemo(
    () => fitView(points, width, height),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [points],
  )
  const [viewState, setViewState] = useState<OrthographicViewState>(home)
  const [hovered, setHovered] = useState<PickingInfo<Point> | null>(null)

  const resetView = useCallback(() => setViewState(home), [home])

  const layer = new ScatterplotLayer<Point>({
    id: 'tracks',
    data: points,
    getPosition: (d) => [d.x, d.y],
    getFillColor: (d) => (hovered?.object?.uri === d.uri ? [255, 255, 255] : [190, 190, 200]),
    // Radius in pixels, not world units, so points stay legible at every zoom rather
    // than dissolving as you pull back.
    radiusUnits: 'pixels',
    getRadius: 2,
    radiusMinPixels: 1.5,
    opacity: 0.75,
    pickable: true,
    onHover: (info: PickingInfo<Point>) => setHovered(info.object ? info : null),
    updateTriggers: { getFillColor: hovered?.object?.uri ?? null },
  })

  // Must compare target as well as zoom. Comparing zoom alone leaves the reset button
  // disabled after a pure pan -- which is exactly the "spatial UI loses you" failure the
  // affordance exists to prevent, and it is invisible until you pan without zooming.
  const moved = hasMoved(viewState, home)

  return (
    <>
      <DeckGL
        views={
          // flipY defaults to true (screen convention, y increasing downward), which
          // renders the map as a vertical mirror of pipeline.preview and every
          // matplotlib/plotly view of the same artifact. Keeping y up means the browser
          // and the preview tool always agree, so "compare against the prototype"
          // stays a meaningful check.
          new OrthographicView({ id: 'ortho', flipY: false })
        }
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs as OrthographicViewState)}
        // scrollZoom.smooth keeps the wheel from teleporting you across scales.
        // deck.gl anchors orthographic wheel-zoom at the pointer by default, which is
        // the behaviour that keeps you from losing the thing you were aiming at.
        controller={{ scrollZoom: { smooth: true, speed: 0.012 }, doubleClickZoom: false }}
        layers={[layer]}
        // Dots are 2px but a cursor is not that precise. A larger pick radius makes
        // hit-testing usable in the dense core, where a quarter of points sit within
        // ~4px of a neighbour (measured overplot 0.256) and exact targeting is hopeless.
        pickingRadius={6}
        getCursor={({ isDragging }) => (isDragging ? 'grabbing' : hovered ? 'pointer' : 'grab')}
        style={{ position: 'absolute', inset: '0', background: '#0b0d12' }}
      />
      {hovered?.object && <Tooltip info={hovered} />}
      <button className="reset" onClick={resetView} disabled={!moved} title="Fit the whole library">
        {moved ? 'reset view' : 'whole library'}
      </button>
    </>
  )
}
