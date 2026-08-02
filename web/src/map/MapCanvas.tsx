import { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { OrthographicView } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'

import { bounds, type Point } from './useMapData'

const PADDING = 0.9 // leave a margin so edge clusters are not flush against the frame

/**
 * Frame the whole cloud on load.
 *
 * Orthographic zoom is log2 scale: 2^zoom pixels per world unit. Fitting means solving
 * for the axis that constrains, so a wide-but-short cloud is not cropped vertically.
 */
export function fitView(points: Point[], width: number, height: number) {
  const b = bounds(points)
  const zoom = Math.log2(
    Math.min(width / Math.max(b.width, 1e-6), height / Math.max(b.height, 1e-6)) * PADDING,
  )
  return {
    target: [b.minX + b.width / 2, b.minY + b.height / 2, 0] as [number, number, number],
    zoom,
  }
}

type Props = {
  points: Point[]
  width: number
  height: number
}

export default function MapCanvas({ points, width, height }: Props) {
  const initialViewState = useMemo(
    () => fitView(points, width, height),
    // Deliberately frames once, on the data. Recomputing on resize would yank the view
    // out from under the user mid-interaction.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [points],
  )

  const layer = new ScatterplotLayer<Point>({
    id: 'tracks',
    data: points,
    getPosition: (d) => [d.x, d.y],
    getFillColor: [190, 190, 200],
    // Radius in pixels, not world units, so points stay legible at every zoom rather
    // than dissolving as you pull back.
    radiusUnits: 'pixels',
    getRadius: 2,
    radiusMinPixels: 1.5,
    opacity: 0.75,
  })

  return (
    <DeckGL
      // flipY defaults to true (screen convention, y increasing downward), which renders
      // the map as a vertical mirror of pipeline.preview and every matplotlib/plotly view
      // of the same artifact. Keeping y up means the browser and the preview tool always
      // agree, so "compare against the prototype" stays a meaningful check.
      views={new OrthographicView({ id: 'ortho', flipY: false })}
      initialViewState={initialViewState}
      controller={false}
      layers={[layer]}
      style={{ position: 'absolute', inset: '0', background: '#0b0d12' }}
    />
  )
}
