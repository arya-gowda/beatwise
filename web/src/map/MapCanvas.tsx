import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import {
  OrthographicView,
  OrthographicViewport,
  type OrthographicViewState,
  type PickingInfo,
} from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'

import { bounds, type Point } from './useMapData'
import Tooltip from './Tooltip'
import SelectionPanel from './SelectionPanel'
import {
  appendVertex,
  MIN_DRAG_PX,
  orderByCentroid,
  pathExtent,
  selectWithin,
  type Vec2,
} from './lasso'

const PADDING = 0.9 // leave a margin so edge clusters are not flush against the frame

const UNSELECTED: [number, number, number] = [190, 190, 200]
// While a selection exists the rest of the library recedes rather than disappearing --
// context is the whole point of a map, and a selection you cannot place is just a list.
const DIMMED: [number, number, number] = [88, 91, 104]
const SELECTED: [number, number, number] = [122, 214, 255]
const HOVERED: [number, number, number] = [255, 255, 255]

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

  // Armed by the button; Shift is the transient shortcut for the same thing, so the
  // pointer can select without leaving navigate mode.
  const [armed, setArmed] = useState(false)
  const [shiftHeld, setShiftHeld] = useState(false)
  const [path, setPath] = useState<Vec2[]>([])
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const frame = useRef<HTMLDivElement>(null)
  const drawing = useRef(false)
  // The ref is the source of truth; `path` exists only to drive the overlay. Reading the
  // state variable on pointerup would risk missing the last move if React batched the two
  // together, which quietly clips the end of the shape.
  const pathRef = useRef<Vec2[]>([])
  // Held so a gesture can be torn down by something other than the pointer that started
  // it -- Escape, or a browser-issued pointercancel.
  const capturedPointer = useRef<number | null>(null)

  const lassoMode = armed || shiftHeld

  // First contact: nobody has drawn this shape before, so the button gets a couple of
  // quiet pulses on a session that has never armed the lasso, then never again. An
  // ambient nudge, not a tutorial wall. Suppressed for good the first time lasso mode is
  // entered, through either door (button or Shift).
  const [hintSeen, setHintSeen] = useState(() => {
    try {
      return localStorage.getItem('beatwise:seen-lasso') === '1'
    } catch {
      return true // storage unavailable (private mode etc.) -- default to no animation
    }
  })
  useEffect(() => {
    if (!lassoMode || hintSeen) return
    setHintSeen(true)
    try {
      localStorage.setItem('beatwise:seen-lasso', '1')
    } catch {
      // Nothing to do -- the hint just won't persist across reloads.
    }
  }, [lassoMode, hintSeen])

  const resetView = useCallback(() => setViewState(home), [home])
  const clearSelection = useCallback(() => setSelected(new Set()), [])

  /**
   * Tear down an in-flight gesture without committing it.
   *
   * The pointer handlers gate on `drawing.current`, a ref, so anything that only flips
   * React state leaves the drag very much alive. Escape used to do exactly that: the chip
   * turned off, the selection cleared, and then the still-live drag finalised on release
   * and repopulated the selection Escape had just cleared -- while the UI had claimed
   * lasso mode was off the whole time.
   */
  const cancelDrag = useCallback(() => {
    if (!drawing.current) return
    drawing.current = false

    const id = capturedPointer.current
    capturedPointer.current = null
    if (id !== null && frame.current?.hasPointerCapture(id)) {
      frame.current.releasePointerCapture(id)
    }

    pathRef.current = []
    setPath([])
  }, [])

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Shift') setShiftHeld(true)
      if (e.key === 'Escape') {
        // Order matters: kill the gesture first, or it finalises over the top of the
        // clear a moment later.
        cancelDrag()
        setArmed(false)
        clearSelection()
      }
    }
    const up = (e: KeyboardEvent) => e.key === 'Shift' && setShiftHeld(false)
    // Losing the keyup while the window is unfocused would leave the map permanently
    // stuck in lasso mode with no visible cause.
    const blur = () => setShiftHeld(false)

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [clearSelection, cancelDrag])

  /** Pointer position relative to the canvas, which is what unproject expects. */
  const toLocal = (e: React.PointerEvent): Vec2 => {
    const r = frame.current?.getBoundingClientRect()
    return [e.clientX - (r?.left ?? 0), e.clientY - (r?.top ?? 0)]
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (!lassoMode || e.button !== 0) return
    // The controls and the selection panel are children of this div, so their clicks
    // bubble here. Without this guard, clicking `clear` while lasso mode is live starts a
    // drag and captures the pointer -- which retargets the pointerup and can swallow the
    // button's own click. P1-07 puts `Create playlist` in that same subtree.
    if ((e.target as HTMLElement).closest('button')) return

    drawing.current = true
    capturedPointer.current = e.pointerId
    e.currentTarget.setPointerCapture(e.pointerId)
    pathRef.current = [toLocal(e)]
    setPath(pathRef.current)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawing.current) return
    pathRef.current = appendVertex(pathRef.current, toLocal(e))
    setPath(pathRef.current)
  }

  const onPointerUp = (e: React.PointerEvent) => {
    if (!drawing.current) return
    drawing.current = false
    capturedPointer.current = null
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }

    const drawn = pathRef.current
    pathRef.current = []
    setPath([])
    // A click that wandered a few pixels is not a lasso. Without this, every stray click
    // on the map silently wipes the selection the user just made.
    if (drawn.length < 3 || pathExtent(drawn) < MIN_DRAG_PX) return

    // Unproject ONCE, here, against the viewport the shape was actually drawn over.
    // From this point the selection is a set of URIs and no longer depends on the view.
    const z = (Array.isArray(viewState.zoom) ? viewState.zoom[0] : viewState.zoom) ?? 0
    const viewport = new OrthographicViewport({
      width,
      height,
      target: (viewState.target ?? [0, 0, 0]) as [number, number, number],
      zoom: z,
      flipY: false,
    })
    const world = drawn.map((v) => viewport.unproject(v).slice(0, 2) as Vec2)
    setSelected(selectWithin(points, world))
    // The sticky toggle is one-shot: a completed drag IS the selection. Leaving `armed`
    // on would mean the very next drag on the canvas -- almost always an attempt to pan
    // over and look at what was just caught -- silently redraws the lasso and replaces
    // it instead. Shift-mode needs no equivalent: it already disarms itself on keyup.
    setArmed(false)
  }

  const layer = new ScatterplotLayer<Point>({
    id: 'tracks',
    data: points,
    getPosition: (d) => [d.x, d.y],
    getFillColor: (d) => {
      if (hovered?.object?.uri === d.uri) return HOVERED
      if (selected.has(d.uri)) return SELECTED
      return selected.size > 0 ? DIMMED : UNSELECTED
    },
    // A selected point sits a little larger as well as brighter. In the dense core colour
    // alone is not enough to pick a selection out of its neighbours.
    getRadius: (d) => (selected.has(d.uri) ? 3 : 2),
    // Radius in pixels, not world units, so points stay legible at every zoom rather
    // than dissolving as you pull back.
    radiusUnits: 'pixels',
    radiusMinPixels: 1.5,
    opacity: 0.75,
    pickable: true,
    onHover: (info: PickingInfo<Point>) => setHovered(info.object ? info : null),
    updateTriggers: {
      getFillColor: [hovered?.object?.uri ?? null, selected],
      getRadius: selected,
    },
  })

  // Must compare target as well as zoom. Comparing zoom alone leaves the reset button
  // disabled after a pure pan -- which is exactly the "spatial UI loses you" failure the
  // affordance exists to prevent, and it is invisible until you pan without zooming.
  const moved = hasMoved(viewState, home)
  const chosen = useMemo(() => orderByCentroid(points, selected), [points, selected])

  return (
    <div
      ref={frame}
      className="frame"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      // Cancel means DISCARD, not "finish early". Sharing the pointerup handler here
      // committed whatever partial shape existed when the browser yanked the pointer
      // stream -- a context menu or a touch conflict would silently hand back a
      // selection the user never drew.
      onPointerCancel={cancelDrag}
    >
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
        // dragPan is what the lasso takes over; the wheel is deliberately left alone so
        // zoom still works mid-selection.
        controller={{
          scrollZoom: { smooth: true, speed: 0.012 },
          doubleClickZoom: false,
          dragPan: !lassoMode,
        }}
        layers={[layer]}
        // Dots are 2px but a cursor is not that precise. A larger pick radius makes
        // hit-testing usable in the dense core, where a quarter of points sit within
        // ~4px of a neighbour (measured overplot 0.256) and exact targeting is hopeless.
        pickingRadius={6}
        getCursor={({ isDragging }) =>
          lassoMode ? 'crosshair' : isDragging ? 'grabbing' : hovered ? 'pointer' : 'grab'
        }
        style={{ position: 'absolute', inset: '0', background: '#0b0d12' }}
      />

      {/* Screen-space overlay: drawn in the coordinates the pointer already reports, so
          the in-progress shape costs no layer churn and no reprojection per frame. */}
      {path.length > 1 && (
        <svg className="lasso" width={width} height={height}>
          <polygon points={path.map(([x, y]) => `${x},${y}`).join(' ')} />
        </svg>
      )}

      {hovered?.object && !lassoMode && <Tooltip info={hovered} />}

      <SelectionPanel tracks={chosen} onClear={clearSelection} />

      <div className="controls">
        <button
          className={`chip chip--lasso${armed ? ' chip--on' : ''}${
            !hintSeen ? ' chip--hint' : ''
          }`}
          onClick={() => setArmed((a) => !a)}
          title="Drag a shape around a region (or hold Shift and drag)"
        >
          {armed ? 'lasso on' : 'lasso'}
        </button>
        <button className="chip" onClick={resetView} disabled={!moved} title="Fit the whole library">
          {moved ? 'home' : 'whole library'}
        </button>
      </div>
    </div>
  )
}
