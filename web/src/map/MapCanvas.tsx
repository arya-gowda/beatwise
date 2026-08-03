import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import {
  OrthographicView,
  OrthographicViewport,
  type OrthographicViewState,
  type PickingInfo,
} from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'

import { bounds, type GenreIndex, type Point } from './useMapData'
import Tooltip, { type GenreBadge } from './Tooltip'
import SelectionPanel from './SelectionPanel'
import Legend from './Legend'
// Every colour the map draws comes from here, including the interaction states -- one
// place decides what a dot looks like, so the legend and the layer cannot disagree.
import { buildScale, inFocus, pointColour, SELECTED, type ColourMode } from './colour'
import {
  appendVertex,
  MIN_DRAG_PX,
  orderByCentroid,
  pathExtent,
  selectWithin,
  type Vec2,
} from './lasso'

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
  /** Embedding artifact version, carried through so an exported playlist can name the
   *  map that produced it. A route is only reproducible if you know which space it ran
   *  through. */
  version: string
  /** The genre artifact, or null when it could not be loaded. Null costs the genre colour
   *  mode and nothing else — see docs/decisions/0004-genre-artifact.md. */
  genre: GenreIndex | null
}

export default function MapCanvas({ points, width, height, version, genre }: Props) {
  const home = useMemo(
    () => fitView(points, width, height),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [points],
  )
  const [viewState, setViewState] = useState<OrthographicViewState>(home)
  const [hovered, setHovered] = useState<PickingInfo<Point> | null>(null)

  // Colour-by is a lens over the same points, so it is state of its own and touches
  // nothing else: switching modes cannot move the camera or change the selection, because
  // neither `viewState` nor `selected` is derived from it.
  const [mode, setMode] = useState<ColourMode>('off')
  const scale = useMemo(() => buildScale(points, mode, genre), [points, mode, genre])
  const colouring = scale.kind !== 'off'

  // Legend focus: one category isolated, everything else receded. Two pieces of state
  // rather than one, because hovering a row must PREVIEW without discarding what is
  // pinned -- moving the mouse off the row goes back to the pinned family, not to nothing.
  const [pinned, setPinned] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const focus = preview ?? pinned
  // A focus key belongs to one scale. Carrying `rock` into the explicit mode would match
  // nothing and silently grey the entire map, so switching lens drops it. Pan, zoom and
  // the selection are untouched -- P1-09's criterion still holds.
  useEffect(() => {
    setPinned(null)
    setPreview(null)
  }, [mode])

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
        // Escape already means "put the map back". A pinned legend category greys most of
        // the library, so it is exactly the kind of state you want one key to undo.
        setPinned(null)
        setPreview(null)
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
    // button's own click.
    //
    // P1-07 added two more kinds of target to that subtree, and buttons alone no longer
    // cover it: the editable playlist name is an `input`, where pointer capture would
    // break click-to-place-caret and drag-to-select-text outright, and the result is an
    // `a`. Anything natively interactive owns its own pointer.
    //
    // P1-09's legend is listed whole rather than by its buttons: it is a block of chrome
    // with gaps and a gradient bar between them, and a drag started on the gap would
    // capture the pointer over a control the user was aiming at.
    if ((e.target as HTMLElement).closest('button, input, a, textarea, select, .legend')) return

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

  /**
   * Draw order, and the reason it is not just `points`.
   *
   * deck.gl draws a ScatterplotLayer in data order, so the last point wins where two
   * overlap — and a measured 0.256 of this cloud is overplotted at the default zoom. With
   * a seven-track family isolated, one or two of those seven sitting UNDER a dimmed
   * neighbour is not a rounding error, it is a fifth of the thing you asked to see. So
   * while a focus is active the in-focus points are moved to the end.
   *
   * A partition, not a sort: O(n) and stable, so nothing else about the drawing changes.
   * At the discovery corpus this is one pass over a million points per focus change, which
   * is fine; what would not be fine is a comparison sort, which is why this is not one.
   */
  const ordered = useMemo(() => {
    if (focus === null) return points
    const out: Point[] = []
    const lit: Point[] = []
    for (const p of points) (inFocus(scale, p, focus) ? lit : out).push(p)
    return out.concat(lit)
  }, [points, scale, focus])

  const layer = new ScatterplotLayer<Point>({
    id: 'tracks',
    data: ordered,
    getPosition: (d) => [d.x, d.y],
    getFillColor: (d) =>
      pointColour(scale, d, {
        hovered: hovered?.object?.uri === d.uri,
        selected: selected.has(d.uri),
        anySelected: selected.size > 0,
        focus,
      }),
    // A selected point sits a little larger as well as brighter. In the dense core colour
    // alone is not enough to pick a selection out of its neighbours. Under a colour mode
    // it needs slightly more room, because the ring below eats into the fill.
    //
    // A focused point grows too, and that is not decoration: reggae is seven dots in 2,389.
    // Dimming the other 2,382 finds them; the extra pixel is what makes them clickable once
    // found. Selection still outranks focus, here as in pointColour.
    getRadius: (d) =>
      selected.has(d.uri)
        ? colouring
          ? 3.4
          : 3
        : focus !== null && inFocus(scale, d, focus)
          ? 3.2
          : 2,
    // Radius in pixels, not world units, so points stay legible at every zoom rather
    // than dissolving as you pull back.
    radiusUnits: 'pixels',
    radiusMinPixels: 1.5,
    // THE SELECTION RING. Under a colour mode the fill has to keep meaning what the
    // legend says it means, so "selected" cannot be carried by hue -- it is carried by a
    // constant cyan outline instead, drawn around the point's own colour. deck.gl centres
    // the stroke on the edge, so a 1.2px ring at radius 3.4 gives an 8px dot with a 5.6px
    // core of real colour: far enough out a selection reads as a cyan mass exactly as it
    // did before, and closer in every selected point still says what it is worth. With no
    // colour mode there is nothing to protect, so the width is 0 -- which the shader
    // treats as no stroke at all -- and P1-05's solid cyan fill stands unchanged.
    stroked: true,
    getLineColor: SELECTED,
    getLineWidth: (d) => (colouring && selected.has(d.uri) ? 1.2 : 0),
    lineWidthUnits: 'pixels',
    // A touch more opaque under a colour mode: hue on a 2-3px dot survives less
    // translucency than a flat grey does.
    opacity: colouring ? 0.85 : 0.75,
    pickable: true,
    onHover: (info: PickingInfo<Point>) => setHovered(info.object ? info : null),
    // Miss one of these and the map keeps the colours of the previous mode.
    updateTriggers: {
      getFillColor: [hovered?.object?.uri ?? null, selected, mode, focus],
      getRadius: [selected, mode, focus],
      getLineWidth: [selected, mode],
    },
  })

  // Must compare target as well as zoom. Comparing zoom alone leaves the reset button
  // disabled after a pure pan -- which is exactly the "spatial UI loses you" failure the
  // affordance exists to prevent, and it is invisible until you pan without zooming.
  const moved = hasMoved(viewState, home)
  const chosen = useMemo(() => orderByCentroid(points, selected), [points, selected])

  // Resolved here so the tooltip stays a renderer: it is handed names and colours, not an
  // index to look things up in. Only under the genre mode -- the card that identifies a
  // track under a tempo ramp has no business listing families.
  const hoveredGenres: GenreBadge[] | null = useMemo(() => {
    if (mode !== 'genre' || !genre || !hovered?.object) return null
    const g = genre.byUri.get(hovered.object.uri)
    const names = new Map(genre.manifest.macro_families.map((f) => [f.id, f.name]))
    // `genre_macro_set` is primary-first and already deduplicated by the pipeline.
    return (g?.genre_macro_set ?? []).map((id) => ({
      key: id,
      label: names.get(id) ?? id,
      colour: scale.kind === 'categorical' ? scale.colourOf(id) : [255, 255, 255],
    }))
  }, [mode, genre, hovered, scale])

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

      {hovered?.object && !lassoMode && <Tooltip info={hovered} genres={hoveredGenres} />}

      {/* `chosen`, not anything the panel renders: the panel caps its list at 300 rows
          and the export must see the whole selection. */}
      <SelectionPanel tracks={chosen} onClear={clearSelection} embeddingVersion={version} />

      <Legend
        mode={mode}
        scale={scale}
        onChange={setMode}
        hasGenre={genre !== null}
        pinned={pinned}
        onPin={setPinned}
        onPreview={setPreview}
      />

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
