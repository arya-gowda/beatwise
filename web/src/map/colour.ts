/**
 * Colour-by: the second lens on the map.
 *
 * Position means sound. Nothing in this file may ever feed back into position — these are
 * read-only accessors over fields the substrate already carried for display (see
 * pipeline/features.py, CARRY). Popularity in particular is deliberately absent from the
 * embedding and present here; that separation is the founding idea of the product, and
 * this module is where it becomes visible.
 *
 * Everything is pure and data-driven so the layer and the legend cannot disagree: both
 * read the same `Scale` object, so a swatch always means what a dot means.
 */
import type { Point } from './useMapData'

export type Rgb = [number, number, number]

export type ColourMode = 'off' | 'popularity' | 'year' | 'tempo' | 'explicit' | 'added'

// --- reserved colours ----------------------------------------------------------------
//
// The interaction owns white and cyan. No palette below may approach either, or a hovered
// or selected point stops being readable as one. Measured minimum CIE76 distance from any
// palette stop: 43 to white, 58 to the selection cyan.

export const UNSELECTED: Rgb = [190, 190, 200]
export const DIMMED: Rgb = [88, 91, 104]
export const SELECTED: Rgb = [122, 214, 255]
export const HOVERED: Rgb = [255, 255, 255]

/** Points the active mode has no value for. Neutral and clearly lighter than DIMMED
 *  (CIE76 27) — absence should be visible, not hidden. Colouring a null as zero would be
 *  a lie about the data. */
const NO_DATA: Rgb = [154, 160, 173]

// --- palettes ------------------------------------------------------------------------
//
// Four sequential ramps, all perceptually ordered and monotone in luminance, all sampled
// from the matplotlib uniform family, all truncated at the dark end so the lowest stop
// still clears the #0b0d12 background (measured contrast >= 1.8:1 at every low end; an
// untruncated viridis bottoms out at 1.28:1, which is invisible at 2-3px).

const hex = (s: string): Rgb => [
  parseInt(s.slice(1, 3), 16),
  parseInt(s.slice(3, 5), 16),
  parseInt(s.slice(5, 7), 16),
]

/** viridis, truncated to t in [0.13, 1]. The prototype's popularity scale. */
const VIRIDIS = ['#472e7c', '#3c4f8a', '#2f6b8e', '#25858e', '#1fa088', '#38b977',
  '#70cf57', '#b8de29', '#fde725'].map(hex)

/** plasma, t in [0.16, 1]. Old is cold, new is hot. */
const PLASMA = ['#5901a5', '#8305a7', '#a72197', '#c5407e', '#dd5e66', '#f07f4f',
  '#fca338', '#fccd25', '#f0f921'].map(hex)

/** inferno, t in [0.30, 0.94]. Slow smoulders, fast blazes. Magma was tried here first
 *  and dropped on measurement: truncated clear of its near-white top it is only 132 units
 *  of CIE76 path long against inferno's 178, and a short ramp is a ramp you cannot read
 *  differences on. */
const INFERNO = ['#6a176e', '#8c2369', '#ab2f5e', '#ca404a', '#e25734', '#f37819',
  '#fb9b06', '#fac42a', '#f2ea69'].map(hex)

/** Hand-built for date added, indigo through teal and green to gold: long-settled adds
 *  sit cool and dark, recent ones glow. Hand-built because the four uniform matplotlib
 *  ramps are spoken for and the only spare one is too short. 165 units of path — nearly
 *  twice the single-hue amber ramp tried first, which put a decade of adds inside a
 *  CIE76 of 8. It shares a stretch of hue path with viridis (mean CIE76 30 apart at equal
 *  t), which is acceptable because only one mode is ever on and the key is on screen. */
const DUSK = ['#3a3d86', '#375c84', '#3a787b', '#489068', '#6fa253', '#9bb245',
  '#c1bc50', '#dece6f', '#f5e39a'].map(hex)

/** Explicitness is boolean, so it gets two colours, not a ramp. Amber against a warm
 *  neutral rather than a red/green pair: the flag is a marking on a minority of tracks
 *  and "not flagged" should read as the absence of a mark. CIE76 75 apart, and the pair
 *  survives every colour vision deficiency because it differs in luminance as well as
 *  hue. The neutral is warm on purpose — the cooler slate tried first landed 28 from the
 *  selection cyan, close enough that an unselected point could read as a selected one at
 *  2-3px. This one is 46 away. */
const EXPLICIT_YES: Rgb = hex('#ff9145')
const EXPLICIT_NO: Rgb = hex('#848478')

/** Sample a ramp. Piecewise linear between stops in sRGB — the stops are close enough
 *  together (9 across the ramp) that the interpolation error against the true colormap is
 *  below a just-noticeable difference. */
export function ramp(palette: Rgb[], t: number): Rgb {
  const u = Math.max(0, Math.min(1, t)) * (palette.length - 1)
  const i = Math.min(palette.length - 2, Math.floor(u))
  const f = u - i
  const a = palette[i]
  const b = palette[i + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ]
}

export function toCss(c: Rgb) {
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

// --- scales --------------------------------------------------------------------------

/**
 * How a value maps onto the 0..1 of the ramp.
 *
 * `linear` is the honest default and the only thing that can reproduce the prototype.
 * `quantile` exists because two of these fields are calendar fields on a library that
 * grew: measured, a linear year ramp puts a whole decile of the library inside a CIE76
 * of 3-6, which is not a colour difference anyone can see on a 2-3px dot. Spacing by rank
 * spends the ramp where the tracks are. It IS a distortion, so a quantile legend carries
 * five value ticks at even positions — reading `1954 · 2012 · 2019 · 2022 · 2025` off an
 * evenly divided bar is the skew disclosing itself rather than being hidden.
 */
export type Transform = 'linear' | 'quantile'

export type Tick = { pos: number; label: string }

export type SequentialScale = {
  kind: 'sequential'
  mode: ColourMode
  blurb: string
  palette: Rgb[]
  min: number
  max: number
  transform: Transform
  /** Null means "this point has no value here" — never 0. */
  value: (p: Point) => number | null
  format: (v: number) => string
  /** Value -> 0..1 along the ramp. */
  position: (v: number) => number
  ticks: Tick[]
  /** Distribution across the domain, for the legend. Linear scales only: under a
   *  quantile transform it is flat by construction and the ticks tell the story instead.
   *  Without it a bare linear gradient would imply a spread the data does not have. */
  bins: number[] | null
  missing: number
  /** Extra plain-language line when the field needed a decision, e.g. partial dates. */
  note: string | null
}

export type CategoricalEntry = { key: string; label: string; colour: Rgb; count: number }

export type CategoricalScale = {
  kind: 'categorical'
  mode: ColourMode
  blurb: string
  entries: CategoricalEntry[]
  keyOf: (p: Point) => string
}

export type OffScale = { kind: 'off'; mode: 'off'; blurb: string }

export type Scale = SequentialScale | CategoricalScale | OffScale

/** Mode order in the picker. Adding a mode is one entry here plus one case in
 *  `buildScale` — P1-12's genre mode is a categorical scale and needs nothing else. */
export const MODES: { id: ColourMode; label: string }[] = [
  { id: 'off', label: 'off' },
  { id: 'popularity', label: 'popularity' },
  { id: 'year', label: 'year' },
  { id: 'tempo', label: 'tempo' },
  { id: 'explicit', label: 'explicit' },
  { id: 'added', label: 'added' },
]

// `Added By` is entirely null in this export (2,389 of 2,389) and is deliberately NOT a
// mode. It is a different column from `Added At`, which is the `added` mode above.

const BINS = 28

/**
 * Release year from a Spotify release date.
 *
 * The export mixes three precisions: `YYYY-MM-DD`, `YYYY-MM`, and bare `YYYY`. Parsing
 * these as instants would pin every partial date to 1 January and quietly place a 1972
 * album earlier in the ramp than a 1972-06-01 one. Reading the leading four digits and
 * colouring at YEAR resolution puts every track on the same footing regardless of how
 * precisely its date was recorded, which is what "handle them without misplacing them"
 * has to mean.
 */
export function releaseYear(release: string): number | null {
  const m = /^(\d{4})/.exec(release ?? '')
  return m ? Number(m[1]) : null
}

/** True when the export gave a year and nothing else. Counted for the legend so the
 *  figure is measured in the browser rather than asserted from the ticket. */
export function isYearOnly(release: string): boolean {
  return /^\d{4}$/.test((release ?? '').trim())
}

function addedAt(p: Point): number | null {
  const t = Date.parse(p.added_at)
  return Number.isFinite(t) ? t : null
}

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

function formatMonth(ms: number) {
  const d = new Date(ms)
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`
}

type SeqSpec = {
  palette: Rgb[]
  blurb: string
  transform: Transform
  /** Ticks under a quantile ramp. Three where the labels are wide, five where they fit. */
  tickCount?: number
  value: (p: Point) => number | null
  format: (v: number) => string
  note?: (points: Point[]) => string | null
}

const SEQUENTIAL: Partial<Record<ColourMode, SeqSpec>> = {
  popularity: {
    palette: VIRIDIS,
    // The founding idea, said out loud where the user can read it.
    blurb: 'spotify popularity. never used for position.',
    // Linear over the data extent, which is exactly what the prototype's plotly view did.
    // This one is not a free choice: the acceptance criterion is that it reproduces that
    // view, and a rank-spaced popularity ramp would not be the same picture.
    transform: 'linear',
    value: (p) => (p.popularity === null ? null : p.popularity),
    format: (v) => String(Math.round(v)),
  },
  year: {
    palette: PLASMA,
    blurb: 'release year. cool is old, hot is new.',
    transform: 'quantile',
    tickCount: 5,
    value: (p) => releaseYear(p.release_date),
    format: (v) => String(Math.round(v)),
    note: (points) => {
      const n = points.filter((p) => isYearOnly(p.release_date)).length
      return n ? `${n.toLocaleString()} give a year only — coloured by that year.` : null
    },
  },
  tempo: {
    palette: INFERNO,
    // Unlike popularity, tempo IS one of the ten features behind the layout. Worth saying:
    // a tempo gradient across a region is partly the map explaining itself.
    blurb: 'tempo in bpm. one of the ten features that set position.',
    // Linear: bpm is a physical quantity and equal steps should look equal. It can afford
    // to be — the spread is wide enough that a linear ramp still separates the deciles.
    transform: 'linear',
    value: (p) => (Number.isFinite(p.tempo) ? p.tempo : null),
    format: (v) => `${Math.round(v)} bpm`,
  },
  added: {
    palette: DUSK,
    blurb: 'when you saved it. cool is long ago, gold is recent.',
    transform: 'quantile',
    tickCount: 3,
    value: addedAt,
    format: formatMonth,
  },
}

/**
 * Rank-spaced breakpoints, deduplicated.
 *
 * Ties collapse: if a fifth of the library shares one value, that value gets one position
 * rather than a flat stretch of ramp nothing else can use. `xs` stays strictly increasing
 * so `positionOf` below never divides by zero.
 */
function quantileBreaks(sorted: number[], k: number) {
  const xs: number[] = []
  const ts: number[] = []
  for (let i = 0; i <= k; i++) {
    const v = sorted[Math.min(sorted.length - 1, Math.round((i / k) * (sorted.length - 1)))]
    const t = i / k
    if (xs.length && v === xs[xs.length - 1]) {
      ts[ts.length - 1] = t
      continue
    }
    xs.push(v)
    ts.push(t)
  }
  if (xs.length === 1) {
    xs.push(xs[0] + 1)
    ts.push(1)
  }
  ts[0] = 0
  ts[ts.length - 1] = 1
  return { xs, ts }
}

/** Value at a percentile of the sorted values, for the legend's ticks. */
function quantileAt(sorted: number[], p: number) {
  return sorted[Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)))]
}

export function buildScale(points: Point[], mode: ColourMode): Scale {
  if (mode === 'explicit') {
    let yes = 0
    for (const p of points) if (p.explicit) yes++
    return {
      kind: 'categorical',
      mode,
      blurb: "spotify's explicit flag.",
      entries: [
        { key: 'yes', label: 'explicit', colour: EXPLICIT_YES, count: yes },
        { key: 'no', label: 'not flagged', colour: EXPLICIT_NO, count: points.length - yes },
      ],
      keyOf: (p) => (p.explicit ? 'yes' : 'no'),
    }
  }

  const spec = SEQUENTIAL[mode]
  if (!spec) return { kind: 'off', mode: 'off', blurb: '' }

  let min = Infinity
  let max = -Infinity
  let missing = 0
  const values: number[] = []
  for (const p of points) {
    const v = spec.value(p)
    if (v === null) {
      missing++
      continue
    }
    values.push(v)
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!values.length) {
    min = 0
    max = 1
  }

  // The domain is always the full data extent. No percentile clipping anywhere: it would
  // drop the 1954 records off the end of the year ramp, and a value that falls outside
  // the key is worse than a value that is hard to tell from its neighbour.
  const span = max - min || 1

  let position: (v: number) => number
  let ticks: Tick[]
  let bins: number[] | null = null
  let note = spec.note?.(points) ?? null

  if (spec.transform === 'quantile') {
    const sorted = [...values].sort((a, b) => a - b)
    const { xs, ts } = quantileBreaks(sorted, 16)
    position = (v) => {
      if (v <= xs[0]) return 0
      // At most 17 segments, so a scan is cheaper than a search. If a corpus of a million
      // points ever makes this hot, precompute a lookup table rather than a binary search.
      for (let i = 1; i < xs.length; i++) {
        if (v <= xs[i]) {
          const f = (v - xs[i - 1]) / (xs[i] - xs[i - 1])
          return ts[i - 1] + (ts[i] - ts[i - 1]) * f
        }
      }
      return 1
    }
    const n = spec.tickCount ?? 5
    ticks = Array.from({ length: n }, (_, i) => {
      const p = i / (n - 1)
      return { pos: p, label: spec.format(quantileAt(sorted, p)) }
    })
    note = note
      ? `${note} even steps by rank, not by value.`
      : 'even steps by rank, not by value.'
  } else {
    position = (v) => (v - min) / span
    ticks = [
      { pos: 0, label: spec.format(min) },
      { pos: 1, label: spec.format(max) },
    ]
    bins = new Array<number>(BINS).fill(0)
    for (const v of values) {
      bins[Math.min(BINS - 1, Math.floor(((v - min) / span) * BINS))]++
    }
  }

  return {
    kind: 'sequential',
    mode,
    blurb: spec.blurb,
    palette: spec.palette,
    min,
    max,
    transform: spec.transform,
    value: spec.value,
    format: spec.format,
    position,
    ticks,
    bins,
    missing,
    note,
  }
}

/** The colour a point carries before any interaction state is applied. */
export function baseColour(scale: Scale, p: Point): Rgb {
  if (scale.kind === 'off') return UNSELECTED
  if (scale.kind === 'categorical') {
    const key = scale.keyOf(p)
    const hit = scale.entries.find((e) => e.key === key)
    return hit ? hit.colour : NO_DATA
  }
  const v = scale.value(p)
  if (v === null) return NO_DATA
  return ramp(scale.palette, scale.position(v))
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ]
}

/** How far an unselected point is pulled toward the dim grey while a selection exists. */
const DIM_STRENGTH = 0.62
/** How far a selected point is pushed toward white. Small — the ring does the work. */
const LIFT_STRENGTH = 0.25

/**
 * Recede a point without erasing what it encodes.
 *
 * The original behaviour replaced the fill outright with one flat grey. Under colour-by
 * that would delete the entire signal the mode exists to show the moment anything is
 * selected — you would lasso a region to inspect it and the map around it would go blank.
 *
 * Contracting toward the dim grey instead of blending toward the background keeps hue and
 * hue ORDER intact while compressing the luminance range hard: measured, the full palette
 * span of 1.8:1..16.9:1 against the background becomes 2.1:1..6.3:1 once dimmed. The
 * field still reads as receded, and a warm point still reads as warm. At strength 1.0
 * this is exactly the old flat grey, which is what `off` mode still uses.
 */
export function dim(c: Rgb): Rgb {
  return mix(c, DIMMED, DIM_STRENGTH)
}

/** Selected points keep their own colour, lifted. The cyan ring, not the fill, is what
 *  says "selected" — see MapCanvas. */
export function lift(c: Rgb): Rgb {
  return mix(c, HOVERED, LIFT_STRENGTH)
}

/**
 * Compose colour-by with hover, selection and dimming. One function so the precedence is
 * stated once: hover beats selection beats dimming.
 */
export function pointColour(
  scale: Scale,
  p: Point,
  opts: { hovered: boolean; selected: boolean; anySelected: boolean },
): Rgb {
  if (opts.hovered) return HOVERED
  // With no colour mode there is nothing to preserve, so the original flat treatment
  // stands — it is stronger, and it is the one P1-05 and P1-07 were verified against.
  if (scale.kind === 'off') {
    if (opts.selected) return SELECTED
    return opts.anySelected ? DIMMED : UNSELECTED
  }
  const base = baseColour(scale, p)
  if (opts.selected) return lift(base)
  return opts.anySelected ? dim(base) : base
}

/** CSS gradient for a sequential legend, sampled from the same ramp the layer uses. */
export function gradientCss(palette: Rgb[]) {
  const stops = palette.map((c, i) => `${toCss(c)} ${((i / (palette.length - 1)) * 100).toFixed(1)}%`)
  return `linear-gradient(90deg, ${stops.join(', ')})`
}
