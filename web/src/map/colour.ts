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
import type { GenreIndex, Point } from './useMapData'

export type Rgb = [number, number, number]

export type ColourMode =
  | 'off'
  | 'popularity'
  | 'year'
  | 'tempo'
  | 'explicit'
  | 'added'
  | 'genre'

// --- reserved colours ----------------------------------------------------------------
//
// The interaction owns white and cyan. No palette below may approach either, or a hovered
// or selected point stops being readable as one.
//
// Measured (2026-08-02) across every stop of every palette in this file, in CIELAB, on the
// COMPOSITED colour -- 0.85 over #0b0d12, which is what a dot actually is on screen:
//
//   minimum CIE76 to the hover white   39.4   (dusk's top stop)
//   minimum CIE76 to the selection cyan 35.2  (viridis's fourth stop)
//
// P1-12's genre palette is held to those measured floors rather than to the wider ones an
// earlier version of this comment claimed (43 / 58, which no longer described the ramps
// after dusk and the truncated viridis landed). It clears both: 42.3 to white, 40.6 to
// cyan. tests/test_genre_palette.py recomputes all of this rather than trusting the
// comment, which is how the stale figures got caught.

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

/**
 * The twelve macro genre families (docs/decisions/0005-macro-genre-taxonomy.md).
 *
 * Built the way Glasbey et al. (2007) build categorical colours -- farthest-point in a
 * perceptual space against an already-chosen set -- with three departures that this
 * particular map forces:
 *
 *  1. The already-chosen set is not empty. The background, the selection cyan, the hover
 *     white, the dim grey and the unlabelled neutral are all fixed and reserved, and the
 *     twelve had to be found in what was left. That is why there is no mid-cyan here: the
 *     one obvious hole in the wheel belongs to the selection.
 *  2. Everything is measured on the COMPOSITED colour, 0.85 over #0b0d12. Optimising the
 *     literal hex values would tune colours nobody ever sees.
 *  3. Small-field tritanopia. At 2-3px the S-cone contribution collapses for EVERY viewer,
 *     not only for the 0.01% with the deficiency, so tritan-simulated distance is part of
 *     the objective rather than a post-hoc check. A pair may be at most 2x closer under
 *     tritan simulation than it is in normal foveal vision.
 *
 * Structurally it is a hue wheel on two lightness tiers, alternating -- L* 46.9 and 61.3
 * composited, contrast 5.0:1 and 8.6:1 against the background. Alternating means
 * hue-ADJACENT colours also differ in luminance, and luminance is the one channel that
 * survives both a 2px dot and every colour vision deficiency.
 *
 * Measured, composited, CIE76:
 *
 *   minimum pairwise, normal vision   30.0   (electronic vs stage & screen)
 *   minimum pairwise, tritan          15.7
 *   minimum pairwise, deutan           7.2   (r&b vs punk & metal)
 *   minimum pairwise, protan           6.9   (the same pair)
 *   minimum to the selection cyan     40.6
 *   minimum to the hover white        42.3
 *   minimum to the unlabelled neutral 43.5
 *
 * For scale: tab10 measures 24.1 / 13.2 / 6.1 / 5.0 on the same four axes and carries ten
 * categories, not twelve; it also sits 18.9 from the selection cyan, which is why it could
 * not simply be adopted. So twelve is not the problem the ticket wondered it might be --
 * these hold further apart than the palette most of this industry reaches for by default.
 *
 * WHAT IS NOT SOLVED: deuteranopia and protanopia. Twelve categories cannot be made
 * red-green safe -- a dichromat's colour space is essentially two-dimensional, and the
 * accepted ceiling is eight to nine (Okabe-Ito is eight; Paul Tol stops at nine and says
 * so). Four of the 66 pairs fall under CIE76 10 under deutan simulation. The honest
 * mitigation is not a better palette, it is the legend focus control below: isolating one
 * family never requires telling its hue from a neighbour's.
 *
 * WHICH FAMILY GETS WHICH COLOUR IS NOT ARBITRARY. Assignment was optimised jointly with
 * the palette, weighting each pair by w = n_i + n_j -- the product of how often two
 * families meet on screen (~ n_i * n_j) and what a confusion costs per track (1/n_i +
 * 1/n_j). The floor stays unweighted so no two swatches in the legend can collapse, but
 * the headroom above it goes where it earns most: the four largest families sit at least
 * 55.6 apart, and every one of the tiny families sits at least 33.0 from every large one.
 */
const GENRE: Record<string, Rgb> = {
  'hip-hop': hex('#d735d7'),
  'rnb-soul': hex('#d35f01'),
  pop: hex('#75bf01'),
  rock: hex('#fd0167'),
  'punk-metal': hex('#868601'),
  electronic: hex('#d0a568'),
  jazz: hex('#0183e5'),
  latin: hex('#14c2a3'),
  reggae: hex('#32933b'),
  'folk-country': hex('#ee8dc2'),
  'stage-screen': hex('#fc8d7f'),
  'holiday-novelty': hex('#be9afe'),
}

/**
 * The 1,137 tracks with no genre at all — 47.6% of the library.
 *
 * Deliberately NOT the `NO_DATA` above, which is a light neutral. That one is right for a
 * handful of missing values: absence as an exception, worth spotting. This is absence as
 * the GROUND — nearly half the map — and at NO_DATA's lightness it would be the loudest
 * thing on screen, which would say something false about a library where the labels are
 * missing rather than the music.
 *
 * So: dark, desaturated, contrast 2.95:1 against the background — visible enough that the
 * shape of the cloud stays readable, dim enough to sit behind the twelve. 43.5 from the
 * nearest family colour and 42.8 from the selection cyan, so it can be neither mistaken
 * for a thirteenth genre nor for a selected point.
 */
export const UNLABELLED: Rgb = hex('#565d6e')

/** The key a track carries when it has no genre. Not a family id — it is deliberately
 *  impossible to collide with one, because `unlabelled` must never look like a decision.
 *  Same reasoning as `macro.UNREVIEWED` in the pipeline.
 *
 *  The parentheses are the mechanism, and they are load-bearing: every declared family id
 *  is kebab-case (`hip-hop`, `stage-screen`), so a key containing a character outside
 *  [\w-] cannot ever be one, whatever the taxonomy grows into.
 *  `tests/test_genre_palette.py` asserts exactly that. A previous revision achieved the
 *  same property with an invisible NUL byte inside this literal, which worked, made the
 *  whole file read as binary to grep and `file`, and would have been undebuggable. */
export const NO_GENRE = '(unlabelled)'

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

export type CategoricalEntry = {
  key: string
  label: string
  colour: Rgb
  /** Tracks this key COLOURS. For genre that is the primary family only. */
  count: number
  /** Tracks that carry this key without being coloured by it — genre's `genre_macro_set`
   *  minus the primary. Null where a track can only belong to one category. Jazz is 42
   *  primary and 75 in set; a legend that showed only the 42 would hide over half the jazz
   *  in the library, which is the thing 0005 kept the set for. */
  also: number | null
  /** The "this track has no value here" entry. Rendered apart from the categories: it is
   *  not one of them and must not read as one. */
  absence: boolean
}

export type CategoricalScale = {
  kind: 'categorical'
  mode: ColourMode
  blurb: string
  entries: CategoricalEntry[]
  keyOf: (p: Point) => string
  /** Colour lookup for the layer. A Map rather than a scan over `entries`: at 2,389 points
   *  a 13-way `find` per point per frame is free, and at the discovery corpus it is 13
   *  million comparisons a frame. Cheap to get right now, awkward later. */
  colourOf: (key: string) => Rgb
  /**
   * Does this point belong to `key` for the purpose of legend focus?
   *
   * Defaults to "it is the key that colours it". Genre overrides it with SET membership,
   * so focusing Jazz lights all 75 tracks that touch jazz rather than only the 42 it
   * colours. The lit points keep their own primary colour, so a jazz-adjacent pop record
   * lights up pop — which is the multi-macro fact stated visually instead of hidden.
   */
  matches: (p: Point, key: string) => boolean
  /** Plain-language line under the key. Genre's is the coverage figure. */
  note: string | null
}

export type OffScale = { kind: 'off'; mode: 'off'; blurb: string }

export type Scale = SequentialScale | CategoricalScale | OffScale

/** Mode order in the picker. Adding a mode is one entry here plus one case in
 *  `buildScale` — P1-12's genre mode is a categorical scale and needed nothing else. */
export const MODES: { id: ColourMode; label: string }[] = [
  { id: 'off', label: 'off' },
  { id: 'popularity', label: 'popularity' },
  { id: 'year', label: 'year' },
  { id: 'tempo', label: 'tempo' },
  { id: 'explicit', label: 'explicit' },
  { id: 'added', label: 'added' },
  { id: 'genre', label: 'genre' },
]

/** Genre rides on a second artifact that can legitimately be absent — it is built by its
 *  own CLI on its own version line. A mode that would colour every point neutral is worse
 *  than a mode that is not offered, so the picker drops it rather than showing it broken. */
export function availableModes(hasGenre: boolean) {
  return MODES.filter((m) => m.id !== 'genre' || hasGenre)
}

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

/** Finish a categorical scale: the colour Map, and the defaults for the two hooks only
 *  genre currently needs. One place, so no caller can forget one. */
function categorical(
  scale: Omit<CategoricalScale, 'kind' | 'colourOf' | 'matches' | 'note'> &
    Partial<Pick<CategoricalScale, 'matches' | 'note'>>,
): CategoricalScale {
  const byKey = new Map(scale.entries.map((e) => [e.key, e.colour]))
  return {
    kind: 'categorical',
    matches: (p, key) => scale.keyOf(p) === key,
    note: null,
    ...scale,
    colourOf: (key) => byKey.get(key) ?? NO_DATA,
  }
}

/**
 * Colour by macro genre.
 *
 * The families, their names and their counts all come from the genre artifact's manifest,
 * never from a list retyped here. A thirteenth family added to `data/genre_macro_map.json`
 * appears in this legend on the next genre build; what it does NOT do is silently acquire
 * a colour, because `GENRE` above would have no entry for it. That case renders in the
 * light `NO_DATA` neutral — visibly wrong rather than plausibly wrong, which is the
 * distinction that matters when the alternative is a point that looks unlabelled.
 * tests/test_genre_palette.py asserts the two lists agree so it should never happen.
 */
function genreScale(points: Point[], genre: GenreIndex): CategoricalScale {
  const { manifest, byUri } = genre
  const missing = manifest.macro_families.filter((f) => !(f.id in GENRE))
  if (missing.length) {
    console.warn(
      `genre families with no colour: ${missing.map((f) => f.id).join(', ')} — ` +
        'add them to GENRE in web/src/map/colour.ts',
    )
  }

  // Counted over the points ON THIS MAP, not read off the manifest. The two artifacts are
  // joined on track_uri and are separately versioned, so they can legitimately be built
  // from different exports; if they were, the manifest's coverage would describe a set of
  // tracks that is not the one on screen. Measuring here means the legend states what is
  // actually drawn, and the disagreement note below fires instead of the number quietly
  // being about something else.
  const primary = new Map<string, number>()
  const inSet = new Map<string, number>()
  let covered = 0
  let multi = 0
  for (const p of points) {
    const g = byUri.get(p.uri)
    const key = g?.genre_macro_primary
    if (!key) continue
    covered++
    if (g!.genre_macro_set.length > 1) multi++
    primary.set(key, (primary.get(key) ?? 0) + 1)
    for (const m of g!.genre_macro_set) inSet.set(m, (inSet.get(m) ?? 0) + 1)
  }

  const entries: CategoricalEntry[] = manifest.macro_families
    .map((f) => ({
      key: f.id,
      label: f.name,
      colour: GENRE[f.id] ?? NO_DATA,
      count: primary.get(f.id) ?? 0,
      also: (inSet.get(f.id) ?? 0) - (primary.get(f.id) ?? 0),
      absence: false,
    }))
    // Largest first. The taxonomy's own order is preserved in the artifact; here the
    // ranking IS the disclosure the ticket asked for -- reading 320, 317, 213, 168 … 15,
    // 12, 7 down the column is what makes a seven-track family legible as a real but tiny
    // thing rather than as a colour that failed.
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))

  const unlabelled = points.length - covered
  entries.push({
    key: NO_GENRE,
    label: 'no genre labels',
    colour: UNLABELLED,
    count: unlabelled,
    also: null,
    absence: true,
  })

  const pct = ((covered / Math.max(1, points.length)) * 100).toFixed(2)
  let note =
    `${pct}% of this map carries a genre — ` +
    `${covered.toLocaleString()} of ${points.length.toLocaleString()}. ` +
    'the rest is a gap in the export, not a thirteenth genre.'
  // Rounding, not disagreement: the manifest stores coverage to four places.
  if (Math.abs(covered / Math.max(1, points.length) - manifest.coverage) > 0.0001) {
    note +=
      ` the genre artifact reports ${(manifest.coverage * 100).toFixed(2)}% over ` +
      `${manifest.n_tracks.toLocaleString()} tracks — it was built from a different export.`
  }

  return categorical({
    mode: 'genre',
    blurb: 'macro genre from the csv labels. colour is the primary family.',
    entries,
    keyOf: (p) => byUri.get(p.uri)?.genre_macro_primary ?? NO_GENRE,
    // Set membership, not primary. See the type's doc comment: this is what makes the 177
    // multi-family tracks findable from the legend as well as from a hover.
    matches: (p, key) => {
      const g = byUri.get(p.uri)
      if (key === NO_GENRE) return !g?.genre_macro_primary
      return g ? g.genre_macro_set.includes(key) : false
    },
    note: `${note} ${multi.toLocaleString()} tracks span more than one family; hover one to see all of them.`,
  })
}

export function buildScale(
  points: Point[],
  mode: ColourMode,
  genre: GenreIndex | null = null,
): Scale {
  if (mode === 'genre') {
    // Defensive: the picker does not offer genre without an index.
    return genre ? genreScale(points, genre) : { kind: 'off', mode: 'off', blurb: '' }
  }

  if (mode === 'explicit') {
    let yes = 0
    for (const p of points) if (p.explicit) yes++
    return categorical({
      mode,
      blurb: "spotify's explicit flag.",
      entries: [
        { key: 'yes', label: 'explicit', colour: EXPLICIT_YES, count: yes, also: null, absence: false },
        {
          key: 'no',
          label: 'not flagged',
          colour: EXPLICIT_NO,
          count: points.length - yes,
          also: null,
          absence: false,
        },
      ],
      keyOf: (p) => (p.explicit ? 'yes' : 'no'),
    })
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
  if (scale.kind === 'categorical') return scale.colourOf(scale.keyOf(p))
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

/** Is this point in the currently focused legend category? `focus` null means no category
 *  is focused, which is not the same as "no point matches". */
export function inFocus(scale: Scale, p: Point, focus: string | null): boolean {
  if (focus === null || scale.kind !== 'categorical') return true
  return scale.matches(p, focus)
}

/**
 * Compose colour-by with hover, selection, legend focus and dimming. One function so the
 * precedence is stated once: hover beats selection beats focus beats dimming.
 *
 * Selection outranks focus deliberately. Focusing a family while a selection exists must
 * not make the selection recede — the selection is a thing the user built by hand, and the
 * focus is a thing they are glancing at. So a selected point keeps its colour and its ring
 * whichever family is focused, and the focus only decides how the rest of the map recedes.
 *
 * Out-of-focus points use the SAME `dim` the selection already uses, rather than a second
 * receded treatment. One way of saying "not this" is enough, and reusing it means a point
 * that is both unselected and out of focus does not double-dim into the background.
 */
export function pointColour(
  scale: Scale,
  p: Point,
  opts: {
    hovered: boolean
    selected: boolean
    anySelected: boolean
    /** Legend category being isolated, or null. */
    focus?: string | null
  },
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
  if (!inFocus(scale, p, opts.focus ?? null)) return dim(base)
  return opts.anySelected ? dim(base) : base
}

/** CSS gradient for a sequential legend, sampled from the same ramp the layer uses. */
export function gradientCss(palette: Rgb[]) {
  const stops = palette.map((c, i) => `${toCss(c)} ${((i / (palette.length - 1)) * 100).toFixed(1)}%`)
  return `linear-gradient(90deg, ${stops.join(', ')})`
}
