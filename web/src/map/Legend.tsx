/**
 * The key to whatever the dots currently mean, plus the switch that changes it.
 *
 * Picker and legend are one block deliberately: a colour scale with no key is decoration,
 * and a switch parked away from its key makes you look in two places to read one map.
 *
 * Bottom-LEFT, stacked above the readout. The controls corner (bottom-right) would have
 * been the obvious home, but the picker needs to open upward and the selection panel owns
 * everything above bottom:4rem on the right — a popover there sits under the panel exactly
 * when a selection exists, which is when you most want to change lens. Flagged for the
 * ux-designer rather than decided as a visual call.
 */
import type { Scale, ColourMode, SequentialScale, CategoricalScale } from './colour'
import { availableModes, gradientCss, toCss } from './colour'

type Props = {
  mode: ColourMode
  scale: Scale
  onChange: (mode: ColourMode) => void
  /** Genre needs a second artifact; without it the mode is not offered at all. */
  hasGenre: boolean
  /** The pinned category, or null. Preview (hover) is reported separately so that letting
   *  go of the mouse returns to the pinned state rather than to nothing. */
  pinned: string | null
  onPin: (key: string | null) => void
  onPreview: (key: string | null) => void
}

/** Distribution across the domain, drawn above the ramp. Without it the gradient implies
 *  an even spread; this library is 90% post-2010 releases and the year bar should say so
 *  rather than let the eye assume otherwise. */
function Density({ bins }: { bins: number[] }) {
  const peak = Math.max(1, ...bins)
  const w = 100 / bins.length
  return (
    <svg className="legend__density" viewBox="0 0 100 16" preserveAspectRatio="none" aria-hidden="true">
      {bins.map((n, i) => {
        const h = (n / peak) * 16
        return <rect key={i} x={i * w} y={16 - h} width={w * 0.85} height={h} />
      })}
    </svg>
  )
}

function Sequential({ scale }: { scale: SequentialScale }) {
  return (
    <>
      {scale.bins && <Density bins={scale.bins} />}
      <div className="legend__ramp" style={{ background: gradientCss(scale.palette) }} />
      {/* Ticks are placed at their own position along the bar rather than spread evenly,
          so a quantile ramp shows its uneven values against an even bar -- which is the
          point. First and last are pulled inside the ends so neither runs off. */}
      <div className="legend__ticks">
        {scale.ticks.map((t) => (
          <span
            key={t.pos}
            className="legend__tick"
            style={{
              left: `${t.pos * 100}%`,
              transform: `translateX(${-t.pos * 100}%)`,
            }}
          >
            {t.label}
          </span>
        ))}
      </div>
      {scale.note && <p className="legend__note">{scale.note}</p>}
      {scale.missing > 0 && (
        <p className="legend__note">
          {scale.missing.toLocaleString()} have no value here — shown neutral, not as zero.
        </p>
      )}
    </>
  )
}

/**
 * The key, and the answer to "I cannot find the rare category".
 *
 * Four of the twelve genre families are under 2% of the labelled tracks — reggae is seven
 * tracks. No palette makes seven dots findable among 2,389; there are not enough pixels of
 * them for colour to be the mechanism. Two things are done about that here, and neither is
 * a merge, because 0005 argues each family earned its place and merging would destroy
 * information to flatter a rendering:
 *
 *   1. Every row states its count, and the rows are ordered largest first. A reader who
 *      cannot spot Reggae on the map can read `7` and know that is the answer rather than
 *      a palette failure. The bar behind each row makes the same point pre-attentively.
 *   2. Every row is a control. Hovering previews, clicking pins: the family stays lit and
 *      everything else recedes. This is the standard answer to a rare category and it
 *      costs nothing in the common case — the map is unchanged until you touch a row.
 *
 * It doubles as the accommodation for red-green colour vision deficiency, which twelve
 * categorical hues cannot be made safe for. Isolating a family never requires telling its
 * hue from its neighbour's.
 */
function Categorical({ scale, pinned, onPin, onPreview }: {
  scale: CategoricalScale
  pinned: string | null
  onPin: (key: string | null) => void
  onPreview: (key: string | null) => void
}) {
  const peak = Math.max(1, ...scale.entries.filter((e) => !e.absence).map((e) => e.count))
  return (
    <>
      <ul className="legend__keys" onPointerLeave={() => onPreview(null)}>
        {scale.entries.map((e) => (
          <li key={e.key} className={e.absence ? 'legend__key--absent' : undefined}>
            <button
              type="button"
              className={`legend__key${pinned === e.key ? ' legend__key--on' : ''}`}
              aria-pressed={pinned === e.key}
              title={
                e.also
                  ? `${e.label}: ${e.count} coloured, ${e.also} more carry it as a second family`
                  : `isolate ${e.label}`
              }
              onClick={() => onPin(pinned === e.key ? null : e.key)}
              onPointerEnter={() => onPreview(e.key)}
              onFocus={() => onPreview(e.key)}
              onBlur={() => onPreview(null)}
            >
              {/* Proportional, so the shape of the distribution is visible without
                  reading twelve numbers. Behind the row, at low alpha: it is context, not
                  a chart. */}
              {!e.absence && (
                <span
                  className="legend__bar"
                  style={{ width: `${(e.count / peak) * 100}%`, background: toCss(e.colour) }}
                  aria-hidden="true"
                />
              )}
              <span className="legend__swatch" style={{ background: toCss(e.colour) }} />
              <span className="legend__label">{e.label}</span>
              <span className="legend__count">
                {e.count.toLocaleString()}
                {/* Jazz is 42 primary and 75 in set. Showing only the 42 would hide over
                    half the jazz in the library. */}
                {e.also ? <span className="legend__also"> +{e.also}</span> : null}
              </span>
            </button>
          </li>
        ))}
      </ul>
      {scale.note && <p className="legend__note">{scale.note}</p>}
      {pinned && (
        <p className="legend__note legend__note--live">
          showing {scale.entries.find((e) => e.key === pinned)?.label ?? pinned} only ·
          <button type="button" className="legend__link" onClick={() => onPin(null)}>
            show all
          </button>
        </p>
      )}
    </>
  )
}

export default function Legend({
  mode,
  scale,
  onChange,
  hasGenre,
  pinned,
  onPin,
  onPreview,
}: Props) {
  return (
    <div className="legend">
      <div className="legend__modes">
        <span className="legend__title">colour</span>
        {availableModes(hasGenre).map((m) => (
          <button
            key={m.id}
            type="button"
            className={`legend__mode${m.id === mode ? ' legend__mode--on' : ''}`}
            aria-pressed={m.id === mode}
            onClick={() => onChange(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {scale.kind !== 'off' && (
        <div className="legend__body">
          <p className="legend__blurb">{scale.blurb}</p>
          {scale.kind === 'sequential' ? (
            <Sequential scale={scale} />
          ) : (
            <Categorical
              scale={scale}
              pinned={pinned}
              onPin={onPin}
              onPreview={onPreview}
            />
          )}
        </div>
      )}
    </div>
  )
}
