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
import { MODES, gradientCss, toCss } from './colour'

type Props = {
  mode: ColourMode
  scale: Scale
  onChange: (mode: ColourMode) => void
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

function Categorical({ scale }: { scale: CategoricalScale }) {
  return (
    <ul className="legend__keys">
      {scale.entries.map((e) => (
        <li key={e.key}>
          <span className="legend__swatch" style={{ background: toCss(e.colour) }} />
          {e.label}
          <span className="legend__count">{e.count.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  )
}

export default function Legend({ mode, scale, onChange }: Props) {
  return (
    <div className="legend">
      <div className="legend__modes">
        <span className="legend__title">colour</span>
        {MODES.map((m) => (
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
            <Categorical scale={scale} />
          )}
        </div>
      )}
    </div>
  )
}
