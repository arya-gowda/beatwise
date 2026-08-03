import { useEffect, useState } from 'react'
import MapCanvas from './map/MapCanvas'
import { useGenreData, useMapData } from './map/useMapData'
import './App.css'

function useViewport() {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight })
  useEffect(() => {
    const onResize = () => setSize({ width: window.innerWidth, height: window.innerHeight })
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return size
}

export default function App() {
  const map = useMapData()
  // Deliberately not awaited alongside the map. The genre artifact is a separate build on
  // a separate version line, and the map must render whether or not it exists -- a genre
  // that can block the map is a genre that can hold the whole product hostage to a
  // taxonomy edit. Called before the early returns below because hooks are unconditional.
  const genre = useGenreData()
  const { width, height } = useViewport()

  if (map.state === 'loading') {
    return <div className="notice">reading the map…</div>
  }

  if (map.state === 'error') {
    return (
      <div className="notice notice--error">
        <strong>could not load the map</strong>
        <span>{map.detail}</span>
        <span className="hint">
          check that uvicorn is running and that an artifact exists —
          <code>python -m pipeline.build</code>
        </span>
      </div>
    )
  }

  const { data } = map
  return (
    <>
      <MapCanvas
        points={data.points}
        width={width}
        height={height}
        version={data.version}
        genre={genre.state === 'ready' ? genre.index : null}
      />
      <div className="readout">
        {data.points.length.toLocaleString()} tracks · {data.version}
      </div>
    </>
  )
}
