import { useEffect, useState } from 'react'
import MapCanvas from './map/MapCanvas'
import { useMapData } from './map/useMapData'
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
      <MapCanvas points={data.points} width={width} height={height} />
      <div className="readout">
        {data.points.length.toLocaleString()} tracks · {data.version}
      </div>
    </>
  )
}
