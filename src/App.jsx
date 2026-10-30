import React, { useRef, useState, useEffect, useCallback } from 'react'
import './App.css'

const MODELS = [
  { value: 'u2netp', label: 'u2netp (best for glasses)' },
  { value: 'u2net', label: 'u2net (default)' },
  { value: 'u2net_human_seg', label: 'u2net human seg' },
]

function App() {
  const videoRef = useRef(null)
  const containerRef = useRef(null)
  const glassesRef = useRef(null)
  const faceMeshRef = useRef(null)
  const cameraRef = useRef(null)
  const scaleRef = useRef(1)
  const [glasses, setGlasses] = useState(null) // { src, naturalWidth, naturalHeight }
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [videoAspect, setVideoAspect] = useState(16 / 9)
  const [scale, setScale] = useState(1)
  const [model, setModel] = useState('u2netp')
  const [matting, setMatting] = useState(false)
  const [lastFile, setLastFile] = useState(null)

  // Keep the tracker reading the latest scale without re-registering callbacks
  useEffect(() => {
    scaleRef.current = scale
  }, [scale])

  // Process a glasses image (from file input OR clipboard paste)
  const handleGlassesFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('That clipboard content is not an image.')
      return
    }
    setLastFile(file)
    setIsProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', file)
    formData.append('model', model)
    formData.append('matting', matting ? '1' : '0')

    try {
      // Same-origin by default; in local dev Vite proxies to the Flask server
      const res = await fetch('/remove-background', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.error || 'Failed to remove background')
      }

      const data = await res.json()
      const img = new Image()
      img.onload = () => {
        setGlasses({ src: data.image, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight })
        setScale(1)
      }
      img.onerror = () => setError('Failed to load the processed image.')
      img.src = data.image
    } catch (err) {
      setError(err.message || 'Error connecting to the background removal server.')
    } finally {
      setIsProcessing(false)
    }
  }, [model, matting])

  // Clipboard paste: Ctrl+V / Cmd+V adds a glasses image
  useEffect(() => {
    const onPaste = (e) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) handleGlassesFile(file)
          e.preventDefault()
          return
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [handleGlassesFile])

  // Re-run the last image with different tweak settings
  const reprocess = () => {
    if (lastFile) handleGlassesFile(lastFile)
  }

  // Start camera + face tracking
  useEffect(() => {
    if (!window.FaceMesh || !window.Camera) {
      setError('MediaPipe libraries failed to load. Check your internet connection and reload.')
      return
    }

    let cancelled = false

    const onResults = (results) => {
      const container = containerRef.current
      const el = glassesRef.current
      if (!container || !el) {
        return
      }

      if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) {
        setFaceDetected(false)
        return
      }
      setFaceDetected(true)

      const lm = results.multiFaceLandmarks[0]
      // Iris centers (refineLandmarks gives us 468/473). Fallback to eye
      // outer corners if iris landmarks are missing.
      const leftEye = lm[468] || lm[33]
      const rightEye = lm[473] || lm[263]
      // Face edges (ear-to-ear) for a much more natural width than eye distance
      const faceLeft = lm[234]
      const faceRight = lm[454]

      const w = container.clientWidth
      const h = container.clientHeight

      // The video is mirrored via CSS (scaleX(-1)).
      // A point at normalized (x, y) appears on screen at ((1 - x), y).
      const lx = (1 - leftEye.x) * w
      const ly = leftEye.y * h
      const rx = (1 - rightEye.x) * w
      const ry = rightEye.y * h
      const flx = (1 - faceLeft.x) * w
      const frx = (1 - faceRight.x) * w

      const faceWidth = Math.abs(frx - flx)

      // Screen-left point is subject's right eye; screen-right is subject's left
      const leftPt = lx < rx ? { x: lx, y: ly } : { x: rx, y: ry }
      const rightPt = lx < rx ? { x: rx, y: ry } : { x: lx, y: ly }

      // Real glasses span roughly the full face width at the temples.
      // The slider multiplies this for fine-tuning.
      const glassesWidth = faceWidth * 0.95 * scaleRef.current
      const cx = (leftPt.x + rightPt.x) / 2
      // Glasses bridge sits ON the eye line — drop it slightly below the
      // iris centers so the lenses cover the eyes instead of the eyebrows.
      const cy = (leftPt.y + rightPt.y) / 2 + glassesWidth * 0.05
      const angle = Math.atan2(rightPt.y - leftPt.y, rightPt.x - leftPt.x)

      el.style.left = `${cx}px`
      el.style.top = `${cy}px`
      el.style.width = `${glassesWidth}px`
      el.style.transform = `translate(-50%, -50%) rotate(${angle}rad)`
    }

    const faceMesh = new window.FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    })
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    faceMesh.onResults(onResults)
    faceMeshRef.current = faceMesh

    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        try {
          await faceMeshRef.current.send({ image: videoRef.current })
        } catch (e) {
          // Skip frame (e.g., video not ready yet)
        }
      },
      width: 1280,
      height: 720
    })

    camera.start()
      .then(() => {
        if (!cancelled) setCameraActive(true)
      })
      .catch((err) => {
        if (!cancelled) setError('Unable to access camera: ' + (err.message || err))
      })
    cameraRef.current = camera

    // Match the container aspect ratio to the actual camera output
    const video = videoRef.current
    const onMeta = () => {
      if (video.videoWidth && video.videoHeight) {
        setVideoAspect(video.videoWidth / video.videoHeight)
      }
    }
    video.addEventListener('loadedmetadata', onMeta)

    return () => {
      cancelled = true
      video.removeEventListener('loadedmetadata', onMeta)
      if (cameraRef.current) {
        cameraRef.current.stop()
      }
      if (faceMeshRef.current) {
        faceMeshRef.current.close()
      }
    }
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <h1>Glasster 🕶️</h1>
        <p>Virtual Glasses Try-On</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="camera-container" ref={containerRef} style={{ aspectRatio: videoAspect }}>
        <video ref={videoRef} autoPlay playsInline muted />
        {!cameraActive && (
          <div className="camera-placeholder">
            <p>Starting camera...</p>
          </div>
        )}
        {glasses && (
          <img
            ref={glassesRef}
            src={glasses.src}
            alt="Glasses overlay"
            className="glasses-overlay"
            draggable={false}
          />
        )}
        {cameraActive && !faceDetected && (
          <div className="face-hint">
            Look at the camera to detect your face
          </div>
        )}
      </div>

      {glasses && (
        <div className="slider-row">
          <span className="slider-label">Size</span>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.05"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            className="size-slider"
          />
          <span className="slider-value">{Math.round(scale * 100)}%</span>
        </div>
      )}

      <div className="controls">
        <label className="upload-btn">
          {isProcessing ? 'Processing...' : '📷 Upload / Paste (Ctrl+V)'}
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => {
              const file = e.target.files[0]
              if (file) handleGlassesFile(file)
              e.target.value = '' // allow re-uploading the same file
            }}
            disabled={isProcessing}
            hidden
          />
        </label>
        {glasses && (
          <button className="remove-btn" onClick={() => setGlasses(null)}>
            ❌ Remove Glasses
          </button>
        )}
      </div>

      {lastFile && (
        <div className="tweak-row">
          <span className="tweak-label">Tweak cutout:</span>
          <select
            className="tweak-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={isProcessing}
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
          <label className="tweak-check">
            <input
              type="checkbox"
              checked={matting}
              onChange={(e) => setMatting(e.target.checked)}
              disabled={isProcessing}
            />
            Alpha matting
          </label>
          <button className="tweak-btn" onClick={reprocess} disabled={isProcessing}>
            {isProcessing ? 'Processing...' : '↻ Re-process'}
          </button>
        </div>
      )}

      <footer>
        <p>Auto-positioned on your eyes • Paste (Ctrl+V), upload or tweak the cutout</p>
      </footer>
    </div>
  )
}

export default App
