'use client'

import Image from 'next/image'
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { CheckCircle2, RotateCcw, Save } from 'lucide-react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000'

type Point = { x: number; y: number }

const DEFAULT_CAMERA_ID = 'cam_01_simulation'

const placeholderCameraFrame =
  'data:image/svg+xml;charset=UTF-8,' +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="55%" stop-color="#111827"/>
          <stop offset="100%" stop-color="#1f2937"/>
        </linearGradient>
        <linearGradient id="glow" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#22c55e" stop-opacity="0.28"/>
          <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.12"/>
        </linearGradient>
      </defs>
      <rect width="1200" height="720" fill="url(#bg)"/>
      <rect x="70" y="70" width="1060" height="580" rx="28" fill="url(#glow)" stroke="#334155" stroke-width="2"/>
      <g opacity="0.18" stroke="#94a3b8" stroke-width="1">
        <path d="M0 120H1200"/>
        <path d="M0 240H1200"/>
        <path d="M0 360H1200"/>
        <path d="M0 480H1200"/>
        <path d="M0 600H1200"/>
        <path d="M200 0V720"/>
        <path d="M400 0V720"/>
        <path d="M600 0V720"/>
        <path d="M800 0V720"/>
        <path d="M1000 0V720"/>
      </g>
      <g transform="translate(600 312)">
        <rect x="-126" y="-54" width="252" height="108" rx="22" fill="#0b1220" stroke="#475569" stroke-width="3"/>
        <circle cx="0" cy="0" r="38" fill="#111827" stroke="#22d3ee" stroke-width="8"/>
        <circle cx="0" cy="0" r="17" fill="#38bdf8" opacity="0.7"/>
        <rect x="-30" y="-108" width="60" height="26" rx="8" fill="#111827" stroke="#475569" stroke-width="2"/>
        <rect x="-16" y="54" width="32" height="90" rx="14" fill="#111827" stroke="#475569" stroke-width="2"/>
      </g>
      <text x="600" y="620" text-anchor="middle" font-family="Arial, sans-serif" font-size="34" fill="#e2e8f0">Aperçu caméra</text>
      <text x="600" y="660" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#94a3b8">Tracez une zone interdite directement sur le flux</text>
    </svg>
  `)

export default function ZoneEditor() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const pendingSingleClickRef = useRef<number | null>(null)
  const [cameraId, setCameraId] = useState(DEFAULT_CAMERA_ID)
  const [points, setPoints] = useState<Point[]>([])
  const [isClosed, setIsClosed] = useState(false)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null)

  const normalizedPoints = useMemo(() => points.map((point) => ({ x: point.x, y: point.y })), [points])

  const getRelativePoint = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return null

    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    return {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const context = canvas.getContext('2d')
    if (!context) return

    const draw = () => {
      const width = canvas.width
      const height = canvas.height

      context.clearRect(0, 0, width, height)
      context.lineJoin = 'round'
      context.lineCap = 'round'

      if (points.length === 0) return

      const canvasPoints = points.map((point) => ({
        x: point.x * width,
        y: point.y * height,
      }))

      context.save()
      context.strokeStyle = 'rgba(248, 113, 113, 0.95)'
      context.fillStyle = 'rgba(239, 68, 68, 0.95)'
      context.lineWidth = 3

      if (canvasPoints.length > 1) {
        context.beginPath()
        context.moveTo(canvasPoints[0].x, canvasPoints[0].y)
        canvasPoints.slice(1).forEach((point) => context.lineTo(point.x, point.y))
        if (isClosed) {
          context.closePath()
        }
        context.stroke()
      }

      canvasPoints.forEach((point, index) => {
        context.beginPath()
        context.arc(point.x, point.y, 7, 0, Math.PI * 2)
        context.fill()
        context.lineWidth = 2
        context.strokeStyle = '#ffffff'
        context.stroke()

        context.fillStyle = '#ef4444'
        context.font = '12px Arial'
        context.fillText(`${index + 1}`, point.x + 10, point.y - 10)
      })

      context.restore()
    }

    const resize = () => {
      const { width, height } = container.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(width))
      canvas.height = Math.max(1, Math.round(height))
      draw()
    }

    const observer = new ResizeObserver(() => resize())
    observer.observe(container)
    resize()

    return () => observer.disconnect()
  }, [points, isClosed])

  useEffect(() => {
    return () => {
      if (pendingSingleClickRef.current) {
        window.clearTimeout(pendingSingleClickRef.current)
      }
    }
  }, [])

  const closePolygon = () => {
    if (points.length < 3) {
      setMessage({ kind: 'error', text: 'Ajoutez au moins 3 points avant de fermer le polygone.' })
      return
    }
    setIsClosed(true)
    setMessage(null)
  }

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (isClosed) return

    if (pendingSingleClickRef.current) {
      window.clearTimeout(pendingSingleClickRef.current)
    }

    pendingSingleClickRef.current = window.setTimeout(() => {
      const point = getRelativePoint(event)
      if (!point) return
      setPoints((current) => [...current, point])
      setMessage(null)
      pendingSingleClickRef.current = null
    }, 140)
  }

  const handleCanvasDoubleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault()
    if (pendingSingleClickRef.current) {
      window.clearTimeout(pendingSingleClickRef.current)
      pendingSingleClickRef.current = null
    }
    closePolygon()
  }

  const handleSave = async () => {
    if (points.length < 3 || !isClosed) {
      setMessage({ kind: 'error', text: 'Fermez le polygone avant de sauvegarder.' })
      return
    }

    setSaving(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_BASE_URL}/zones/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          camera_id: cameraId.trim(),
          zone_name: 'Zone Interdite',
          points: normalizedPoints,
          active: true,
          updated_at: new Date().toISOString(),
        }),
      })

      const data = await response.json().catch(() => null)

      if (!response.ok) {
        const detail = data?.detail ?? 'Erreur inconnue'
        setMessage({ kind: 'error', text: `Erreur API: ${detail}` })
        return
      }

      setMessage({ kind: 'ok', text: 'Zone sauvegardée dans Supabase.' })
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Erreur réseau'
      setMessage({ kind: 'error', text: `Erreur API: ${detail}` })
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    if (pendingSingleClickRef.current) {
      window.clearTimeout(pendingSingleClickRef.current)
      pendingSingleClickRef.current = null
    }
    setPoints([])
    setIsClosed(false)
    setMessage(null)
  }

  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-5 py-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Zone Editor
        </h1>
        <p className="max-w-2xl text-sm text-zinc-500 dark:text-zinc-400">
          Cliquez sur l’image pour tracer une zone polygonale. Les points sont enregistrés en
          coordonnées normalisées entre 0 et 1.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.8fr)_minmax(280px,0.8fr)]">
        <div className="overflow-hidden rounded-3xl border border-zinc-200 bg-zinc-950 shadow-sm dark:border-zinc-800">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 text-xs text-zinc-300">
            <span>Caméra: {cameraId}</span>
            <span>{points.length} point{points.length > 1 ? 's' : ''}</span>
          </div>

          <div ref={containerRef} className="relative aspect-[16/9] w-full bg-zinc-900">
            <Image
              src={placeholderCameraFrame}
              alt="Aperçu caméra"
              fill
              className="absolute inset-0 h-full w-full object-cover opacity-95"
              draggable={false}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 h-full w-full cursor-crosshair"
              onClick={handleCanvasClick}
              onDoubleClick={handleCanvasDoubleClick}
            />
            <div className="pointer-events-none absolute inset-x-4 bottom-4 rounded-2xl border border-white/10 bg-black/35 px-4 py-3 text-xs text-zinc-100 backdrop-blur-sm">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1">
                  <span className="h-2 w-2 rounded-full bg-red-500" /> clic gauche = point
                </span>
                <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1">
                  <CheckCircle2 className="h-3.5 w-3.5" /> double clic = fermer le polygone
                </span>
              </div>
            </div>
          </div>
        </div>

        <aside className="flex flex-col gap-4 rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <div>
            <label className="mb-1.5 block text-xs font-medium text-zinc-700 dark:text-zinc-300">
              camera_id
            </label>
            <input
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              className="w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition focus:border-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100"
            />
          </div>

          <div className="rounded-2xl bg-zinc-50 p-4 text-sm text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">
            <p className="font-medium text-zinc-900 dark:text-zinc-100">Règles d’édition</p>
            <ul className="mt-2 space-y-1.5">
              <li>• Cliquez pour ajouter un point rouge visible.</li>
              <li>• Double-cliquez pour fermer le polygone.</li>
              <li>• Sauvegarde en coordonnées normalisées dans Supabase.</li>
            </ul>
          </div>

          {message && (
            <div
              className={`rounded-2xl border px-4 py-3 text-sm ${
                message.kind === 'ok'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/40 dark:text-emerald-300'
                  : 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/40 dark:bg-rose-950/40 dark:text-rose-300'
              }`}
            >
              {message.text}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={handleSave}
              disabled={saving}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
            >
              <Save className="h-4 w-4" />
              {saving ? 'Sauvegarde…' : 'Sauvegarder'}
            </button>

            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-zinc-200 bg-white px-4 py-2.5 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </button>
          </div>

          <div className="rounded-2xl border border-dashed border-zinc-200 p-4 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <p className="mb-2 font-medium text-zinc-700 dark:text-zinc-200">Coordonnées actuelles</p>
            {normalizedPoints.length === 0 ? (
              <p>Aucun point pour le moment.</p>
            ) : (
              <ol className="space-y-1 font-mono">
                {normalizedPoints.map((point, index) => (
                  <li key={`${index}-${point.x}-${point.y}`}>
                    {index + 1}. ({point.x.toFixed(3)}, {point.y.toFixed(3)})
                  </li>
                ))}
              </ol>
            )}
          </div>
        </aside>
      </div>
    </section>
  )
}