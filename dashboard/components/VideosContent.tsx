'use client'

import { useState, useEffect, useRef } from 'react'
import { Video, Circle } from 'lucide-react'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

export default function VideosContent() {
  const { lang } = useLang()
  const tr = t[lang].videos

  const [videoFrame, setVideoFrame] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [recordingFilename, setRecordingFilename] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Connecter au WebSocket /ws/video
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//localhost:8000/ws/video`
    
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      console.log('✅ Connecté au flux vidéo')
      setIsConnected(true)
      setError(null)
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.event === 'frame' && data.data) {
          // data.data est base64 JPEG, on l'affiche dans l'img
          setVideoFrame(`data:image/jpeg;base64,${data.data}`)
        } else if (data.event === 'connected') {
          console.log('📹 Flux vidéo prêt')
        }
      } catch (e) {
        console.error('Erreur parsing WebSocket message:', e)
      }
    }
    
    ws.onerror = () => {
      setIsConnected(false)
      setError('Erreur de connexion WebSocket')
    }
    
    ws.onclose = () => {
      setIsConnected(false)
      console.log('WebSocket fermé')
    }
    
    wsRef.current = ws
    
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [])

  // Démarrer l'enregistrement
  const handleStartRecording = async () => {
    try {
      const res = await fetch('http://localhost:8000/video/record/start/', {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json() as { detail?: string }
        throw new Error(err.detail || 'Erreur démarrage enregistrement')
      }
      const data = await res.json() as { output_path: string }
      setIsRecording(true)
      setRecordingFilename(null)
      setError(null)
      console.log('🎥 Enregistrement démarré:', data.output_path)
    } catch (err: unknown) {
      const error = err instanceof Error ? err.message : 'Erreur démarrage enregistrement'
      setError(error)
      console.error(err)
    }
  }

  // Arrêter l'enregistrement
  const handleStopRecording = async () => {
    try {
      const res = await fetch('http://localhost:8000/video/record/stop/', {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json() as { detail?: string }
        throw new Error(err.detail || 'Erreur arrêt enregistrement')
      }
      const data = await res.json() as { filename: string }
      setIsRecording(false)
      setRecordingFilename(data.filename)
      setError(null)
      console.log('🎬 Enregistrement arrêté:', data.filename)
    } catch (err: unknown) {
      const error = err instanceof Error ? err.message : 'Erreur arrêt enregistrement'
      setError(error)
      console.error(err)
    }
  }

  return (
    <div className="flex h-screen flex-col bg-zinc-50 dark:bg-zinc-950">
      {/* En-tête */}
      <div className="border-b border-zinc-200 bg-white px-8 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Video className="h-6 w-6 text-zinc-900 dark:text-zinc-100" />
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              {tr.title}
            </h1>
          </div>
          
          {/* Statut connexion */}
          <div className="flex items-center gap-2">
            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
              {isConnected ? 'En ligne' : 'Hors ligne'}
            </span>
          </div>
        </div>
      </div>

      {/* Zone vidéo */}
      <div className="flex-1 overflow-auto px-8 py-8">
        {videoFrame ? (
          <div className="flex flex-col items-center gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={videoFrame}
              alt="Flux vidéo en direct"
              className="max-h-[600px] w-full max-w-4xl rounded-lg border border-zinc-300 object-contain dark:border-zinc-700"
            />
            <div className="text-sm text-zinc-500 dark:text-zinc-400">
              Flux vidéo en temps réel
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Video className="mx-auto h-12 w-12 text-zinc-300 dark:text-zinc-700" />
              <p className="mt-2 text-sm font-medium text-zinc-500">
                {isConnected ? 'En attente de frames...' : 'Connexion en cours...'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Contrôles enregistrement */}
      <div className="border-t border-zinc-200 bg-white px-8 py-6 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex flex-col gap-4">
          {/* Messages - ligne 1 */}
          <div className="flex flex-wrap items-center gap-3">
            {error && (
              <div className="rounded bg-red-50 px-3 py-2 text-sm font-medium text-red-700 dark:bg-red-900/20 dark:text-red-400">
                ⚠️ {error}
              </div>
            )}
            {recordingFilename && (
              <div className="rounded bg-green-50 px-3 py-2 text-sm font-medium text-green-700 dark:bg-green-900/20 dark:text-green-400">
                ✅ Enregistré: {recordingFilename}
              </div>
            )}
            {isRecording && (
              <div className="flex items-center gap-2 rounded bg-blue-50 px-3 py-2 dark:bg-blue-900/20">
                <Circle className="h-3 w-3 animate-pulse fill-red-500 text-red-500" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-400">
                  En cours d&apos;enregistrement...
                </span>
              </div>
            )}
          </div>

          {/* Boutons - ligne 2 */}
          <div className="flex gap-3">
            <button
              onClick={handleStartRecording}
              disabled={!isConnected || isRecording}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-2 font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-blue-500 dark:hover:bg-blue-600"
            >
              ⏺️ Enregistrer
            </button>
            <button
              onClick={handleStopRecording}
              disabled={!isRecording}
              className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-6 py-2 font-medium text-white transition-colors hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-red-500 dark:hover:bg-red-600"
            >
              ⏹️ Stopper
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
