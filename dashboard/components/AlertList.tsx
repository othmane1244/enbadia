// ============================================================
// components/AlertList.tsx — Alertes temps réel
// Écoute Supabase Realtime — mise à jour automatique
// ============================================================

'use client'

import { useEffect, useState } from 'react'
import { supabase, Alert } from '@/lib/supabase'
import AlertBadge from './AlertBadge'

interface AlertListProps {
  initialAlerts: Alert[]
}

export default function AlertList({ initialAlerts }: AlertListProps) {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    // ── Supabase Realtime — écoute les nouvelles insertions ──
    const channel = supabase
      .channel('alerts-realtime')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'alerts' },
        (payload) => {
          const newAlert = payload.new as Alert
          setAlerts(prev => [newAlert, ...prev].slice(0, 100))
        }
      )
      .subscribe((status) => {
        setConnected(status === 'SUBSCRIBED')
      })

    return () => { supabase.removeChannel(channel) }
  }, [])

  const formatTime = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleDateString('fr-FR')
  }

  return (
    <div className="bg-white rounded-xl shadow-sm p-4">
      {/* En-tête */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">
          Alertes en temps réel
        </h2>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
          <span className="text-xs text-gray-500">
            {connected ? 'Connecté' : 'Connexion...'}
          </span>
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
            {alerts.length} alerte{alerts.length > 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Liste */}
      {alerts.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-2">📷</div>
          <p className="text-sm">Aucune alerte — surveillance active</p>
          <p className="text-xs mt-1">Lance le simulateur pour voir les alertes apparaître</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`rounded-lg border p-3 transition-all
                ${!alert.is_resolved
                  ? 'border-red-100 bg-red-50'
                  : 'border-gray-100 bg-gray-50'
                }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <AlertBadge type={alert.alert_type} />
                    <span className="text-xs text-gray-400">
                      📷 {alert.camera_id}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 truncate">
                    {alert.description}
                  </p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-gray-400">
                      {formatDate(alert.created_at)} — {formatTime(alert.created_at)}
                    </span>
                    <span className="text-xs text-gray-400">
                      conf: {(alert.confidence_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className={`text-xs px-2 py-1 rounded-full shrink-0
                  ${alert.is_resolved
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                  }`}
                >
                  {alert.is_resolved ? '✓ Résolu' : '● Actif'}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
