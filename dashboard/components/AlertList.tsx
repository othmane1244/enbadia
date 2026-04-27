'use client'

import { Alert } from '@/lib/supabase'
import AlertBadge from './AlertBadge'
import { Camera } from 'lucide-react'

interface AlertListProps {
  alerts: Alert[]
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export default function AlertList({ alerts }: AlertListProps) {
  return (
    <section className="overflow-hidden rounded-lg border border-zinc-900 bg-zinc-950">
      <header className="flex items-center justify-between border-b border-zinc-900 px-5 py-3">
        <h2 className="text-sm font-medium text-zinc-100">Alertes récentes</h2>
        <span className="text-xs tabular-nums text-zinc-500">
          {alerts.length}
        </span>
      </header>

      {alerts.length === 0 ? (
        <div className="flex flex-col items-center gap-2 px-6 py-20 text-center">
          <Camera className="h-5 w-5 text-zinc-700" />
          <p className="text-sm text-zinc-500">Aucune alerte</p>
          <p className="text-xs text-zinc-600">
            Le flux est actif — en attente de détections.
          </p>
        </div>
      ) : (
        <ul className="max-h-[560px] divide-y divide-zinc-900 overflow-y-auto">
          {alerts.map((alert) => (
            <li
              key={alert.id}
              className="group flex items-start gap-4 px-5 py-3.5 transition-colors hover:bg-zinc-900/30"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-3">
                  <AlertBadge type={alert.alert_type} />
                  <span className="text-xs text-zinc-600">
                    · {alert.camera_id}
                  </span>
                </div>
                <p className="mt-1 truncate text-sm text-zinc-400">
                  {alert.description}
                </p>
              </div>

              <div className="flex flex-col items-end gap-1 text-xs tabular-nums">
                <span className="text-zinc-400">
                  {formatTime(alert.created_at)}
                </span>
                <span className="text-zinc-600">
                  {(alert.confidence_score * 100).toFixed(0)}%
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
