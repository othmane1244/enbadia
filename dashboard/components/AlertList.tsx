'use client'

import { Alert } from '@/lib/supabase'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'
import AlertBadge from './AlertBadge'
import { Camera } from 'lucide-react'

interface Props {
  alerts: Alert[]
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export default function AlertList({ alerts }: Props) {
  const { lang } = useLang()
  const tr = t[lang].alerts

  return (
    <section className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 dark:border-zinc-900 dark:bg-zinc-950">
      <header className="flex items-center justify-between border-b border-zinc-200 px-5 py-3 dark:border-zinc-900">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{tr.title}</h2>
        <span className="text-xs tabular-nums text-zinc-500">{alerts.length}</span>
      </header>

      {alerts.length === 0 ? (
        <div className="flex flex-col items-center gap-2 px-6 py-20 text-center">
          <Camera className="h-5 w-5 text-zinc-300 dark:text-zinc-700" />
          <p className="text-sm text-zinc-500">{tr.empty}</p>
          <p className="text-xs text-zinc-400 dark:text-zinc-600">{tr.waiting}</p>
        </div>
      ) : (
        <ul className="max-h-[560px] divide-y divide-zinc-200 overflow-y-auto dark:divide-zinc-900">
          {alerts.map((alert) => (
            <li
              key={alert.id}
              className="flex items-start gap-4 px-5 py-3.5 transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-900/30"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-3">
                  <AlertBadge type={alert.alert_type} />
                  <span className="text-xs text-zinc-400 dark:text-zinc-600">· {alert.camera_id}</span>
                </div>
                <p className="mt-1 truncate text-sm text-zinc-600 dark:text-zinc-400">
                  {alert.description}
                </p>
              </div>
              <div className="flex flex-col items-end gap-1 text-xs tabular-nums">
                <span className="text-zinc-600 dark:text-zinc-400">{formatTime(alert.created_at)}</span>
                <span className="text-zinc-400 dark:text-zinc-600">
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
