'use client'

import { useState } from 'react'
import { Alert } from '@/lib/supabase'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'
import AlertBadge from './AlertBadge'

interface Props {
  initialAlerts: Alert[]
}

const TYPES = ['Intrusion', 'Chute', 'Objet_Abandonne', 'Attroupement']

function formatDateTime(iso: string, lang: string) {
  return new Date(iso).toLocaleString(lang === 'fr' ? 'fr-FR' : 'en-GB', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function LogsShell({ initialAlerts }: Props) {
  const [filter, setFilter] = useState<string | null>(null)
  const { lang } = useLang()
  const tr = t[lang]

  const filtered = filter
    ? initialAlerts.filter((a) => a.alert_type === filter)
    : initialAlerts

  return (
    <div className="px-8 py-8">
      <h1 className="mb-6 text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
        {tr.logs.title}
      </h1>

      <div className="mb-4 flex flex-wrap gap-2">
        <button
          onClick={() => setFilter(null)}
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            filter === null
              ? 'bg-zinc-900 text-zinc-100 dark:bg-zinc-100 dark:text-zinc-900'
              : 'border border-zinc-200 text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-900/50'
          }`}
        >
          {tr.logs.all}
        </button>
        {TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setFilter(filter === type ? null : type)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              filter === type
                ? 'bg-zinc-900 text-zinc-100 dark:bg-zinc-100 dark:text-zinc-900'
                : 'border border-zinc-200 text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-900/50'
            }`}
          >
            {tr.alerts.type[type] ?? type}
          </button>
        ))}
      </div>

      <section className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 dark:border-zinc-900 dark:bg-zinc-950">
        {filtered.length === 0 ? (
          <div className="px-6 py-16 text-center text-sm text-zinc-500">
            {tr.logs.noResults}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-900">
                  {(['type', 'camera', 'time', 'confidence', 'status'] as const).map((col) => (
                    <th
                      key={col}
                      className="px-5 py-3 text-left text-xs font-medium uppercase tracking-wider text-zinc-500"
                    >
                      {tr.logs.columns[col]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-900">
                {filtered.map((alert) => (
                  <tr
                    key={alert.id}
                    className="transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-900/30"
                  >
                    <td className="px-5 py-3">
                      <AlertBadge type={alert.alert_type} />
                    </td>
                    <td className="px-5 py-3 text-zinc-600 dark:text-zinc-400">
                      {alert.camera_id}
                    </td>
                    <td className="px-5 py-3 tabular-nums text-zinc-600 dark:text-zinc-400">
                      {formatDateTime(alert.created_at, lang)}
                    </td>
                    <td className="px-5 py-3 tabular-nums text-zinc-600 dark:text-zinc-400">
                      {(alert.confidence_score * 100).toFixed(0)}%
                    </td>
                    <td className="px-5 py-3">
                      <span className="flex items-center gap-1.5">
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            alert.is_resolved ? 'bg-emerald-500' : 'bg-rose-500'
                          }`}
                        />
                        <span className="text-xs text-zinc-500">
                          {alert.is_resolved ? tr.logs.resolved : tr.logs.active}
                        </span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <p className="mt-3 text-xs text-zinc-400 dark:text-zinc-600">
        {tr.logs.results(filtered.length)}
      </p>
    </div>
  )
}
