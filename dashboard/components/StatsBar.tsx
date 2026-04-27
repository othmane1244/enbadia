'use client'

import { Alert } from '@/lib/supabase'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

interface Props {
  alerts: Alert[]
}

export default function StatsBar({ alerts }: Props) {
  const { lang } = useLang()
  const tr = t[lang].stats

  const total = alerts.length
  const active = alerts.filter((a) => !a.is_resolved).length
  const today = alerts.filter((a) => {
    const d = new Date(a.created_at)
    const now = new Date()
    return (
      d.getDate() === now.getDate() &&
      d.getMonth() === now.getMonth() &&
      d.getFullYear() === now.getFullYear()
    )
  }).length

  const items = [
    { label: tr.active, value: active, accent: active > 0 },
    { label: tr.today,  value: today,  accent: false },
    { label: tr.total,  value: total,  accent: false },
  ]

  return (
    <div className="grid grid-cols-3 divide-x divide-zinc-200 overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 dark:divide-zinc-900 dark:border-zinc-900 dark:bg-zinc-950">
      {items.map((item) => (
        <div key={item.label} className="px-6 py-5">
          <div className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            {item.label}
          </div>
          <div
            className={`mt-2 text-3xl font-semibold tabular-nums tracking-tight ${
              item.accent ? 'text-rose-600 dark:text-rose-400' : 'text-zinc-900 dark:text-zinc-100'
            }`}
          >
            {item.value}
          </div>
        </div>
      ))}
    </div>
  )
}
