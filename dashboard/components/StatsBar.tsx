import { Alert } from '@/lib/supabase'

interface Props {
  alerts: Alert[]
}

export default function StatsBar({ alerts }: Props) {
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
    { label: 'Actives', value: active, accent: active > 0 },
    { label: "Aujourd'hui", value: today, accent: false },
    { label: 'Total', value: total, accent: false },
  ]

  return (
    <div className="grid grid-cols-3 divide-x divide-zinc-900 overflow-hidden rounded-lg border border-zinc-900 bg-zinc-950">
      {items.map((item) => (
        <div key={item.label} className="px-6 py-5">
          <div className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            {item.label}
          </div>
          <div
            className={`mt-2 text-3xl font-semibold tabular-nums tracking-tight ${
              item.accent ? 'text-rose-400' : 'text-zinc-100'
            }`}
          >
            {item.value}
          </div>
        </div>
      ))}
    </div>
  )
}
