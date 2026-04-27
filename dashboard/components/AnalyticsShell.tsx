'use client'

import { Alert } from '@/lib/supabase'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'
import {
  AreaChart, Area,
  BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'

interface Props {
  alerts: Alert[]
}

const TYPE_COLORS: Record<string, string> = {
  Intrusion:       '#f43f5e',
  Chute:           '#f97316',
  Objet_Abandonne: '#f59e0b',
  Attroupement:    '#0ea5e9',
}

function byType(alerts: Alert[]) {
  const map: Record<string, number> = {}
  for (const a of alerts) map[a.alert_type] = (map[a.alert_type] ?? 0) + 1
  return Object.entries(map).map(([type, count]) => ({ type, count }))
}

function byDay(alerts: Alert[]) {
  const now = new Date()
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(now)
    d.setDate(d.getDate() - (6 - i))
    const key = `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`
    return { date: key, count: 0 }
  })
  for (const a of alerts) {
    const d = new Date(a.created_at)
    const key = `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`
    const entry = days.find((x) => x.date === key)
    if (entry) entry.count++
  }
  return days
}

const tooltipStyle = {
  background: '#18181b',
  border: '1px solid #27272a',
  borderRadius: 6,
  fontSize: 12,
  color: '#a1a1aa',
}

export default function AnalyticsShell({ alerts }: Props) {
  const { lang } = useLang()
  const tr = t[lang].analytics

  const typeData = byType(alerts)
  const dayData = byDay(alerts)

  return (
    <div className="px-8 py-8">
      <h1 className="mb-8 text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
        {tr.title}
      </h1>

      {alerts.length === 0 ? (
        <p className="text-sm text-zinc-500">{tr.noData}</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <section className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 p-5 dark:border-zinc-900 dark:bg-zinc-950">
            <h2 className="mb-4 text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {tr.overTime}
            </h2>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={dayData}>
                <CartesianGrid stroke="#71717a" strokeOpacity={0.15} vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: '#71717a', strokeOpacity: 0.3 }} />
                <Area type="monotone" dataKey="count" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.1} strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </section>

          <section className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 p-5 dark:border-zinc-900 dark:bg-zinc-950">
            <h2 className="mb-4 text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {tr.byType}
            </h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={typeData}>
                <CartesianGrid stroke="#71717a" strokeOpacity={0.15} vertical={false} />
                <XAxis dataKey="type" tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#71717a' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#71717a', fillOpacity: 0.08 }} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {typeData.map((entry) => (
                    <Cell key={entry.type} fill={TYPE_COLORS[entry.type] ?? '#71717a'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </section>
        </div>
      )}
    </div>
  )
}
