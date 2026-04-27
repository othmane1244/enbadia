'use client'

import { useEffect, useMemo, useState } from 'react'
import { supabase, isConfigured, Alert } from '@/lib/supabase'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'
import StatsBar from './StatsBar'
import AlertList from './AlertList'
import RulesLegend from './RulesLegend'

interface Props {
  initialAlerts: Alert[]
}

export default function DashboardShell({ initialAlerts }: Props) {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
  const [connected, setConnected] = useState(false)
  const { lang } = useLang()
  const tr = t[lang]

  useEffect(() => {
    const channel = supabase
      .channel('alerts-realtime')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'alerts' },
        (payload) => {
          setAlerts((prev) => [payload.new as Alert, ...prev].slice(0, 100))
        },
      )
      .subscribe((status) => setConnected(status === 'SUBSCRIBED'))

    return () => { supabase.removeChannel(channel) }
  }, [])

  const unresolved = useMemo(
    () => alerts.filter((a) => !a.is_resolved).length,
    [alerts],
  )

  return (
    <div className="px-8 py-8">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            {tr.overview.title}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {unresolved === 0
              ? tr.overview.noAlerts
              : tr.overview.activeAlerts(unresolved)}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {!isConfigured && (
            <span className="rounded border border-zinc-300 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 dark:border-zinc-800">
              {tr.demo}
            </span>
          )}
          <div className="flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                connected ? 'animate-pulse bg-emerald-500' : 'bg-zinc-400 dark:bg-zinc-600'
              }`}
            />
            <span className="text-xs text-zinc-500">
              {connected ? tr.live : tr.offline}
            </span>
          </div>
        </div>
      </div>

      <StatsBar alerts={alerts} />

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <AlertList alerts={alerts} />
        </div>
        <div>
          <RulesLegend />
        </div>
      </div>
    </div>
  )
}
