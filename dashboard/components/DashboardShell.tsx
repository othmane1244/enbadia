'use client'

import { useEffect, useMemo, useState } from 'react'
import { supabase, isConfigured, Alert } from '@/lib/supabase'
import StatsBar from './StatsBar'
import AlertList from './AlertList'
import RulesLegend from './RulesLegend'
import { Activity } from 'lucide-react'

interface Props {
  initialAlerts: Alert[]
}

export default function DashboardShell({ initialAlerts }: Props) {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
  const [connected, setConnected] = useState(false)

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

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  const unresolved = useMemo(
    () => alerts.filter((a) => !a.is_resolved).length,
    [alerts],
  )

  return (
    <main className="min-h-screen bg-[#0a0a0b]">
      <header className="sticky top-0 z-10 border-b border-zinc-900 bg-[#0a0a0b]/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <Activity className="h-4 w-4 text-zinc-400" />
            <span className="text-sm font-medium tracking-tight text-zinc-100">
              Surveillance
            </span>
            <span className="ml-2 text-xs text-zinc-600">
              ENSA Béni Mellal
            </span>
          </div>
          <div className="flex items-center gap-3">
            {!isConfigured && (
              <span className="rounded border border-zinc-800 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Demo
              </span>
            )}
            <div className="flex items-center gap-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  connected ? 'bg-emerald-500' : 'bg-zinc-600'
                } ${connected ? 'animate-pulse' : ''}`}
              />
              <span className="text-xs text-zinc-400">
                {connected ? 'Live' : 'Hors ligne'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-8 flex items-baseline justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
              Vue d'ensemble
            </h1>
            <p className="mt-1 text-sm text-zinc-500">
              {unresolved === 0
                ? 'Aucune alerte active.'
                : `${unresolved} alerte${unresolved > 1 ? 's' : ''} en cours.`}
            </p>
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

      <footer className="mx-auto max-w-6xl px-6 py-8 text-xs text-zinc-600">
        IA & Cybersécurité · 2025—2026
      </footer>
    </main>
  )
}
