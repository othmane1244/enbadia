'use client'

import { useEffect, useMemo, useState } from 'react'
import AppShell from './AppShell'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

type HealthPayload = {
  status?: string
  uptime_seconds?: number
  frames_processed?: number
  alerts_total?: number
  ws_clients?: number
}

type StatsPayload = {
  uptime_seconds?: number
  frames_processed?: number
  alerts_total?: number
  ws_clients?: number
}

type WatchdogPayload = {
  cpu_temperature_c?: number | null
  ram_used_percent?: number | null
  disk_used_percent?: number | null
  last_supabase_ping?: string | null
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000'

function formatSeconds(totalSeconds?: number) {
  if (totalSeconds == null) return 'N/A'
  const seconds = Math.max(0, Math.floor(totalSeconds))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  return `${hours}h ${minutes}m ${secs}s`
}

function formatMaybeNumber(value?: number | null, suffix = '') {
  if (value == null || Number.isNaN(value)) return 'N/A'
  return `${value.toFixed(1)}${suffix}`
}

function StatCard({ label, value, tone }: { label: string; value: string; tone: 'emerald' | 'rose' | 'zinc' }) {
  const toneClasses = {
    emerald: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/40 dark:text-emerald-300',
    rose: 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/40 dark:bg-rose-950/40 dark:text-rose-300',
    zinc: 'border-zinc-200 bg-white text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300',
  }

  return (
    <div className={`rounded-2xl border p-4 ${toneClasses[tone]}`}>
      <p className="text-xs uppercase tracking-wide opacity-70">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-tight">{value}</p>
    </div>
  )
}

export default function SystemShell() {
  const { lang } = useLang()
  const tr = t[lang]
  const system = tr.system

  const [health, setHealth] = useState<HealthPayload | null>(null)
  const [stats, setStats] = useState<StatsPayload | null>(null)
  const [watchdog, setWatchdog] = useState<WatchdogPayload | null>(null)
  const [apiOnline, setApiOnline] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const fetchAll = async () => {
      try {
        const [healthRes, statsRes, watchdogRes] = await Promise.all([
          fetch(`${API_BASE_URL}/health/`, { cache: 'no-store' }),
          fetch(`${API_BASE_URL}/stats/`, { cache: 'no-store' }),
          fetch(`${API_BASE_URL}/watchdog/status/`, { cache: 'no-store' }),
        ])

        const [healthJson, statsJson, watchdogJson] = await Promise.all([
          healthRes.json().catch(() => null),
          statsRes.json().catch(() => null),
          watchdogRes.json().catch(() => null),
        ])

        if (cancelled) return

        setHealth(healthRes.ok ? healthJson : null)
        setStats(statsRes.ok ? statsJson : null)
        setWatchdog(watchdogRes.ok ? watchdogJson : null)
        setApiOnline(healthRes.ok && statsRes.ok && watchdogRes.ok)
        setLastUpdated(new Date().toLocaleTimeString())
      } catch {
        if (cancelled) return
        setHealth(null)
        setStats(null)
        setWatchdog(null)
        setApiOnline(false)
        setLastUpdated(new Date().toLocaleTimeString())
      }
    }

    fetchAll()
    const interval = window.setInterval(fetchAll, 5000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  const merged = useMemo(() => ({
    uptimeSeconds: health?.uptime_seconds ?? stats?.uptime_seconds,
    framesProcessed: stats?.frames_processed ?? health?.frames_processed,
    alertsTotal: stats?.alerts_total ?? health?.alerts_total,
    wsClients: stats?.ws_clients ?? health?.ws_clients,
  }), [health, stats])

  return (
    <AppShell>
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-5 py-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
            {system.title}
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Polling toutes les 5 secondes sur `/health/`, `/stats/` et `/watchdog/status/`.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label={apiOnline ? system.apiUp : system.apiDown}
            value={apiOnline ? 'OK' : 'DOWN'}
            tone={apiOnline ? 'emerald' : 'rose'}
          />
          <StatCard label={system.uptime} value={formatSeconds(merged.uptimeSeconds)} tone="zinc" />
          <StatCard label={system.framesProcessed} value={`${merged.framesProcessed ?? 0}`} tone="zinc" />
          <StatCard label={system.alertsTotal} value={`${merged.alertsTotal ?? 0}`} tone="zinc" />
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <StatCard label={system.wsClients} value={`${merged.wsClients ?? 0}`} tone="zinc" />
          <StatCard label={system.cpuTemp} value={formatMaybeNumber(watchdog?.cpu_temperature_c, '°C')} tone="zinc" />
          <StatCard label={system.ram} value={formatMaybeNumber(watchdog?.ram_used_percent, '%')} tone="zinc" />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <StatCard label={system.disk} value={formatMaybeNumber(watchdog?.disk_used_percent, '%')} tone="zinc" />
          <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
            <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">{system.lastUpdate}</p>
            <p className="mt-2 text-sm text-zinc-900 dark:text-zinc-100">
              {lastUpdated ?? system.unknown}
            </p>
            <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
              Dernier ping Supabase: {watchdog?.last_supabase_ping ?? system.unknown}
            </p>
          </div>
        </div>
      </section>
    </AppShell>
  )
}