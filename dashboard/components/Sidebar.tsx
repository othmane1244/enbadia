'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Activity, LayoutDashboard, FileText, Video, BarChart2, Pentagon, Cpu } from 'lucide-react'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

export default function Sidebar() {
  const pathname = usePathname()
  const { lang } = useLang()
  const tr = t[lang]

  const items = [
    { href: '/',          icon: LayoutDashboard, label: tr.nav.overview },
    { href: '/logs',      icon: FileText,         label: tr.nav.logs },
    { href: '/videos',    icon: Video,            label: tr.nav.videos },
    { href: '/analytics', icon: BarChart2,        label: tr.nav.analytics },
    { href: '/zones',     icon: Pentagon,         label: tr.nav.zones },
    { href: '/system',    icon: Cpu,              label: tr.nav.system },
  ]

  return (
    <aside className="flex h-screen w-52 shrink-0 flex-col border-r border-zinc-200 bg-white dark:border-zinc-900 dark:bg-[#0a0a0b]">
      <div className="flex h-14 items-center gap-2.5 border-b border-zinc-200 px-5 dark:border-zinc-900">
        <Activity className="h-4 w-4 text-zinc-600 dark:text-zinc-400" />
        <span className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100">
          Surveillance
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 p-2 pt-3">
        {items.map(({ href, icon: Icon, label }) => {
          const active = pathname === href
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                active
                  ? 'bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-50 hover:text-zinc-800 dark:hover:bg-zinc-900/50 dark:hover:text-zinc-300'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          )
        })}
      </nav>

    </aside>
  )
}
