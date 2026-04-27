'use client'

import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

export default function RulesLegend() {
  const { lang } = useLang()
  const tr = t[lang].rules

  return (
    <section className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 dark:border-zinc-900 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 px-5 py-3 dark:border-zinc-900">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{tr.title}</h2>
      </header>
      <ul className="divide-y divide-zinc-200 dark:divide-zinc-900">
        {tr.items.map((r) => (
          <li key={r.label} className="flex items-start gap-3 px-5 py-3.5">
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${r.dot}`} />
            <div className="min-w-0">
              <div className="text-sm text-zinc-800 dark:text-zinc-200">{r.label}</div>
              <div className="text-xs text-zinc-500">{r.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
