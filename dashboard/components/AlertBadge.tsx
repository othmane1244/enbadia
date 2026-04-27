'use client'

import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

interface Props {
  type: string
}

const dots: Record<string, string> = {
  Intrusion:       'bg-rose-500',
  Chute:           'bg-orange-500',
  Objet_Abandonne: 'bg-amber-500',
  Attroupement:    'bg-sky-500',
}

export default function AlertBadge({ type }: Props) {
  const { lang } = useLang()
  const label = t[lang].alerts.type[type] ?? type
  const dot = dots[type] ?? 'bg-zinc-500'
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-700 dark:text-zinc-300">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  )
}
