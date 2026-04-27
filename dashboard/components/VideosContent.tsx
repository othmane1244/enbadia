'use client'

import { Video } from 'lucide-react'
import { useLang } from '@/lib/i18n'
import { t } from '@/lib/translations'

export default function VideosContent() {
  const { lang } = useLang()
  const tr = t[lang].videos

  return (
    <div className="px-8 py-8">
      <h1 className="mb-8 text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
        {tr.title}
      </h1>
      <div className="flex flex-col items-center gap-3 py-24 text-center">
        <Video className="h-6 w-6 text-zinc-300 dark:text-zinc-700" />
        <p className="text-sm font-medium text-zinc-500">{tr.soon}</p>
        <p className="max-w-xs text-xs text-zinc-400 dark:text-zinc-600">{tr.description}</p>
      </div>
    </div>
  )
}
