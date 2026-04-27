'use client'

import { useLang } from '@/lib/i18n'

export default function LanguageToggle() {
  const { lang, setLang } = useLang()
  return (
    <button
      type="button"
      onClick={() => setLang(lang === 'fr' ? 'en' : 'fr')}
      className="rounded px-2 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
    >
      {lang === 'fr' ? 'EN' : 'FR'}
    </button>
  )
}
