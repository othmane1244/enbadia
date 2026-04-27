'use client'

import { LangProvider } from '@/lib/i18n'
import Sidebar from './Sidebar'
import ThemeToggle from './ThemeToggle'
import LanguageToggle from './LanguageToggle'
import LogoutButton from './LogoutButton'

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <LangProvider>
      <div className="flex h-screen overflow-hidden">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden">
          <header className="flex h-14 shrink-0 items-center justify-end gap-1 border-b border-zinc-200 px-5 dark:border-zinc-900">
            <LanguageToggle />
            <ThemeToggle />
            <LogoutButton />
          </header>
          <div className="flex-1 overflow-y-auto">
            {children}
          </div>
        </div>
      </div>
    </LangProvider>
  )
}
