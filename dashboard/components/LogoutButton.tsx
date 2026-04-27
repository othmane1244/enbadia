'use client'

import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase-browser'
import { isConfigured } from '@/lib/supabase'
import { LogOut } from 'lucide-react'

export default function LogoutButton() {
  const router = useRouter()

  async function handleLogout() {
    if (isConfigured) {
      const supabase = createClient()
      await supabase.auth.signOut()
    }
    router.push('/login')
    router.refresh()
  }

  return (
    <button
      type="button"
      onClick={handleLogout}
      aria-label="Se déconnecter"
      className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
    >
      <LogOut className="h-3.5 w-3.5" />
    </button>
  )
}
