import { supabase } from '@/lib/supabase'
import type { Alert } from '@/lib/supabase'
import AppShell from '@/components/AppShell'
import LogsShell from '@/components/LogsShell'

async function getAlerts(): Promise<Alert[]> {
  const { data, error } = await supabase
    .from('alerts')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(500)
  if (error) return []
  return data ?? []
}

export default async function LogsPage() {
  const alerts = await getAlerts()
  return (
    <AppShell>
      <LogsShell initialAlerts={alerts} />
    </AppShell>
  )
}
