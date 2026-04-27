import { supabase } from '@/lib/supabase'
import type { Alert } from '@/lib/supabase'
import AppShell from '@/components/AppShell'
import AnalyticsShell from '@/components/AnalyticsShell'

async function getAlerts(): Promise<Alert[]> {
  const { data, error } = await supabase
    .from('alerts')
    .select('*')
    .order('created_at', { ascending: false })
  if (error) return []
  return data ?? []
}

export default async function AnalyticsPage() {
  const alerts = await getAlerts()
  return (
    <AppShell>
      <AnalyticsShell alerts={alerts} />
    </AppShell>
  )
}
