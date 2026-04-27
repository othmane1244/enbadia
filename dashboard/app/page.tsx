import { supabase, Alert } from '@/lib/supabase'
import DashboardShell from '@/components/DashboardShell'

async function getAlerts(): Promise<Alert[]> {
  const { data, error } = await supabase
    .from('alerts')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(100)

  if (error) return []
  return data ?? []
}

export default async function DashboardPage() {
  const alerts = await getAlerts()
  return <DashboardShell initialAlerts={alerts} />
}
