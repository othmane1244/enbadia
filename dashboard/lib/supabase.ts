import { createClient, SupabaseClient } from '@supabase/supabase-js'

const url = process.env.NEXT_PUBLIC_SUPABASE_URL
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

export const isConfigured = Boolean(url && key)

function createStub(): SupabaseClient {
  const emptyQuery: any = {
    select: () => emptyQuery,
    order: () => emptyQuery,
    limit: () => Promise.resolve({ data: [], error: null }),
  }
  const stubChannel: any = {
    on: () => stubChannel,
    subscribe: (cb?: (s: string) => void) => {
      cb?.('CLOSED')
      return stubChannel
    },
  }
  return {
    from: () => emptyQuery,
    channel: () => stubChannel,
    removeChannel: () => {},
  } as unknown as SupabaseClient
}

if (!isConfigured && typeof window === 'undefined') {
  console.warn(
    '[supabase] NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY missing — running in demo mode (no data).',
  )
}

export const supabase: SupabaseClient = isConfigured
  ? createClient(url!, key!)
  : createStub()

export interface Alert {
  id:               string
  created_at:       string
  camera_id:        string
  alert_type:       string
  description:      string
  confidence_score: number
  detection_info:   any[]
  is_resolved:      boolean
}
