// ============================================================
// lib/supabase.ts — Client Supabase
// Système de Surveillance Intelligente — ENSA Béni Mellal
// ============================================================

import { createClient } from '@supabase/supabase-js'

const supabaseUrl  = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnon)

// Types TypeScript pour la table alerts
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
