// ============================================================
// app/page.tsx — Dashboard principal
// Système de Surveillance Intelligente — ENSA Béni Mellal
// ============================================================

import { supabase, Alert } from '@/lib/supabase'
import AlertList from '@/components/AlertList'
import StatsCards from '@/components/StatsCards'

// Récupère les alertes côté serveur (SSR)
async function getAlerts(): Promise<Alert[]> {
  const { data, error } = await supabase
    .from('alerts')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(100)

  if (error) {
    console.error('Erreur Supabase:', error)
    return []
  }
  return data ?? []
}

export default async function DashboardPage() {
  const alerts = await getAlerts()

  return (
    <main className="min-h-screen bg-gray-50">

      {/* ── Header ── */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              🎥 Surveillance Intelligente
            </h1>
            <p className="text-xs text-gray-500 mt-0.5">
              ENSA Béni Mellal — RPi 5 + Hailo-8 + YOLO11n
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Statut API */}
            <a
              href="http://127.0.0.1:8000/health/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs bg-green-50 text-green-700 border border-green-200 px-3 py-1.5 rounded-lg hover:bg-green-100 transition"
            >
              🟢 API FastAPI
            </a>
            {/* Lien Swagger */}
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs bg-blue-50 text-blue-700 border border-blue-200 px-3 py-1.5 rounded-lg hover:bg-blue-100 transition"
            >
              📖 Swagger
            </a>
          </div>
        </div>
      </header>

      {/* ── Contenu principal ── */}
      <div className="max-w-7xl mx-auto px-4 py-6">

        {/* Bandeau simulation */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 mb-6 flex items-center gap-2">
          <span className="text-amber-600 text-sm">⚠️</span>
          <p className="text-xs text-amber-700">
            <strong>Mode simulation PC</strong> — Pipeline webcam + ONNX.
            Sur RPi 5 + Hailo-8 : libcamera + HEF (~25–35 FPS).
          </p>
        </div>

        {/* Cartes statistiques */}
        <StatsCards alerts={alerts} />

        {/* Grille principale */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Liste alertes — 2/3 de la largeur */}
          <div className="lg:col-span-2">
            <AlertList initialAlerts={alerts} />
          </div>

          {/* Panneau infos système — 1/3 */}
          <div className="space-y-4">

            {/* Info pipeline */}
            <div className="bg-white rounded-xl shadow-sm p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                Pipeline Edge-to-Cloud
              </h3>
              <div className="space-y-2">
                {[
                  { step: '1', label: 'Capture',     detail: 'Webcam / libcamera',        ms: '<10ms',      ok: true },
                  { step: '2', label: 'Inférence',   detail: 'YOLO11n ONNX / HEF',        ms: '~5–8ms',     ok: true },
                  { step: '3', label: 'Post-traitement', detail: 'NMS + BoT-SORT',         ms: '~3–5ms',     ok: true },
                  { step: '4', label: 'API FastAPI', detail: 'HTTPS + WebSocket',          ms: '~50–150ms',  ok: true },
                  { step: '5', label: 'Dashboard',   detail: 'Rendu React',                ms: '<100ms',     ok: true },
                ].map(item => (
                  <div key={item.step} className="flex items-center gap-2 text-xs">
                    <span className={`w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold shrink-0
                      ${item.ok ? 'bg-green-500' : 'bg-gray-300'}`}>
                      {item.step}
                    </span>
                    <span className="text-gray-700 font-medium w-24 shrink-0">{item.label}</span>
                    <span className="text-gray-400 flex-1">{item.detail}</span>
                    <span className="text-gray-500 font-mono">{item.ms}</span>
                  </div>
                ))}
                <div className="border-t pt-2 mt-2 flex justify-between text-xs">
                  <span className="font-semibold text-gray-700">Latence totale</span>
                  <span className="font-bold text-green-700">&lt;400ms</span>
                </div>
              </div>
            </div>

            {/* Règles de détection */}
            <div className="bg-white rounded-xl shadow-sm p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                Règles actives
              </h3>
              <div className="space-y-2 text-xs">
                {[
                  { icon: '🚨', label: 'Intrusion',       rule: 'Zone interdite : 40% gauche' },
                  { icon: '🆘', label: 'Chute',            rule: 'Ratio bbox w/h > 1.2' },
                  { icon: '🎒', label: 'Objet abandonné',  rule: 'Bagage sans personne <120px' },
                  { icon: '👥', label: 'Attroupement',     rule: '≥ 3 personnes dans le frame' },
                ].map(r => (
                  <div key={r.label} className="flex items-start gap-2">
                    <span>{r.icon}</span>
                    <div>
                      <span className="font-medium text-gray-700">{r.label}</span>
                      <p className="text-gray-400">{r.rule}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Stack technique */}
            <div className="bg-white rounded-xl shadow-sm p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                Stack technique
              </h3>
              <div className="space-y-1 text-xs text-gray-600">
                {[
                  '🤖 YOLO11n — 2.6M params, 39.0 mAP',
                  '⚡ Hailo-8 — 26 TOPS, INT8',
                  '🐍 FastAPI + Pydantic',
                  '🗄️  Supabase PostgreSQL',
                  '⚛️  Next.js 14 + Tailwind',
                ].map(item => (
                  <div key={item} className="flex items-center gap-1">
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="text-center py-4 text-xs text-gray-400 border-t border-gray-200 mt-8">
        ENSA Béni Mellal — IA & Cybersécurité 2025–2026 — Pr. ABOUHILAL Abdelmoula
      </footer>

    </main>
  )
}
