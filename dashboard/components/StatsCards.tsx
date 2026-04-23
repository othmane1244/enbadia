// ============================================================
// components/StatsCards.tsx — Cartes statistiques
// ============================================================

import { Alert } from '@/lib/supabase'

interface StatsCardsProps {
  alerts: Alert[]
}

export default function StatsCards({ alerts }: StatsCardsProps) {
  const total       = alerts.length
  const intrusions  = alerts.filter(a => a.alert_type === 'Intrusion').length
  const chutes      = alerts.filter(a => a.alert_type === 'Chute').length
  const abandons    = alerts.filter(a => a.alert_type === 'Objet_Abandonne').length
  const attroupements = alerts.filter(a => a.alert_type === 'Attroupement').length
  const nonResolues = alerts.filter(a => !a.is_resolved).length

  const cards = [
    { label: 'Total Alertes',      value: total,          icon: '📊', color: 'border-l-gray-400' },
    { label: 'Non résolues',       value: nonResolues,    icon: '⚠️',  color: 'border-l-red-500' },
    { label: 'Intrusions',         value: intrusions,     icon: '🚨', color: 'border-l-red-400' },
    { label: 'Chutes',             value: chutes,         icon: '🆘', color: 'border-l-orange-400' },
    { label: 'Objets abandonnés',  value: abandons,       icon: '🎒', color: 'border-l-yellow-400' },
    { label: 'Attroupements',      value: attroupements,  icon: '👥', color: 'border-l-blue-400' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
      {cards.map(card => (
        <div
          key={card.label}
          className={`bg-white rounded-xl shadow-sm border-l-4 ${card.color} p-4`}
        >
          <div className="text-2xl mb-1">{card.icon}</div>
          <div className="text-2xl font-bold text-gray-800">{card.value}</div>
          <div className="text-xs text-gray-500 mt-1">{card.label}</div>
        </div>
      ))}
    </div>
  )
}
