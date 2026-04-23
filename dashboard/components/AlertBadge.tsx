// ============================================================
// components/AlertBadge.tsx — Badge type d'alerte
// ============================================================

interface AlertBadgeProps {
  type: string
}

const config: Record<string, { label: string; color: string }> = {
  Intrusion:        { label: '🚨 Intrusion',        color: 'bg-red-100 text-red-800 border border-red-300' },
  Chute:            { label: '🆘 Chute',             color: 'bg-orange-100 text-orange-800 border border-orange-300' },
  Objet_Abandonne:  { label: '🎒 Objet Abandonné',  color: 'bg-yellow-100 text-yellow-800 border border-yellow-300' },
  Attroupement:     { label: '👥 Attroupement',      color: 'bg-blue-100 text-blue-800 border border-blue-300' },
}

export default function AlertBadge({ type }: AlertBadgeProps) {
  const c = config[type] ?? { label: type, color: 'bg-gray-100 text-gray-800 border border-gray-300' }
  return (
    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${c.color}`}>
      {c.label}
    </span>
  )
}
