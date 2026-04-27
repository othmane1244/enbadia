interface AlertBadgeProps {
  type: string
}

const config: Record<string, { label: string; dot: string }> = {
  Intrusion:       { label: 'Intrusion',        dot: 'bg-rose-500' },
  Chute:           { label: 'Chute',            dot: 'bg-orange-500' },
  Objet_Abandonne: { label: 'Objet abandonné',  dot: 'bg-amber-500' },
  Attroupement:    { label: 'Attroupement',     dot: 'bg-sky-500' },
}

export default function AlertBadge({ type }: AlertBadgeProps) {
  const c = config[type] ?? { label: type, dot: 'bg-zinc-500' }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-300">
      <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  )
}
