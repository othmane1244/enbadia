const rules = [
  { label: 'Intrusion',       detail: 'Présence en zone interdite',  dot: 'bg-rose-500' },
  { label: 'Chute',           detail: 'Ratio bbox w/h > 1.2',         dot: 'bg-orange-500' },
  { label: 'Objet abandonné', detail: 'Bagage isolé (< 120 px)',      dot: 'bg-amber-500' },
  { label: 'Attroupement',    detail: '≥ 3 personnes détectées',      dot: 'bg-sky-500' },
]

export default function RulesLegend() {
  return (
    <section className="overflow-hidden rounded-lg border border-zinc-900 bg-zinc-950">
      <header className="border-b border-zinc-900 px-5 py-3">
        <h2 className="text-sm font-medium text-zinc-100">Règles de détection</h2>
      </header>
      <ul className="divide-y divide-zinc-900">
        {rules.map((r) => (
          <li key={r.label} className="flex items-start gap-3 px-5 py-3.5">
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${r.dot}`} />
            <div className="min-w-0">
              <div className="text-sm text-zinc-200">{r.label}</div>
              <div className="text-xs text-zinc-500">{r.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
