import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchAlerts } from '../lib/api'

function Sparkline({ data, color = '#00d4ff' }) {
  if (!data.length) return null
  const max = Math.max(...data, 1)
  const width = 300
  const height = 40
  const points = data.map((value, index) => {
    const x = (index / Math.max(data.length - 1, 1)) * width
    const y = height - (value / max) * height
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" opacity="0.8" />
    </svg>
  )
}

function RiskHistogram({ alerts }) {
  const buckets = new Array(10).fill(0)
  for (const alert of alerts) {
    const bucket = Math.min(9, Math.floor((alert.risk_score ?? 0) / 10))
    buckets[bucket] += 1
  }
  const max = Math.max(...buckets, 1)

  return (
    <div className="flex h-16 items-end gap-1">
      {buckets.map((value, index) => {
        const pct = (value / max) * 100
        const color = index >= 8 ? '#ff4d4d' : index >= 6 ? '#ffb020' : '#00e5ff'

        return (
          <div key={index} className="flex flex-1 flex-col items-center gap-0.5">
            <div className="w-full rounded-sm" style={{ height: `${pct}%`, background: color, minHeight: value > 0 ? 2 : 0 }} />
            <span className="font-mono text-[7px] text-muted">{index * 10}</span>
          </div>
        )
      })}
    </div>
  )
}

function ThreatRelationshipGraph({ alerts }) {
  const graph = useMemo(() => {
    const sourceCounts = {}
    const typeCounts = {}
    const edgeCounts = {}

    for (const alert of alerts) {
      const source = alert.src_ip
      const attackType = alert.attack_type
      if (!source || !attackType || attackType === 'BENIGN') continue

      sourceCounts[source] = (sourceCounts[source] ?? 0) + 1
      typeCounts[attackType] = (typeCounts[attackType] ?? 0) + 1
      const edgeKey = `${source}|||${attackType}`
      edgeCounts[edgeKey] = (edgeCounts[edgeKey] ?? 0) + 1
    }

    const topSources = Object.entries(sourceCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([label, count]) => ({ id: `src:${label}`, label, count, side: 'left' }))

    const topTypes = Object.entries(typeCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([label, count]) => ({ id: `type:${label}`, label, count, side: 'right' }))

    const allowedSources = new Set(topSources.map((node) => node.label))
    const allowedTypes = new Set(topTypes.map((node) => node.label))

    const edges = Object.entries(edgeCounts)
      .map(([key, count]) => {
        const [source, attackType] = key.split('|||')
        return { source, attackType, count }
      })
      .filter((edge) => allowedSources.has(edge.source) && allowedTypes.has(edge.attackType))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)

    return { topSources, topTypes, edges }
  }, [alerts])

  if (!graph.topSources.length || !graph.topTypes.length) {
    return <p className="font-mono text-[10px] text-muted/60 italic">Not enough alert variety to render a relationship graph yet.</p>
  }

  const width = 760
  const height = 360
  const leftX = 160
  const rightX = 600
  const topPadding = 44
  const leftGap = graph.topSources.length > 1 ? (height - topPadding * 2) / (graph.topSources.length - 1) : 0
  const rightGap = graph.topTypes.length > 1 ? (height - topPadding * 2) / (graph.topTypes.length - 1) : 0

  const leftPositions = Object.fromEntries(
    graph.topSources.map((node, index) => [
      node.label,
      { x: leftX, y: topPadding + index * leftGap, count: node.count },
    ])
  )

  const rightPositions = Object.fromEntries(
    graph.topTypes.map((node, index) => [
      node.label,
      { x: rightX, y: topPadding + index * rightGap, count: node.count },
    ])
  )

  const maxEdge = Math.max(...graph.edges.map((edge) => edge.count), 1)
  const maxNode = Math.max(
    ...graph.topSources.map((node) => node.count),
    ...graph.topTypes.map((node) => node.count),
    1
  )

  return (
    <div className="space-y-4">
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="rounded-md border border-border bg-base/30">
        <defs>
          <linearGradient id="threat-edge" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--safe)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--critical)" stopOpacity="0.45" />
          </linearGradient>
        </defs>

        {graph.edges.map((edge) => {
          const from = leftPositions[edge.source]
          const to = rightPositions[edge.attackType]
          if (!from || !to) return null

          const controlOffset = Math.max(90, Math.abs(to.x - from.x) / 2.2)
          const path = `M ${from.x} ${from.y} C ${from.x + controlOffset} ${from.y}, ${to.x - controlOffset} ${to.y}, ${to.x} ${to.y}`
          const strokeWidth = 1.5 + (edge.count / maxEdge) * 5

          return (
            <path
              key={`${edge.source}-${edge.attackType}`}
              d={path}
              fill="none"
              stroke="url(#threat-edge)"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
          )
        })}

        {graph.topSources.map((node) => {
          const pos = leftPositions[node.label]
          const radius = 10 + (node.count / maxNode) * 10
          return (
            <g key={node.id}>
              <circle cx={pos.x} cy={pos.y} r={radius} fill="rgba(0, 229, 255, 0.12)" stroke="var(--safe)" strokeWidth="1.5" />
              <text x={pos.x - 20} y={pos.y - 14} textAnchor="end" fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">
                {node.label}
              </text>
              <text x={pos.x - 20} y={pos.y + 2} textAnchor="end" fill="var(--text-dim)" fontSize="9" fontFamily="var(--font-mono)">
                {node.count} alerts
              </text>
            </g>
          )
        })}

        {graph.topTypes.map((node) => {
          const pos = rightPositions[node.label]
          const radius = 10 + (node.count / maxNode) * 10
          return (
            <g key={node.id}>
              <circle cx={pos.x} cy={pos.y} r={radius} fill="rgba(255, 77, 77, 0.12)" stroke="var(--critical)" strokeWidth="1.5" />
              <text x={pos.x + 20} y={pos.y - 14} textAnchor="start" fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">
                {node.label}
              </text>
              <text x={pos.x + 20} y={pos.y + 2} textAnchor="start" fill="var(--text-dim)" fontSize="9" fontFamily="var(--font-mono)">
                {node.count} alerts
              </text>
            </g>
          )
        })}
      </svg>

      <div className="grid gap-2 md:grid-cols-2">
        <div className="rounded-sm border border-safe/20 bg-safe/5 px-3 py-2">
          <p className="font-mono text-[9px] uppercase tracking-widest text-safe">Left nodes</p>
          <p className="mt-1 text-sm text-secondary">Top source IPs by alert count</p>
        </div>
        <div className="rounded-sm border border-critical/20 bg-critical/5 px-3 py-2">
          <p className="font-mono text-[9px] uppercase tracking-widest text-critical">Right nodes</p>
          <p className="mt-1 text-sm text-secondary">Top attack types linked from those sources</p>
        </div>
      </div>
    </div>
  )
}

export default function StatsPage() {
  const { data: alerts = [] } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
  })

  const last100 = alerts.slice(0, 100)

  const cicAvg = last100.length
    ? ((last100.reduce((sum, alert) => sum + (alert.cic_confidence ?? 0), 0) / last100.length) * 100).toFixed(1)
    : '-'

  const autoBlockRate = alerts.length
    ? ((alerts.filter((alert) => alert.auto_blocked).length / alerts.length) * 100).toFixed(1)
    : '0.0'

  const now = Date.now()
  const minuteSlots = new Array(60).fill(0)
  for (const alert of alerts) {
    const ageMinutes = (now - new Date(alert.timestamp).getTime()) / 60000
    if (ageMinutes >= 0 && ageMinutes < 60) {
      minuteSlots[59 - Math.min(59, Math.floor(ageMinutes))] += 1
    }
  }

  const typeCounts = {}
  for (const alert of alerts) {
    if (alert.attack_type && alert.attack_type !== 'BENIGN') {
      typeCounts[alert.attack_type] = (typeCounts[alert.attack_type] || 0) + 1
    }
  }

  const sortedTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])
  const total = alerts.length || 1

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <h1 className="font-mono text-[11px] uppercase tracking-widest text-muted">Model, Detection, and Graph Analytics</h1>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {[
          { label: 'CIC Avg Confidence', value: `${cicAvg}%` },
          { label: 'Total Detections', value: `${alerts.length}` },
          { label: 'Auto-Block Rate', value: `${autoBlockRate}%` },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col gap-2 border border-border bg-panel p-4">
            <span className="font-mono text-[9px] uppercase tracking-widest text-muted">{label}</span>
            <span className="font-mono text-2xl text-white font-600">{value}</span>
          </div>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="border border-border bg-panel p-4">
          <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted">Threat Relationship Graph</p>
          <ThreatRelationshipGraph alerts={alerts} />
        </div>

        <div className="space-y-6">
          <div className="border border-border bg-panel p-4">
            <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted">Alert Volume - Last 60 Minutes</p>
            <Sparkline data={minuteSlots} />
          </div>

          <div className="border border-border bg-panel p-4">
            <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted">Risk Score Distribution</p>
            <RiskHistogram alerts={alerts} />
          </div>
        </div>
      </div>

      <div className="border border-border bg-panel p-4">
        <p className="mb-3 font-mono text-[9px] uppercase tracking-widest text-muted">Attack Type Distribution</p>
        {sortedTypes.length === 0 ? (
          <p className="font-mono text-[10px] text-muted/50 italic">No alerts yet</p>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                {['Attack Type', 'Count', '%'].map((heading) => (
                  <th key={heading} className="pb-2 text-left font-mono text-[8px] uppercase tracking-widest text-muted">{heading}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedTypes.map(([type, count]) => (
                <tr key={type} className="border-b border-border/40">
                  <td className="py-1.5 font-mono text-[10px] text-white/80">{type}</td>
                  <td className="py-1.5 font-mono text-[10px] text-white/60">{count}</td>
                  <td className="py-1.5 font-mono text-[10px] text-muted">{((count / total) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
