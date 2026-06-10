import { useMemo } from 'react'

const CHART_COLORS = [
  '#ff4d4d', '#ffb020', '#00e5ff', '#3b82f6', '#10b981',
  '#a855f7', '#ec4899', '#6366f1', '#14b8a6', '#f97316',
]

function SectionTitle({ text }) {
  return (
    <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: '10px' }}>
      {text}
    </div>
  )
}

const panelStyle = {
  padding: '14px',
  background: 'var(--bg-panel)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
}

function DonutChart({ data, size = 130 }) {
  const total = data.reduce((s, d) => s + d.value, 0)
  if (!total) return null
  const cx = size / 2, cy = size / 2, r = size * 0.36, strokeW = size * 0.12
  let cumAngle = -90

  const segments = data.map((d, i) => {
    const pct = d.value / total
    const angle = pct * 360
    const startRad = (cumAngle * Math.PI) / 180
    const endRad   = ((cumAngle + angle) * Math.PI) / 180
    const x1 = cx + r * Math.cos(startRad), y1 = cy + r * Math.sin(startRad)
    const x2 = cx + r * Math.cos(endRad),   y2 = cy + r * Math.sin(endRad)
    const large = angle > 180 ? 1 : 0
    cumAngle += angle
    return { ...d, path: `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`, color: CHART_COLORS[i % CHART_COLORS.length], pct }
  })

  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
      <svg width={size} height={size} style={{ flexShrink: 0 }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={strokeW}/>
        {segments.map((s, i) => (
          <path key={i} d={s.path} fill="none" stroke={s.color} strokeWidth={strokeW} strokeLinecap="butt"/>
        ))}
        <text x={cx} y={cy - 4} textAnchor="middle" fill="var(--text-primary)" fontSize="18" fontWeight="700" fontFamily="var(--font-mono)">{total}</text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-display)">total</text>
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', minWidth: 0 }}>
        {segments.slice(0, 8).map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: s.color, flexShrink: 0 }}/>
            <span style={{ color: 'var(--text-secondary)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.label}</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginLeft: 'auto', flexShrink: 0 }}>{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function HorizontalBarChart({ data }) {
  const max = Math.max(...data.map((d) => d.value), 1)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {data.map((d, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', width: '110px', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.label}</span>
          <div style={{ flex: 1, height: '6px', borderRadius: '3px', background: 'var(--border)' }}>
            <div style={{ height: '100%', borderRadius: '3px', width: `${(d.value / max) * 100}%`, background: `linear-gradient(90deg, ${CHART_COLORS[i % CHART_COLORS.length]}88, ${CHART_COLORS[i % CHART_COLORS.length]})`, transition: 'width 0.5s ease' }}/>
          </div>
          <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', width: '28px', textAlign: 'right', flexShrink: 0 }}>{d.value}</span>
        </div>
      ))}
    </div>
  )
}

function Sparkline({ data, width = 280, height = 48 }) {
  if (!data || data.length < 2) return null
  const max = Math.max(...data, 1)
  const stepX = width / (data.length - 1)
  const points = data.map((v, i) => `${i * stepX},${height - 4 - (v / max) * (height - 8)}`).join(' ')
  const areaPoints = points + ` ${width},${height - 4} 0,${height - 4}`
  const color = 'var(--safe)'
  const lastX = (data.length - 1) * stepX
  const lastY = height - 4 - (data[data.length - 1] / max) * (height - 8)
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <polygon points={areaPoints} fill="url(#sparkGrad)"/>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"/>
      <circle cx={lastX} cy={lastY} r="3" fill={color}/>
    </svg>
  )
}

export default function AttackChart({ alerts }) {
  const attackDist = useMemo(() => {
    const counts = {}
    alerts.forEach((a) => { if (a.attack_type) counts[a.attack_type] = (counts[a.attack_type] || 0) + 1 })
    return Object.entries(counts).map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value)
  }, [alerts])

  const topIPs = useMemo(() => {
    const counts = {}
    alerts.forEach((a) => { if (a.src_ip) counts[a.src_ip] = (counts[a.src_ip] || 0) + 1 })
    return Object.entries(counts).map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value).slice(0, 6)
  }, [alerts])

  const timeline = useMemo(() => {
    const now = Date.now()
    const buckets = new Array(30).fill(0)
    alerts.forEach((a) => {
      const age = (now - new Date(a.timestamp).getTime()) / 60000
      const bucket = Math.floor(age / 2)
      if (bucket >= 0 && bucket < 30) buckets[29 - bucket]++
    })
    return buckets
  }, [alerts])

  const sevDist = useMemo(() => {
    const c = { critical: 0, suspicious: 0, unknown: 0 }
    alerts.forEach((a) => {
      if (a.threat_level === 2) c.critical++
      else if (a.threat_level === 1) c.suspicious++
      else c.unknown++
    })
    return c
  }, [alerts])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '12px' }}>
      {/* Severity summary */}
      <div style={{ ...panelStyle, position: 'relative' }} className="hud-corners">
        <SectionTitle text="Threat Level Distribution" />
        <div style={{ display: 'flex', gap: '12px' }}>
          {[
            { label: 'Critical',   value: sevDist.critical,   color: 'var(--critical)' },
            { label: 'Suspicious', value: sevDist.suspicious, color: 'var(--suspicious)' },
            { label: 'Unknown',    value: sevDist.unknown,    color: 'var(--muted)' },
          ].map((s) => (
            <div key={s.label} style={{ flex: 1, textAlign: 'center' }}>
              <div style={{ fontSize: '20px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: s.color }}>{s.value}</div>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Attack type donut */}
      {attackDist.length > 0 && (
        <div style={panelStyle}>
          <SectionTitle text="Attack Type Distribution" />
          <DonutChart data={attackDist} size={130} />
        </div>
      )}

      {/* Top IPs */}
      {topIPs.length > 0 && (
        <div style={panelStyle}>
          <SectionTitle text="Top Source IPs" />
          <HorizontalBarChart data={topIPs} />
        </div>
      )}

      {/* Timeline sparkline */}
      <div style={panelStyle}>
        <SectionTitle text="Alert Volume (Last 60 min)" />
        <Sparkline data={timeline} width={300} height={50} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
          <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>-60m</span>
          <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>now</span>
        </div>
      </div>
    </div>
  )
}
