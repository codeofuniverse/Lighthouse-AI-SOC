import { useMemo, useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { fetchAttackerAlerts, fetchAttackers, postAction } from '../lib/api'
import AlertDetail from '../components/AlertDetail'

const SEV = { 2: 'text-critical', 1: 'text-suspicious', 0: 'text-muted' }
const THREAT_LABEL = { 2: 'CRITICAL', 1: 'SUSPICIOUS', 0: 'LOW' }
const THREAT_CLASS = { 2: 'text-critical', 1: 'text-suspicious', 0: 'text-muted' }

function StatCard({ label, value }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '12px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>{value ?? '-'}</span>
    </div>
  )
}

function AbuseGauge({ score }) {
  const pct = Math.min(Math.max(score ?? 0, 0), 100)
  const color = pct >= 75 ? 'var(--critical)' : pct >= 40 ? 'var(--suspicious)' : 'var(--safe)'
  const r = 28
  const circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <svg width="80" height="80" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={r} fill="none" stroke="var(--bg-base)" strokeWidth="8" />
        <circle cx="40" cy="40" r={r} fill="none" stroke={color}
          strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
          strokeWidth="8" transform="rotate(-90 40 40)" />
        <text x="40" y="44" textAnchor="middle" fontSize="13" fill="var(--text-primary)" fontFamily="monospace">{pct}</text>
      </svg>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)' }}>Abuse Score</span>
    </div>
  )
}

function AttackTimelineGraph({ alerts }) {
  if (!alerts.length) {
    return <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', opacity: 0.5 }}>No timeline data available.</p>
  }
  const sorted = [...alerts].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
  const minTime = new Date(sorted[0].timestamp).getTime()
  const maxTime = new Date(sorted[sorted.length - 1].timestamp).getTime()
  const range = Math.max(maxTime - minTime, 1)
  const lanes = [...new Set(sorted.map(a => a.attack_type || 'Unknown'))]
  const laneMap = Object.fromEntries(lanes.map((l, i) => [l, i]))
  const width = 900
  const laneHeight = 48
  const chartHeight = Math.max(160, lanes.length * laneHeight + 36)
  const levelColor = (l) => l === 2 ? 'var(--critical)' : l === 1 ? 'var(--suspicious)' : 'var(--safe)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <svg width="100%" height={chartHeight} viewBox={`0 0 ${width} ${chartHeight}`}
        style={{ borderRadius: '2px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.1)' }}>
        {lanes.map((lane, i) => {
          const y = 28 + i * laneHeight
          return (
            <g key={lane}>
              <line x1="140" y1={y} x2="840" y2={y} stroke="var(--border)" strokeDasharray="4 5" />
              <text x="18" y={y + 4} fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">{lane}</text>
            </g>
          )
        })}
        <line x1="140" y1="16" x2="140" y2={chartHeight - 20} stroke="var(--border)" />
        {sorted.map((alert, i) => {
          const lane = laneMap[alert.attack_type || 'Unknown']
          const y = 28 + lane * laneHeight
          const t = (new Date(alert.timestamp).getTime() - minTime) / range
          const x = 160 + t * 650
          const radius = 7 + ((alert.confidence ?? 0.4) * 8)
          return (
            <g key={alert.id}>
              {i > 0 && laneMap[sorted[i - 1].attack_type || 'Unknown'] === lane && (
                <line
                  x1={160 + ((new Date(sorted[i - 1].timestamp).getTime() - minTime) / range) * 650}
                  y1={y} x2={x} y2={y}
                  stroke={levelColor(alert.threat_level)} strokeOpacity="0.35" strokeWidth="2" />
              )}
              <circle cx={x} cy={y} r={radius} fill={levelColor(alert.threat_level)}
                fillOpacity="0.18" stroke={levelColor(alert.threat_level)} strokeWidth="1.5" />
              <text x={x} y={y - 14} textAnchor="middle" fill="var(--text-muted)" fontSize="8" fontFamily="var(--font-mono)">
                {Math.round(alert.risk_score ?? 0)}
              </text>
            </g>
          )
        })}
        <text x="160" y={chartHeight - 6} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
          {new Date(minTime).toLocaleTimeString()}
        </text>
        <text x="810" y={chartHeight - 6} textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
          {new Date(maxTime).toLocaleTimeString()}
        </text>
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
        <span style={{ color: 'var(--critical)' }}>■ Critical</span>
        <span style={{ color: 'var(--suspicious)' }}>■ Suspicious</span>
        <span style={{ color: 'var(--safe)' }}>■ Low</span>
        <span>Bubble size = model confidence · Label = risk score</span>
      </div>
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      gap: '12px', padding: '60px 24px', color: 'var(--text-muted)',
    }}>
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.3">
        <circle cx="12" cy="12" r="10" /><path d="M12 8v4m0 4h.01" />
      </svg>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textAlign: 'center', opacity: 0.5 }}>{message}</p>
    </div>
  )
}

function AttackerList() {
  const navigate = useNavigate()
  const [attackers, setAttackers] = useState(null)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchAttackers()
      .then(data => setAttackers(Array.isArray(data) ? data : []))
      .catch(err => { setError(err?.message ?? 'Failed to load'); setAttackers([]) })
  }, [])

  const filtered = useMemo(() => {
    if (!attackers) return []
    const q = search.trim().toLowerCase()
    if (!q) return attackers
    return attackers.filter(a =>
      a.src_ip?.includes(q) ||
      (a.geoip?.country ?? '').toLowerCase().includes(q) ||
      (a.attack_types ?? []).some(t => t.toLowerCase().includes(q))
    )
  }, [attackers, search])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '24px', overflowY: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Attackers</h1>
          {attackers && (
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', margin: '4px 0 0' }}>
              {attackers.length} unique source IPs detected
            </p>
          )}
        </div>
        <input
          style={{
            width: '220px', height: '30px', padding: '0 10px',
            fontFamily: 'var(--font-mono)', fontSize: '11px',
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            color: 'var(--text-primary)', outline: 'none',
          }}
          placeholder="Filter by IP, country, type..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {error && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--critical)', padding: '8px 12px', border: '1px solid var(--critical)', background: 'rgba(255,59,59,0.05)' }}>
          Error: {error}
        </div>
      )}

      {attackers === null ? (
        <EmptyState message="Loading attackers..." />
      ) : filtered.length === 0 ? (
        <EmptyState message={
          attackers.length === 0
            ? "No attackers detected yet.\nAlerts will appear here once attacks are detected from Suricata."
            : "No attackers match your filter."
        } />
      ) : (
        <div style={{ border: '1px solid var(--border)' }}>
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 70px 80px 70px 140px 90px',
            gap: '8px', padding: '8px 12px',
            background: 'var(--bg-panel)', borderBottom: '1px solid var(--border)',
          }}>
            {['IP Address', 'Alerts', 'Severity', 'Max Risk', 'Last Seen', 'Country'].map(h => (
              <span key={h} style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)' }}>{h}</span>
            ))}
          </div>
          {filtered.map(attacker => (
            <div
              key={attacker.src_ip}
              onClick={() => navigate(`/attackers/${attacker.src_ip}`)}
              style={{
                display: 'grid', gridTemplateColumns: '1fr 70px 80px 70px 140px 90px',
                gap: '8px', padding: '10px 12px', cursor: 'pointer',
                borderBottom: '1px solid var(--border)', alignItems: 'center',
                transition: 'background var(--transition-fast)',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {attacker.src_ip}
                </span>
                {attacker.is_known_attacker && (
                  <span style={{ flexShrink: 0, fontFamily: 'var(--font-mono)', fontSize: '7px', padding: '2px 5px', border: '1px solid rgba(255,59,59,0.4)', background: 'rgba(255,59,59,0.1)', color: 'var(--critical)', textTransform: 'uppercase' }}>Known</span>
                )}
                {attacker.auto_blocked && (
                  <span style={{ flexShrink: 0, fontFamily: 'var(--font-mono)', fontSize: '7px', padding: '2px 5px', border: '1px solid rgba(0,229,255,0.3)', background: 'rgba(0,229,255,0.1)', color: 'var(--safe)', textTransform: 'uppercase' }}>Blocked</span>
                )}
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-secondary)' }}>{attacker.alert_count}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600, color: attacker.max_threat === 2 ? 'var(--critical)' : attacker.max_threat === 1 ? 'var(--suspicious)' : 'var(--text-muted)' }}>
                {THREAT_LABEL[attacker.max_threat] ?? 'LOW'}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-secondary)' }}>{Math.round(attacker.max_risk)}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)' }}>
                {attacker.last_seen ? new Date(attacker.last_seen).toLocaleString() : '-'}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {attacker.geoip?.country ?? '-'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function AttackerPage() {
  const { ip } = useParams()
  const navigate = useNavigate()
  const [alerts, setAlerts] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [blocking, setBlocking] = useState(false)
  const [blockResult, setBlockResult] = useState(null)
  const [timelineExpanded, setTimelineExpanded] = useState(false)

  useEffect(() => {
    if (!ip) return
    setAlerts(null)
    setError(null)
    fetchAttackerAlerts(ip)
      .then(data => setAlerts(Array.isArray(data) ? data : []))
      .catch(err => { setError(err?.message ?? 'Failed to load'); setAlerts([]) })
  }, [ip])

  if (!ip) return <AttackerList />

  const loading = alerts === null
  const latest = alerts?.[0] ?? {}
  const geo = latest.geoip ?? {}
  const abuse = latest.abuse_score ?? 0
  const isKnown = latest.is_known_attacker ?? false

  const uniqueTypes = [...new Set((alerts ?? []).map(a => a.attack_type).filter(Boolean))]
  const allTechniques = [...new Map(
    (alerts ?? [])
      .flatMap(a => Array.isArray(a.mitre_techniques) ? a.mitre_techniques : [])
      .map(t => [t.technique_id, t])
  ).values()]

  const firstSeen = alerts?.length ? new Date(alerts[alerts.length - 1]?.timestamp).toLocaleString() : '-'
  const lastSeen  = alerts?.length ? new Date(alerts[0]?.timestamp).toLocaleString() : '-'
  const averageRisk = useMemo(() => {
    if (!alerts?.length) return '-'
    return Math.round(alerts.reduce((s, a) => s + (a.risk_score ?? 0), 0) / alerts.length)
  }, [alerts])

  const handleBlockAll = async () => {
    if (!alerts?.length) return
    setBlocking(true)
    setBlockResult(null)
    try {
      for (const alert of alerts.filter(a => !a.auto_blocked)) {
        await postAction(alert.id, 'block')
      }
      setBlockResult('success')
      setAlerts(prev => prev?.map(a => ({ ...a, auto_blocked: true })) ?? prev)
    } catch {
      setBlockResult('error')
    } finally {
      setBlocking(false)
    }
  }

  const row = (label, value) => value ? (
    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', padding: '5px 0' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-secondary)' }}>{value}</span>
    </div>
  ) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', overflowY: 'auto', padding: '24px', height: '100%' }}>
      {/* Back button */}
      <button
        onClick={() => navigate('/attackers')}
        style={{ alignSelf: 'flex-start', fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
        onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
      >
        ← Back to Attackers
      </button>

      {error && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--critical)', padding: '8px 12px', border: '1px solid var(--critical)', background: 'rgba(255,59,59,0.05)' }}>
          Error loading alerts: {error}
        </div>
      )}

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <h1 style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>{ip}</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            {geo.country && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', padding: '2px 8px', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                {geo.country}{geo.city ? ` · ${geo.city}` : ''}
              </span>
            )}
            {isKnown && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', padding: '2px 8px', border: '1px solid rgba(255,59,59,0.4)', background: 'rgba(255,59,59,0.1)', color: 'var(--critical)' }}>Known Attacker</span>
            )}
            {abuse >= 50 && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', padding: '2px 8px', border: '1px solid rgba(245,158,11,0.4)', background: 'rgba(245,158,11,0.1)', color: 'var(--suspicious)' }}>High Abuse Score</span>
            )}
          </div>
        </div>
        <AbuseGauge score={abuse} />
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
        <StatCard label="Total Alerts" value={loading ? '...' : alerts.length} />
        <StatCard label="Attack Types" value={loading ? '...' : uniqueTypes.length} />
        <StatCard label="Avg Risk" value={loading ? '...' : averageRisk} />
        <StatCard label="First Seen" value={loading ? '...' : firstSeen} />
        <StatCard label="Last Seen" value={loading ? '...' : lastSeen} />
      </div>

      {/* Timeline graph */}
      <div style={{ border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '16px' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>Attack Timeline Graph</p>
        {loading ? <EmptyState message="Loading..." /> : <AttackTimelineGraph alerts={alerts} />}
      </div>

      {/* GeoIP + AbuseIPDB */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '16px' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>GeoIP</p>
          {geo.country ? (
            <div>
              {row('Country', geo.country)}
              {row('City', geo.city)}
              {row('Coordinates', geo.lat != null ? `${geo.lat?.toFixed(2)}, ${geo.lon?.toFixed(2)}` : null)}
              {row('Tor Exit', geo.is_tor ? 'YES' : null)}
              {row('VPN', geo.is_vpn ? 'YES' : null)}
            </div>
          ) : (
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', fontStyle: 'italic', opacity: 0.5 }}>
              {loading ? 'Loading...' : 'No GeoIP data — GeoLite2 database not configured'}
            </p>
          )}
        </div>

        <div style={{ border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '16px' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '12px' }}>AbuseIPDB</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)' }}>Score</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: 600, color: abuse >= 75 ? 'var(--critical)' : abuse >= 40 ? 'var(--suspicious)' : 'var(--safe)' }}>
                {abuse}/100
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)' }}>Known Attacker</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: isKnown ? 'var(--critical)' : 'var(--safe)' }}>
                {isKnown ? 'YES' : 'NO'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* MITRE techniques */}
      {allTechniques.length > 0 && (
        <div>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '8px' }}>MITRE ATT&CK Observed</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {allTechniques.map(t => (
              <span key={t.technique_id} style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', padding: '4px 8px', border: '1px solid rgba(0,229,255,0.25)', background: 'rgba(0,229,255,0.05)', color: 'var(--safe)' }}>
                {t.technique_id} · {t.technique_name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Alert timeline list */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', margin: 0 }}>
            Alert History {!loading && alerts.length > 0 && `(${alerts.length})`}
          </p>
          {!loading && alerts.length > 5 && (
            <button
              onClick={() => setTimelineExpanded(v => !v)}
              style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              {timelineExpanded ? 'Show less ↑' : `Show all ${alerts.length} ↓`}
            </button>
          )}
        </div>

        {loading ? (
          <EmptyState message="Loading alerts..." />
        ) : alerts.length === 0 ? (
          <EmptyState message="No alerts found for this IP address." />
        ) : (
          <div style={{ border: '1px solid var(--border)', overflowY: 'auto', maxHeight: timelineExpanded ? 'none' : '260px' }}>
            {alerts.map(alert => (
              <div
                key={alert.id}
                onClick={() => setSelected(alert)}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid var(--border)',
                  transition: 'background var(--transition-fast)',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-muted)', minWidth: '140px' }}>
                  {new Date(alert.timestamp).toLocaleString()}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', fontWeight: 600, color: alert.threat_level === 2 ? 'var(--critical)' : alert.threat_level === 1 ? 'var(--suspicious)' : 'var(--text-muted)', flex: 1, textAlign: 'center' }}>
                  {alert.attack_type}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-secondary)' }}>
                    Risk: {Math.round(alert.risk_score ?? 0)}
                  </span>
                  {alert.auto_blocked && (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', padding: '2px 6px', border: '1px solid rgba(0,229,255,0.3)', background: 'rgba(0,229,255,0.1)', color: 'var(--safe)' }}>Blocked</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Block action */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingBottom: '24px' }}>
        <button
          onClick={handleBlockAll}
          disabled={blocking || !alerts?.length}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase',
            letterSpacing: '0.1em', padding: '8px 16px', cursor: blocking || !alerts?.length ? 'not-allowed' : 'pointer',
            border: '1px solid rgba(255,59,59,0.4)', background: 'transparent', color: 'var(--critical)',
            opacity: blocking || !alerts?.length ? 0.4 : 1, transition: 'background var(--transition-fast)',
          }}
          onMouseEnter={e => { if (!blocking && alerts?.length) e.currentTarget.style.background = 'rgba(255,59,59,0.1)' }}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          {blocking ? 'Blocking...' : 'Block All Traffic from IP'}
        </button>
        {blockResult === 'success' && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--safe)' }}>All alerts blocked successfully.</span>}
        {blockResult === 'error' && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--critical)' }}>Block failed — check backend logs.</span>}
      </div>

      <AnimatePresence>
        {selected && <AlertDetail key={selected.id} alert={selected} onClose={() => setSelected(null)} />}
      </AnimatePresence>
    </div>
  )
}
