import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchAlerts, MOCK_ALERTS } from '../lib/api'
import { useWebSocket } from '../hooks/useWebSocket'
import AlertCard from './AlertCard'

function SearchIcon({ size = 13 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
    </svg>
  )
}
function ShieldOffIcon({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M19.69 14a6.9 6.9 0 0 0 .31-2V5l-8-3-3.16 1.18"/>
      <path d="M4.73 4.73 4 5v7c0 6 8 10 8 10a20.29 20.29 0 0 0 5.62-4.38"/>
      <line x1="2" y1="2" x2="22" y2="22"/>
    </svg>
  )
}
function XIcon({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18"/><path d="m6 6 12 12"/>
    </svg>
  )
}

const FILTERS = [
  { key: 'all',       label: 'All',       color: 'var(--safe)' },
  { key: 'critical',  label: 'Critical',  color: 'var(--critical)' },
  { key: 'suspicious',label: 'Suspicious',color: 'var(--suspicious)' },
  { key: 'unknown',   label: 'Unknown',   color: 'var(--muted)' },
  { key: 'blocked',   label: 'Blocked',   color: 'var(--critical)' },
  { key: 'dismissed', label: 'Dismissed', color: 'var(--text-dim)' },
]

function sortAlerts(alerts, sortBy) {
  return [...alerts].sort((a, b) => {
    if (sortBy === 'timestamp')  return new Date(b.timestamp) - new Date(a.timestamp)
    if (sortBy === 'risk_score') return (b.risk_score ?? 0) - (a.risk_score ?? 0)
    if (sortBy === 'confidence') return (b.confidence ?? 0) - (a.confidence ?? 0)
    if (b.threat_level !== a.threat_level) return b.threat_level - a.threat_level
    return new Date(b.timestamp) - new Date(a.timestamp)
  })
}

function filterAlerts(alerts, filter, query) {
  let next = alerts
  if (filter === 'critical')   next = next.filter((a) => a.threat_level === 2)
  if (filter === 'suspicious') next = next.filter((a) => a.threat_level === 1)
  if (filter === 'unknown')    next = next.filter((a) => a.threat_level === 0)
  if (filter === 'blocked')    next = next.filter((a) => a.auto_blocked === true)
  if (filter === 'dismissed')  next = next.filter((a) => a.status === 'dismissed')

  const q = query.trim().toLowerCase()
  if (!q) return next
  return next.filter((a) =>
    [a.src_ip, a.agent_name, a.attack_type, a.rule_description, a.ai_explanation]
      .filter(Boolean).some((v) => String(v).toLowerCase().includes(q))
  )
}

export default function AlertFeed({ onSelectAlert, selectedId, onConnectedChange }) {
  const queryClient = useQueryClient()
  const feedRef = useRef(null)
  const seenIds = useRef(new Set())
  const [filter, setFilter] = useState('all')
  const [sortBy, setSortBy] = useState('threat_level')
  const [search, setSearch] = useState('')
  const [liveAlerts, setLiveAlerts] = useState([])
  const [freshIds, setFreshIds] = useState(new Set())
  const [newCount, setNewCount] = useState(0)

  const { data, isError } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: 30_000,
    retry: 1,
  })

  useEffect(() => {
    if (!data) return
    data.forEach((a) => seenIds.current.add(a.id))
  }, [data])

  const { connected } = useWebSocket((alert) => {
    if (seenIds.current.has(alert.id)) return
    seenIds.current.add(alert.id)
    setLiveAlerts((cur) => [alert, ...cur])
    setFreshIds((cur) => { const s = new Set(cur); s.add(alert.id); return s })
    setNewCount((c) => c + 1)
    queryClient.invalidateQueries({ queryKey: ['stats'] })
    window.setTimeout(() => {
      setFreshIds((cur) => { const s = new Set(cur); s.delete(alert.id); return s })
    }, 2000)
  })

  useEffect(() => { onConnectedChange?.(connected) }, [connected, onConnectedChange])

  const baseAlerts = isError ? MOCK_ALERTS : (data ?? [])
  const mergedAlerts = useMemo(() => {
    const byId = new Map()
    for (const a of [...liveAlerts, ...baseAlerts]) byId.set(a.id, a)
    return sortAlerts([...byId.values()], sortBy)
  }, [baseAlerts, liveAlerts, sortBy])

  const counts = useMemo(() => ({
    all:       mergedAlerts.length,
    critical:  mergedAlerts.filter((a) => a.threat_level === 2).length,
    suspicious:mergedAlerts.filter((a) => a.threat_level === 1).length,
    unknown:   mergedAlerts.filter((a) => a.threat_level === 0).length,
    blocked:   mergedAlerts.filter((a) => a.auto_blocked === true).length,
    dismissed: mergedAlerts.filter((a) => a.status === 'dismissed').length,
  }), [mergedAlerts])

  const visible = useMemo(() => filterAlerts(mergedAlerts, filter, search), [mergedAlerts, filter, search])

  const clearNew = () => {
    feedRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
    setNewCount(0)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0 }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
        padding: '8px 16px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-surface)', flexShrink: 0, flexWrap: 'wrap',
      }}>
        {/* Filter pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '2px', flexWrap: 'wrap' }}>
          {FILTERS.map((f) => {
            const active = filter === f.key
            return (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  padding: '5px 12px', borderRadius: '3px',
                  background: active ? 'var(--bg-elevated)' : 'transparent',
                  border: active ? '1px solid var(--border-hover)' : '1px solid transparent',
                  borderBottom: active ? `2px solid ${f.color}` : '2px solid transparent',
                  color: active ? f.color : 'var(--text-muted)',
                  fontSize: '12px', fontWeight: active ? 600 : 400,
                  fontFamily: 'var(--font-display)', cursor: 'pointer',
                  transition: 'all var(--transition-fast)',
                }}
              >
                {f.label}
                <span style={{
                  fontSize: '10px', fontFamily: 'var(--font-mono)', padding: '0 5px',
                  borderRadius: '8px',
                  background: active ? `${f.color}22` : 'var(--bg-panel)',
                  color: active ? f.color : 'var(--text-dim)',
                }}>
                  {counts[f.key]}
                </span>
              </button>
            )
          })}
        </div>

        {/* Right: live badge + search + sort */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {newCount > 0 && (
            <button
              onClick={clearNew}
              style={{
                display: 'flex', alignItems: 'center', gap: '4px',
                padding: '3px 10px', borderRadius: '12px',
                background: 'var(--safe-bg)', border: '1px solid rgba(0,229,255,0.3)',
                color: 'var(--safe)', fontSize: '11px', fontFamily: 'var(--font-mono)',
                fontWeight: 600, cursor: 'pointer',
                animation: 'pulse-dot 1.5s ease-in-out infinite',
              }}
            >
              +{newCount} new
            </button>
          )}

          <div style={{ position: 'relative' }}>
            <span style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }}>
              <SearchIcon size={13} />
            </span>
            <input
              className="inp inp-mono"
              placeholder="Filter IP, type, text..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ paddingLeft: '28px', width: '200px', fontSize: '11px', height: '30px' }}
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                style={{ position: 'absolute', right: '6px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px' }}
              >
                <XIcon size={12} />
              </button>
            )}
          </div>

          <select className="sel" value={sortBy} onChange={(e) => setSortBy(e.target.value)} style={{ height: '30px', fontSize: '11px' }}>
            <option value="threat_level">Sort: Severity</option>
            <option value="timestamp">Sort: Newest</option>
            <option value="risk_score">Sort: Risk Score</option>
            <option value="confidence">Sort: Confidence</option>
          </select>
        </div>
      </div>

      {/* Feed */}
      <div ref={feedRef} style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
        {visible.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '12px' }}>
            {search ? <SearchIcon size={32} /> : <ShieldOffIcon size={32} />}
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              {search ? `No alerts matching "${search}"` : `No ${filter !== 'all' ? filter : ''} alerts`}
            </span>
          </div>
        ) : (
          visible.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              isNew={freshIds.has(alert.id)}
              isSelected={alert.id === selectedId}
              onClick={() => onSelectAlert(alert)}
              onActionComplete={() => {
                queryClient.invalidateQueries({ queryKey: ['alerts'] })
                queryClient.invalidateQueries({ queryKey: ['stats'] })
              }}
            />
          ))
        )}
      </div>
    </div>
  )
}
