import { useState, useCallback } from 'react'
import { postAction } from '../lib/api'

function timeAgo(ts) {
  const diff = (Date.now() - new Date(ts).getTime()) / 1000
  if (diff < 5) return 'just now'
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function severityColor(level) {
  if (level === 2) return 'var(--critical)'
  if (level === 1) return 'var(--suspicious)'
  return 'var(--muted)'
}

function severityLabel(level) {
  if (level === 2) return 'CRITICAL'
  if (level === 1) return 'SUSPICIOUS'
  return 'UNKNOWN'
}

function severityBadgeClass(level) {
  if (level === 2) return 'badge badge-critical'
  if (level === 1) return 'badge badge-suspicious'
  return 'badge badge-muted'
}

function riskColor(score) {
  if (score >= 70) return 'var(--critical)'
  if (score >= 40) return 'var(--suspicious)'
  return 'var(--safe)'
}

function ConfidenceBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
      <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', width: '32px', flexShrink: 0 }}>{label}</span>
      <div className="confidence-bar" style={{ flex: 1, minWidth: '40px' }}>
        <div className="confidence-bar-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}66, ${color})` }} />
      </div>
      <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color, width: '28px', textAlign: 'right', flexShrink: 0 }}>{pct}%</span>
    </div>
  )
}

function LockIcon({ size = 10 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
      <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    </svg>
  )
}
function UserXIcon({ size = 10 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <line x1="17" y1="8" x2="23" y2="14"/><line x1="23" y1="8" x2="17" y2="14"/>
    </svg>
  )
}
function XCircleIcon({ size = 10 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="m15 9-6 6"/><path d="m9 9 6 6"/>
    </svg>
  )
}

export default function AlertCard({ alert, isNew, isSelected, onClick, onActionComplete }) {
  const [status, setStatus] = useState(alert.status)
  const [autoBlocked, setAutoBlocked] = useState(alert.auto_blocked)
  const [loading, setLoading] = useState(null)

  const color = severityColor(alert.threat_level)
  const isDismissed = status === 'dismissed'
  const isIsolated = status === 'isolated'

  const handleAction = useCallback(async (e, action) => {
    e.stopPropagation()
    setLoading(action)
    const prevStatus = status
    const prevBlocked = autoBlocked
    if (action === 'block') setAutoBlocked(true)
    if (action === 'dismiss') setStatus('dismissed')
    if (action === 'isolate') setStatus('isolated')
    try {
      await postAction(alert.id, action)
      onActionComplete?.()
    } catch {
      setStatus(prevStatus)
      setAutoBlocked(prevBlocked)
    } finally {
      setLoading(null)
    }
  }, [alert.id, status, autoBlocked, onActionComplete])

  const flashClass = isNew
    ? alert.threat_level === 2 ? 'animate-flash-critical'
      : alert.threat_level === 1 ? 'animate-flash-suspicious'
      : 'animate-flash'
    : ''

  return (
    <div
      className={flashClass}
      onClick={onClick}
      style={{
        display: 'flex', flexDirection: 'column', gap: '8px',
        padding: '12px 14px',
        background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-panel)',
        borderLeft: `3px solid ${color}`,
        borderBottom: '1px solid var(--border)',
        borderRight: isSelected ? '1px solid var(--border-hover)' : '1px solid transparent',
        borderTop: '1px solid transparent',
        cursor: 'pointer',
        transition: 'all var(--transition-fast)',
        opacity: isDismissed ? 0.4 : 1,
      }}
      onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.background = 'var(--bg-elevated)' }}
      onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.background = 'var(--bg-panel)' }}
    >
      {/* Row 1: severity + attack type + time */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
          <span className={severityBadgeClass(alert.threat_level)} style={{ fontSize: '9px', padding: '1px 6px' }}>
            {severityLabel(alert.threat_level)}
          </span>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>{alert.attack_type ?? 'Unknown'}</span>
          {autoBlocked && (
            <span className="badge badge-blocked" style={{ fontSize: '8px', padding: '1px 5px', display: 'flex', alignItems: 'center', gap: '3px' }}>
              <LockIcon size={8} /> BLOCKED
            </span>
          )}
          {isIsolated && <span className="badge badge-suspicious" style={{ fontSize: '8px', padding: '1px 5px' }}>ISOLATED</span>}
          {isDismissed && <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontStyle: 'italic' }}>dismissed</span>}
        </div>
        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', flexShrink: 0 }}>
          {timeAgo(alert.timestamp)}
        </span>
      </div>

      {/* Row 2: src → dest + risk score */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {alert.src_ip}
          <span style={{ color: 'var(--text-dim)', margin: '0 4px' }}>→</span>
          {alert.agent_name}
          {alert.dst_port && <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>:{alert.dst_port}</span>}
        </span>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: '36px', height: '24px', borderRadius: '4px', flexShrink: 0,
          background: `${riskColor(alert.risk_score)}11`,
          border: `1px solid ${riskColor(alert.risk_score)}33`,
        }}>
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: riskColor(alert.risk_score) }}>
            {Math.round(alert.risk_score ?? 0)}
          </span>
        </div>
      </div>

      {/* Row 3: AI explanation */}
      <div className="line-clamp-2" style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
        {alert.ai_explanation ?? alert.rule_description ?? 'No analysis available.'}
      </div>

      {/* Row 4: confidence bars + actions */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', flex: 1, minWidth: 0 }}>
          <ConfidenceBar label="CIC" value={alert.cic_confidence} color={color} />
        </div>

        <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
          {!autoBlocked && !isDismissed && (
            <button
              className="btn btn-danger"
              style={{ padding: '3px 8px', fontSize: '10px', gap: '4px' }}
              disabled={loading === 'block'}
              onClick={(e) => handleAction(e, 'block')}
            >
              <LockIcon size={10} />
              {loading === 'block' ? '...' : 'Block'}
            </button>
          )}
          {!isIsolated && !isDismissed && (
            <button
              className="btn btn-warn"
              style={{ padding: '3px 8px', fontSize: '10px', gap: '4px' }}
              disabled={loading === 'isolate'}
              onClick={(e) => handleAction(e, 'isolate')}
            >
              <UserXIcon size={10} />
              {loading === 'isolate' ? '...' : 'Isolate'}
            </button>
          )}
          {!isDismissed && (
            <button
              className="btn btn-ghost"
              style={{ padding: '3px 8px', fontSize: '10px', gap: '4px' }}
              disabled={loading === 'dismiss'}
              onClick={(e) => handleAction(e, 'dismiss')}
            >
              <XCircleIcon size={10} />
              {loading === 'dismiss' ? '...' : 'Dismiss'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
