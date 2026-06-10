import { useState, useEffect, useRef } from 'react'

function AnimatedCounter({ value, duration = 800 }) {
  const [display, setDisplay] = useState(0)
  const prev = useRef(0)
  useEffect(() => {
    const start = prev.current
    const diff = value - start
    if (diff === 0) return
    const startTime = performance.now()
    const step = (now) => {
      const t = Math.min((now - startTime) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      setDisplay(Math.round(start + diff * ease))
      if (t < 1) requestAnimationFrame(step)
      else prev.current = value
    }
    requestAnimationFrame(step)
  }, [value, duration])
  return <span>{display.toLocaleString()}</span>
}

function AlertTriangleIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <path d="M12 9v4"/><path d="M12 17h.01"/>
    </svg>
  )
}
function EyeIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  )
}
function LockIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
      <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    </svg>
  )
}
function ActivityIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  )
}
function DatabaseIcon({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M3 5V19A9 3 0 0 0 21 19V5"/>
      <path d="M3 12A9 3 0 0 0 21 12"/>
    </svg>
  )
}

function LighthouseLogo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
      <path d="M16 4L8 8" stroke="var(--safe)" strokeWidth="1" opacity="0.4"/>
      <path d="M16 4L24 8" stroke="var(--safe)" strokeWidth="1" opacity="0.4"/>
      <path d="M16 4L4 6" stroke="var(--safe)" strokeWidth="0.5" opacity="0.2"/>
      <path d="M16 4L28 6" stroke="var(--safe)" strokeWidth="0.5" opacity="0.2"/>
      <circle cx="16" cy="5" r="2.5" fill="var(--safe)" opacity="0.3"/>
      <circle cx="16" cy="5" r="1.5" fill="var(--safe)" opacity="0.7"/>
      <path d="M13 10L11 28H21L19 10H13Z" fill="var(--text-muted)" opacity="0.6"/>
      <path d="M14 10L12.5 28H19.5L18 10H14Z" fill="var(--text-secondary)" opacity="0.4"/>
      <rect x="12" y="7" width="8" height="4" rx="1" fill="var(--bg-elevated)" stroke="var(--safe)" strokeWidth="0.8" opacity="0.8"/>
      <rect x="11" y="11" width="10" height="1" rx="0.5" fill="var(--text-muted)" opacity="0.5"/>
      <rect x="14.5" y="15" width="3" height="2" rx="0.5" fill="var(--safe)" opacity="0.15"/>
      <rect x="14.5" y="20" width="3" height="2" rx="0.5" fill="var(--safe)" opacity="0.1"/>
      <rect x="10" y="27" width="12" height="2" rx="1" fill="var(--text-muted)" opacity="0.4"/>
    </svg>
  )
}

export default function StatsBar({ connected, stats, health, theme, onToggleTheme, onTogglePanel, panelOpen }) {
  const s = stats ?? { total_today: 0, critical: 0, suspicious: 0, auto_blocked: 0 }

  const statItems = [
    { label: 'ALERTS TODAY', value: s.total_today, color: 'var(--text-primary)',  Icon: ActivityIcon,      pulse: false },
    { label: 'CRITICAL',     value: s.critical,    color: 'var(--critical)',       Icon: AlertTriangleIcon, pulse: s.critical > 0 },
    { label: 'SUSPICIOUS',   value: s.suspicious,  color: 'var(--suspicious)',     Icon: EyeIcon,           pulse: false },
    { label: 'AUTO-BLOCKED', value: s.auto_blocked,color: 'var(--safe)',           Icon: LockIcon,          pulse: false },
  ]

  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 20px', height: '56px',
      background: 'linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-surface) 100%)',
      borderBottom: '1px solid var(--border)',
      position: 'relative', overflow: 'hidden', flexShrink: 0,
    }}>
      <div className="scan-line" />

      {/* Logo */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '16px',
        letterSpacing: '2px', color: 'var(--text-primary)', userSelect: 'none',
      }}>
        <LighthouseLogo size={28} />
        <span>LIGHTHOUSE</span>
        <span style={{ fontSize: '9px', fontWeight: 500, color: 'var(--text-muted)', letterSpacing: '1px', marginLeft: '-4px', marginTop: '2px' }}>SOC</span>
      </div>

      {/* Stat boxes */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        {statItems.map(({ label, value, color, Icon, pulse }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'center', gap: '10px',
            padding: '6px 16px', borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            animation: pulse ? 'pulse-glow 2s ease-in-out infinite' : 'none',
          }}>
            <div style={{ color, opacity: 0.7 }}><Icon size={14} /></div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <span style={{ fontSize: '10px', fontWeight: 500, color: 'var(--text-muted)', letterSpacing: '0.5px', lineHeight: 1 }}>{label}</span>
              <span style={{ fontSize: '18px', fontWeight: 700, color, fontFamily: 'var(--font-mono)', lineHeight: 1.2 }}>
                <AnimatedCounter value={value} />
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Right controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        {/* WS live indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: connected ? 'var(--success)' : 'var(--text-muted)' }}>
          <span style={{
            width: '7px', height: '7px', borderRadius: '50%',
            background: connected ? 'var(--success)' : 'var(--muted)',
            animation: connected ? 'pulse-dot 2s ease-in-out infinite' : 'none',
            boxShadow: connected ? '0 0 6px var(--success)' : 'none',
          }} />
          <span>{connected ? 'LIVE' : 'RECONNECTING'}</span>
        </div>

        {health && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)' }}>
            <DatabaseIcon size={12} />
            <span>{health.db_alerts?.toLocaleString()} records</span>
          </div>
        )}

        {/* Theme toggle */}
        <button className="btn btn-ghost" onClick={onToggleTheme} style={{ padding: '4px 8px' }}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
          {theme === 'dark' ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>

        {onTogglePanel && (
          <button className="btn btn-ghost" onClick={onTogglePanel} style={{ padding: '4px 8px' }}
            title={panelOpen ? 'Collapse panel' : 'Expand panel'}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              {panelOpen
                ? <><rect width="18" height="18" x="3" y="3" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/></>
                : <><rect width="18" height="18" x="3" y="3" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/><path d="m8 9 3 3-3 3"/></>
              }
            </svg>
          </button>
        )}
      </div>
    </header>
  )
}
