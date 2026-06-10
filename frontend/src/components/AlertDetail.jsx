import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { postAction } from '../lib/api'

// ── Icons ──────────────────────────────────────────────────────────────────
function Icon({ children, size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  )
}
const LockIcon    = ({ size = 13 }) => <Icon size={size}><rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></Icon>
const UserXIcon   = ({ size = 13 }) => <Icon size={size}><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="17" y1="8" x2="23" y2="14"/><line x1="23" y1="8" x2="17" y2="14"/></Icon>
const XCircleIcon = ({ size = 13 }) => <Icon size={size}><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></Icon>
const XIcon       = ({ size = 16 }) => <Icon size={size}><path d="M18 6 6 18"/><path d="m6 6 12 12"/></Icon>
const ClockIcon   = ({ size = 12 }) => <Icon size={size}><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></Icon>
const ZapIcon     = ({ size = 12 }) => <Icon size={size}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></Icon>
const FileIcon    = ({ size = 12 }) => <Icon size={size}><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></Icon>
const GlobeIcon   = ({ size = 12 }) => <Icon size={size}><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></Icon>
const TargetIcon  = ({ size = 12 }) => <Icon size={size}><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></Icon>
const ServerIcon  = ({ size = 12 }) => <Icon size={size}><rect width="20" height="8" x="2" y="2" rx="2"/><rect width="20" height="8" x="2" y="14" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></Icon>
const PlayIcon    = ({ size = 13 }) => <Icon size={size}><polygon points="5 3 19 12 5 21 5 3"/></Icon>

// ── Helpers ────────────────────────────────────────────────────────────────
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
function riskColor(score) {
  if (score >= 70) return 'var(--critical)'
  if (score >= 40) return 'var(--suspicious)'
  return 'var(--safe)'
}

// ── Sub-components ─────────────────────────────────────────────────────────
function MetaRow({ label, value, mono = true }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '5px 0', borderBottom: '1px solid var(--border)', gap: '12px' }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 500, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: '11px', fontFamily: mono ? 'var(--font-mono)' : 'var(--font-display)', color: 'var(--text-secondary)', textAlign: 'right', wordBreak: 'break-all' }}>{value ?? '—'}</span>
    </div>
  )
}

function SectionTitle({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
      {icon} {text}
    </div>
  )
}

function RiskGauge({ score }) {
  const color = riskColor(score)
  const angle = (score / 100) * 180
  const rad = (a) => (a * Math.PI) / 180
  const r = 50
  const endX = 60 + r * Math.cos(rad(180 - angle))
  const endY = 62 - r * Math.sin(rad(180 - angle))
  const arcPath = `M 10 62 A ${r} ${r} 0 0 1 ${endX} ${endY}`
  const bgPath  = `M 10 62 A ${r} ${r} 0 0 1 110 62`
  const decision =
    score >= 81 ? 'AUTO-BLOCK' :
    score >= 61 ? 'REVIEW' :
    score >= 31 ? 'ALERT' : 'LOG ONLY'
  const decisionColor =
    score >= 81 ? 'var(--critical)' :
    score >= 61 ? 'var(--suspicious)' :
    score >= 31 ? 'var(--safe)' : 'var(--text-muted)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <svg width="120" height="72" viewBox="0 0 120 72">
        <path d={bgPath} fill="none" stroke="var(--border)" strokeWidth="6" strokeLinecap="round"/>
        <path d={arcPath} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${color}66)` }}/>
        <text x="60" y="56" textAnchor="middle" fill={color} fontSize="22" fontWeight="700" fontFamily="var(--font-mono)">{score}</text>
        <text x="60" y="70" textAnchor="middle" fill="var(--text-muted)" fontSize="8" fontFamily="var(--font-mono)">/ 100</text>
      </svg>
      <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: decisionColor, letterSpacing: '0.5px' }}>{decision}</span>
    </div>
  )
}

function DetailConfidenceBar({ label, sublabel, value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span>
          {sublabel && <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: '6px' }}>{sublabel}</span>}
        </div>
        <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', fontWeight: 600, color }}>{pct}%</span>
      </div>
      <div className="confidence-bar" style={{ height: '6px' }}>
        <div className="confidence-bar-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }}/>
      </div>
    </div>
  )
}

function AbuseScore({ score }) {
  const color = score > 50 ? 'var(--critical)' : score > 20 ? 'var(--suspicious)' : 'var(--success)'
  const label = score > 50 ? 'HIGH RISK' : score > 20 ? 'MODERATE' : 'LOW'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <div style={{ width: '42px', height: '42px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: `${color}15`, border: `2px solid ${color}44` }}>
        <span style={{ fontSize: '14px', fontFamily: 'var(--font-mono)', fontWeight: 700, color }}>{score}</span>
      </div>
      <div>
        <div style={{ fontSize: '11px', fontWeight: 600, color }}>{label}</div>
        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>AbuseIPDB Score</div>
      </div>
    </div>
  )
}

function ActionTimeline({ history }) {
  if (!history || history.length === 0) {
    return <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic', padding: '8px 0' }}>No actions recorded</div>
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {history.map((h, i) => (
        <div key={i} style={{ display: 'flex', gap: '10px', padding: '6px 0', borderLeft: '2px solid var(--border)', marginLeft: '6px', paddingLeft: '14px', position: 'relative' }}>
          <div style={{ position: 'absolute', left: '-4px', top: '10px', width: '8px', height: '8px', borderRadius: '50%', background: 'var(--bg-panel)', border: '2px solid var(--safe)' }}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {String(h.action).replace(/_/g, ' ').toUpperCase()}
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {h.analyst} · {new Date(h.time).toLocaleTimeString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Tabs ───────────────────────────────────────────────────────────────────
const TABS = ['Overview', 'Enrichment', 'MITRE', 'Session']

// ── Main Component ─────────────────────────────────────────────────────────
export default function AlertDetail({ alert, onClose }) {
  const [tab, setTab] = useState('Overview')
  const [loading, setLoading] = useState(null)
  const [localStatus, setLocalStatus] = useState(alert.status)
  const [localBlocked, setLocalBlocked] = useState(alert.auto_blocked)
  const navigate = useNavigate()

  const color = severityColor(alert.threat_level)

  const handleAction = useCallback(async (action) => {
    setLoading(action)
    const prev = { status: localStatus, blocked: localBlocked }
    if (action === 'block') setLocalBlocked(true)
    if (action === 'dismiss') setLocalStatus('dismissed')
    if (action === 'isolate') setLocalStatus('isolated')
    try {
      await postAction(alert.id, action)
    } catch {
      setLocalStatus(prev.status)
      setLocalBlocked(prev.blocked)
    } finally {
      setLoading(null)
    }
  }, [alert.id, localStatus, localBlocked])

  const riskExplanation =
    alert.risk_score >= 81 ? `Auto-blocked: risk score ${Math.round(alert.risk_score)} exceeded threshold 81. ML×40% + AbuseIPDB×20% + behavior×30% + asset×10%.` :
    alert.risk_score >= 61 ? `Flagged for review: risk score ${Math.round(alert.risk_score)} is in the review range (61–80). Analyst action recommended.` :
    alert.risk_score >= 31 ? `Alert generated: risk score ${Math.round(alert.risk_score)} crossed alert threshold (31). Monitor for escalation.` :
    `Logged silently: risk score ${Math.round(alert.risk_score ?? 0)} below alert threshold (31). Low priority.`

  const sectionStyle = { padding: '14px 0', borderBottom: '1px solid var(--border)' }

  return (
    <div className="animate-slide-right" style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      borderLeft: '1px solid var(--border)', background: 'var(--bg-surface)',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 16px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-panel)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className={`badge ${alert.threat_level === 2 ? 'badge-critical' : alert.threat_level === 1 ? 'badge-suspicious' : 'badge-muted'}`}>
            {severityLabel(alert.threat_level)}
          </span>
          <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{alert.attack_type}</span>
          {localBlocked && (
            <span className="badge badge-blocked" style={{ fontSize: '8px', display: 'flex', alignItems: 'center', gap: '3px' }}>
              <LockIcon size={8} /> AUTO-BLOCKED
            </span>
          )}
        </div>
        <button className="btn btn-ghost" onClick={onClose} style={{ padding: '4px' }}>
          <XIcon size={16} />
        </button>
      </div>

      {/* Tab bar */}
      <div style={{
        display: 'flex', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-panel)', flexShrink: 0,
      }}>
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1, padding: '8px 4px', fontSize: '11px', fontWeight: tab === t ? 600 : 400,
              color: tab === t ? 'var(--text-primary)' : 'var(--text-muted)',
              background: 'transparent', border: 'none', cursor: 'pointer',
              borderBottom: tab === t ? `2px solid ${color}` : '2px solid transparent',
              fontFamily: 'var(--font-display)', transition: 'all var(--transition-fast)',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 16px' }}>

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && (
          <>
            {/* Risk + confidence */}
            <div style={{ ...sectionStyle, display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
              <RiskGauge score={Math.round(alert.risk_score ?? 0)} />
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <DetailConfidenceBar label="CIC-IDS 2017" sublabel="XGBoost + LightGBM" value={alert.cic_confidence} color={color} />
                <DetailConfidenceBar label="Overall" value={alert.confidence} color="var(--accent)" />
              </div>
            </div>

            {/* Risk explanation */}
            <div style={sectionStyle}>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.5, padding: '8px 10px', background: 'var(--bg-panel)', borderRadius: 'var(--radius-sm)', borderLeft: `2px solid ${riskColor(alert.risk_score ?? 0)}` }}>
                {riskExplanation}
              </div>
            </div>

            {/* Threat Intel */}
            <div style={sectionStyle}>
              <SectionTitle icon={<GlobeIcon />} text="Threat Intelligence" />
              <AbuseScore score={alert.abuse_score ?? 0} />
            </div>

            {/* AI Analysis */}
            <div style={sectionStyle}>
              <SectionTitle icon={<ZapIcon />} text="AI Analysis" />
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6, padding: '10px 12px', background: 'var(--bg-panel)', borderRadius: 'var(--radius-sm)', borderLeft: '2px solid var(--success)' }}>
                {alert.ai_explanation ?? 'No analysis available.'}
              </div>
            </div>

            {/* Metadata */}
            <div style={sectionStyle}>
              <SectionTitle icon={<FileIcon />} text="Alert Metadata" />
              <MetaRow label="Alert ID"    value={alert.id} />
              <MetaRow label="Timestamp"   value={new Date(alert.timestamp).toLocaleString()} />
              <MetaRow label="Source IP"   value={alert.src_ip} />
              <MetaRow label="Destination" value={`${alert.dst_ip ?? '—'}${alert.dst_port ? `:${alert.dst_port}` : ''}`} />
              <MetaRow label="Protocol"    value={`${alert.proto ?? '—'}${alert.dst_port ? `/${alert.dst_port}` : ''}`} />
              <MetaRow label="Agent"       value={`${alert.agent_name ?? '—'} (${alert.agent_id ?? '—'})`} />
              <MetaRow label="Rule Level"  value={alert.rule_level} />
              <MetaRow label="Rule"        value={alert.rule_description} mono={false} />
              <MetaRow label="Status"      value={localStatus?.toUpperCase()} />
              <MetaRow label="Ingested"    value={alert.ingested_at ? new Date(alert.ingested_at).toLocaleString() : '—'} />
            </div>

            {/* Action History */}
            <div style={sectionStyle}>
              <SectionTitle icon={<ClockIcon />} text="Action History" />
              <ActionTimeline history={alert.action_history} />
            </div>

            <div style={{ height: '16px' }} />
          </>
        )}

        {/* ── ENRICHMENT TAB ── */}
        {tab === 'Enrichment' && (
          <>
            {/* GeoIP */}
            <div style={sectionStyle}>
              <SectionTitle icon={<GlobeIcon />} text="GeoIP" />
              {alert.geoip && Object.keys(alert.geoip).length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {alert.geoip.country && <MetaRow label="Country" value={alert.geoip.country_name ?? alert.geoip.country} />}
                  {alert.geoip.city    && <MetaRow label="City"    value={alert.geoip.city} />}
                  {alert.geoip.lat != null && <MetaRow label="Coords" value={`${alert.geoip.lat?.toFixed(2)}, ${alert.geoip.lon?.toFixed(2)}`} />}
                  {alert.geoip.isp     && <MetaRow label="ISP"     value={alert.geoip.isp} />}
                  {alert.geoip.is_tor  && <MetaRow label="Tor Exit" value="YES" />}
                  {alert.geoip.is_vpn  && <MetaRow label="VPN"     value="YES" />}
                </div>
              ) : (
                <p style={{ fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic' }}>No GeoIP data available</p>
              )}
            </div>

            {/* Abuse */}
            <div style={sectionStyle}>
              <SectionTitle icon={<GlobeIcon />} text="Threat Intelligence" />
              <AbuseScore score={alert.abuse_score ?? 0} />
              {alert.is_known_attacker && (
                <div style={{ marginTop: '8px' }}>
                  <span className="badge badge-critical" style={{ fontSize: '9px' }}>Known Attacker</span>
                </div>
              )}
            </div>

            {/* Asset */}
            <div style={sectionStyle}>
              <SectionTitle icon={<ServerIcon />} text="Asset Context" />
              {alert.asset_name ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <MetaRow label="Asset" value={alert.asset_name} />
                  <MetaRow label="Criticality" value={alert.asset_crit ?? '—'} />
                  <MetaRow label="Owner" value={alert.asset_owner ?? '—'} />
                </div>
              ) : (
                <p style={{ fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic' }}>No asset data available</p>
              )}
            </div>
            <div style={{ height: '16px' }} />
          </>
        )}

        {/* ── MITRE TAB ── */}
        {tab === 'MITRE' && (
          <>
            <div style={sectionStyle}>
              <SectionTitle icon={<TargetIcon />} text="ATT&CK Techniques" />
              {alert.mitre_techniques && alert.mitre_techniques.length > 0 ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {alert.mitre_techniques.map((t, i) => (
                    <div key={i} style={{ padding: '4px 10px', borderRadius: '3px', background: 'var(--accent-bg)', border: '1px solid rgba(59,130,246,0.2)' }}>
                      <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontWeight: 600 }}>
                        {t.technique_id}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{t.technique_name}</div>
                      {t.tactic && <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{t.tactic}</div>}
                    </div>
                  ))}
                </div>
              ) : (
                <div>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', opacity: 0.4, marginBottom: '8px' }}>
                    {['T1046 — Network Service Discovery', 'T1110 — Brute Force', 'T1498 — Network DoS'].map((t) => (
                      <span key={t} className="badge badge-muted" style={{ fontSize: '9px' }}>{t}</span>
                    ))}
                  </div>
                  <p style={{ fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic' }}>Live MITRE mapping requires backend enrichment</p>
                </div>
              )}
            </div>
            <div style={{ height: '16px' }} />
          </>
        )}

        {/* ── SESSION TAB ── */}
        {tab === 'Session' && (
          <>
            <div style={sectionStyle}>
              <SectionTitle icon={<ClockIcon />} text="Session Info" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                {[
                  { label: 'Session ID', value: alert.session_id || '—' },
                  { label: 'Events', value: alert.session_count ?? 1 },
                  { label: 'Duration', value: alert.session_dur ? `${alert.session_dur}s` : '—' },
                  { label: 'Status', value: localStatus?.toUpperCase() ?? '—' },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: '8px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px' }}>{label}</div>
                    <div style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ height: '16px' }} />
          </>
        )}
      </div>

      {/* Footer actions */}
      <div style={{ display: 'flex', gap: '8px', padding: '12px 16px', borderTop: '1px solid var(--border)', background: 'var(--bg-panel)', flexShrink: 0 }}>
        {!localBlocked && localStatus !== 'dismissed' && (
          <button className="btn btn-danger" style={{ flex: 1 }} disabled={loading === 'block'} onClick={() => handleAction('block')}>
            <LockIcon size={13} /> {loading === 'block' ? 'Blocking...' : 'Block IP'}
          </button>
        )}
        {localStatus !== 'isolated' && localStatus !== 'dismissed' && (
          <button className="btn btn-warn" style={{ flex: 1 }} disabled={loading === 'isolate'} onClick={() => handleAction('isolate')}>
            <UserXIcon size={13} /> {loading === 'isolate' ? 'Isolating...' : 'Isolate Agent'}
          </button>
        )}
        {localStatus !== 'dismissed' && (
          <button className="btn" style={{ flex: 1 }} disabled={loading === 'dismiss'} onClick={() => handleAction('dismiss')}>
            <XCircleIcon size={13} /> {loading === 'dismiss' ? 'Dismissing...' : 'Dismiss'}
          </button>
        )}
        {alert.src_ip && (
          <button
            className="btn btn-ghost"
            style={{ flexShrink: 0 }}
            onClick={() => { onClose?.(); navigate(`/attackers/${alert.src_ip}`) }}
            title="View attacker profile"
          >
            Profile
          </button>
        )}
      </div>
    </div>
  )
}
