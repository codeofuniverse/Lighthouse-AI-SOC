import { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchHealth } from '../lib/api'

const THRESHOLDS = [
  { label: 'Auto Block', value: 81, color: '#ff3b3b' },
  { label: 'Review',     value: 61, color: '#f59e0b' },
  { label: 'Alert',      value: 31, color: '#00d4ff' },
  { label: 'Log',        value: 0,  color: '#4a5568' },
]

const REFRESH_OPTIONS = [
  { label: '5 seconds',  value: 5000 },
  { label: '10 seconds', value: 10000 },
  { label: '30 seconds', value: 30000 },
  { label: '60 seconds', value: 60000 },
]

function ThresholdSlider({ label, value, color }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between">
        <span className="font-mono text-[9px] uppercase tracking-widest text-muted">{label} threshold</span>
        <span className="font-mono text-[10px]" style={{ color }}>{value}</span>
      </div>
      <input
        type="range" min="0" max="100" value={value} disabled
        className="w-full h-1 rounded-full appearance-none cursor-default opacity-60"
        style={{ accentColor: color }}
      />
    </div>
  )
}

function Badge({ value, positive }) {
  return (
    <span className={`font-mono text-[9px] uppercase px-2 py-0.5 border rounded-sm ${positive ? 'border-safe/40 bg-safe/10 text-safe' : 'border-critical/40 bg-critical/10 text-critical'}`}>
      {value}
    </span>
  )
}

function Row({ label, children }) {
  return (
    <div className="flex justify-between items-center">
      <span className="font-mono text-[10px] text-white/70">{label}</span>
      {children}
    </div>
  )
}

export default function SettingsPage() {
  const queryClient = useQueryClient()
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: fetchHealth, refetchInterval: 15000 })

  const [refreshInterval, setRefreshInterval] = useState(
    () => parseInt(localStorage.getItem('lh_refresh_interval') ?? '10000', 10)
  )
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    localStorage.setItem('lh_refresh_interval', String(refreshInterval))
    // Invalidate alerts query so new interval is picked up on next component mount
    queryClient.invalidateQueries({ queryKey: ['alerts'] })
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    setRefreshInterval(10000)
    localStorage.removeItem('lh_refresh_interval')
    localStorage.removeItem('lh_theme')
    setSaved(false)
  }

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      <h1 className="font-mono text-[11px] uppercase tracking-widest text-muted">Configuration</h1>

      {/* Dashboard preferences */}
      <div className="p-4 bg-panel border border-border space-y-4">
        <p className="font-mono text-[9px] uppercase tracking-widest text-muted">Dashboard Preferences</p>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Alert Feed Refresh Interval</label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="bg-base border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50"
          >
            {REFRESH_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <p className="font-mono text-[9px] text-muted/50 italic">Applies on next page load</p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleSave}
            className="px-4 py-1.5 border border-safe/40 text-safe hover:bg-safe/10 font-mono text-[10px] uppercase tracking-wider transition-colors"
          >
            {saved ? 'Saved ✓' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-1.5 border border-border text-muted hover:text-white hover:bg-white/5 font-mono text-[10px] uppercase tracking-wider transition-colors"
          >
            Reset to Defaults
          </button>
        </div>
      </div>

      {/* Decision thresholds */}
      <div className="p-4 bg-panel border border-border space-y-4">
        <p className="font-mono text-[9px] uppercase tracking-widest text-muted">Decision Thresholds (read-only)</p>
        {THRESHOLDS.map(t => <ThresholdSlider key={t.label} {...t} />)}
        <p className="font-mono text-[9px] text-muted/50 italic">Adjust in backend/decision_engine.py</p>
      </div>

      {/* System status */}
      <div className="p-4 bg-panel border border-border">
        <p className="font-mono text-[9px] uppercase tracking-widest text-muted mb-3">System Status</p>
        <div className="space-y-3">
          <Row label="Backend">
            <Badge value={health ? 'Online' : 'Offline'} positive={!!health} />
          </Row>
          <Row label="Redis">
            <Badge value={health?.redis ? 'Connected' : 'Offline'} positive={!!health?.redis} />
          </Row>
          <Row label="LLM Service">
            <Badge value={health?.llm_failures === 0 ? 'OK' : `${health?.llm_failures ?? '?'} failures`} positive={!health?.llm_failures} />
          </Row>
          <Row label="DB Alert Count">
            <span className="font-mono text-[10px] text-white/60">{health?.db_alerts?.toLocaleString() ?? '—'}</span>
          </Row>
          <Row label="SOAR Mode">
            <Badge value="Dry-Run (see .env)" positive={false} />
          </Row>
        </div>
      </div>

      {/* Dual-model config */}
      <div className="p-4 bg-panel border border-border">
        <p className="font-mono text-[9px] uppercase tracking-widest text-muted mb-3">ML Pipeline</p>
        <div className="space-y-2">
          {[
            ['CIC-IDS-2017 (18-feat, 7-class)', 'cic2017_pipeline_smote.joblib', 'F1 0.99'],
            ['Web-Attack booster (HTTP)',       'cic2017_webattack_v3_http.joblib', 'Recall 0.99'],
          ].map(([name, file, perf]) => (
            <div key={name} className="flex justify-between items-start border-b border-border/40 pb-2 last:border-0">
              <div>
                <p className="font-mono text-[10px] text-white/80">{name}</p>
                <p className="font-mono text-[8px] text-muted">{file}</p>
              </div>
              <span className="font-mono text-[9px] text-safe">{perf}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Risk scoring formula */}
      <div className="p-4 bg-panel border border-border">
        <p className="font-mono text-[9px] uppercase tracking-widest text-muted mb-3">Risk Scoring Formula</p>
        <p className="font-mono text-[10px] text-white/70 leading-relaxed">
          ML × 0.40 + AbuseIPDB × 0.20 + behavior × 0.30 + asset × 0.10
        </p>
        <p className="font-mono text-[9px] text-muted/50 mt-2 italic">Modify weights in pipeline/risk_scorer.py</p>
      </div>
    </div>
  )
}
