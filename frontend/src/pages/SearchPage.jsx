import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { fetchSearch } from '../lib/api'
import AlertDetail from '../components/AlertDetail'

const ATTACK_TYPES = ['DDoS', 'DoS', 'PortScan', 'Brute Force', 'Bot', 'Web Attack', 'Shellcode', 'Recon', 'Exploits']
const STATUSES = ['active', 'dismissed', 'isolated']

const SEV = {
  2: 'text-critical',
  1: 'text-suspicious',
  0: 'text-muted',
}

function ResultRow({ alert, onClick }) {
  const navigate = useNavigate()
  return (
    <tr
      onClick={() => onClick(alert)}
      className="border-b border-border hover:bg-white/3 cursor-pointer transition-colors"
    >
      <td className="px-3 py-2 font-mono text-[10px] text-muted/70">
        {new Date(alert.timestamp).toLocaleString()}
      </td>
      <td className={`px-3 py-2 font-mono text-[10px] font-600 ${SEV[alert.threat_level] ?? 'text-muted'}`}>
        {alert.attack_type}
      </td>
      <td className="px-3 py-2 font-mono text-[10px]">
        <span
          className="text-safe hover:underline cursor-pointer"
          onClick={(e) => { e.stopPropagation(); navigate(`/attackers/${alert.src_ip}`) }}
        >
          {alert.src_ip}
        </span>
      </td>
      <td className="px-3 py-2 font-mono text-[10px] text-white/60">{Math.round(alert.risk_score ?? 0)}</td>
      <td className="px-3 py-2 font-mono text-[10px] text-muted/60">{alert.status}</td>
      <td className="px-3 py-2">
        {alert.auto_blocked && (
          <span className="font-mono text-[8px] uppercase px-1.5 py-0.5 border border-safe/30 text-safe bg-safe/10">
            Blocked
          </span>
        )}
      </td>
    </tr>
  )
}

export default function SearchPage() {
  const [filters, setFilters] = useState({ src_ip: '', attack_type: '', threat_level: '', status: '', since: '', auto_blocked: '' })
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)

  const set = (k, v) => setFilters(f => ({ ...f, [k]: v }))

  const handleSearch = async () => {
    setLoading(true)
    try {
      const params = {}
      if (filters.src_ip)       params.src_ip       = filters.src_ip
      if (filters.attack_type)  params.attack_type  = filters.attack_type
      if (filters.threat_level !== '') params.threat_level = parseInt(filters.threat_level)
      if (filters.status)       params.status       = filters.status
      if (filters.since)        params.since        = filters.since
      if (filters.auto_blocked) params.auto_blocked = filters.auto_blocked === 'true'
      const data = await fetchSearch(params)
      setResults(Array.isArray(data) ? data : [])
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const exportJSON = () => {
    if (!results) return
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `lighthouse-alerts-${Date.now()}.json`
    a.click()
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-border">
        <h1 className="font-mono text-[11px] uppercase tracking-widest text-muted mb-4">Alert Search</h1>

        {/* Filters */}
        <div className="grid grid-cols-3 gap-3 mb-3">
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Source IP</label>
            <input
              value={filters.src_ip}
              onChange={e => set('src_ip', e.target.value)}
              placeholder="e.g. 1.2.3.4"
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50 placeholder:text-muted/40"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Attack Type</label>
            <select
              value={filters.attack_type}
              onChange={e => set('attack_type', e.target.value)}
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50"
            >
              <option value="">All</option>
              {ATTACK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Threat Level</label>
            <select
              value={filters.threat_level}
              onChange={e => set('threat_level', e.target.value)}
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50"
            >
              <option value="">All</option>
              <option value="2">Critical</option>
              <option value="1">Suspicious</option>
              <option value="0">Unknown</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Status</label>
            <select
              value={filters.status}
              onChange={e => set('status', e.target.value)}
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50"
            >
              <option value="">All</option>
              {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Since (ISO)</label>
            <input
              value={filters.since}
              onChange={e => set('since', e.target.value)}
              placeholder="2026-05-25T00:00:00"
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50 placeholder:text-muted/40"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase tracking-widest text-muted">Auto Blocked</label>
            <select
              value={filters.auto_blocked}
              onChange={e => set('auto_blocked', e.target.value)}
              className="bg-panel border border-border px-3 py-1.5 font-mono text-[11px] text-white outline-none focus:border-safe/50"
            >
              <option value="">Any</option>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-4 py-1.5 border border-safe/40 text-safe hover:bg-safe/10 font-mono text-[10px] uppercase tracking-wider transition-colors disabled:opacity-40"
          >
            {loading ? '···' : 'Search'}
          </button>
          {results?.length > 0 && (
            <button
              onClick={exportJSON}
              className="px-4 py-1.5 border border-border text-muted hover:text-white hover:bg-white/5 font-mono text-[10px] uppercase tracking-wider transition-colors"
            >
              Export JSON
            </button>
          )}
        </div>
      </div>

      {/* Results table */}
      <div className="flex-1 overflow-y-auto">
        {results === null ? (
          <div className="flex items-center justify-center h-full">
            <p className="font-mono text-[10px] text-muted/50">Run a search to view results</p>
          </div>
        ) : results.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2">
            <p className="font-mono text-[10px] text-muted/50">No alerts match your filters</p>
            <p className="font-mono text-[9px] text-muted/30 italic">Try removing a filter or broadening your criteria</p>
          </div>
        ) : (
          <>
          <div className="px-3 py-2 border-b border-border">
            <span className="font-mono text-[9px] text-muted/60">{results.length} result{results.length !== 1 ? 's' : ''} found</span>
          </div>
          <table className="w-full">
            <thead className="sticky top-0 bg-surface border-b border-border">
              <tr>
                {['Timestamp', 'Attack Type', 'Source IP', 'Risk', 'Status', ''].map(h => (
                  <th key={h} className="px-3 py-2 text-left font-mono text-[8px] uppercase tracking-widest text-muted">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map(a => <ResultRow key={a.id} alert={a} onClick={setSelected} />)}
            </tbody>
          </table>
          </>
        )}
      </div>

      <AnimatePresence>
        {selected && (
          <AlertDetail key={selected.id} alert={selected} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>
    </div>
  )
}
