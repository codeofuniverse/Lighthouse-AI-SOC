import { useState } from 'react'
import { motion } from 'framer-motion'
import { fetchSearch } from '../lib/api'

const ATTACK_TYPES = ['DDoS', 'DoS', 'PortScan', 'Brute Force', 'Bot', 'Web Attack', 'Shellcode', 'Recon', 'Exploits']

export default function SearchOverlay({ onClose }) {
  const [filters, setFilters] = useState({
    src_ip: '',
    attack_type: '',
    threat_level: '',
    status: '',
    since: '',
    auto_blocked: '',
  })
  const [limit, setLimit] = useState(200)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)

  const updateFilter = (key, value) => {
    setFilters((current) => ({ ...current, [key]: value }))
  }

  const handleSearch = async () => {
    setLoading(true)
    try {
      const params = { limit }
      if (filters.src_ip) params.src_ip = filters.src_ip
      if (filters.attack_type) params.attack_type = filters.attack_type
      if (filters.threat_level !== '') params.threat_level = Number(filters.threat_level)
      if (filters.status) params.status = filters.status
      if (filters.since) params.since = filters.since
      if (filters.auto_blocked !== '') params.auto_blocked = filters.auto_blocked === 'true'
      const data = await fetchSearch(params)
      setResults(Array.isArray(data) ? data : [])
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const exportJson = () => {
    if (!results?.length) return
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `lighthouse-export-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/65 p-4 backdrop-blur-sm"
    >
      <motion.div
        initial={{ y: 16, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 16, opacity: 0 }}
        className="hud-corners mx-auto flex h-[calc(100vh-2rem)] max-w-6xl flex-col overflow-hidden border border-border bg-surface"
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-safe">Advanced Search</p>
            <h2 className="mt-1 text-lg font-semibold text-primary">Historical alert lookup</h2>
          </div>
          <button className="btn btn-ghost" onClick={onClose}>Close</button>
        </div>

        <div className="grid gap-3 border-b border-border px-5 py-4 md:grid-cols-3 xl:grid-cols-6">
          <input className="inp inp-mono" placeholder="Source IP" value={filters.src_ip} onChange={(e) => updateFilter('src_ip', e.target.value)} />
          <select className="sel" value={filters.attack_type} onChange={(e) => updateFilter('attack_type', e.target.value)}>
            <option value="">All types</option>
            {ATTACK_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
          </select>
          <select className="sel" value={filters.threat_level} onChange={(e) => updateFilter('threat_level', e.target.value)}>
            <option value="">All severities</option>
            <option value="2">Critical</option>
            <option value="1">Suspicious</option>
            <option value="0">Unknown</option>
          </select>
          <select className="sel" value={filters.status} onChange={(e) => updateFilter('status', e.target.value)}>
            <option value="">All statuses</option>
            <option value="active">Active</option>
            <option value="dismissed">Dismissed</option>
            <option value="isolated">Isolated</option>
          </select>
          <input className="inp" type="datetime-local" value={filters.since} onChange={(e) => updateFilter('since', e.target.value)} />
          <select className="sel" value={filters.auto_blocked} onChange={(e) => updateFilter('auto_blocked', e.target.value)}>
            <option value="">Any blocking state</option>
            <option value="true">Blocked</option>
            <option value="false">Not blocked</option>
          </select>
        </div>

        <div className="flex items-center gap-2 border-b border-border px-5 py-3">
          <button className="btn" onClick={handleSearch} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
          <select className="sel" value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            <option value="100">Limit: 100</option>
            <option value="200">Limit: 200</option>
            <option value="500">Limit: 500</option>
            <option value="1000">Limit: 1000</option>
          </select>
          {results && <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">{results.length} results</span>}
          {!!results?.length && <button className="btn btn-ghost ml-auto" onClick={exportJson}>Export JSON</button>}
        </div>

        <div className="flex-1 overflow-auto">
          {results === null ? (
            <div className="flex h-full items-center justify-center text-sm text-dim">Set filters and run a search.</div>
          ) : results.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-dim">No alerts match your filters.</div>
          ) : (
            <table className="w-full border-collapse">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-border">
                  {['Severity', 'Attack Type', 'Source IP', 'Host', 'Risk', 'Status', 'Time'].map((heading) => (
                    <th key={heading} className="px-4 py-3 text-left font-mono text-[9px] uppercase tracking-[0.28em] text-muted">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((alert) => (
                  <tr key={alert.id} className="border-b border-border/70 hover:bg-panel">
                    <td className="px-4 py-3">
                      <span className={`badge ${alert.threat_level === 2 ? 'badge-critical' : alert.threat_level === 1 ? 'badge-suspicious' : 'badge-muted'}`}>
                        {alert.threat_level === 2 ? 'Critical' : alert.threat_level === 1 ? 'Suspicious' : 'Unknown'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-secondary">{alert.attack_type}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-secondary">{alert.src_ip}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-muted">{alert.agent_name}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-secondary">{Math.round(alert.risk_score ?? 0)}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-muted">{alert.status}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-dim">{new Date(alert.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}
