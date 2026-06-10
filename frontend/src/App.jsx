import { useEffect, useMemo, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import StatsBar from './components/StatsBar'
import AlertFeed from './components/AlertFeed'
import AlertDetail from './components/AlertDetail'
import AttackChart from './components/AttackChart'
import SearchOverlay from './components/SearchOverlay'
import SearchPage from './pages/SearchPage'
import AttackerPage from './pages/AttackerPage'
import StatsPage from './pages/StatsPage'
import SettingsPage from './pages/SettingsPage'
import { fetchAlerts, fetchHealth, fetchStats } from './lib/api'

const NAV = [
  { to: '/', label: 'Live Feed' },
  { to: '/search', label: 'Search' },
  { to: '/attackers', label: 'Attackers' },
  { to: '/stats', label: 'Statistics' },
  { to: '/settings', label: 'Settings' },
]

function NavBar({ onOpenSearch, hasSelectedAlert, onCloseAlert }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 16px', height: '36px',
      background: 'var(--bg-surface)', borderBottom: '1px solid var(--border)',
      flexShrink: 0,
    }}>
      <nav style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              display: 'inline-flex', alignItems: 'center',
              padding: '4px 10px', borderRadius: '3px', fontSize: '11px',
              fontFamily: 'var(--font-display)', fontWeight: isActive ? 600 : 400,
              color: isActive ? 'var(--safe)' : 'var(--text-muted)',
              background: isActive ? 'var(--safe-bg)' : 'transparent',
              border: isActive ? '1px solid rgba(0,229,255,0.25)' : '1px solid transparent',
              textDecoration: 'none', transition: 'all var(--transition-fast)',
            })}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button
          className="btn btn-ghost"
          style={{ padding: '4px 10px', fontSize: '11px' }}
          onClick={onOpenSearch}
        >
          Advanced Search
        </button>
        {hasSelectedAlert && (
          <button
            className="btn btn-ghost"
            style={{ padding: '4px 10px', fontSize: '11px' }}
            onClick={onCloseAlert}
          >
            Close Detail
          </button>
        )}
      </div>
    </div>
  )
}

function StatusBar({ health, connected, alertCount }) {
  return (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-dot" style={{ background: 'var(--success)' }} />
        System: {(health?.status ?? 'ok').toUpperCase()}
      </div>
      <div className="status-item">
        <span
          className="status-dot"
          style={{
            background: connected ? 'var(--success)' : 'var(--muted)',
            animation: connected ? 'pulse-dot 2s infinite' : 'none',
          }}
        />
        WebSocket: {connected ? 'Connected' : 'Reconnecting'}
      </div>
      <div className="status-item">
        DB: {(health?.db_alerts ?? 0).toLocaleString()} alerts
      </div>
      {health?.redis === false && (
        <div className="status-item" style={{ color: 'var(--suspicious)' }}>
          Redis: offline
        </div>
      )}
      {health?.llm_failures > 0 && (
        <div className="status-item" style={{ color: 'var(--suspicious)' }}>
          LLM failures: {health.llm_failures}
        </div>
      )}
      <div className="status-item" style={{ marginLeft: 'auto' }}>
        Lighthouse v2.0 · {alertCount.toLocaleString()} alerts loaded
      </div>
    </div>
  )
}

function Dashboard({ connected, onConnectedChange, selectedAlert, setSelectedAlert }) {
  const { data: alerts = [] } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: 10_000,
  })

  const safeAlerts = useMemo(() => (Array.isArray(alerts) ? alerts : []), [alerts])
  const selectedId = selectedAlert?.id ?? null

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Alert feed */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <AlertFeed
          selectedId={selectedId}
          onSelectAlert={setSelectedAlert}
          connected={connected}
          onConnectedChange={onConnectedChange}
        />
      </div>

      {/* Right panel */}
      <div style={{
        width: selectedAlert ? '420px' : '360px', flexShrink: 0,
        display: 'flex', flexDirection: 'column',
        transition: 'width var(--transition-normal)', overflow: 'hidden',
      }}>
        {selectedAlert ? (
          <AlertDetail
            key={selectedAlert.id}
            alert={selectedAlert}
            onClose={() => setSelectedAlert(null)}
          />
        ) : (
          <div style={{ flex: 1, overflowY: 'auto', borderLeft: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
            <AttackChart alerts={safeAlerts} />
          </div>
        )}
      </div>
    </div>
  )
}

function AppShell() {
  const [theme, setTheme] = useState(() => localStorage.getItem('lh_theme') ?? 'dark')
  const [connected, setConnected] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [selectedAlert, setSelectedAlert] = useState(null)
  const location = useLocation()
  const queryClient = useQueryClient()

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 30_000,
  })

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 60_000,
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('lh_theme', theme)
  }, [theme])

  // Derive alert count from shared query cache — no extra fetch
  const cachedAlerts = queryClient.getQueryData(['alerts']) ?? []
  const alertCount = Array.isArray(cachedAlerts) ? cachedAlerts.length : 0

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh',
      background: 'var(--bg-base)', color: 'var(--text-primary)',
    }}>
      <StatsBar
        connected={connected}
        stats={stats}
        health={health}
        theme={theme}
        onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
        panelOpen={true}
        onTogglePanel={() => {}}
      />

      <NavBar
        onOpenSearch={() => setSearchOpen(true)}
        hasSelectedAlert={!!selectedAlert && location.pathname === '/'}
        onCloseAlert={() => setSelectedAlert(null)}
      />

      {/* Page content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Routes>
          <Route
            path="/"
            element={
              <Dashboard
                connected={connected}
                onConnectedChange={setConnected}
                selectedAlert={selectedAlert}
                setSelectedAlert={setSelectedAlert}
              />
            }
          />
          <Route path="/search" element={
            <div style={{ flex: 1, overflow: 'hidden', height: '100%' }}>
              <SearchPage />
            </div>
          } />
          <Route path="/attackers" element={
            <div style={{ flex: 1, overflow: 'auto', height: '100%' }}>
              <AttackerPage />
            </div>
          } />
          <Route path="/attackers/:ip" element={
            <div style={{ flex: 1, overflow: 'auto', height: '100%' }}>
              <AttackerPage />
            </div>
          } />
          <Route path="/stats" element={
            <div style={{ flex: 1, overflow: 'auto', height: '100%' }}>
              <StatsPage />
            </div>
          } />
          <Route path="/settings" element={
            <div style={{ flex: 1, overflow: 'auto', height: '100%' }}>
              <SettingsPage />
            </div>
          } />
        </Routes>
      </div>

      <StatusBar health={health} connected={connected} alertCount={alertCount} />

      <AnimatePresence>
        {searchOpen && <SearchOverlay onClose={() => setSearchOpen(false)} />}
      </AnimatePresence>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
