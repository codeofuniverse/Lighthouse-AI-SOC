import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

const rootHttp = axios.create({
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

let backendReachable = true

function isNetworkError(error) {
  return !error?.response
}

function markBackendOffline(error) {
  if (isNetworkError(error)) {
    backendReachable = false
  }
}

function reviveBackend() {
  backendReachable = true
}

async function safeGet(path, fallback, transform = (data) => data, options) {
  if (!backendReachable) {
    return typeof fallback === 'function' ? fallback() : fallback
  }

  try {
    const { data } = await http.get(path, options)
    reviveBackend()
    return transform(data)
  } catch (error) {
    markBackendOffline(error)
    if (isNetworkError(error)) {
      return typeof fallback === 'function' ? fallback() : fallback
    }
    throw error
  }
}

async function safeRootGet(path, fallback, transform = (data) => data, options) {
  if (!backendReachable) {
    return typeof fallback === 'function' ? fallback() : fallback
  }

  try {
    const { data } = await rootHttp.get(path, options)
    reviveBackend()
    return transform(data)
  } catch (error) {
    markBackendOffline(error)
    if (isNetworkError(error)) {
      return typeof fallback === 'function' ? fallback() : fallback
    }
    throw error
  }
}

async function safePost(path, fallback) {
  if (!backendReachable) {
    return typeof fallback === 'function' ? fallback() : fallback
  }

  try {
    const { data } = await http.post(path)
    reviveBackend()
    return data
  } catch (error) {
    markBackendOffline(error)
    if (isNetworkError(error)) {
      return typeof fallback === 'function' ? fallback() : fallback
    }
    throw error
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}

const CIC_CONFS  = [0.94, 0.88, 0.71, 0.82, 0.43]
const PROTOS     = ['TCP', 'TCP', 'TCP', 'TCP', 'UDP']

function buildMockAlert(alert, index) {
  // UNSW retired (2026) — CIC is the sole ML detector. unsw_confidence stays null.
  const cic  = alert.cic_confidence  ?? CIC_CONFS[index % 5]
  return {
    dst_ip: alert.dst_ip ?? '172.28.0.10',
    dst_port: alert.dst_port ?? 443,
    proto: alert.proto ?? PROTOS[index % 5],
    risk_score: alert.risk_score ?? Math.round((alert.confidence ?? 0.5) * 100),
    cic_confidence: cic,
    unsw_confidence: null,
    abuse_score: alert.abuse_score ?? [82, 67, 55, 34, 18][index % 5],
    is_known_attacker: alert.is_known_attacker ?? index < 2,
    ingested_at: alert.ingested_at ?? alert.timestamp,
    geoip: alert.geoip ?? {
      country: ['Germany', 'United States', 'Singapore', 'Netherlands', 'India'][index % 5],
      city: ['Frankfurt', 'Ashburn', 'Singapore', 'Amsterdam', 'Mumbai'][index % 5],
      lat: [50.11, 39.04, 1.35, 52.37, 19.07][index % 5],
      lon: [8.68, -77.49, 103.82, 4.9, 72.87][index % 5],
      is_tor: index === 0,
      is_vpn: index < 3,
    },
    mitre_techniques: alert.mitre_techniques ?? [
      { technique_id: 'T1110', technique_name: 'Brute Force', tactic: 'Credential Access' },
      { technique_id: 'T1046', technique_name: 'Network Service Discovery', tactic: 'Discovery' },
    ].slice(0, index % 2 === 0 ? 1 : 2),
    session_count: alert.session_count ?? 4 + index * 3,
    session_dur: alert.session_dur ?? 90 + index * 45,
    action_history: alert.action_history ?? [],
    ...alert,
    cic_confidence: cic,
    unsw_confidence: null,
  }
}

const MOCK_ALERT_STORE = [
  {
    id: 'a1b2c3d4',
    timestamp: new Date(Date.now() - 2 * 60000).toISOString(),
    rule_level: 12,
    rule_description: 'Multiple authentication failures followed by success',
    agent_name: 'prod-web-01',
    src_ip: '185.220.101.47',
    threat_level: 2,
    attack_type: 'Brute Force',
    confidence: 0.94,
    ai_explanation: 'Detected 47 failed SSH login attempts from 185.220.101.47 over 3 minutes, followed by a successful authentication.',
    auto_blocked: false,
    status: 'active',
  },
  {
    id: 'e5f6g7h8',
    timestamp: new Date(Date.now() - 8 * 60000).toISOString(),
    rule_level: 10,
    rule_description: 'Outbound connection to known C2 infrastructure',
    agent_name: 'corp-laptop-14',
    src_ip: '10.0.1.42',
    threat_level: 2,
    attack_type: 'C2 Beacon',
    confidence: 0.88,
    ai_explanation: 'Host 10.0.1.42 is making periodic HTTP requests to 93.184.216.34 every 60 seconds with consistent payload size.',
    auto_blocked: true,
    status: 'blocked',
  },
  {
    id: 'i9j0k1l2',
    timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
    rule_level: 8,
    rule_description: 'Unusual outbound data volume detected',
    agent_name: 'db-server-02',
    src_ip: '10.0.2.15',
    threat_level: 1,
    attack_type: 'Data Exfiltration',
    confidence: 0.71,
    ai_explanation: 'Database server transmitted 2.3GB to an external IP over 12 minutes via port 443.',
    auto_blocked: false,
    status: 'active',
  },
  {
    id: 'm3n4o5p6',
    timestamp: new Date(Date.now() - 23 * 60000).toISOString(),
    rule_level: 7,
    rule_description: 'Port scan detected from internal host',
    agent_name: 'dev-workstation-08',
    src_ip: '10.0.3.88',
    threat_level: 1,
    attack_type: 'PortScan',
    confidence: 0.82,
    ai_explanation: 'Internal host scanned 1,247 ports across 12 subnet hosts over 4 minutes using SYN packets.',
    auto_blocked: false,
    status: 'active',
  },
  {
    id: 'q7r8s9t0',
    timestamp: new Date(Date.now() - 41 * 60000).toISOString(),
    rule_level: 5,
    rule_description: 'Anomalous DNS query volume',
    agent_name: 'corp-laptop-22',
    src_ip: '10.0.1.77',
    threat_level: 0,
    attack_type: 'DNS Anomaly',
    confidence: 0.43,
    ai_explanation: 'Host generated 3,200 DNS queries in 5 minutes, with 60% resolving to non-existent domains.',
    auto_blocked: false,
    status: 'active',
  },
].map(buildMockAlert)

function getMockAlerts() {
  return clone(MOCK_ALERT_STORE)
}

function getMockStats() {
  const alerts = MOCK_ALERT_STORE
  return {
    total_today: alerts.length,
    critical: alerts.filter((alert) => alert.threat_level === 2).length,
    suspicious: alerts.filter((alert) => alert.threat_level === 1).length,
    auto_blocked: alerts.filter((alert) => alert.auto_blocked).length,
  }
}

function getMockHealth() {
  return {
    status: 'mock',
    db_alerts: MOCK_ALERT_STORE.length,
  }
}

function filterMockSearch(params = {}) {
  return getMockAlerts().filter((alert) => {
    if (params.src_ip && alert.src_ip !== params.src_ip) return false
    if (params.attack_type && alert.attack_type !== params.attack_type) return false
    if (params.threat_level != null && alert.threat_level !== params.threat_level) return false
    if (params.status && alert.status !== params.status) return false
    if (params.auto_blocked != null && alert.auto_blocked !== params.auto_blocked) return false
    if (params.since && new Date(alert.timestamp) < new Date(params.since)) return false
    return true
  })
}

export const fetchAlerts = async () =>
  safeGet('/alerts', getMockAlerts, (data) => (Array.isArray(data) ? data : data.alerts ?? []))

export const fetchAlert = async (id) =>
  safeGet(`/alerts/${id}`, () => getMockAlerts().find((alert) => alert.id === id) ?? null)

export const fetchStats = async () =>
  safeGet('/stats', getMockStats)

export const postAction = async (id, action) =>
  safePost(`/alerts/${id}/${action}`, () => {
    const alert = MOCK_ALERT_STORE.find((entry) => entry.id === id)
    if (!alert) return null
    if (action === 'block') {
      alert.auto_blocked = true
      alert.status = 'blocked'
    }
    if (action === 'dismiss') {
      alert.status = 'dismissed'
    }
    if (action === 'isolate') {
      alert.status = 'isolated'
    }
    alert.action_history = [
      ...(alert.action_history ?? []),
      { action, analyst: 'mock-user', time: new Date().toISOString() },
    ]
    return clone(alert)
  })

export const fetchSearch = (params) =>
  safeGet('/alerts/search', () => filterMockSearch(params), (data) => data, { params })

export const fetchAttackers = () =>
  safeGet('/attackers', () => {
    const byIp = {}
    getMockAlerts().forEach((alert, i) => {
      const ip = alert.src_ip
      if (!byIp[ip]) byIp[ip] = { src_ip: ip, alert_count: 0, max_threat: 0, max_risk: 0, last_seen: '', attack_types: [], is_known_attacker: false, abuse_score: 0, geoip: alert.geoip, auto_blocked: false }
      byIp[ip].alert_count++
      byIp[ip].max_threat = Math.max(byIp[ip].max_threat, alert.threat_level ?? 0)
      byIp[ip].max_risk = Math.max(byIp[ip].max_risk, alert.risk_score ?? 0)
      if (alert.timestamp > byIp[ip].last_seen) byIp[ip].last_seen = alert.timestamp
      if (alert.attack_type && !byIp[ip].attack_types.includes(alert.attack_type)) byIp[ip].attack_types.push(alert.attack_type)
      byIp[ip].auto_blocked = byIp[ip].auto_blocked || !!alert.auto_blocked
    })
    return Object.values(byIp).sort((a, b) => b.max_threat - a.max_threat || b.max_risk - a.max_risk)
  })

export const fetchAttackerAlerts = (ip) =>
  safeGet(`/sessions/${ip}`, () => getMockAlerts().filter((alert) => alert.src_ip === ip), (data) => data)

export const fetchGeoIP = (ip) =>
  safeGet(`/enrichment/geoip/${ip}`, () => {
    const match = getMockAlerts().find((alert) => alert.src_ip === ip)
    return { ip, result: match?.geoip ?? null }
  })

export const postUnblock = (id) =>
  safePost(`/alerts/${id}/unblock`, () => {
    const alert = MOCK_ALERT_STORE.find((entry) => entry.id === id)
    if (!alert) return null
    alert.auto_blocked = false
    alert.status = 'active'
    return clone(alert)
  })

export const fetchHealth = () =>
  safeRootGet('/health', getMockHealth)

export const MOCK_ALERTS = getMockAlerts()
export const MOCK_STATS = getMockStats()
export const isBackendOffline = () => !backendReachable
