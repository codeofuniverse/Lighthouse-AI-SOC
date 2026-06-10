# Lighthouse AI-SOC — Full Documentation

**Version:** 1.0  
**Last updated:** 2026-05-25

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [ML Models](#2-ml-models)
3. [Detection Pipeline](#3-detection-pipeline)
4. [Risk Scoring](#4-risk-scoring)
5. [Decision Engine](#5-decision-engine)
6. [SOAR Actions](#6-soar-actions)
7. [API Reference](#7-api-reference)
8. [Data Storage](#8-data-storage)
9. [Testing Guide](#9-testing-guide)
10. [Deployment Guide](#10-deployment-guide)
11. [Attack Simulation](#11-attack-simulation)
12. [Model Performance](#12-model-performance)

---

## 1. System Overview

Lighthouse is an AI-powered Security Operations Centre (SOC) platform. It captures live network traffic via Suricata, extracts ML features from each flow, runs a dual-model detection pipeline, scores risk, and surfaces real-time alerts on a React dashboard — all with automated blocking capability.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NETWORK TRAFFIC                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │    SURICATA      │  Packet capture on network interface
                    │  (IDS sensor)    │  Writes flow records → eve.json
                    └────────┬─────────┘
                             │  /var/log/suricata/eve.json
                             ▼
                    ┌──────────────────┐
                    │  SURICATA BRIDGE │  Tails eve.json, extracts 19 CIC
                    │ suricata_bridge  │  features per flow, runs ML models
                    └────────┬─────────┘
                             │  DetectionEvent
                             ▼
              ┌──────────────────────────┐
              │     DUAL ML PIPELINE     │
              │  CIC 2017 (19 features)  │  XGBoost binary + LightGBM 6-class
              │  UNSW-NB15 (18 features) │  XGBoost binary + LightGBM 9-class
              │  Consensus gate          │  Both must agree for is_threat=True
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │     RISK SCORER          │  Weighted formula → 0–100 score
              │  + AbuseIPDB enrichment  │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │    DECISION ENGINE       │  log / alert / review / auto_block
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │    LLM ASSISTANT         │  1–2 sentence natural language
              │  (Ollama / Groq / cloud) │  explanation per alert
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │    FASTAPI BACKEND       │  REST API + WebSocket broadcast
              │    + SQLite / PostgreSQL │  Alert persistence across restarts
              └──────────┬───────────────┘
                         │  WebSocket /ws/alerts
                         ▼
              ┌──────────────────────────┐
              │   REACT DASHBOARD        │  Terminal Noir aesthetic
              │   (SOC analyst view)     │  Real-time alert feed + actions
              └──────────────────────────┘
```

### Docker Services

| Service | Purpose | Port |
|---------|---------|------|
| `lh-suricata` | Packet capture → eve.json | — |
| `lh-detector` | ML detection pipeline | — |
| `lh-backend` | FastAPI REST + WebSocket | 8000 |
| `lh-victim` | Attack target (nginx) | 8080, 8443 |
| `lh-attacker-hping` | SYN floods, port scans, brute force | — |
| `lh-attacker-msf` | Metasploit exploits | — |
| `lh-kafka` | Alert streaming | 9092 |
| `lh-redis` | GeoIP/threat intel cache | 6379 |
| `wazuh-manager` | SIEM / agent coordination | 55000, 1514, 1515 |
| `wazuh-agent` | Endpoint log collection | — |
| `lh-zookeeper` | Kafka coordination | — |

---

## 2. ML Models

Lighthouse runs two independent ML models on every network flow. They operate as a
dual-expert system — CIC 2017 is the primary classifier, UNSW-NB15 is the second opinion.

### 2.1 CIC-IDS-2017 Model

**Dataset:** Canadian Institute for Cybersecurity IDS 2017  
**Source:** Real network traffic captured over 5 days, labelled via CICFlowMeter  
**Training size:** 2,830,696 flows (after filtering Heartbleed/Infiltration)  
**Train/test split:** 80/20 stratified holdout  

**19 Features extracted from Suricata eve.json:**

| Feature | Source field |
|---------|-------------|
| Flow Duration | `flow.start` / `flow.end` |
| Total Fwd Packets | `flow.pkts_toserver` |
| Total Backward Packets | `flow.pkts_toclient` |
| Total Length of Fwd Packets | `flow.bytes_toserver` |
| Total Length of Bwd Packets | `flow.bytes_toclient` |
| Fwd Packet Length Mean | bytes_fwd / pkts_fwd |
| Bwd Packet Length Mean | bytes_bwd / pkts_bwd |
| Bwd Packet Length Max | bwd_mean × 1.5 (estimated) |
| Flow Bytes/s | total_bytes / duration |
| Flow Packets/s | total_pkts / duration |
| Flow IAT Mean | duration / total_pkts |
| Flow IAT Std | iat_mean × 0.3 |
| Fwd IAT Total | duration |
| Fwd IAT Mean | duration / pkts_fwd |
| Bwd IAT Mean | duration / pkts_bwd |
| FIN Flag Count | from `tcp.tcp_flags_ts` |
| SYN Flag Count | from `tcp.tcp_flags_ts` |
| PSH Flag Count | from `tcp.tcp_flags_ts` |
| ACK Flag Count | from `tcp.tcp_flags_ts` |

**Architecture:**
- **Stage 1:** XGBoost binary classifier (BENIGN vs Attack), scale_pos_weight≥2.0
- **Stage 2:** LightGBM 6-class classifier (attack family), trained on attack rows only
- **SMOTE:** Applied to training split only — Bot (1,573→10,000), Web Attack (1,744→10,000)
- **BENIGN cap:** 300,000 rows (downsampled from 2.27M to keep training tractable)

**Attack families (6):** Bot, Brute Force, DDoS, DoS, PortScan, Web Attack  
**Excluded:** Heartbleed (11 samples), Infiltration (36 samples) — too few for reliable training

**Validation Metrics (5-fold CV on training split):**

![CIC Validation Metrics](../reports/model_evaluation/cic_validation_metrics.png)

**ROC / AUC Curves (held-out 20% test set):**

![CIC ROC AUC](../reports/model_evaluation/cic_roc_auc.png)

---

### 2.2 UNSW-NB15 Model

**Dataset:** University of New South Wales NB15 (2015)  
**Source:** Synthetic attacks generated in the UNSW Canberra Cyber Range lab  
**Training size:** 175,341 flows | **Test size:** 82,332 flows (pre-split CSVs)  

**18 Features:**

| Feature | Description |
|---------|-------------|
| `dur` | Flow duration (seconds) |
| `spkts` | Source → dest packet count |
| `dpkts` | Dest → source packet count |
| `sbytes` | Source → dest bytes |
| `dbytes` | Dest → source bytes |
| `smeansz` | Mean packet size, source → dest |
| `dmeansz` | Mean packet size, dest → source |
| `rate` | Flow rate (pkts/s) |
| `sload` | Source bits per second |
| `dload` | Dest bits per second |
| `sjit` | Source jitter (ms) |
| `djit` | Dest jitter (ms) |
| `sintpkt` | Source inter-packet arrival time (ms) |
| `dintpkt` | Dest inter-packet arrival time (ms) |
| `synack` | Time between SYN and SYN-ACK |
| `ackdat` | Time between SYN-ACK and ACK |
| `ct_srv_src` | # connections to same service from same source |
| `ct_dst_ltm` | # connections to same dest in last time window |

**Architecture:**
- **Stage 1:** XGBoost binary (Normal vs Attack), scale_pos_weight≥2.0
- **Stage 2:** LightGBM 9-class (native UNSW category labels)
- **SMOTE:** Applied to Stage 2 attack rows — Analysis, Backdoor, Shellcode, Worms upsampled to 8,000
- **Native labels** (not forced into CIC names): Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms

**Validation Metrics (5-fold CV):**

![UNSW Validation Metrics](../reports/model_evaluation/unsw_validation_metrics.png)

**ROC / AUC Curves (held-out test set):**

![UNSW ROC AUC](../reports/model_evaluation/unsw_roc_auc.png)

---

### 2.3 Consensus Gate

A flow is marked `is_threat=True` only when:

```
CIC prediction == attack (any non-BENIGN label)
    OR
(UNSW attack_prob >= 0.70 AND CIC attack_prob >= 0.15)
```

This means:
- CIC alone can trigger a threat (it's the primary model)
- UNSW alone cannot — it needs at least 15% suspicion from CIC too
- This prevents UNSW's high false positive rate (35% of benign traffic) from flooding the dashboard

---

## 3. Detection Pipeline

```
Suricata eve.json (flow record)
    │
    ├─ _extract_cic_features()   → 19-element dict
    │       detection/suricata_bridge.py:96
    │
    ├─ SuricataBridge._predict()  → (label, attack_prob)
    │       Stage 1: XGBoost binary
    │       Stage 2: LightGBM multi-class (if Stage 1 = attack)
    │
    ├─ SuricataBridge._unsw_predict()  → (label, prob) | None
    │       UnswFeatureBridge.transform() → 18-element DataFrame
    │       Stage 1 + Stage 2 same pattern
    │
    └─ yield DetectionEvent(
            timestamp, src_ip, dst_ip, dst_port, proto, app_proto,
            prediction,          # CIC label: BENIGN | DDoS | DoS | ...
            is_threat,           # consensus gate result
            stage1_attack_prob,  # CIC binary probability
            unsw_prediction,     # UNSW label: Normal | Generic | Exploits | ...
            unsw_attack_prob,    # UNSW binary probability
            suricata_alert,      # Suricata rule signature (if any)
            cic_features         # raw 19-feature dict
       )

**Note:** Suricata can emit `alert` events for flows that never complete (RST mid-stream).
The bridge maintains a bounded buffer (`_MAX_PENDING = 2000`) of alert→flow_id associations
and discards the oldest when the cap is reached to prevent unbounded memory growth.
```

**Key files:**
- `detection/suricata_bridge.py` — `SuricataBridge`, `DetectionEvent`, `_extract_cic_features()`
- `detection/unsw_feature_bridge.py` — `UnswFeatureBridge.transform()`

---

## 4. Risk Scoring

**Formula** (`pipeline/risk_scorer.py`):

```
risk_score = (ML_confidence × 0.4)
           + (threat_intel  × 0.2)
           + (behavior_sev  × 0.3)
           + (asset_weight  × 0.1)
```

Scaled 0–100.

| Component | Source | Weight |
|-----------|--------|--------|
| `ML_confidence` | `stage1_attack_prob × 100` | 40% |
| `threat_intel` | AbuseIPDB `abuse_score` (0–100) | 20% |
| `behavior_sev` | `rule_level / 15 × 100` (Wazuh rule level 0–15) | 30% |
| `asset_weight` | Static map: server=1.0, workstation=0.7, unknown=0.5 | 10% |

**Threat level mapping:**

| Risk Score | Threat Level | Label |
|-----------|-------------|-------|
| 0–40 | 0 | Unknown |
| 41–70 | 1 | Suspicious |
| 71–100 | 2 | Critical |

**Example:** ML=90%, AbuseIPDB=80, rule_level=10, asset=server
```
= (90 × 0.4) + (80 × 0.2) + (10/15×100 × 0.3) + (100 × 0.1)
= 36 + 16 + 20 + 10 = 82  →  threat_level=2 (Critical)
```

---

## 5. Decision Engine

**Thresholds** (`pipeline/decision_engine.py`):

| Risk Score | Action | Description |
|-----------|--------|-------------|
| 0–30 | `log` | Write to file only. Not shown on dashboard. |
| 31–60 | `alert` | Normal dashboard alert. Analyst reviews manually. |
| 61–80 | `review` | Alert flagged "Pending Review". Human approval needed before action. |
| 81–100 | `auto_block` | Source IP automatically blocked via iptables/Wazuh. `auto_blocked=true` on alert. |

**Decision dataclass:**
```python
@dataclass
class Decision:
    action: Literal["log", "alert", "review", "auto_block"]
    risk_score: float
    threat_level: int    # 0 | 1 | 2
    auto_blocked: bool
```

---

## 6. SOAR Actions

SOAR (Security Orchestration, Automation and Response) actions are available via the
dashboard's action buttons and are also triggered automatically at `risk_score >= 81`.

**`backend/soar.py` — `SoarEngine`:**

| Action | Production (`SOAR_DRY_RUN=0`) | Dev (`SOAR_DRY_RUN=1`) |
|--------|-------------------------------|------------------------|
| Block IP | `iptables -I INPUT -s <ip> -j DROP` (exec, not shell) | Log only |
| Unblock IP | `iptables -D INPUT -s <ip> -j DROP` (exec, not shell) | Log only |
| Isolate agent | Wazuh REST API `PUT /active-response` → firewall-drop | Log only |

`block_ip` and `unblock_ip` use `asyncio.create_subprocess_exec` (argument list, not a shell string)
to prevent any shell-injection risk from attacker-controlled IP values.

All actions are recorded in `action_history` on the alert:
```json
[{"action": "block", "analyst": "auto", "time": "2026-05-25T14:23:01Z"}]
```

**Setting `SOAR_DRY_RUN=0` requires root** (iptables access). In Docker, the backend container
runs with `CAP_NET_ADMIN`. In production Linux, run the backend as root or with sudoers entry.

---

## 7. API Reference

Base URL: `http://localhost:8000`

### GET /api/alerts

Returns the last 200 alerts sorted by threat_level DESC, timestamp DESC.

```bash
curl http://localhost:8000/api/alerts
```

**Response:**
```json
[
  {
    "id": "a1b2c3d4e5f6",
    "timestamp": "2026-05-25T14:23:01.123456+00:00",
    "attack_type": "DDoS",
    "src_ip": "192.168.1.100",
    "dst_ip": "172.28.0.10",
    "dst_port": 80,
    "proto": "TCP",
    "agent_name": "192.168.1.100",
    "rule_level": 12,
    "rule_description": "ET DOS Suricata SYN flood",
    "status": "active",
    "auto_blocked": true,
    "confidence": 0.9923,
    "threat_level": 2,
    "risk_score": 87.4,
    "ai_explanation": "SYN flood detected from 192.168.1.100 targeting port 80 with 99% ML confidence. AbuseIPDB score 80 — known attacker. Auto-blocked.",
    "cic_confidence": 0.9923,
    "unsw_confidence": 0.8341,
    "abuse_score": 80,
    "action_history": [{"action": "block", "analyst": "auto", "time": "..."}],
    "ingested_at": "2026-05-25T14:23:01.123456+00:00"
  }
]
```

### GET /api/stats

```bash
curl http://localhost:8000/api/stats
```

**Response:**
```json
{"total_today": 47, "critical": 12, "suspicious": 23, "auto_blocked": 5}
```

### POST /api/alerts/{id}/block

Blocks the source IP of an alert.

```bash
curl -X POST http://localhost:8000/api/alerts/a1b2c3d4e5f6/block
```

**Response:** Updated alert dict with `auto_blocked=true`.

### POST /api/alerts/{id}/isolate

Isolates the Wazuh agent associated with the alert.

```bash
curl -X POST http://localhost:8000/api/alerts/a1b2c3d4e5f6/isolate
```

### POST /api/alerts/{id}/dismiss

Marks the alert as dismissed.

```bash
curl -X POST http://localhost:8000/api/alerts/a1b2c3d4e5f6/dismiss
```

### WebSocket /ws/alerts

Real-time alert stream. On connect: receives last 50 alerts. Then receives each new alert
as it arrives from the detection pipeline.

```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8000/ws/alerts');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

### GET /api/alerts/search

Query full SQLite history (up to 10,000 rows) with optional filters.

```bash
curl "http://localhost:8000/api/alerts/search?attack_type=DDoS&threat_level=2"
curl "http://localhost:8000/api/alerts/search?src_ip=192.168.1.5"
curl "http://localhost:8000/api/alerts/search?auto_blocked=true&limit=50"
curl "http://localhost:8000/api/alerts/search?since=2026-05-25T00:00:00"
```

All parameters are optional and combined with AND. `limit` max is 1000.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src_ip` | string | Filter by exact source IP |
| `attack_type` | string | e.g. `DDoS`, `PortScan`, `Brute Force` |
| `threat_level` | 0/1/2 | 0=Unknown, 1=Suspicious, 2=Critical |
| `status` | string | `active`, `dismissed`, `isolated` |
| `since` | ISO-8601 | Lower bound timestamp |
| `auto_blocked` | bool | `true` for auto-blocked alerts only |
| `limit` | int | Max results (default 200, max 1000) |

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "ok", "db_alerts": 4231}
```

---

## 8. Data Storage

### Current (SQLite — local PC)

| Data | Location | Format | Gitignored? |
|------|----------|--------|-------------|
| Suricata raw flows | `logs/eve.json` | JSON lines | Yes |
| ML detections | `logs/detections.json` | JSON lines | Yes |
| **Alert database** | `data/lighthouse_alerts.db` | SQLite | Yes |
| Redis cache | Redis container | Key-value | N/A |
| Wazuh alerts | Wazuh container `/var/ossec/logs/alerts/` | JSON + plain | N/A |
| Model files | `data/models/*.joblib` | Joblib binary | Yes |
| Model graphs | `reports/model_evaluation/*.png` | PNG | Yes |

**SQLite schema (`backend/db.py`):**
```sql
CREATE TABLE alerts (
    id           TEXT PRIMARY KEY,
    timestamp    TEXT    NOT NULL DEFAULT '',
    src_ip       TEXT    NOT NULL DEFAULT '',
    dst_ip       TEXT    NOT NULL DEFAULT '',
    attack_type  TEXT    NOT NULL DEFAULT '',
    threat_level INTEGER NOT NULL DEFAULT 0,
    risk_score   REAL    NOT NULL DEFAULT 0.0,
    auto_blocked INTEGER NOT NULL DEFAULT 0,   -- 0/1 boolean
    status       TEXT    NOT NULL DEFAULT 'active',
    data         TEXT    NOT NULL DEFAULT '{}'  -- full alert JSON blob
);
CREATE INDEX idx_ts   ON alerts (timestamp DESC);
CREATE INDEX idx_tl   ON alerts (threat_level DESC);
CREATE INDEX idx_src  ON alerts (src_ip);
CREATE INDEX idx_type ON alerts (attack_type);
CREATE INDEX idx_stat ON alerts (status);
```

WAL mode enabled — reads never block writes. Single persistent connection per process with
`threading.Lock`. Prune runs every 500 inserts (not every insert) to avoid COUNT(*) overhead.

Alerts persist across backend restarts. On startup, the last 500 alerts reload from SQLite
into the in-memory deque automatically. Maximum 10,000 rows stored — oldest are pruned automatically.

**Database CLI (`scripts/manage_db.py`):**

```powershell
# Show size, row counts, and breakdowns
python scripts/manage_db.py status

# Search with filters
python scripts/manage_db.py search --attack_type DDoS --threat_level 2 --limit 20
python scripts/manage_db.py search --src_ip 192.168.1.5 -v   # -v shows AI explanation

# Export all alerts to JSON
python scripts/manage_db.py export --out alerts_backup.json

# Delete oldest rows, keep last 5000
python scripts/manage_db.py prune --keep 5000

# Reclaim disk space after bulk deletes
python scripts/manage_db.py vacuum

# Delete ALL alerts (prompts for confirmation)
python scripts/manage_db.py clear
```

### Future (PostgreSQL — online server)

When deploying to a cloud server (Supabase, Railway, Render, self-hosted VPS):

1. Set `DATABASE_URL=postgresql://user:pass@host/lighthouse` in `.env`
2. `backend/db.py` connection string switches — **no other code changes needed**
3. Schema is identical; SQLAlchemy Core handles both dialects

Recommended hosting options:
- **Supabase** — free tier, 500MB, PostgreSQL, REST API included
- **Railway** — $5/mo, simple git deploy
- **Render** — free tier available, PostgreSQL add-on

---

## 9. Testing Guide

### Layer 1 — Unit Tests

```powershell
cd C:\Users\avane\Desktop\Lighthouse
.venv\Scripts\activate
pytest tests/ -v
```

Covers: alert schema validation, ML classifier, enrichment pipeline, Wazuh bridge.

### Layer 2 — Synthetic Flow Injection (recommended for development)

Bypasses Suricata entirely — writes synthetic eve.json records directly.
No Docker, no network interface, works on any machine.

```powershell
# Terminal 1 — start backend
$env:LIGHTHOUSE_DEV = "0"
$env:EVE_JSON_PATH  = "logs/test_eve.json"
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — inject attacks
python tests/simulate_suricata_attack.py --attack ddos       --count 50  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack portscan   --count 30  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack bruteforce --count 20  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack bot        --count 10  --eve logs/test_eve.json

# Terminal 3 — start frontend
cd frontend && npm run dev     # http://localhost:5173
```

Available profiles: `ddos`, `ddos-https`, `dos`, `portscan`, `bruteforce`, `bot`,
`fuzzer`, `exploit`, `recon`, `shellcode`, `worm`

### Layer 3 — Docker Stack (real Suricata capture)

```powershell
docker compose up -d
Start-Sleep 15

# Launch attacks from attacker containers
docker exec lh-attacker-hping sh /home/attacks.sh ddos       172.28.0.10
docker exec lh-attacker-hping sh /home/attacks.sh portscan   172.28.0.10
docker exec lh-attacker-hping sh /home/attacks.sh bruteforce 172.28.0.10

# Watch live
docker logs -f lh-detector
docker exec lh-suricata tail -f /var/log/suricata/fast.log
```

### Layer 4 — API Endpoint Testing

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/alerts
curl -X POST http://localhost:8000/api/alerts/ALERT_ID/block
curl -X POST http://localhost:8000/api/alerts/ALERT_ID/dismiss
```

### Layer 5 — End-to-End Validation Checklist

| Test | Expected | Pass When |
|------|----------|-----------|
| DDoS injection (50 flows) | CIC=DDoS, confidence≥0.99 | `threat_level=2` in dashboard |
| PortScan injection | CIC=PortScan, 100% detection | `attack_type="PortScan"` in alert |
| Brute Force injection | ≥87% detection | `confidence≥0.87` |
| Benign flow injection | BENIGN prediction | No alert in dashboard |
| Auto-block trigger | risk_score≥81 | `auto_blocked=true` |
| Restart backend | Alerts reload from SQLite | Previous alerts still visible |
| WebSocket | New alert within 2s | Browser console receives JSON |

---

## 10. Deployment Guide

### 10.1 Real-World Network Architecture

```
[Internet / Corporate LAN]
       │
       ▼
[Network TAP or switch SPAN port]
       │
       ▼
[Suricata sensor]            ← Dedicated VM or physical NIC in promiscuous mode
  /var/log/suricata/eve.json
       │  (NFS mount or scp/rsync to detector host)
       ▼
[Lighthouse server]
  - Python 3.11+ venv
  - Both ML models (data/models/*.joblib)
  - FastAPI backend on :8000
  - Redis on :6379
  - Wazuh manager on :55000
       │
       ▼
[SOC Analyst workstations]
  http://LIGHTHOUSE_IP:5173
```

### 10.2 Suricata Setup (Linux sensor)

```bash
sudo apt install suricata
sudo cp infra/suricata/suricata.yaml /etc/suricata/suricata.yaml
sudo cp infra/suricata/lighthouse.rules /etc/suricata/rules/

# Edit suricata.yaml line ~65: change docker0 → your interface (eth0, ens3, etc.)
sudo nano /etc/suricata/suricata.yaml

sudo systemctl enable --now suricata
sudo tail -f /var/log/suricata/eve.json   # verify flows appear
```

### 10.3 Lighthouse Backend Setup (Linux)

```bash
git clone <your-repo> /opt/lighthouse
cd /opt/lighthouse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env
# Set:
#   EVE_JSON_PATH=/var/log/suricata/eve.json   (or NFS path)
#   LIGHTHOUSE_DEV=0
#   SOAR_DRY_RUN=0                              (requires root)
#   LLM_PROVIDER=groq                           (or ollama, openai)
#   ABUSEIPDB_API_KEY=your-key

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 10.4 Frontend (Production Build)

```bash
cd frontend
npm install
npm run build          # outputs to dist/

# Serve with nginx
sudo cp -r dist/ /var/www/lighthouse/
# Or serve directly:
npx serve dist -l 5173
```

### 10.5 Wazuh Agent on Monitored Hosts

```bash
# Ubuntu/Debian
wget https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.14.5-1_amd64.deb
dpkg -i wazuh-agent.deb

# Set manager IP
sed -i 's/<address>MANAGER_IP<\/address>/<address>YOUR_LIGHTHOUSE_IP<\/address>/' \
    /var/ossec/etc/ossec.conf

systemctl enable --now wazuh-agent
```

### 10.6 Environment Variables Reference

| Variable | Default | Description |
|---------|---------|-------------|
| `EVE_JSON_PATH` | (empty) | Path to Suricata eve.json. Empty = dev mode (mock alerts) |
| `LIGHTHOUSE_DEV` | `1` | `1` = mock alerts; `0` = real Suricata pipeline |
| `SOAR_DRY_RUN` | `1` | `1` = log only; `0` = real iptables/Wazuh actions |
| `CIC_MODEL_PATH` | `data/models/cic2017_pipeline_smote.joblib` | CIC model path |
| `UNSW_MODEL_PATH` | `data/models/unsw_nb15_pipeline.joblib` | UNSW model path |
| `LLM_PROVIDER` | `ollama_cloud` | `groq` \| `openai` \| `ollama_cloud` \| `ollama` |
| `LLM_MODEL` | `gemma3:4b` | Model name for LLM provider |
| `ABUSEIPDB_API_KEY` | — | Free key from abuseipdb.com |
| `REDIS_HOST` | `localhost` | Redis host |
| `WAZUH_HOST` | `localhost` | Wazuh manager host |

---

## 11. Attack Simulation

> All tools below are legitimate security research tools.
> **Only run against systems you own or have explicit written permission to test.**

### 11.1 Tier 1 — Packet Crafters & Traffic Generators

#### hping3 (in Docker as `lh-attacker-hping`)

```bash
# SYN flood → CIC=DDoS
sudo hping3 -S -p 80 --flood --rand-source TARGET_IP

# Controlled rate (1000 pps)
sudo hping3 -S -p 80 -i u1000 TARGET_IP

# Port scan → CIC=PortScan
sudo hping3 -S --scan 1-1024 TARGET_IP

# SSH brute force simulation
sudo hping3 -S -p 22 -i u500 TARGET_IP
```

Expected: `attack_type=DDoS`, `confidence≥0.99`, `risk_score≥75`

#### Scapy (Python, fine-grained control)

```python
from scapy.all import *

# SYN flood with spoofed sources
for i in range(1000):
    send(IP(src=RandIP(), dst="TARGET_IP") / TCP(dport=80, flags="S"), verbose=0)

# Slowloris-style partial HTTP
for i in range(200):
    send(IP(dst="TARGET_IP") / TCP(dport=80, flags="PA") /
         "GET / HTTP/1.1\r\nHost: target\r\n", verbose=0)
```

#### Ostinato (GUI traffic generator)
Create stream: Ethernet → IP → TCP → dest=TARGET_IP, port=80  
Set rate: 10,000 pps for DDoS simulation  
Use case: reproduce exact CIC dataset flow statistics

#### TRex (Cisco, high-speed L4–L7)
```bash
git clone https://github.com/cisco-system-traffic-generator/trex-core
cd trex-core/scripts && sudo ./t-rex-64 -i
# Stress-test at >100K flows/min to validate detector throughput
```

### 11.2 Tier 2 — Attack Frameworks

#### Nmap

```bash
sudo nmap -sS -p 1-1024 TARGET_IP           # SYN stealth → CIC=PortScan
sudo nmap -A --min-rate 5000 TARGET_IP      # Aggressive → UNSW=Reconnaissance
```

#### Hydra (brute force)

```bash
hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://TARGET_IP -t 4
hydra -l admin -P wordlist.txt ftp://TARGET_IP
```

Expected: `attack_type=Brute Force`, `confidence≥0.87`

#### Metasploit (in Docker as `lh-attacker-msf`)

```bash
docker exec -it lh-attacker-msf msfconsole

msf> use auxiliary/scanner/portscan/syn
msf> set RHOSTS TARGET_IP
msf> run

# Exploit (lab VM only)
msf> use exploit/unix/ftp/vsftpd_234_backdoor
msf> set RHOSTS TARGET_IP
msf> run
```

Expected: UNSW=Exploits (64%), CIC sees anomalous flow

#### Slowloris

```bash
pip install slowloris
slowloris TARGET_IP --port 80 --sockets 200
```

Expected: `attack_type=DoS`, `confidence≥0.99`

### 11.3 Tier 3 — Breach & Attack Simulation (BAS) Platforms

#### Atomic Red Team (MITRE ATT&CK mapped tests)

```powershell
Install-Module -Name invoke-atomicredteam -Scope CurrentUser
Invoke-AtomicTest T1046   # Network Service Discovery (PortScan)
Invoke-AtomicTest T1110   # Brute Force
Invoke-AtomicTest T1498   # Network Denial of Service
```

#### CALDERA (MITRE automated adversary emulation)

```bash
git clone https://github.com/mitre/caldera.git
cd caldera && pip install -r requirements.txt
python server.py --insecure
# Open http://localhost:8888 → create operation → deploy agent on victim
```

Detection phases: Reconnaissance → CIC=PortScan, UNSW=Reconnaissance  
Lateral movement → UNSW=Exploits  
Impact → CIC=DDoS

#### Infection Monkey (Zero Trust validation)

Deploy from https://github.com/guardicore/monkey  
Tests: network scanning, credential guessing, lateral movement  
Validates consensus gate catches multi-stage attacks

### 11.4 What to Observe During Every Attack

| Observation | Check |
|-------------|-------|
| Alert latency | Appears within 1–3 seconds |
| attack_type | Matches the attack launched |
| confidence | ≥0.85 for volumetric attacks |
| threat_level | 2 (Critical) for DDoS/DoS/PortScan |
| ai_explanation | Natural language description |
| auto_blocked | `true` if risk_score ≥81 |
| Suricata rule | `docker exec lh-suricata tail -f /var/log/suricata/fast.log` |
| ML log | `docker logs -f lh-detector` |

---

## 12. Model Performance

### CIC 2017 — Held-out Test Set (20% stratified split)

**Binary macro F1: 0.9520**

| Class | Detected | Total | Detection Rate | Notes |
|-------|---------|-------|---------------|-------|
| PortScan | 31,774 | 31,786 | **100.0%** | Perfect |
| DDoS | 25,588 | 25,606 | **99.9%** | Near-perfect |
| DoS | 50,299 | 50,532 | **99.5%** | Near-perfect |
| BENIGN | 437,099 | 454,620 | **96.1%** | Low false positive rate |
| Web Attack | 396 | 436 | **90.8%** | Very good |
| Brute Force | 2,421 | 2,767 | **87.5%** | Good |
| Bot | 338 | 393 | **86.0%** | Good (only 1,966 real training samples) |

### UNSW-NB15 — Pre-split Test Set

**Binary macro F1: 0.8248**

| Class | Detected | Total | Detection Rate | Notes |
|-------|---------|-------|---------------|-------|
| Generic | 18,305 | 18,871 | **97.0%** | Excellent |
| Reconnaissance | 2,855 | 3,496 | **81.7%** | Good |
| Fuzzers | 4,446 | 6,062 | **73.3%** | Acceptable |
| Exploits | 7,145 | 11,132 | **64.2%** | Acceptable |
| Shellcode | 238 | 378 | **63.0%** | Acceptable |
| Normal | 24,149 | 37,000 | **65.3%** | False positives filtered by consensus gate |
| Worms | 21 | 44 | **47.7%** | Low — only 44 test samples |
| DoS | 1,000 | 4,089 | **24.5%** | Low — CIC covers this gap |
| Backdoor | 103 | 583 | **17.7%** | Low — limited training data |
| Analysis | 56 | 677 | **8.3%** | Very low — rare pattern |

### Why the Dual-Model Approach Works

CIC 2017 and UNSW-NB15 have complementary strengths:

| Attack Type | CIC 2017 | UNSW-NB15 | Combined Coverage |
|-------------|----------|-----------|------------------|
| DDoS / Floods | 99.9% | 97% (Generic) | Near-perfect |
| Port Scan | 100% | 81.7% (Recon) | Perfect primary |
| Brute Force | 87.5% | — | Good |
| DoS | 99.5% | 24.5% | CIC dominates |
| Exploits / 0-days | — | 64.2% | UNSW fills gap |
| Backdoors | — | 17.7% | Partial coverage |
| Bot Traffic | 86% | — | Good |

The consensus gate (`unsw_prob ≥ 0.70 AND cic_prob ≥ 0.15`) means UNSW false positives
are suppressed, while UNSW's exploit/backdoor detections that CIC misses still surface
when CIC sees at least minor suspicion.

---

## Retraining Models

To retrain after collecting new data or fixing label issues:

```powershell
.venv\Scripts\activate

# UNSW-NB15 (~10 min)
python scripts/train_unsw_nb15.py

# CIC 2017 (~20 min)
python scripts/retrain_cic_smote.py

# Regenerate performance graphs
python scripts/generate_model_reports.py
```

Model files saved to `data/models/` (gitignored — do not commit binaries).
