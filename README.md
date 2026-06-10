<div align="center">

```
 ██▓     ██▓  ▄████  ██░ ██ ▄▄▄█████▓ ██░ ██  ▒█████   █    ██   ██████ ▓█████
▓██▒    ▓██▒ ██▒ ▀█▒▓██░ ██▒▓  ██▒ ▓▒▓██░ ██▒▒██▒  ██▒ ██  ▓██▒▒██    ▒ ▓█   ▀
▒██░    ▒██▒▒██░▄▄▄░▒██▀▀██░▒ ▓██░ ▒░▒██▀▀██░▒██░  ██▒▓██  ▒██░░ ▓██▄   ▒███
▒██░    ░██░░▓█  ██▓░▓█ ░██ ░ ▓██▓ ░ ░▓█ ░██ ▒██   ██░▓▓█  ░██░  ▒   ██▒▒▓█  ▄
░██████▒░██░░▒▓███▀▒░▓█▒░██▓  ▒██▒ ░ ░▓█▒░██▓░ ████▓▒░▒▒█████▓ ▒██████▒▒░▒████▒
░ ▒░▓  ░░▓   ░▒   ▒  ▒ ░░▒░▒  ▒ ░░    ▒ ░░▒░▒░ ▒░▒░▒░ ░▒▓▒ ▒ ▒ ▒ ▒▓▒ ▒ ░░░ ▒░ ░
░ ░ ▒  ░ ▒ ░  ░   ░  ▒ ░▒░ ░    ░     ▒ ░▒░ ░  ░ ▒ ▒░ ░░▒░ ░ ░ ░ ░▒  ░ ░ ░ ░  ░
  ░ ░    ▒ ░░ ░   ░  ░  ░░ ░  ░       ░  ░░ ░░ ░ ░ ▒   ░░░ ░ ░ ░  ░  ░     ░
    ░  ░ ░        ░  ░  ░  ░          ░  ░  ░    ░ ░     ░           ░     ░  ░
```

**AI-Powered Security Operations Centre**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-Binary_F1_0.952-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-6--class_classifier-31A354?style=flat-square)](https://lightgbm.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-WAL_mode-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Docker](https://img.shields.io/badge/Docker-11_services-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What Is Lighthouse?

Lighthouse captures live network traffic through Suricata, extracts flow-level features, runs two independent ML models in parallel, scores risk on a 0–100 scale, and surfaces real-time alerts on a Terminal Noir React dashboard — with automated IP blocking when risk crosses the auto-block threshold.

```
 Network Traffic                                              SOC Analyst
 ┌──────────┐   eve.json    ┌─────────────┐   WebSocket    ┌───────────────┐
 │ Suricata │ ─────────────▶│ Lighthouse  │ ─────────────▶ │               │
 │  (IDS)   │               │  Backend    │                │  ██ CRITICAL  │
 └──────────┘               │             │    REST API    │  ▓▓ SUSPICIOUS│
                             │  CIC 2017   │ ◀──────────── │  ░░ Auto-blk  │
                             │  UNSW-NB15  │               │               │
                             │  Risk Score │  Block / ISO  │  [Block IP]   │
                             │  LLM Expl.  │ ─────────────▶│  [Isolate]    │
                             │  SQLite DB  │               │  [Dismiss]    │
                             └─────────────┘               └───────────────┘
```

---

## Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🛡 LIGHTHOUSE  SOC    47 Total Today    12 Critical ●   23 Suspicious ●     │
│                       5 Auto-Blocked 🛡                              ● LIVE │
├──────────┬────────────────────┬──────────────────────────────┬───────┬──────┤
│ CRITICAL │ DDoS               │ High-rate SYN flood from     │ Conf. │      │
│  2s ago  │ 172.28.0.20 → web  │ 172.28.0.20 targeting port   │ ████░ │ AUTO │
│          │                    │ 80 with 4,800 packets/s.     │  97%  │BLOCK │
├──────────┼────────────────────┼──────────────────────────────┼───────┼──────┤
│ CRITICAL │ PortScan           │ SYN-only flows targeting     │ Conf. │[Blk] │
│  9s ago  │ 10.0.0.199 → web   │ 1,200 sequential ports.     │ ████░ │[Iso] │
│          │                    │ Zero bytes returned.         │  91%  │[Dis] │
├──────────┼────────────────────┼──────────────────────────────┼───────┼──────┤
│SUSPICIOUS│ Brute Force        │ 47 FIN+PSH+ACK flows to      │ Conf. │[Blk] │
│  16s ago │ 172.28.0.21 → :22  │ port 22. SSH pattern match.  │ ███░░ │[Iso] │
│          │                    │                              │  78%  │[Dis] │
├──────────┼────────────────────┼──────────────────────────────┼───────┼──────┤
│SUSPICIOUS│ DoS                │ Sustained high-byte flows    │ Conf. │[Blk] │
│  23s ago │ 185.220.101.5→:80  │ (420KB avg). Hulk variant.  │ ██░░░ │[Iso] │
│          │                    │                              │  65%  │[Dis] │
└──────────┴────────────────────┴──────────────────────────────┴───────┴──────┘
```

**Alert Detail Drawer** (opens on click):
```
                                    ┌──────────────────────────────────┐
                                    │ [CRITICAL]  [Blocked]        ✕   │
                                    │ DDoS                             │
                                    ├──────────────────────────────────┤
                                    │ ALERT METADATA                   │
                                    │ ID          a1b2c3d4e5f6         │
                                    │ Timestamp   2026-05-25 14:23:01  │
                                    │ Source IP   172.28.0.20          │
                                    │ Rule Level  13                   │
                                    ├──────────────────────────────────┤
                                    │ ML CONFIDENCE                    │
                                    │ Overall    ████████████░░  97%   │
                                    │ CIC 2017   ████████████░░  97%   │
                                    │ UNSW-NB15  █████████░░░░░  83%   │
                                    ├──────────────────────────────────┤
                                    │ AI ANALYSIS                      │
                                    │ │ High-rate SYN flood from       │
                                    │ │ 172.28.0.20 targeting port 80  │
                                    │ │ with 4,800 packets/s. CIC      │
                                    │ │ model 97% — matches DDoS       │
                                    │ │ training distribution.         │
                                    ├──────────────────────────────────┤
                                    │ [  Block IP  ] [Isolate] [Dismiss]│
                                    └──────────────────────────────────┘
```

---

## ML Model Performance

Two independent models run on every network flow. Results on held-out test sets:

### CIC-IDS-2017 — Binary macro F1: **0.9520**

| Attack Family | Detection Rate | Notes |
|---------------|:--------------:|-------|
| PortScan | **100.0%** | Perfect |
| DDoS | **99.9%** | Near-perfect |
| DoS | **99.5%** | Near-perfect |
| BENIGN | **96.1%** | Low false-positive rate |
| Web Attack | **90.8%** | Very good |
| Brute Force | **87.5%** | Good |
| Bot | **86.0%** | Good (limited training samples) |

**Stage 1 ROC (Binary — BENIGN vs Attack):**

![CIC ROC Stage 1](reports/model_evaluation/cic_roc_stage1.png)

**Stage 2 ROC (Attack families only — no BENIGN):**

![CIC ROC Stage 2](reports/model_evaluation/cic_roc_stage2.png)

**Confusion Matrix:**

![CIC Confusion Matrix](reports/model_evaluation/cic_confusion_matrix.png)

**Classification Report:**

![CIC Classification Report](reports/model_evaluation/cic_classification_report.png)

**Cross-Validation Metrics (SMOTE + StandardScaler fitted inside each fold):**

![CIC Validation Metrics](reports/model_evaluation/cic_validation_metrics.png)

---

### UNSW-NB15 — Binary macro F1: **0.8248**

| Attack Category | Detection Rate | Notes |
|-----------------|:--------------:|-------|
| Generic | **97.0%** | Excellent |
| Reconnaissance | **81.7%** | Good |
| Fuzzers | **73.3%** | Acceptable |
| Exploits | **64.2%** | Fills CIC gap |
| Shellcode | **63.0%** | Acceptable |
| DoS | **24.5%** | CIC covers this gap |
| Backdoor | **17.7%** | Limited training data |

**Stage 1 ROC (Binary):**

![UNSW ROC Stage 1](reports/model_evaluation/unsw_roc_stage1.png)

**Stage 2 ROC (Attack categories only):**

![UNSW ROC Stage 2](reports/model_evaluation/unsw_roc_stage2.png)

**Confusion Matrix:**

![UNSW Confusion Matrix](reports/model_evaluation/unsw_confusion_matrix.png)

**Classification Report:**

![UNSW Classification Report](reports/model_evaluation/unsw_classification_report.png)

**Cross-Validation Metrics:**

![UNSW Validation Metrics](reports/model_evaluation/unsw_validation_metrics.png)

---

### Why Dual-Model?

```
Attack type         CIC 2017    UNSW-NB15    Combined
─────────────────── ────────    ─────────    ────────
DDoS / Floods       99.9% ✓     97.0% ✓      Near-perfect
Port Scan           100%  ✓     81.7% ✓      Perfect primary
Brute Force         87.5% ✓       —          Good via CIC
DoS attacks         99.5% ✓     24.5%        CIC dominates
Exploits / 0-days     —         64.2% ✓      UNSW fills gap
Backdoors             —         17.7%        Partial coverage
Bot traffic         86.0% ✓       —          Good via CIC
```

**Consensus gate:** A flow is `is_threat=True` when CIC detects an attack, OR when
UNSW attack probability ≥ 70% AND CIC attack probability ≥ 15%. This prevents
UNSW's higher false-positive rate from flooding the dashboard while still
surfacing exploit/intrusion patterns that CIC misses.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          NETWORK TRAFFIC                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │    SURICATA      │  Packet capture on network interface
                   │   (IDS sensor)   │  Writes flow records → eve.json
                   └────────┬─────────┘
                            │  /var/log/suricata/eve.json
                            ▼
                   ┌──────────────────┐
                   │ SURICATA BRIDGE  │  Tails eve.json, extracts 19 CIC
                   │  (detection/)    │  features per flow (2 bounded buffers:
                   │                  │  pending_alerts max 2000 entries)
                   └────────┬─────────┘
                            │  DetectionEvent
                            ▼
             ┌──────────────────────────────┐
             │       DUAL ML PIPELINE       │
             │                              │
             │  CIC 2017 (19 features)      │  XGBoost binary
             │    Stage 1: BENIGN vs Attack │  + LightGBM 6-class
             │    Stage 2: Attack family    │
             │                              │
             │  UNSW-NB15 (18 features)     │  XGBoost binary
             │    Stage 1: Normal vs Attack │  + LightGBM 9-class
             │    Stage 2: Attack category  │
             │                              │
             │  Consensus gate              │  Both must agree
             └────────────┬─────────────────┘
                          │
                          ▼
             ┌──────────────────────────────┐
             │        RISK SCORER           │
             │  ML(40%) + Intel(20%)        │  → 0–100 score
             │  + Behavior(30%) + Asset(10%)│
             └────────────┬─────────────────┘
                          │
                          ▼
             ┌──────────────────────────────┐
             │      DECISION ENGINE         │
             │  0-30:  log (silent)         │
             │  31-60: alert                │
             │  61-80: review               │
             │  81-100: AUTO BLOCK          │
             └────────────┬─────────────────┘
                          │
                          ▼
             ┌──────────────────────────────┐
             │       LLM ASSISTANT          │  Groq / Ollama / OpenAI
             │  1-2 sentence explanation    │  Falls back to rule-based
             └────────────┬─────────────────┘
                          │
                          ▼
             ┌──────────────────────────────┐
             │      FASTAPI BACKEND         │  REST + WebSocket
             │      + SQLite (WAL mode)     │  10,000 row max, 5 indexes
             │      + SOAR engine           │  iptables / Wazuh API
             └────────────┬─────────────────┘
                          │  WebSocket /ws/alerts
                          ▼
             ┌──────────────────────────────┐
             │    REACT DASHBOARD           │  Terminal Noir aesthetic
             │    (SOC analyst view)        │  Real-time alert feed
             │    http://localhost:5173     │  Block / Isolate / Dismiss
             └──────────────────────────────┘
```

---

## Quick Start

### Option A — Docker (full stack with real Suricata)

```powershell
cp .env.example .env      # edit LLM_PROVIDER, ABUSEIPDB_API_KEY if needed

docker compose up -d

# Open dashboard
start http://localhost:5173

# Launch test attacks (from attacker containers)
docker exec lh-attacker-hping sh /home/attacks.sh ddos       172.28.0.10
docker exec lh-attacker-hping sh /home/attacks.sh portscan   172.28.0.10
docker exec lh-attacker-hping sh /home/attacks.sh bruteforce 172.28.0.10

# Watch ML detections live
docker logs -f lh-detector
```

### Option B — Local Dev (mock alerts, no Docker needed)

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — backend (mock alert mode)
$env:LIGHTHOUSE_DEV = "1"
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

### Option C — Synthetic Attack Injection (real ML, no Suricata)

```powershell
# Terminal 1 — backend watching a log file
$env:LIGHTHOUSE_DEV = "0"
$env:EVE_JSON_PATH  = "logs/test_eve.json"
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — inject synthetic Suricata flow records
python tests/simulate_suricata_attack.py --attack ddos       --count 50  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack portscan   --count 30  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack bruteforce --count 20  --eve logs/test_eve.json
python tests/simulate_suricata_attack.py --attack bot        --count 10  --eve logs/test_eve.json

# Terminal 3 — frontend
cd frontend && npm run dev
```

Available attack profiles: `ddos`, `ddos-https`, `dos`, `portscan`, `bruteforce`,
`bot`, `fuzzer`, `exploit`, `recon`, `shellcode`, `worm`

---

## Docker Services

| Service | Purpose | Port |
|---------|---------|------|
| `lh-suricata` | Packet capture → eve.json | — |
| `lh-detector` | Dual ML detection pipeline | — |
| `lh-backend` | FastAPI REST + WebSocket | **8000** |
| `lh-victim` | Attack target (nginx) | 8080, 8443 |
| `lh-attacker-hping` | SYN floods, scans, brute force | — |
| `lh-attacker-msf` | Metasploit framework | — |
| `lh-kafka` | Alert event stream | 9092 |
| `lh-redis` | GeoIP / threat intel cache | 6379 |
| `wazuh-manager` | SIEM + agent coordination | 55000 |
| `wazuh-agent` | Endpoint log collection | — |
| `lh-zookeeper` | Kafka coordination | — |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/alerts` | Last 200 alerts (threat_level DESC) |
| `GET` | `/api/alerts/search` | Filter by src_ip, attack_type, threat_level, status, since, auto_blocked |
| `GET` | `/api/stats` | `{total_today, critical, suspicious, auto_blocked}` |
| `POST` | `/api/alerts/{id}/block` | Block source IP via iptables / Wazuh |
| `POST` | `/api/alerts/{id}/isolate` | Isolate Wazuh agent |
| `POST` | `/api/alerts/{id}/dismiss` | Mark alert dismissed |
| `WS` | `/ws/alerts` | Real-time alert stream |
| `GET` | `/health` | `{status, db_alerts}` |

```bash
# Examples
curl http://localhost:8000/api/stats
curl "http://localhost:8000/api/alerts/search?attack_type=DDoS&threat_level=2"
curl "http://localhost:8000/api/alerts/search?auto_blocked=true&limit=50"
curl -X POST http://localhost:8000/api/alerts/a1b2c3d4/block
```

---

## Data Storage

| Data | Location | Format | Gitignored |
|------|----------|--------|-----------|
| Alert database | `data/lighthouse_alerts.db` | SQLite WAL | ✓ |
| Suricata flows | `logs/eve.json` | JSON lines | ✓ |
| ML models | `data/models/*.joblib` | Joblib binary | ✓ |
| Model graphs | `reports/model_evaluation/*.png` | PNG | ✓ |
| Redis cache | Redis container | Key-value | — |

**SQLite:** WAL mode, persistent connection, 10 columns, 5 indexes.
Max 10,000 rows — oldest pruned automatically every 500 inserts.
Last 500 alerts reload into memory on restart.

**Database CLI:**
```powershell
python scripts/manage_db.py status                              # size + breakdowns
python scripts/manage_db.py search --attack_type DDoS -v       # search + AI explanation
python scripts/manage_db.py export --out backup.json           # JSON export
python scripts/manage_db.py prune --keep 5000                  # keep last 5000 rows
python scripts/manage_db.py vacuum                             # reclaim disk space
```

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app — REST API, WebSocket, ingestion loop |
| `backend/store.py` | In-memory deque + SQLite persistence |
| `backend/db.py` | SQLite WAL layer — schema, CRUD, search, prune |
| `backend/soar.py` | Block / isolate / unblock (iptables exec + Wazuh API) |
| `backend/llm_assistant.py` | LLM explanations — Groq, Ollama, OpenAI, fallback |
| `detection/suricata_bridge.py` | eve.json → 19 CIC features → ML → DetectionEvent |
| `detection/unsw_feature_bridge.py` | Suricata flow → 18 UNSW features |
| `pipeline/risk_scorer.py` | Weighted risk formula 0–100 |
| `pipeline/decision_engine.py` | Route by score: log / alert / review / auto_block |
| `pipeline/enrichment/` | GeoIP, AbuseIPDB, MITRE mapper, sessionizer |
| `scripts/retrain_cic_smote.py` | Retrain CIC 2017 model (~20 min) |
| `scripts/train_unsw_nb15.py` | Retrain UNSW-NB15 model (~10 min) |
| `scripts/generate_model_reports.py` | Generate all 10 model performance graphs |
| `scripts/manage_db.py` | SQLite CLI — status, search, export, prune, vacuum |
| `tests/simulate_suricata_attack.py` | Synthetic Suricata flow injection |

---

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|---------|---------|-------------|
| `EVE_JSON_PATH` | — | Suricata eve.json path. Empty = dev/mock mode |
| `LIGHTHOUSE_DEV` | `1` | `1` = mock alerts, `0` = real pipeline |
| `SOAR_DRY_RUN` | `1` | `1` = log only, `0` = real iptables/Wazuh (root required) |
| `CIC_MODEL_PATH` | `data/models/cic2017_pipeline_smote.joblib` | CIC model |
| `UNSW_MODEL_PATH` | `data/models/unsw_nb15_pipeline.joblib` | UNSW model |
| `LLM_PROVIDER` | `ollama_cloud` | `groq` \| `openai` \| `ollama_cloud` \| `ollama` |
| `LLM_API_KEY` | — | API key for chosen LLM provider |
| `LLM_MODEL` | `gemma3:4b` | Model name |
| `ABUSEIPDB_API_KEY` | — | Free key from abuseipdb.com |
| `REDIS_HOST` | `localhost` | Redis host |
| `WAZUH_HOST` | `localhost` | Wazuh manager host |

---

## Retraining Models

```powershell
.venv\Scripts\activate

python scripts/train_unsw_nb15.py          # ~10 min — UNSW-NB15 dual-stage model
python scripts/retrain_cic_smote.py        # ~20 min — CIC 2017 with SMOTE
python scripts/generate_model_reports.py   # regenerate all 10 graphs
```

Both training scripts apply SMOTE and StandardScaler **inside** each cross-validation
fold (no data leakage). Models saved to `data/models/` (gitignored — do not commit binaries).

---

## Attack Simulation

> Run only against systems you own or have explicit written permission to test.

```bash
# Tier 1 — Packet crafters (hping3 already in Docker)
docker exec lh-attacker-hping hping3 -S -p 80 --flood 172.28.0.10          # DDoS
docker exec lh-attacker-hping hping3 -S --scan 1-1024 172.28.0.10          # PortScan
docker exec lh-attacker-hping hping3 -S -p 22 -i u500 172.28.0.10          # Brute Force

# Tier 2 — Attack frameworks (Metasploit already in Docker)
docker exec -it lh-attacker-msf msfconsole
sudo nmap -sS -p 1-1024 172.28.0.10                                         # PortScan
hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://172.28.0.10 -t 4    # Brute Force

# Tier 3 — BAS platforms (MITRE ATT&CK mapped)
Invoke-AtomicTest T1046    # Network Service Discovery
Invoke-AtomicTest T1110    # Brute Force
Invoke-AtomicTest T1498    # Network DoS
```

---

## Full Documentation

See **[docs/LIGHTHOUSE_FULL_DOCUMENTATION.md](docs/LIGHTHOUSE_FULL_DOCUMENTATION.md)** for:

- Complete architecture diagrams
- Feature engineering details (all 19 CIC + 18 UNSW features mapped to Suricata fields)
- Risk scoring formula and examples
- Full API reference with request/response samples
- Real-world deployment guide (Suricata on dedicated sensor, nginx frontend, systemd)
- Wazuh agent installation on monitored endpoints
- Complete attack simulation guide (hping3, Scapy, Ostinato, TRex, Nmap, Metasploit, CALDERA, Infection Monkey, Atomic Red Team)
- Model performance tables and embedded graphs
- Database management CLI reference

---

<div align="center">

Built for UNSW — AI-SOC capstone project

</div>
