"""Lighthouse SOC — FastAPI backend.

Endpoints:
  GET  /api/alerts               → last 200 alerts (sorted threat_level desc)
  GET  /api/stats                → total_today, critical, suspicious, auto_blocked
  POST /api/alerts/{id}/block    → block IP via SOAR
  POST /api/alerts/{id}/isolate  → isolate agent via SOAR
  POST /api/alerts/{id}/dismiss  → mark alert dismissed
  WS   /ws/alerts                → real-time alert stream to React dashboard
  GET  /health                   → liveness check
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import redis as redis_lib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend import asset_service
from backend.alert_builder import build_alert
from backend.db import db_count
from backend.llm_assistant import llm
from backend.soar import soar
from backend.store import store
from backend.ws_manager import ws_manager
from detection.rate_aggregator import RateAggregator
from detection.wazuh_alerts import HostSeenTracker, WazuhAlert, tail_wazuh_alerts
from pipeline.decision_engine import Decision, DecisionEngine
from pipeline.enrichment.geoip import GeoIPEnricher
from pipeline.enrichment.mitre_mapper import MitreMapper
from pipeline.enrichment.sessionizer import Sessionizer
from pipeline.enrichment.threat_intel import ThreatIntelEnricher
from pipeline.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

_EVE_JSON_PATH  = os.getenv("EVE_JSON_PATH",  "/var/log/suricata/eve.json")
_CIC_MODEL_PATH = os.getenv("CIC_MODEL_PATH", "data/models/cic2017_pipeline_smote.joblib")
_ZEEK_CONN_PATH = os.getenv("ZEEK_CONN_PATH", "/zeek/logs/current/conn.log")

_risk_scorer     = RiskScorer()
_decision_engine = DecisionEngine()
_rate_agg        = RateAggregator()  # volumetric (DDoS/PortScan/DoS) detection
_host_seen       = HostSeenTracker()  # IPs recently seen on the Wazuh host layer

# ── Enrichment pipeline (graceful degradation if deps missing) ────────────────
try:
    _redis = redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _redis.ping()
    _redis_ok = True
except Exception:
    _redis = None  # type: ignore[assignment]
    _redis_ok = False

_geoip   = GeoIPEnricher(os.getenv("GEOIP_DB_PATH", "data/GeoLite2-City.mmdb"), _redis)
_threat  = ThreatIntelEnricher(os.getenv("ABUSEIPDB_API_KEY", ""), _redis)
_mitre   = MitreMapper("data/mitre_rule_mapping.yaml")
_session = Sessionizer(_redis) if _redis else None

asset_service.load()


# ── Dev-mode seed ─────────────────────────────────────────────────────────────

async def _seed_dev_alerts() -> None:
    """Seed 5 realistic mock alerts when running without Suricata (dev mode)."""
    profiles = [
        ("DDoS",       "172.28.0.20",    2, 0.97, 88.0, "172.28.0.20",
         "High-rate SYN flood from 172.28.0.20 targeting port 80 with 4,800 packets/s. "
         "CIC model confidence 97% — matches DDoS training distribution. Risk score 88, auto-block triggered."),
        ("PortScan",   "10.0.0.199",     2, 0.91, 72.0, "10.0.0.199",
         "SYN-only flows targeting 1,200 sequential ports from 10.0.0.199. "
         "Zero bytes returned — all probes unanswered. Reconnaissance pattern confirmed."),
        ("Brute Force","172.28.0.21",    1, 0.78, 61.0, "172.28.0.21",
         "47 FIN+PSH+ACK flows to port 22 from 172.28.0.21 over 90 seconds. "
         "Packet symmetry and flow duration match SSH brute-force training profile."),
        ("DoS",        "185.220.101.5",  1, 0.65, 52.0, "185.220.101.5",
         "Sustained high-byte flows (420KB avg) to port 80 with 35s duration. "
         "PSH+ACK flag pattern and asymmetric byte ratio indicate DoS Hulk variant."),
        ("Bot",        "10.0.0.55",      0, 0.43, 28.0, "10.0.0.55",
         "Periodic low-rate flows every 60s from 10.0.0.55 — consistent beaconing interval. "
         "CIC confidence 43%, below alert threshold. Monitoring recommended."),
    ]

    base_time = datetime.now(timezone.utc)
    for i, (attack, src, level, conf, risk, agent, explanation) in enumerate(profiles):
        ts = (base_time - timedelta(minutes=i * 7)).isoformat()
        alert_id = hashlib.sha1(f"seed:{src}:{attack}".encode()).hexdigest()[:16]
        if await store.get(alert_id):
            continue  # already seeded — skip on restart
        decision = Decision(
            action="auto_block" if risk >= 81 else ("review" if risk >= 61 else ("alert" if risk >= 31 else "log")),
            risk_score=risk,
            threat_level=level,
            auto_blocked=risk >= 81,
        )
        alert: dict[str, Any] = {
            "id":               alert_id,
            "timestamp":        ts,
            "attack_type":      attack,
            "src_ip":           src,
            "dst_ip":           "172.28.0.10",
            "dst_port":         80,
            "proto":            "TCP",
            "agent_name":       agent,
            "agent_id":         f"00{i + 1}",
            "rule_level":       int(risk / 100 * 15),
            "rule_description": f"Suricata flow analysis — {attack} pattern",
            "status":           "active",
            "auto_blocked":     decision.auto_blocked,
            "confidence":       round(conf, 4),
            "threat_level":     level,
            "risk_score":       risk,
            "ai_explanation":   explanation,
            "cic_confidence":   round(conf, 4),
            "unsw_confidence":  None,
            "abuse_score":      0,
            "action_history":   [],
            "ingested_at":      ts,
        }
        await store.add(alert)

    logger.info("Dev mode: seeded %d mock alerts", len(profiles))


# ── Background ingestion ──────────────────────────────────────────────────────

def _zeek_available() -> bool:
    """Zeek is the primary feature sensor when its conn.log exists. The path may be
    the live `current/` symlink or a flat file depending on the Zeek launch mode."""
    if os.getenv("LIGHTHOUSE_ZEEK", "1") != "1":
        return False
    p = _ZEEK_CONN_PATH
    if os.path.exists(p):
        return True
    # also accept a conn.log directly under the logs dir (non-rotating live mode)
    alt = os.path.join(os.path.dirname(os.path.dirname(p)), "conn.log")
    return os.path.exists(alt)


async def _ingest_loop() -> None:
    """Ingest network flows. Topology:

      Zeek-primary (default): Zeek scores flows (rich 18-feat CIC + 28-feat UNSW);
        Suricata runs signatures-only and its alerts are joined onto Zeek flows by
        community-id (Layer 3). This is the strongest hybrid: behavioural + ML +
        signature.
      Suricata fallback: if Zeek's conn.log is absent (Zeek down / not deployed),
        Suricata does the ML scoring itself — detection stays alive.
    """
    if not os.path.exists(_EVE_JSON_PATH) and not _zeek_available():
        return  # dev mode — store already seeded

    loop = asyncio.get_event_loop()
    _QUEUE_SIZE   = int(os.getenv("INGEST_QUEUE_SIZE", "2000"))
    _NUM_CONSUMERS = int(os.getenv("INGEST_CONSUMERS", "3"))
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_SIZE)
    _drop_count = {"n": 0}
    # Shared {community_id: signature} populated by the Suricata signatures-only
    # reader and consumed (joined onto flows) by the Zeek bridge.
    sig_pending: dict[str, str] = {}

    def _pump(stream, source: str) -> None:
        """Push a bridge's DetectionEvents into the shared queue (runs in executor)."""
        try:
            for ev in stream:
                future = asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
                try:
                    future.result(timeout=10)
                except Exception:
                    _drop_count["n"] += 1
                    if _drop_count["n"] <= 10 or _drop_count["n"] % 100 == 0:
                        logger.warning("Queue full, dropping %s event from %s (total drops=%d)",
                                        source, getattr(ev, "src_ip", "?"), _drop_count["n"])
        except Exception as exc:
            logger.error("%s ingestion producer error: %s", source, exc)

    async def _producer() -> None:
        tasks = []
        if _zeek_available():
            # ── Zeek-primary: Zeek scores; Suricata feeds signatures only ──
            from detection.zeek_bridge import ZeekBridge
            zbridge = ZeekBridge(cic_model_path=_CIC_MODEL_PATH)
            logger.info("Zeek-primary ingestion from %s", zbridge.conn_path)
            tasks.append(loop.run_in_executor(
                None, lambda: _pump(zbridge.stream(sig_pending), "zeek")))
            if os.path.exists(_EVE_JSON_PATH):
                from detection.suricata_bridge import tail_signatures
                logger.info("Suricata signatures-only reader on %s", _EVE_JSON_PATH)
                tasks.append(loop.run_in_executor(
                    None, lambda: tail_signatures(_EVE_JSON_PATH, sig_pending)))
        else:
            # ── Fallback: Suricata does the ML scoring (Zeek absent) ──
            from detection.suricata_bridge import SuricataBridge
            bridge = SuricataBridge(model_path=_CIC_MODEL_PATH, eve_path=_EVE_JSON_PATH)
            logger.info("Zeek unavailable — Suricata-ML fallback on %s", _EVE_JSON_PATH)
            tasks.append(loop.run_in_executor(None, _pump, bridge.stream(), "suricata"))
        await asyncio.gather(*tasks)

    # IPs to never alert on — VirtualBox gateway, loopback, link-local
    _WHITELIST = {"192.168.56.1", "127.0.0.1", "0.0.0.0", "::1"}
    _WHITELIST_PREFIXES = ("fe80:", "fc00:", "fd")

    # ── Concept-drift monitor ──────────────────────────────────────────────────
    # Track the rolling attack-probability distribution. A sustained shift in the
    # mean CIC attack_prob signals the live traffic has drifted from the training
    # distribution → log a "retrain recommended" warning (DDM-style trigger; full
    # online retraining per CEUR Vol-3962 / ReCDA KDD 2024 is future work).
    _drift = {"sum": 0.0, "n": 0, "window": 5000, "baseline": None}

    def _track_drift(attack_prob: float) -> None:
        _drift["sum"] += attack_prob
        _drift["n"] += 1
        if _drift["n"] >= _drift["window"]:
            mean = _drift["sum"] / _drift["n"]
            if _drift["baseline"] is None:
                _drift["baseline"] = mean
                logger.info("Drift monitor baseline attack_prob mean=%.4f", mean)
            elif abs(mean - _drift["baseline"]) > 0.15:
                logger.warning(
                    "Concept drift detected: attack_prob mean %.4f vs baseline %.4f "
                    "(>0.15 shift) — retrain recommended", mean, _drift["baseline"])
            _drift["sum"], _drift["n"] = 0.0, 0

    async def _consumer() -> None:
        while True:
            event = await queue.get()
            try:
                _track_drift(event.stage1_attack_prob)
                # Drop whitelisted IPs (VirtualBox gateway, loopback, link-local IPv6)
                src = event.src_ip or ""
                if src in _WHITELIST or any(src.lower().startswith(p) for p in _WHITELIST_PREFIXES):
                    continue

                # ── Volumetric detection: feed EVERY flow to the rate aggregator ──
                # Must run before benign-dropping so rates are counted accurately.
                cf = event.cic_features or {}
                verdict = _rate_agg.observe(
                    src_ip=src,
                    dst_ip=event.dst_ip,
                    dst_port=event.dst_port,
                    syn=cf.get("syn_flag", 0) > 0,
                    ack=cf.get("ack_flag", 0) > 0,
                    bwd_pkts=cf.get("bwd_pkts", 0),
                    total_bytes=cf.get("fwd_bytes", 0) + cf.get("bwd_bytes", 0),
                )

                # ── Fuse three initiators (research-backed hybrid NIDS):
                #   Layer 1 rate aggregator — volumetric truth (SYN flood, scan)
                #   Layer 2 CIC ML          — structured/app-layer attacks
                #   Layer 3 Suricata signature — known IOCs / thresholded floods
                # (UNSW removed 2026 — it only corroborated and was dataset-limited)
                cic_attack  = event.prediction != "BENIGN"
                rate_attack = verdict is not None
                sig_attack  = event.sig_attack_type is not None

                # Truly benign: no initiator fired → drop
                if not (cic_attack or rate_attack or sig_attack):
                    continue

                # Priority: confident rate verdict (measured volumetric truth) >
                # Suricata signature (Suricata's own threshold engine agrees) >
                # CIC family label > low-confidence rate verdict. When rate and
                # signature disagree we prefer rate and log it (never suppress).
                if rate_attack and verdict.confidence >= 0.6:
                    attack_label = verdict.attack_type
                    ml_conf      = verdict.confidence
                    if sig_attack and event.sig_attack_type != verdict.attack_type:
                        logger.info("Detector disagreement: rate=%s sig=%s (using rate) src=%s",
                                    verdict.attack_type, event.sig_attack_type, src)
                elif sig_attack:
                    attack_label = event.sig_attack_type
                    ml_conf      = max(0.85, event.stage1_attack_prob)  # signature is high-confidence
                elif cic_attack:
                    attack_label = event.prediction
                    ml_conf      = event.stage1_attack_prob
                else:  # low-confidence rate verdict
                    attack_label = verdict.attack_type
                    ml_conf      = verdict.confidence

                rule_level = 12  # an initiator fired → treat as high-severity behavior

                risk = _risk_scorer.score(
                    ml_conf=ml_conf,
                    abuse_score=0,
                    rule_level=rule_level,
                    agent_type="unknown",
                    attack_label=attack_label,
                )
                decision = _decision_engine.decide(risk)

                if decision.action == "log":
                    continue

                try:
                    explanation = await asyncio.wait_for(
                        llm.explain(
                            prediction=attack_label,
                            src_ip=event.src_ip,
                            confidence=ml_conf,
                            risk_score=risk,
                        ),
                        timeout=8.0,
                    )
                except asyncio.TimeoutError:
                    explanation = "AI explanation unavailable (timeout)"

                alert = build_alert(event, decision, explanation)
                # Override label/description when a rate verdict or UNSW drove the alert
                alert["attack_type"] = attack_label
                if rate_attack:
                    d = verdict.detail
                    alert["rule_description"] = (
                        f"Volumetric {verdict.attack_type}: {d.get('flows_per_sec')} flows/s, "
                        f"{d.get('unique_dst_ports')} ports, half-open {d.get('half_open_ratio')}"
                    )
                    alert["rate_detail"] = d

                # ── Enrichment ────────────────────────────────────────────────
                src_ip = event.src_ip

                geo = _geoip.enrich(src_ip) or {}
                alert["geoip"]         = geo
                alert["geoip_country"] = geo.get("country", "")
                alert["geoip_city"]    = geo.get("city", "")

                ti = _threat.enrich(src_ip) or {}
                alert["abuse_score"]       = int(ti.get("abuse_score", 0))
                alert["is_known_attacker"] = bool(ti.get("is_known_attacker", False))
                alert["threat_intel"]      = ti

                alert["mitre_techniques"] = _mitre.map([attack_label.lower()])

                if _session:
                    alert = _session.enrich(alert)
                    alert["session_count"] = alert.get("session_event_count", 1)
                    alert["session_dur"]   = int(alert.get("session_duration_seconds", 0))

                asset = asset_service.lookup(
                    agent_id=alert.get("agent_id") or None,
                    agent_ip=src_ip,
                )
                if asset:
                    alert["asset_name"]  = asset.get("asset_name", "")
                    alert["asset_crit"]  = asset.get("asset_criticality", "unknown")
                    alert["asset_owner"] = asset.get("asset_owner", "")

                # Re-score with full enrichment data + host/network correlation
                agent_type = alert.get("asset_crit", "unknown")
                ip_hit_count = int(alert.get("session_count", 1))
                host_corr = _host_seen.is_correlated(src_ip)
                if host_corr:
                    alert["host_correlated"] = True
                risk = _risk_scorer.score(
                    ml_conf=ml_conf,
                    abuse_score=alert["abuse_score"],
                    rule_level=rule_level,
                    agent_type=agent_type,
                    attack_label=attack_label,
                    ip_hit_count=ip_hit_count,
                    host_correlated=host_corr,
                )
                alert["risk_score"]   = risk
                decision              = _decision_engine.decide(risk)
                alert["threat_level"] = decision.threat_level
                alert["auto_blocked"] = decision.auto_blocked
                # ─────────────────────────────────────────────────────────────

                if decision.auto_blocked:
                    await soar.block_ip(event.src_ip)
                    alert["action_history"].append({
                        "action":  "auto_block",
                        "analyst": "system",
                        "time":    datetime.now(timezone.utc).isoformat(),
                    })

                await store.add(alert)
                await ws_manager.broadcast(alert)
                logger.info("Produced alert id=%s type=%s risk=%.0f src=%s",
                            alert["id"], alert["attack_type"], risk, src_ip)

            except Exception as exc:
                logger.error("Alert processing error: %s", exc)
            finally:
                queue.task_done()

    # ── Wazuh host-alert ingestion (Layer: host correlation) ──────────────────
    async def _wazuh_loop() -> None:
        """Tail Wazuh alerts, mark host-seen IPs (for correlation), and surface
        host alerts on the dashboard through the same enrichment + scoring path."""
        if os.getenv("WAZUH_FUSION", "1").strip().lower() in {"0", "false", "no", "off"}:
            logger.info("Wazuh fusion disabled (WAZUH_FUSION=0)")
            return

        wq: asyncio.Queue[WazuhAlert] = asyncio.Queue(maxsize=500)

        def _blocking_tail() -> None:
            try:
                for wa in tail_wazuh_alerts():
                    if wa is None:
                        continue
                    fut = asyncio.run_coroutine_threadsafe(wq.put(wa), loop)
                    try:
                        fut.result(timeout=10)
                    except Exception:
                        pass  # host-alert queue full — drop, network path unaffected
            except Exception as exc:
                logger.error("Wazuh tail error: %s", exc)

        async def _wazuh_consumer() -> None:
            while True:
                wa = await wq.get()
                try:
                    if wa.src_ip:
                        _host_seen.mark(wa.src_ip)  # enables host+network correlation
                    await _handle_wazuh_alert(wa)
                except Exception as exc:
                    logger.error("Wazuh alert processing error: %s", exc)
                finally:
                    wq.task_done()

        loop.run_in_executor(None, _blocking_tail)
        asyncio.create_task(_wazuh_consumer())
        logger.info("Wazuh host-alert fusion started")

    async def _handle_wazuh_alert(wa: WazuhAlert) -> None:
        src = wa.src_ip or wa.agent_ip
        if src in _WHITELIST or any(src.lower().startswith(p) for p in _WHITELIST_PREFIXES):
            return

        # Enrichment (reuse the same enrichers as the network path)
        geo = _geoip.enrich(src) or {} if src else {}
        ti  = _threat.enrich(src) or {} if src else {}
        abuse = int(ti.get("abuse_score", 0))

        # Score using the REAL Wazuh rule level (1–15) — no synthetic value
        risk = _risk_scorer.score(
            ml_conf=0.0,
            abuse_score=abuse,
            rule_level=wa.rule_level,
            agent_type="server",  # host alerts are about a monitored asset
            attack_label=wa.attack_type,
        )
        decision = _decision_engine.decide(risk)

        alert = {
            "id": __import__("hashlib").sha1(
                f"wazuh:{wa.agent_id}:{wa.timestamp}:{wa.rule_description}:{os.urandom(4).hex()}".encode()
            ).hexdigest()[:16],
            "timestamp": wa.timestamp,
            "attack_type": wa.attack_type,
            "src_ip": src,
            "dst_ip": wa.agent_ip,
            "dst_port": 0,
            "proto": "host",
            "agent_name": wa.agent_name or wa.agent_id,
            "agent_id": wa.agent_id,
            "rule_level": wa.rule_level,
            "rule_description": f"[Wazuh] {wa.rule_description}",
            "status": "active",
            "auto_blocked": decision.auto_blocked,
            "confidence": None,
            "threat_level": decision.threat_level,
            "risk_score": risk,
            "ai_explanation": f"Host alert from Wazuh agent {wa.agent_name or wa.agent_id}: {wa.rule_description}",
            "cic_confidence": None,
            "unsw_confidence": None,
            "abuse_score": abuse,
            "geoip": geo,
            "geoip_country": geo.get("country", ""),
            "geoip_city": geo.get("city", ""),
            "is_known_attacker": bool(ti.get("is_known_attacker", False)),
            "threat_intel": ti,
            "mitre_techniques": _mitre.map([g.lower() for g in wa.rule_groups]),
            "source": "wazuh",
            "action_history": [],
            "ingested_at": wa.timestamp,
        }

        if decision.action == "log":
            return
        if decision.auto_blocked and src:
            await soar.block_ip(src)
            alert["action_history"].append(
                {"action": "auto_block", "analyst": "system",
                 "time": datetime.now(timezone.utc).isoformat()})

        await store.add(alert)
        await ws_manager.broadcast(alert)
        logger.info("Produced Wazuh alert id=%s type=%s risk=%.0f agent=%s",
                    alert["id"], alert["attack_type"], risk, wa.agent_name or wa.agent_id)

    asyncio.create_task(_producer())
    for _ in range(_NUM_CONSUMERS):
        asyncio.create_task(_consumer())
    asyncio.create_task(_wazuh_loop())
    logger.info("Ingestion started: queue=%d consumers=%d", _QUEUE_SIZE, _NUM_CONSUMERS)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _redis_ok:
        logger.info("Redis connected at %s:%s", os.getenv("REDIS_HOST", "localhost"), os.getenv("REDIS_PORT", "6379"))
    else:
        logger.warning("Redis unavailable — enrichment caching disabled")

    if not os.path.exists(_EVE_JSON_PATH) and not _zeek_available():
        await _seed_dev_alerts()
    asyncio.create_task(_ingest_loop())
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Lighthouse SOC API", version="1.0.0", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(limit: int = 200):
    return await store.list_alerts(limit=limit)


@app.get("/api/stats")
async def get_stats():
    return await store.stats()


@app.post("/api/alerts/{alert_id}/block")
async def block_alert(alert_id: str):
    alert = await store.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    success = await soar.block_ip(alert.get("src_ip", ""))
    if not success:
        raise HTTPException(status_code=500, detail="SOAR block action failed")
    entry = {"action": "block", "analyst": "analyst", "time": datetime.now(timezone.utc).isoformat()}
    updated = await store.update(alert_id, {
        "auto_blocked":   True,
        "action_history": alert.get("action_history", []) + [entry],
    })
    await ws_manager.broadcast(updated)
    return updated


@app.post("/api/alerts/{alert_id}/isolate")
async def isolate_alert(alert_id: str):
    alert = await store.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    success = await soar.isolate_agent(alert.get("agent_id", ""))
    if not success:
        raise HTTPException(status_code=500, detail="SOAR isolate action failed")
    entry = {"action": "isolate", "analyst": "analyst", "time": datetime.now(timezone.utc).isoformat()}
    updated = await store.update(alert_id, {
        "status":         "isolated",
        "action_history": alert.get("action_history", []) + [entry],
    })
    await ws_manager.broadcast(updated)
    return updated


@app.post("/api/alerts/{alert_id}/dismiss")
async def dismiss_alert(alert_id: str):
    alert = await store.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    entry = {"action": "dismiss", "analyst": "analyst", "time": datetime.now(timezone.utc).isoformat()}
    updated = await store.update(alert_id, {
        "status":         "dismissed",
        "action_history": alert.get("action_history", []) + [entry],
    })
    await ws_manager.broadcast(updated)
    return updated


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws_manager.connect(ws)
    alerts = await store.list_alerts(limit=50)
    for alert in reversed(alerts):
        try:
            await ws.send_json(alert, mode="text")
        except Exception:
            break
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@app.get("/api/alerts/search")
async def search_alerts(
    src_ip:       str  | None = None,
    attack_type:  str  | None = None,
    threat_level: int  | None = None,
    status:       str  | None = None,
    since:        str  | None = None,
    auto_blocked: bool | None = None,
    limit:        int         = 200,
):
    """Query full SQLite history with optional filters.

    Examples:
      /api/alerts/search?src_ip=1.2.3.4
      /api/alerts/search?attack_type=DDoS&threat_level=2
      /api/alerts/search?auto_blocked=true&limit=50
      /api/alerts/search?since=2026-05-25T00:00:00
    """
    return await store.search(
        src_ip=src_ip,
        attack_type=attack_type,
        threat_level=threat_level,
        status=status,
        since=since,
        auto_blocked=auto_blocked,
        limit=min(limit, 1000),
    )


@app.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: str):
    alert = await store.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.post("/api/alerts/{alert_id}/unblock")
async def unblock_alert(alert_id: str):
    alert = await store.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    success = await soar.unblock_ip(alert.get("src_ip", ""))
    if not success:
        raise HTTPException(status_code=500, detail="SOAR unblock action failed")
    entry = {"action": "unblock", "analyst": "analyst", "time": datetime.now(timezone.utc).isoformat()}
    updated = await store.update(alert_id, {
        "auto_blocked":   False,
        "status":         "active",
        "action_history": alert.get("action_history", []) + [entry],
    })
    await ws_manager.broadcast(updated)
    return updated


@app.get("/api/attackers")
async def get_attackers(limit: int = 100):
    """Return unique attacker IPs with aggregated stats, sorted by max threat level."""
    alerts = await store.search(limit=1000)
    seen: dict[str, dict] = {}
    for alert in alerts:
        ip = alert.get("src_ip", "")
        if not ip:
            continue
        if ip not in seen:
            seen[ip] = {
                "src_ip":          ip,
                "alert_count":     0,
                "max_threat":      0,
                "max_risk":        0.0,
                "last_seen":       "",
                "attack_types":    set(),
                "is_known_attacker": False,
                "abuse_score":     0,
                "geoip":           alert.get("geoip") or {},
                "auto_blocked":    False,
            }
        entry = seen[ip]
        entry["alert_count"] += 1
        entry["max_threat"]   = max(entry["max_threat"], alert.get("threat_level", 0))
        entry["max_risk"]     = max(entry["max_risk"],   alert.get("risk_score", 0.0))
        entry["auto_blocked"] = entry["auto_blocked"] or bool(alert.get("auto_blocked"))
        entry["is_known_attacker"] = entry["is_known_attacker"] or bool(alert.get("is_known_attacker"))
        entry["abuse_score"]  = max(entry["abuse_score"], int(alert.get("abuse_score") or 0))
        ts = alert.get("timestamp", "")
        if ts > entry["last_seen"]:
            entry["last_seen"] = ts
            entry["geoip"]     = alert.get("geoip") or entry["geoip"]
        if alert.get("attack_type"):
            entry["attack_types"].add(alert["attack_type"])

    result = []
    for entry in seen.values():
        entry["attack_types"] = sorted(entry["attack_types"])
        result.append(entry)

    result.sort(key=lambda x: (x["max_threat"], x["max_risk"]), reverse=True)
    return result[:limit]


@app.get("/api/sessions/{src_ip}")
async def get_sessions(src_ip: str, limit: int = 50):
    """Return all alerts for a given source IP, newest first."""
    return await store.search(src_ip=src_ip, limit=min(limit, 500))


@app.get("/api/enrichment/geoip/{ip}")
async def get_geoip(ip: str):
    result = _geoip.enrich(ip)
    if result is None:
        return {"ip": ip, "result": None, "note": "private or unknown"}
    return {"ip": ip, "result": result}


@app.get("/api/enrichment/threat-intel/{ip}")
async def get_threat_intel(ip: str):
    result = _threat.enrich(ip)
    if result is None:
        return {"ip": ip, "result": None, "note": "private, unknown, or rate limited"}
    return {"ip": ip, "result": result}


@app.get("/health")
async def health():
    redis_ok = False
    if _redis:
        try:
            _redis.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "status":       "ok",
        "db_alerts":    db_count(),
        "redis":        redis_ok,
        "llm_failures": llm.consecutive_failures,
    }


# ── Dev-only seed endpoint ────────────────────────────────────────────────────

if os.getenv("LIGHTHOUSE_DEV", "1") == "1":
    @app.post("/api/dev/seed")
    async def dev_seed(alert: dict[str, Any]):
        await store.add(alert)
        await ws_manager.broadcast(alert)
        return {"seeded": alert.get("id")}
