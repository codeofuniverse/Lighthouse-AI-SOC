"""SQLite persistence layer for Lighthouse alerts.

Design decisions:
  - WAL mode: readers never block writers; concurrent FastAPI + ingestion loop
    can both access the DB without locking each other out
  - Single persistent connection per process: opened once at module load,
    reused for every operation — no per-call connect/disconnect overhead
  - Proper column schema: id, timestamp, src_ip, attack_type, threat_level,
    risk_score, auto_blocked, status — queryable without JSON parsing
  - data TEXT column: full alert JSON for fields not in the schema
  - Prune runs every 500 inserts (not every insert) — avoids COUNT(*) overhead
  - stats() queries SQLite directly — consistent after restart when deque
    has been partially filled from DB

Schema:
    alerts (
        id           TEXT PRIMARY KEY,
        timestamp    TEXT,          -- ISO-8601, indexed
        src_ip       TEXT,
        dst_ip       TEXT,
        attack_type  TEXT,          -- CIC/UNSW label
        threat_level INTEGER,       -- 0/1/2, indexed
        risk_score   REAL,
        auto_blocked INTEGER,       -- 0/1 boolean
        status       TEXT,          -- active/dismissed/isolated
        data         TEXT           -- full alert JSON blob
    )

Future PostgreSQL migration:
    Replace _DB_PATH / _get_conn() with a psycopg2/asyncpg connection.
    Column names and query shapes are identical; only ? -> %s placeholders change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH  = Path("data/lighthouse_alerts.db")
_MAX_ROWS = 10_000      # hard cap — oldest rows pruned when exceeded
_PRUNE_EVERY = 500      # check prune frequency (inserts between checks)

# ── Single persistent connection per process ──────────────────────────────────
# sqlite3 with check_same_thread=False is safe when access is serialised through
# a threading.Lock (which we do below). WAL mode lets reads proceed during writes.
_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()
_insert_count = 0


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")       # concurrent readers + writer
        _conn.execute("PRAGMA synchronous=NORMAL")     # safe + faster than FULL
        _conn.execute("PRAGMA cache_size=-8000")       # 8 MB page cache
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.commit()
    return _conn


def init_db() -> None:
    """Create tables and indexes. Safe to call on every startup."""
    with _conn_lock:
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id                 TEXT PRIMARY KEY,
                timestamp          TEXT NOT NULL DEFAULT '',
                src_ip             TEXT NOT NULL DEFAULT '',
                dst_ip             TEXT NOT NULL DEFAULT '',
                attack_type        TEXT NOT NULL DEFAULT '',
                threat_level       INTEGER NOT NULL DEFAULT 0,
                risk_score         REAL    NOT NULL DEFAULT 0.0,
                auto_blocked       INTEGER NOT NULL DEFAULT 0,
                status             TEXT    NOT NULL DEFAULT 'active',
                data               TEXT    NOT NULL DEFAULT '{}',
                geoip_country      TEXT    DEFAULT '',
                geoip_city         TEXT    DEFAULT '',
                geoip_is_tor       INTEGER DEFAULT 0,
                geoip_is_vpn       INTEGER DEFAULT 0,
                abuse_score        INTEGER DEFAULT 0,
                is_known_attacker  INTEGER DEFAULT 0,
                mitre_techniques   TEXT    DEFAULT '[]',
                session_id         TEXT    DEFAULT '',
                session_count      INTEGER DEFAULT 1,
                session_dur        INTEGER DEFAULT 0,
                asset_name         TEXT    DEFAULT '',
                asset_crit         TEXT    DEFAULT 'unknown',
                asset_owner        TEXT    DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts    ON alerts (timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tl    ON alerts (threat_level DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_src   ON alerts (src_ip)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type  ON alerts (attack_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stat  ON alerts (status)")

        # Migration: add new columns to existing tables (SQLite has no IF NOT EXISTS for columns)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        new_cols = {
            "geoip_country":     "TEXT DEFAULT ''",
            "geoip_city":        "TEXT DEFAULT ''",
            "geoip_is_tor":      "INTEGER DEFAULT 0",
            "geoip_is_vpn":      "INTEGER DEFAULT 0",
            "abuse_score":       "INTEGER DEFAULT 0",
            "is_known_attacker": "INTEGER DEFAULT 0",
            "mitre_techniques":  "TEXT DEFAULT '[]'",
            "session_id":        "TEXT DEFAULT ''",
            "session_count":     "INTEGER DEFAULT 1",
            "session_dur":       "INTEGER DEFAULT 0",
            "asset_name":        "TEXT DEFAULT ''",
            "asset_crit":        "TEXT DEFAULT 'unknown'",
            "asset_owner":       "TEXT DEFAULT ''",
        }
        for col, definition in new_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} {definition}")
                logger.info("Migration: added column %s", col)

        conn.commit()
    logger.info("SQLite ready at %s (WAL mode)", _DB_PATH)


# ── Write operations ──────────────────────────────────────────────────────────

def insert_alert(alert: dict[str, Any]) -> None:
    """Insert a new alert. Silently ignores duplicate IDs."""
    global _insert_count
    geo = alert.get("geoip") or {}
    mitre = alert.get("mitre_techniques", [])
    try:
        with _conn_lock:
            _get_conn().execute(
                """INSERT OR IGNORE INTO alerts
                   (id, timestamp, src_ip, dst_ip, attack_type,
                    threat_level, risk_score, auto_blocked, status, data,
                    geoip_country, geoip_city, geoip_is_tor, geoip_is_vpn,
                    abuse_score, is_known_attacker, mitre_techniques,
                    session_id, session_count, session_dur,
                    asset_name, asset_crit, asset_owner)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert["id"],
                    alert.get("timestamp", ""),
                    alert.get("src_ip", ""),
                    alert.get("dst_ip", ""),
                    alert.get("attack_type", ""),
                    int(alert.get("threat_level", 0)),
                    float(alert.get("risk_score", 0.0)),
                    1 if alert.get("auto_blocked") else 0,
                    alert.get("status", "active"),
                    json.dumps(alert),
                    geo.get("country", ""),
                    geo.get("city", ""),
                    1 if geo.get("is_tor") else 0,
                    1 if geo.get("is_vpn") else 0,
                    int(alert.get("abuse_score", 0)),
                    1 if alert.get("is_known_attacker") else 0,
                    json.dumps(mitre) if isinstance(mitre, list) else mitre,
                    alert.get("session_id", ""),
                    int(alert.get("session_count", 1)),
                    int(alert.get("session_dur", 0)),
                    alert.get("asset_name", ""),
                    alert.get("asset_crit", "unknown"),
                    alert.get("asset_owner", ""),
                ),
            )
            _get_conn().commit()
            _insert_count += 1
        if _insert_count % _PRUNE_EVERY == 0:
            _prune()
    except Exception as exc:
        logger.warning("insert_alert failed for %s: %s", alert.get("id"), exc)


def update_alert(alert_id: str, alert: dict[str, Any]) -> None:
    """Overwrite all queryable columns + JSON blob for an existing alert."""
    geo = alert.get("geoip") or {}
    mitre = alert.get("mitre_techniques", [])
    try:
        with _conn_lock:
            _get_conn().execute(
                """UPDATE alerts SET
                   threat_level=?, risk_score=?, auto_blocked=?, status=?, data=?,
                   geoip_country=?, geoip_city=?, geoip_is_tor=?, geoip_is_vpn=?,
                   abuse_score=?, is_known_attacker=?, mitre_techniques=?,
                   session_id=?, session_count=?, session_dur=?,
                   asset_name=?, asset_crit=?, asset_owner=?
                   WHERE id=?""",
                (
                    int(alert.get("threat_level", 0)),
                    float(alert.get("risk_score", 0.0)),
                    1 if alert.get("auto_blocked") else 0,
                    alert.get("status", "active"),
                    json.dumps(alert),
                    geo.get("country", ""),
                    geo.get("city", ""),
                    1 if geo.get("is_tor") else 0,
                    1 if geo.get("is_vpn") else 0,
                    int(alert.get("abuse_score", 0)),
                    1 if alert.get("is_known_attacker") else 0,
                    json.dumps(mitre) if isinstance(mitre, list) else mitre,
                    alert.get("session_id", ""),
                    int(alert.get("session_count", 1)),
                    int(alert.get("session_dur", 0)),
                    alert.get("asset_name", ""),
                    alert.get("asset_crit", "unknown"),
                    alert.get("asset_owner", ""),
                    alert_id,
                ),
            )
            _get_conn().commit()
    except Exception as exc:
        logger.warning("update_alert failed for %s: %s", alert_id, exc)


# ── Read operations ───────────────────────────────────────────────────────────

def load_recent(limit: int = 500) -> list[dict[str, Any]]:
    """Return the most recent `limit` alerts, newest first."""
    try:
        with _conn_lock:
            rows = _get_conn().execute(
                "SELECT data FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]
    except Exception as exc:
        logger.warning("load_recent failed: %s", exc)
        return []


def query_alerts(
    *,
    limit: int = 200,
    src_ip: str | None = None,
    attack_type: str | None = None,
    threat_level: int | None = None,
    status: str | None = None,
    since: str | None = None,          # ISO-8601 timestamp lower bound
    auto_blocked: bool | None = None,
) -> list[dict[str, Any]]:
    """Flexible alert query — all filters are optional, combined with AND."""
    clauses: list[str] = []
    params: list[Any]  = []

    if src_ip is not None:
        clauses.append("src_ip = ?");       params.append(src_ip)
    if attack_type is not None:
        clauses.append("attack_type = ?");  params.append(attack_type)
    if threat_level is not None:
        clauses.append("threat_level = ?"); params.append(threat_level)
    if status is not None:
        clauses.append("status = ?");       params.append(status)
    if since is not None:
        try:
            datetime.fromisoformat(since)
            clauses.append("timestamp >= ?"); params.append(since)
        except ValueError:
            logger.warning("query_alerts: invalid 'since' timestamp %r — filter ignored", since)
    if auto_blocked is not None:
        clauses.append("auto_blocked = ?"); params.append(1 if auto_blocked else 0)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT data FROM alerts {where} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    try:
        with _conn_lock:
            rows = _get_conn().execute(sql, params).fetchall()
        return [json.loads(r["data"]) for r in rows]
    except Exception as exc:
        logger.warning("query_alerts failed: %s", exc)
        return []


def db_stats(today_iso: str) -> dict[str, int]:
    """Compute dashboard stats directly from SQLite — consistent after restarts."""
    try:
        with _conn_lock:
            conn = _get_conn()
            total_today  = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE substr(timestamp,1,10) >= ?", (today_iso,)
            ).fetchone()[0]
            critical     = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE threat_level = 2"
            ).fetchone()[0]
            suspicious   = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE threat_level = 1"
            ).fetchone()[0]
            auto_blocked = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE auto_blocked = 1"
            ).fetchone()[0]
        return {
            "total_today":  total_today,
            "critical":     critical,
            "suspicious":   suspicious,
            "auto_blocked": auto_blocked,
        }
    except Exception as exc:
        logger.warning("db_stats failed: %s", exc)
        return {"total_today": 0, "critical": 0, "suspicious": 0, "auto_blocked": 0}


def db_count() -> int:
    """Return total number of stored alerts."""
    try:
        with _conn_lock:
            return _get_conn().execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    except Exception:
        return 0


# ── Maintenance ───────────────────────────────────────────────────────────────

def _prune() -> None:
    """Delete oldest rows when total exceeds _MAX_ROWS."""
    try:
        with _conn_lock:
            count = _get_conn().execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            if count > _MAX_ROWS:
                excess = count - _MAX_ROWS
                _get_conn().execute("""
                    DELETE FROM alerts WHERE id IN (
                        SELECT id FROM alerts ORDER BY timestamp ASC LIMIT ?
                    )
                """, (excess,))
                _get_conn().commit()
                logger.info("Pruned %d old alerts (DB was at %d rows)", excess, count)
    except Exception as exc:
        logger.debug("Prune failed: %s", exc)


def vacuum() -> None:
    """Reclaim disk space after bulk deletes. Run manually or via init_db script."""
    try:
        with _conn_lock:
            _get_conn().execute("VACUUM")
        logger.info("VACUUM complete")
    except Exception as exc:
        logger.warning("VACUUM failed: %s", exc)
