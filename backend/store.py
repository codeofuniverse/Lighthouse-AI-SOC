"""In-memory alert store backed by a bounded deque, persisted to SQLite.

On startup : loads the last 500 alerts from SQLite so alerts survive restarts.
On add()   : writes to in-memory deque + SQLite.
On update(): updates both the in-memory dict and the SQLite row.
stats()    : queries SQLite directly (consistent after restart, not capped by deque size).

Future PostgreSQL migration: only backend/db.py needs to change.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from functools import partial
from typing import Any

from backend.db import (
    db_stats,
    init_db,
    insert_alert,
    load_recent,
    query_alerts,
    update_alert as db_update_alert,
)


class AlertStore:

    def __init__(self, maxsize: int = 500) -> None:
        self._alerts: deque[dict[str, Any]] = deque(maxlen=maxsize)
        self._lock = asyncio.Lock()
        self._index: dict[str, dict[str, Any]] = {}

        init_db()
        for alert in reversed(load_recent(maxsize)):
            self._alerts.appendleft(alert)
            self._index[alert["id"]] = alert

    # ── Writes ────────────────────────────────────────────────────────────────

    async def add(self, alert: dict[str, Any]) -> None:
        async with self._lock:
            self._alerts.appendleft(alert)
            self._index[alert["id"]] = alert
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, insert_alert, alert)

    async def update(self, alert_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        async with self._lock:
            alert = self._index.get(alert_id)
            if alert is None:
                return None
            alert.update(fields)
            updated = dict(alert)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db_update_alert, alert_id, updated)
        return updated

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def get(self, alert_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._index.get(alert_id)

    async def list_alerts(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return alerts from in-memory deque, sorted by threat_level then timestamp."""
        async with self._lock:
            alerts = list(self._alerts)[:limit]
        return sorted(
            alerts,
            key=lambda a: (a.get("threat_level", 0), a.get("timestamp", "")),
            reverse=True,
        )

    async def search(
        self,
        *,
        src_ip: str | None = None,
        attack_type: str | None = None,
        threat_level: int | None = None,
        status: str | None = None,
        since: str | None = None,
        auto_blocked: bool | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query the full SQLite history with optional filters (runs in thread executor)."""
        loop = asyncio.get_event_loop()
        fn = partial(
            query_alerts,
            limit=limit,
            src_ip=src_ip,
            attack_type=attack_type,
            threat_level=threat_level,
            status=status,
            since=since,
            auto_blocked=auto_blocked,
        )
        return await loop.run_in_executor(None, fn)

    async def stats(self) -> dict[str, int]:
        """Dashboard stats from SQLite — accurate across full history and restarts."""
        today_iso = datetime.now(timezone.utc).date().isoformat()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, db_stats, today_iso)


store = AlertStore()
