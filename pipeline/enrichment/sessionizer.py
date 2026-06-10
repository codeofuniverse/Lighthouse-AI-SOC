"""Session tracking and enrichment for alerts."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, cast

import redis

logger = logging.getLogger(__name__)


class Sessionizer:
    """Track user sessions and attach session metadata to alerts."""

    def __init__(self, redis_client: redis.Redis) -> None:
        """Initialize the sessionizer.

        Args:
            redis_client: Redis client for session storage.
        """
        self.redis_client = redis_client
        self.session_window = 300  # 5 minutes
        self.session_ttl = 3600  # 1 hour

    def enrich(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Enrich an alert with session information.

        Args:
            alert: Normalized alert dict.

        Returns:
            Alert dict with session_id, session_event_count, session_duration_seconds attached.
        """
        src_ip = alert.get("src_ip", "")
        if not src_ip:
            # No source IP, create a transient session
            alert["session_id"] = str(uuid.uuid4())
            alert["session_event_count"] = 1
            alert["session_duration_seconds"] = 0
            return alert

        session_key = f"session:{src_ip}"
        now = datetime.now(timezone.utc).timestamp()
        window_start = now - self.session_window

        try:
            # Get events in current window using sorted sets
            events = cast(list[Any], self.redis_client.zrangebyscore(session_key, window_start, now))
            if events:
                # Active session exists, reuse it
                session_data_str = events[0].decode() if isinstance(events[0], bytes) else events[0]
                session_data = json.loads(session_data_str)
                session_id = session_data.get("session_id")
                first_seen = session_data.get("first_seen")
                event_count = session_data.get("event_count", 0) + 1
                unique_dst_ips = set(session_data.get("unique_dst_ips", []))
                unique_ports = set(session_data.get("unique_ports", []))
            else:
                # New session
                session_id = str(uuid.uuid4())
                first_seen = now
                event_count = 1
                unique_dst_ips = set()
                unique_ports = set()

            # Update session metadata
            if dst_ip := alert.get("dst_ip"):
                unique_dst_ips.add(dst_ip)
            if src_port := alert.get("src_port"):
                unique_ports.add(str(src_port))

            session_data = {
                "session_id": session_id,
                "first_seen": first_seen,
                "last_seen": now,
                "event_count": event_count,
                "unique_dst_ips": list(unique_dst_ips),
                "unique_ports": list(unique_ports),
            }

            # Store session in sorted set with timestamp as score
            self.redis_client.zadd(session_key, {json.dumps(session_data): now})
            self.redis_client.expire(session_key, self.session_ttl)

            # Attach session fields to alert
            alert["session_id"] = session_id
            alert["session_event_count"] = event_count
            alert["event_count_in_session"] = event_count
            alert["session_duration_seconds"] = int(now - first_seen)
        except Exception as exc:
            logger.warning("Session enrichment failed for %s: %s", src_ip, exc)
            alert["session_id"] = str(uuid.uuid4())
            alert["session_event_count"] = 1
            alert["event_count_in_session"] = 1
            alert["session_duration_seconds"] = 0

        return alert
