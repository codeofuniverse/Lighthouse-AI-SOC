"""Redis-aware wrapper around RuleEngine to evaluate session-level metrics like unique ports/IPs."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from detection.rule_engine import RuleEngine, MatchedRule

logger = logging.getLogger(__name__)


class RuleEngineRedis(RuleEngine):
    """Extends RuleEngine to resolve session-based metrics from Redis when rules reference them."""

    def __init__(self, rules_dir: str = "detection/sigma_rules", redis_client: redis.Redis | None = None) -> None:
        super().__init__(rules_dir=rules_dir)
        self.redis_client = redis_client

    def _get_session_metadata(self, session_id: str) -> dict[str, Any]:
        """Retrieve session metadata stored by the sessionizer in Redis.

        Expects a sorted set at key `session:{src_ip}` containing JSON blobs; we return the latest one.
        """
        if not self.redis_client or not session_id:
            return {}

        try:
            # Session key is stored as session:{session_id} or session:{src_ip} depending on implementation
            # First try session key by session_id
            key = f"session:{session_id}"
            data = self.redis_client.zrange(key, -1, -1)
            if not data:
                # Try scanning keys for any session:* containing session_id in the JSON
                for k in self.redis_client.scan_iter(match="session:*"):
                    values = self.redis_client.zrange(k, -1, -1)
                    if not values:
                        continue
                    try:
                        payload = json.loads(values[0]) if isinstance(values[0], (str, bytes)) else {}
                        if payload.get("session_id") == session_id:
                            return payload
                    except Exception:
                        continue
                return {}

            raw = data[0]
            if isinstance(raw, bytes):
                raw = raw.decode()
            payload = json.loads(raw)
            return payload
        except Exception as exc:
            logger.warning("Failed to fetch session metadata from Redis: %s", exc)
            return {}

    def evaluate(self, alert: dict[str, Any]) -> list[MatchedRule]:
        """Evaluate rules, resolving session-based metrics from Redis when needed."""
        matched = super().evaluate(alert)

        # Additionally handle rules that include unique_dst_ports or unique_dst_ips conditions
        # by re-evaluating rules manually if needed
        try:
            for rule in self.rules:
                detection = rule.get("detection", {})
                if not detection:
                    continue
                condition = detection.get("condition") or {}
                # Check if rule references unique_dst_ports or unique_dst_ips
                needs_session = False
                keys = []
                if isinstance(condition, list):
                    for c in condition:
                        if isinstance(c, dict):
                            for k in c.keys():
                                if k in ("unique_dst_ports", "unique_dst_ips"):
                                    needs_session = True
                                    keys.append(k)
                elif isinstance(condition, dict):
                    for k in condition.keys():
                        if k in ("unique_dst_ports", "unique_dst_ips"):
                            needs_session = True
                            keys.append(k)

                if not needs_session:
                    continue

                # Fetch session metadata
                session_id = alert.get("session_id") or alert.get("src_ip")
                session_meta = self._get_session_metadata(session_id)
                # Build augmented alert with unique counts
                augmented = dict(alert)
                augmented["unique_dst_ports"] = len(session_meta.get("unique_ports", []))
                augmented["unique_dst_ips"] = len(session_meta.get("unique_dst_ips", []))

                # Re-evaluate the rule against augmented alert using parent helpers
                if self._evaluate_condition(condition, augmented):
                    # If not already present in matched, append
                    if not any(m.rule_id == rule.get("id") for m in matched):
                        matched.append(
                            MatchedRule(
                                rule_id=rule.get("id", "unknown"),
                                rule_name=rule.get("title", "Unknown Rule"),
                                severity=rule.get("severity", "medium"),
                                description=rule.get("description", ""),
                                rule_level=rule.get("rule_level", 5),
                            )
                        )
        except Exception as exc:
            logger.warning("RuleEngineRedis evaluation error: %s", exc)

        return matched
