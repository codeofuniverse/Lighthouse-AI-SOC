"""Threat intelligence enrichment using AbuseIPDB API."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis
import requests

logger = logging.getLogger(__name__)


class ThreatIntelEnricher:
    """Enrich alerts with threat intelligence from AbuseIPDB."""

    def __init__(self, api_key: str, redis_client: redis.Redis[str]) -> None:
        """Initialize the threat intelligence enricher.

        Args:
            api_key: AbuseIPDB API key.
            redis_client: Redis client for caching and rate limiting.
        """
        self.api_key = api_key
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour
        self.rate_limit_key = "abuseipdb:daily_calls"
        self.rate_limit_max = 1000  # Free tier limit

    def enrich(self, ip: str) -> dict[str, Any] | None:
        """Enrich a single IP address with threat intelligence.

        Args:
            ip: IP address to check.

        Returns:
            Dictionary with abuse_score, is_known_attacker, last_reported or None if private/reserved or error.
        """
        if not ip or not self.api_key:
            return None

        # Check cache first
        cache_key = f"threat_intel:{ip}"
        if self.redis_client is not None:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as exc:
                logger.warning("Redis cache lookup failed: %s", exc)

        # Check rate limit
        if self.redis_client is not None:
            try:
                current_calls = self.redis_client.incr(self.rate_limit_key)
                if current_calls == 1:
                    self.redis_client.expireat(self.rate_limit_key, int(datetime.now(timezone.utc).timestamp()) + 86400)
                if current_calls > self.rate_limit_max:
                    logger.warning("AbuseIPDB rate limit reached for the day")
                    return None
            except Exception as exc:
                logger.warning("Failed to check rate limit: %s", exc)

        # Query AbuseIPDB API
        try:
            response = requests.get(
                "https://api.abuseipdb.com/api/v2/check",
                headers={
                    "Key": self.api_key,
                    "Accept": "application/json",
                },
                params={"ipAddress": ip, "maxAgeInDays": 90},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            result = {
                "abuse_score": data.get("abuseConfidenceScore", 0),
                "is_known_attacker": data.get("abuseConfidenceScore", 0) > 50,
                "last_reported": data.get("lastReportedAt") or None,
            }
            # Cache the result
            if self.redis_client is not None:
                try:
                    self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
                except Exception as exc:
                    logger.warning("Failed to cache threat intel result: %s", exc)
            return result
        except requests.RequestException as exc:
            logger.warning("AbuseIPDB API call failed for %s: %s", ip, exc)
            return None
