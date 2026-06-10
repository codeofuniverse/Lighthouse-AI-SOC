"""GeoIP enrichment for alerts."""

from __future__ import annotations

import ipaddress
import json
import logging
from typing import Any

import geoip2.database
import redis

logger = logging.getLogger(__name__)


class GeoIPEnricher:
    """Enrich alerts with GeoIP data from MaxMind GeoLite2 database."""

    def __init__(self, db_path: str, redis_client: redis.Redis[str]) -> None:
        """Initialize the GeoIP enricher.

        Args:
            db_path: Path to GeoLite2-City.mmdb database file.
            redis_client: Redis client for caching results.
        """
        self.db_path = db_path
        self.redis_client = redis_client
        self.cache_ttl = 24 * 3600  # 24 hours
        try:
            self.reader = geoip2.database.Reader(db_path)
        except Exception as exc:
            logger.warning("Failed to open GeoIP database: %s", exc)
            self.reader = None

    def enrich(self, ip: str) -> dict[str, Any] | None:
        """Enrich a single IP address with GeoIP data.

        Args:
            ip: IP address to enrich.

        Returns:
            Dictionary with country, city, lat, lon, is_tor, is_vpn or None if private/reserved.
        """
        if not ip:
            return None

        # Check cache first
        cache_key = f"geoip:{ip}"
        if self.redis_client is not None:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as exc:
                logger.warning("Redis cache lookup failed: %s", exc)

        # Check if private/reserved IP
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local:
                logger.debug("Skipping private/reserved IP: %s", ip)
                return None
        except ValueError:
            logger.debug("Invalid IP address: %s", ip)
            return None

        # Query GeoIP database
        if not self.reader:
            logger.debug("GeoIP database not initialized")
            return None

        try:
            response = self.reader.city(ip)
            result = {
                "country": response.country.iso_code or "Unknown",
                "city": response.city.name or "Unknown",
                "lat": response.location.latitude,
                "lon": response.location.longitude,
                "is_tor": False,
                "is_vpn": False,
            }
            # Cache the result
            if self.redis_client is not None:
                try:
                    self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
                except Exception as exc:
                    logger.warning("Failed to cache GeoIP result: %s", exc)
            return result
        except Exception as exc:
            logger.warning("GeoIP lookup failed for %s: %s", ip, exc)
            return None

    def close(self) -> None:
        """Close the GeoIP reader."""
        if self.reader:
            self.reader.close()
