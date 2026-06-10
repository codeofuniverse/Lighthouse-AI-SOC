"""Tests for enrichment modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import redis

from pipeline.enrichment.geoip import GeoIPEnricher
from pipeline.enrichment.mitre_mapper import MitreMapper
from pipeline.enrichment.sessionizer import Sessionizer
from pipeline.enrichment.threat_intel import ThreatIntelEnricher


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create a mock Redis client."""
    return MagicMock(spec=redis.Redis)


def test_geoip_enricher_skips_private_ip(mock_redis: MagicMock) -> None:
    """Test that private IPs are skipped gracefully."""
    enricher = GeoIPEnricher("data/GeoLite2-City.mmdb", mock_redis)
    result = enricher.enrich("192.168.1.1")
    assert result is None


def test_geoip_enricher_skips_invalid_ip(mock_redis: MagicMock) -> None:
    """Test that invalid IPs are skipped gracefully."""
    enricher = GeoIPEnricher("data/GeoLite2-City.mmdb", mock_redis)
    result = enricher.enrich("invalid")
    assert result is None


def test_threat_intel_enricher_returns_none_without_key(mock_redis: MagicMock) -> None:
    """Test that enricher returns None without API key."""
    enricher = ThreatIntelEnricher("", mock_redis)
    result = enricher.enrich("1.2.3.4")
    assert result is None


def test_mitre_mapper_loads_mapping() -> None:
    """Test that MITRE mapper loads the YAML mapping correctly."""
    mapper = MitreMapper("data/mitre_rule_mapping.yaml")
    techniques = mapper.map(["sshd"])
    assert len(techniques) > 0
    assert techniques[0]["technique_id"] == "T1110"


def test_mitre_mapper_returns_empty_for_unknown_group() -> None:
    """Test that mapper returns empty list for unknown rule groups."""
    mapper = MitreMapper("data/mitre_rule_mapping.yaml")
    techniques = mapper.map(["unknown_group"])
    assert len(techniques) == 0


def test_mitre_mapper_deduplicates_techniques() -> None:
    """Test that mapper deduplicates techniques across rule groups."""
    mapper = MitreMapper("data/mitre_rule_mapping.yaml")
    techniques = mapper.map(["authentication", "syslog"])
    # Both have T1078, should only appear once
    technique_ids = [t["technique_id"] for t in techniques]
    assert technique_ids.count("T1078") == 1


def test_sessionizer_creates_new_session_without_src_ip(mock_redis: MagicMock) -> None:
    """Test that sessionizer creates a transient session without src_ip."""
    mock_redis.zrangebyscore.return_value = []
    sessionizer = Sessionizer(mock_redis)
    alert = {"id": "123", "timestamp": "2026-05-14T10:00:00Z"}
    enriched = sessionizer.enrich(alert)
    assert enriched["session_id"]
    assert enriched["session_event_count"] == 1
    assert enriched["session_duration_seconds"] == 0


def test_sessionizer_tracks_event_count(mock_redis: MagicMock) -> None:
    """Test that sessionizer increments event count for active sessions."""
    import json
    from datetime import datetime

    now = datetime.utcnow().timestamp()
    session_data = {
        "session_id": "sess-1",
        "first_seen": now - 100,
        "last_seen": now,
        "event_count": 3,
        "unique_dst_ips": ["10.0.0.1"],
        "unique_ports": ["443"],
    }
    mock_redis.zrangebyscore.return_value = [json.dumps(session_data).encode()]
    mock_redis.zadd.return_value = 1
    mock_redis.expire.return_value = True

    sessionizer = Sessionizer(mock_redis)
    alert = {"id": "124", "src_ip": "192.168.1.1", "dst_ip": "10.0.0.2", "src_port": 443}
    enriched = sessionizer.enrich(alert)
    assert enriched["session_id"] == "sess-1"
    assert enriched["session_event_count"] == 4
