"""Kafka utilities and constants."""

from __future__ import annotations

import json
from typing import Any


KAFKA_BROKERS = "kafka:29092"
RAW_ALERTS_TOPIC = "raw-alerts"
ENRICHED_ALERTS_TOPIC = "enriched-alerts"
DETECTIONS_TOPIC = "detections"
ENRICHMENT_GROUP = "enrichment-workers"


def serialize_alert(alert: dict[str, Any]) -> str:
    """Serialize an alert dict to JSON."""
    return json.dumps(alert, default=str)


def deserialize_alert(message: str) -> dict[str, Any]:
    """Deserialize a JSON alert message."""
    return json.loads(message)
