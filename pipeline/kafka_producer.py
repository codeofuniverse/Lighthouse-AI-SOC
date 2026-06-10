"""Kafka producer for normalized Wazuh alerts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from confluent_kafka import KafkaError, Producer

from pipeline.kafka_utils import RAW_ALERTS_TOPIC, serialize_alert

logger = logging.getLogger(__name__)


class KafkaAlertProducer:
    """Produce normalized alerts to Kafka raw-alerts topic."""

    def __init__(self, brokers: str) -> None:
        """Initialize the Kafka producer.

        Args:
            brokers: Comma-separated list of Kafka brokers (e.g., 'kafka:29092').
        """
        self.brokers = brokers
        self.producer = Producer(
            {
                "bootstrap.servers": brokers,
                "client.id": "wazuh-bridge-producer",
                "acks": "all",
            }
        )

    def _delivery_report(self, err: KafkaError | None, msg: Any) -> None:
        """Handle message delivery results."""
        if err is not None:
            logger.error("Message delivery failed: %s", err)
        else:
            logger.debug("Message delivered to %s [%d]", msg.topic(), msg.partition())

    def produce_alert(self, alert: dict[str, Any]) -> None:
        """Produce a single alert to Kafka.

        Args:
            alert: Normalized alert dictionary.
        """
        try:
            # Use src_ip as key for partition locality
            key = alert.get("src_ip", "")
            if not key:
                key = alert.get("agent_id", "")

            key_bytes = key.encode("utf-8") if key else alert.get("id", "unknown").encode("utf-8")

            value = serialize_alert(alert).encode("utf-8")
            self.producer.produce(
                RAW_ALERTS_TOPIC,
                key=key_bytes,
                value=value,
                callback=self._delivery_report,
            )
            self.producer.poll(0)
        except Exception as exc:
            logger.error("Failed to produce alert: %s", exc)

    def flush(self) -> None:
        """Flush pending messages."""
        self.producer.flush()

    def close(self) -> None:
        """Close the producer."""
        self.flush()
        self.producer.close()
