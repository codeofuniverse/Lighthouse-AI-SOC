"""Kafka consumer for alert enrichment pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from datetime import datetime, timezone

import redis
from confluent_kafka import Consumer, KafkaError

from pipeline.enrichment.geoip import GeoIPEnricher
from pipeline.enrichment.mitre_mapper import MitreMapper
from pipeline.enrichment.sessionizer import Sessionizer
from pipeline.enrichment.threat_intel import ThreatIntelEnricher
from pipeline.kafka_utils import (
    ENRICHED_ALERTS_TOPIC,
    ENRICHMENT_GROUP,
    RAW_ALERTS_TOPIC,
    deserialize_alert,
    serialize_alert,
)

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """Orchestrate alert enrichment and produce enriched alerts to Kafka."""

    def __init__(
        self,
        brokers: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        geoip_db_path: str = "data/GeoLite2-City.mmdb",
        mitre_mapping_path: str = "data/mitre_rule_mapping.yaml",
        abuseipdb_api_key: str = "",
        asset_db_path: str = "data/asset_db.json",
    ) -> None:
        """Initialize the enrichment pipeline.

        Args:
            brokers: Kafka brokers.
            redis_host: Redis host.
            redis_port: Redis port.
            redis_db: Redis database number.
            geoip_db_path: Path to GeoIP database.
            mitre_mapping_path: Path to MITRE mapping YAML.
            abuseipdb_api_key: AbuseIPDB API key.
        """
        self.brokers = brokers
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

        # Initialize enrichers
        self.geoip_enricher = GeoIPEnricher(geoip_db_path, self.redis_client)
        self.threat_intel_enricher = ThreatIntelEnricher(abuseipdb_api_key, self.redis_client)
        self.mitre_mapper = MitreMapper(mitre_mapping_path)
        self.sessionizer = Sessionizer(self.redis_client)
        self.asset_db = self._load_asset_db(asset_db_path)

        # Initialize consumer
        self.consumer = Consumer(
            {
                "bootstrap.servers": brokers,
                "group.id": ENRICHMENT_GROUP,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self.consumer.subscribe([RAW_ALERTS_TOPIC])

        # Initialize producer for enriched alerts
        from confluent_kafka import Producer

        self.producer = Producer(
            {
                "bootstrap.servers": brokers,
                "client.id": "enrichment-producer",
            }
        )

    def _enrich_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Apply all enrichment layers to a single alert.

        Args:
            alert: Normalized alert from Kafka.

        Returns:
            Enriched alert with geoip, threat_intel, mitre_techniques, and session fields.
        """
        # GeoIP enrichment for src_ip and agent_ip
        alert["geoip"] = {}
        if src_ip := alert.get("src_ip"):
            if geoip_data := self.geoip_enricher.enrich(src_ip):
                alert["geoip"]["src"] = geoip_data
        if agent_ip := alert.get("agent_ip"):
            if geoip_data := self.geoip_enricher.enrich(agent_ip):
                alert["geoip"]["agent"] = geoip_data

        # Threat intelligence enrichment
        alert["threat_intel"] = {}
        if src_ip := alert.get("src_ip"):
            if threat_data := self.threat_intel_enricher.enrich(src_ip):
                alert["threat_intel"]["src"] = threat_data

        # MITRE technique mapping
        rule_groups = alert.get("rule_groups", [])
        alert["mitre_techniques"] = self.mitre_mapper.map(rule_groups)

        # Time-of-day feature extraction
        self._add_time_features(alert)

        # Session tracking
        alert = self.sessionizer.enrich(alert)

        # Asset lookup
        asset = self._lookup_asset(alert)
        if asset:
            alert.update(asset)

        # Default asset criticality
        if "asset_criticality" not in alert:
            alert["asset_criticality"] = "medium"

        return alert

    def _produce_enriched_alert(self, alert: dict[str, Any]) -> bool:
        """Produce an enriched alert to Kafka.

        Args:
            alert: Enriched alert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            key = alert.get("src_ip", alert.get("agent_id", "unknown")).encode("utf-8")
            value = serialize_alert(alert).encode("utf-8")
            self.producer.produce(ENRICHED_ALERTS_TOPIC, key=key, value=value)
            self.producer.poll(0)
            return True
        except Exception as exc:
            logger.error("Failed to produce enriched alert: %s", exc)
            return False

    def _load_asset_db(self, asset_db_path: str) -> dict[str, Any]:
        """Load asset metadata from a local JSON DB."""
        try:
            with open(asset_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("Asset DB not found at %s", asset_db_path)
        except Exception as exc:
            logger.warning("Failed to load asset DB: %s", exc)
        return {}

    def _lookup_asset(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Lookup asset metadata by agent_id or agent_ip."""
        if not self.asset_db:
            return {}

        agents = self.asset_db.get("agents", {})
        ips = self.asset_db.get("ips", {})

        agent_id = alert.get("agent_id")
        agent_ip = alert.get("agent_ip")

        if agent_id and agent_id in agents:
            return agents[agent_id]
        if agent_ip and agent_ip in ips:
            return ips[agent_ip]
        return {}

    def _add_time_features(self, alert: dict[str, Any]) -> None:
        """Attach hour-of-day and time-of-day bucket."""
        timestamp = alert.get("timestamp")
        if not isinstance(timestamp, str):
            return

        try:
            if timestamp.endswith("Z"):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(timestamp)
            dt = dt.astimezone(timezone.utc)
            hour = dt.hour
            alert["hour_of_day"] = hour
            if 0 <= hour < 6:
                bucket = "night"
            elif 6 <= hour < 12:
                bucket = "morning"
            elif 12 <= hour < 18:
                bucket = "afternoon"
            else:
                bucket = "evening"
            alert["time_of_day"] = bucket
        except ValueError:
            return

    def run(self) -> None:
        """Run the enrichment pipeline consumer loop."""
        logger.info("Enrichment pipeline started")
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error("Consumer error: %s", msg.error())
                        continue

                try:
                    alert = deserialize_alert(msg.value().decode("utf-8"))
                    enriched_alert = self._enrich_alert(alert)
                    if self._produce_enriched_alert(enriched_alert):
                        self.consumer.commit(asynchronous=False)
                        logger.info("Enriched and committed alert id=%s", alert.get("id"))
                    else:
                        logger.warning("Failed to produce enriched alert id=%s", alert.get("id"))
                except Exception as exc:
                    logger.error("Failed to process message: %s", exc)
        except KeyboardInterrupt:
            logger.info("Enrichment pipeline shutting down")
        finally:
            self.consumer.close()
            self.producer.flush()
            self.producer.close()
            self.geoip_enricher.close()
            self.redis_client.close()

    def close(self) -> None:
        """Close all resources."""
        self.consumer.close()
        self.producer.close()
        self.geoip_enricher.close()
        self.redis_client.close()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    pipeline = EnrichmentPipeline(
        brokers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        geoip_db_path=os.getenv("GEOIP_DATABASE_PATH", "data/GeoLite2-City.mmdb"),
        mitre_mapping_path="data/mitre_rule_mapping.yaml",
        abuseipdb_api_key=os.getenv("ABUSEIPDB_API_KEY", ""),
        asset_db_path=os.getenv("ASSET_DB_PATH", "data/asset_db.json"),
    )
    pipeline.run()
