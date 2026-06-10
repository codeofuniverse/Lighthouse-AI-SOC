"""Detection result aggregation and Kafka production."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from confluent_kafka import Producer, KafkaError

logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    """Detection result combining all detection sources."""

    alert_id: str
    timestamp: str
    attack_type: str = Field(description="Attack classification: benign, suspicious, malicious")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence 0-1")
    anomaly_score: float = Field(ge=0, le=1, description="Anomaly score 0-1")
    matched_rules: list[str] = Field(default_factory=list, description="List of rule IDs that matched")
    mitre_techniques: list[dict[str, str]] = Field(
        default_factory=list,
        description="MITRE ATT&CK techniques",
    )
    threat_intel_severity: str = Field(
        default="none",
        description="Severity: none, low, medium, high",
    )
    detection_sources: list[str] = Field(
        default_factory=list,
        description="Which detection engines fired: rule_engine, ml_classifier, anomaly_detector",
    )
    source_ip: str = Field(default="", description="Source IP from alert")
    dest_ip: str = Field(default="", description="Destination IP from alert")
    session_id: str = Field(default="", description="Session ID for grouping")


class DetectionAggregator:
    """Aggregate detections from multiple sources and produce to Kafka."""

    def __init__(self, kafka_brokers: str = "localhost:9092") -> None:
        """Initialize the aggregator.

        Args:
            kafka_brokers: Kafka broker addresses.
        """
        self.kafka_brokers = kafka_brokers
        self.producer = Producer(
            {
                "bootstrap.servers": kafka_brokers,
                "client.id": "detection-aggregator",
            }
        )

    def _threat_severity_from_abuse_score(self, abuse_score: int) -> str:
        """Map abuse score to severity level.

        Args:
            abuse_score: Score from 0-100.

        Returns:
            Severity level: none, low, medium, high.
        """
        if abuse_score >= 75:
            return "high"
        elif abuse_score >= 50:
            return "medium"
        elif abuse_score >= 25:
            return "low"
        return "none"

    def aggregate(
        self,
        alert: dict[str, Any],
        rule_matches: list[Any] | None = None,
        ml_result: dict[str, Any] | None = None,
        anomaly_result: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Aggregate detection results from all sources.

        Args:
            alert: Original enriched alert.
            rule_matches: List of matched rule objects.
            ml_result: ML classifier result dict.
            anomaly_result: Anomaly detector result dict.

        Returns:
            Aggregated DetectionResult.
        """
        start_time = time.time()

        # Initialize detection sources tracker
        detection_sources = []
        confidence_scores = []

        # Extract rule matches
        matched_rules = []
        if rule_matches:
            detection_sources.append("rule_engine")
            matched_rules = [rule.rule_id for rule in rule_matches]
            # Rule confidence: use max rule_level as proxy
            if rule_matches:
                max_level = max(rule.rule_level for rule in rule_matches)
                confidence_scores.append(min(1.0, max_level / 10.0))

        # Extract ML result
        attack_type = "unknown"
        if ml_result:
            detection_sources.append("ml_classifier")
            attack_type = ml_result.get("attack_type", "unknown")
            ml_confidence = ml_result.get("confidence", 0.0)
            confidence_scores.append(ml_confidence)

        # Extract anomaly result
        if anomaly_result:
            detection_sources.append("anomaly_detector")
            anomaly_score = anomaly_result.get("anomaly_score", 0.0)
            if anomaly_result.get("is_anomaly"):
                confidence_scores.append(anomaly_score)

        # Compute overall confidence as weighted average
        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.0

        # Threat severity from threat intelligence
        threat_intel = alert.get("threat_intel", {})
        if isinstance(threat_intel, dict):
            if "abuse_score" in threat_intel:
                abuse_score = threat_intel.get("abuse_score", 0)
            else:
                abuse_score = threat_intel.get("src", {}).get("abuse_score", 0)
        else:
            abuse_score = 0
        threat_severity = self._threat_severity_from_abuse_score(abuse_score)

        # Anomaly score
        anomaly_score = anomaly_result.get("anomaly_score", 0.0) if anomaly_result else 0.0

        # Build detection result
        result = DetectionResult(
            alert_id=alert.get("id", "unknown"),
            timestamp=alert.get("timestamp", ""),
            attack_type=attack_type,
            confidence_score=overall_confidence,
            anomaly_score=anomaly_score,
            matched_rules=matched_rules,
            mitre_techniques=alert.get("mitre_techniques", []),
            threat_intel_severity=threat_severity,
            detection_sources=detection_sources,
            source_ip=alert.get("src_ip", ""),
            dest_ip=alert.get("dst_ip", ""),
            session_id=alert.get("session_id", ""),
        )

        elapsed = (time.time() - start_time) * 1000  # ms
        if elapsed > 100:
            logger.warning("Detection aggregation took %.1f ms (exceeded 100 ms target)", elapsed)

        return result

    def produce_detection(self, detection_result: DetectionResult) -> bool:
        """Produce detection result to Kafka.

        Args:
            detection_result: Detection result to produce.

        Returns:
            True if successful.
        """
        try:
            key = detection_result.source_ip.encode("utf-8") if detection_result.source_ip else b"unknown"
            value = detection_result.model_dump_json().encode("utf-8")
            self.producer.produce(
                "detections",
                key=key,
                value=value,
                callback=self._delivery_report,
            )
            self.producer.poll(0)
            return True
        except Exception as exc:
            logger.error("Failed to produce detection: %s", exc)
            return False

    def _delivery_report(self, err: KafkaError | None, msg: Any) -> None:
        """Handle delivery reports."""
        if err is not None:
            logger.error("Detection delivery failed: %s", err)
        else:
            logger.debug("Detection delivered to %s [%d]", msg.topic(), msg.partition())

    def flush(self) -> None:
        """Flush pending messages."""
        self.producer.flush()

    def close(self) -> None:
        """Close the producer."""
        self.flush()
        self.producer.close()
