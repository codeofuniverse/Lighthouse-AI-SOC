"""Integrated detection pipeline orchestrator for Phase 3."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import pandas as pd
from confluent_kafka import Consumer, KafkaError

from detection.aggregator import DetectionAggregator
from detection.anomaly_detector import AnomalyDetector
from detection.ml_classifier import MLClassifier
from detection.rule_engine import RuleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Orchestrate detection on enriched alerts from Kafka."""

    def __init__(
        self,
        kafka_brokers: str = "localhost:9092",
        rule_engine_dir: str = "detection/sigma_rules",
        model_dir: str = "detection/models",
        consumer_group_id: str = "detection-engine",
    ) -> None:
        """Initialize detection pipeline.

        Args:
            kafka_brokers: Kafka broker addresses.
            rule_engine_dir: Directory with Sigma rules.
            model_dir: Directory with trained models.
        """
        self.kafka_brokers = kafka_brokers
        self.rule_engine = RuleEngine(rule_engine_dir)
        self.ml_classifier = MLClassifier(model_dir)
        self.anomaly_detector = AnomalyDetector()
        self.aggregator = DetectionAggregator(kafka_brokers)
        self.consumer_group_id = consumer_group_id

        # Try to load pre-trained models
        if not self.ml_classifier.load():
            logger.warning("No pre-trained ML model found, inference will use defaults")

        # Initialize Kafka consumer for enriched alerts
        self.consumer = Consumer(
            {
                "bootstrap.servers": kafka_brokers,
                "group.id": consumer_group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self.consumer.subscribe(["enriched-alerts"])

    def run(self) -> None:
        """Run the detection pipeline consumer loop."""
        logger.info("Detection pipeline started")
        detection_count = 0
        total_inference_ms = 0

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
                    # Deserialize enriched alert
                    alert_json = msg.value().decode("utf-8")
                    alert = json.loads(alert_json)

                    # Run detection pipeline with timing
                    inference_start = time.time()

                    # 1. Sigma rule evaluation
                    rule_matches = self.rule_engine.evaluate(alert)

                    # 2. ML classification
                    ml_result = self.ml_classifier.predict(alert)

                    # 3. Anomaly detection
                    anomaly_result = self.anomaly_detector.score(alert)

                    # 4. Aggregate results
                    detection_result = self.aggregator.aggregate(
                        alert,
                        rule_matches=rule_matches,
                        ml_result=ml_result,
                        anomaly_result=anomaly_result,
                    )

                    inference_ms = (time.time() - inference_start) * 1000
                    total_inference_ms += inference_ms

                    # Produce detection
                    if self.aggregator.produce_detection(detection_result):
                        self.consumer.commit(asynchronous=False)
                        detection_count += 1

                        # Log summary
                        if detection_count % 10 == 0:
                            avg_inference_ms = total_inference_ms / detection_count
                            logger.info(
                                "Processed %d detections, avg inference: %.2f ms, "
                                "rule_matches: %d, attack_type: %s, confidence: %.2f%%",
                                detection_count,
                                avg_inference_ms,
                                len(rule_matches),
                                detection_result.attack_type,
                                detection_result.confidence_score * 100,
                            )
                    else:
                        logger.warning("Failed to produce detection for alert id=%s", alert.get("id"))

                except Exception as exc:
                    logger.error("Failed to process alert: %s", exc)

        except KeyboardInterrupt:
            logger.info("Detection pipeline shutting down")
        finally:
            logger.info(
                "Detection pipeline stopped. Processed %d detections in total.",
                detection_count,
            )
            self.consumer.close()
            self.aggregator.close()


def main() -> None:
    """Main entry point for detection pipeline."""
    kafka_brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")
    rule_engine_dir = os.getenv("SIGMA_RULES_DIR", "detection/sigma_rules")
    model_dir = os.getenv("MODEL_DIR", "detection/models")
    consumer_group_id = os.getenv("DETECTION_GROUP_ID", "detection-engine")

    pipeline = DetectionPipeline(
        kafka_brokers=kafka_brokers,
        rule_engine_dir=rule_engine_dir,
        model_dir=model_dir,
        consumer_group_id=consumer_group_id,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
