"""Tests for Phase 3 detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from detection.aggregator import DetectionAggregator, DetectionResult
from detection.anomaly_detector import AnomalyDetector
from detection.ml_classifier import MLClassifier
from detection.rule_engine import RuleEngine


@pytest.fixture
def sample_alerts() -> list[dict]:
    """Generate sample enriched alerts for testing."""
    now = datetime.utcnow().isoformat() + "Z"
    alerts = [
        {
            "id": "alert-001",
            "timestamp": now,
            "rule_level": 3,
            "rule_description": "Normal SSH activity",
            "rule_groups": ["sshd"],
            "agent_id": "agent-001",
            "agent_name": "web-server-01",
            "agent_ip": "10.0.0.1",
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "protocol": "ssh",
            "geoip": {
                "src": {"country": "US", "city": "New York", "is_tor": False, "is_vpn": False},
            },
            "threat_intel": {
                "src": {"abuse_score": 5, "is_known_attacker": False, "last_reported": None},
            },
            "mitre_techniques": [],
            "session_id": "sess-001",
            "session_event_count": 1,
            "session_duration_seconds": 30,
            "asset_criticality": "medium",
        },
        {
            "id": "alert-002",
            "timestamp": now,
            "rule_level": 8,
            "rule_description": "SSH Brute Force",
            "rule_groups": ["sshd", "authentication"],
            "agent_id": "agent-001",
            "agent_name": "web-server-01",
            "agent_ip": "10.0.0.1",
            "src_ip": "203.0.113.50",
            "dst_ip": "10.0.0.1",
            "src_port": 54321,
            "protocol": "ssh",
            "geoip": {
                "src": {"country": "CN", "city": "Unknown", "is_tor": True, "is_vpn": False},
            },
            "threat_intel": {
                "src": {"abuse_score": 85, "is_known_attacker": True, "last_reported": now},
            },
            "mitre_techniques": [
                {"technique_id": "T1110", "technique_name": "Brute Force", "tactic": "Credential Access"},
            ],
            "session_id": "sess-002",
            "session_event_count": 25,
            "session_duration_seconds": 45,
            "asset_criticality": "high",
        },
    ]
    return alerts


def test_rule_engine_loads_rules() -> None:
    """Test that rule engine loads rules from directory."""
    engine = RuleEngine("detection/sigma_rules")
    assert len(engine.rules) >= 5, f"Expected at least 5 rules, got {len(engine.rules)}"


def test_rule_engine_evaluates_conditions() -> None:
    """Test rule condition evaluation."""
    engine = RuleEngine("detection/sigma_rules")
    alert = {
        "rule_level": 8,
        "session_event_count": 25,
        "session_duration_seconds": 45,
        "threat_intel": {"is_known_attacker": True},
    }
    matches = engine.evaluate(alert)
    # Should match SSH brute force rule
    assert len(matches) > 0, "Expected SSH brute force rule to match"
    assert any("brute" in rule.rule_name.lower() for rule in matches)


def test_rule_engine_no_false_positives() -> None:
    """Test that benign alerts don't trigger rules."""
    engine = RuleEngine("detection/sigma_rules")
    benign_alert = {
        "rule_level": 2,
        "rule_groups": ["sshd"],
        "session_event_count": 1,
        "session_duration_seconds": 5,
        "threat_intel": {"is_known_attacker": False},
    }
    matches = engine.evaluate(benign_alert)
    # Should have few or no matches for benign alert
    assert len(matches) == 0, f"Expected no rule matches for benign alert, got {len(matches)}"


def test_ml_classifier_feature_engineering() -> None:
    """Test feature engineering for ML classifier."""
    classifier = MLClassifier()
    alert_df = pd.DataFrame([
        {
            "rule_level": 5,
            "src_port": 22,
            "session_event_count": 10,
            "session_duration_seconds": 60,
            "geoip": {"src": {"is_tor": True, "is_vpn": False}},
            "threat_intel": {"src": {"abuse_score": 75, "is_known_attacker": True}},
            "mitre_techniques": [{"technique_id": "T1110"}],
            "rule_groups": ["sshd"],
            "asset_criticality": "high",
            "protocol": "ssh",
        }
    ])
    features_df = classifier._engineer_features(alert_df)
    assert features_df.shape[0] == 1
    assert features_df.shape[1] >= 8, f"Expected at least 8 features, got {features_df.shape[1]}"


def test_ml_classifier_trains_models(sample_alerts: list[dict]) -> None:
    """Test ML classifier training."""
    classifier = MLClassifier()
    alert_df = pd.DataFrame(sample_alerts)
    
    # Create synthetic labels
    y = np.array([0, 2])  # benign, malicious
    
    results = classifier.train(alert_df, y)
    assert results["xgb_f1"] >= 0.0
    assert results["lgb_f1"] >= 0.0
    assert results["best_model"] in ["xgb", "lgb"]


def test_ml_classifier_predicts() -> None:
    """Test ML classifier prediction."""
    classifier = MLClassifier()
    # Use 5 samples (< 6) so the CV path is skipped — this test targets predict(), not CV
    alert_df = pd.DataFrame([
        {
            "rule_level": i * 2,
            "src_port": 22 + i * 100,
            "session_event_count": i * 3,
            "session_duration_seconds": i * 20,
            "geoip": {"src": {"is_tor": i > 2, "is_vpn": i > 3}},
            "threat_intel": {"src": {"abuse_score": i * 20, "is_known_attacker": i > 3}},
            "mitre_techniques": [{"technique_id": "T1110"}] if i > 2 else [],
            "rule_groups": ["sshd"],
            "asset_criticality": "high" if i > 2 else "low",
            "protocol": "tcp",
        }
        for i in range(5)
    ])
    y = np.array([0, 0, 1, 1, 2])

    classifier.train(alert_df, y)
    
    test_alert = {
        "rule_level": 8,
        "src_port": 22,
        "session_event_count": 15,
        "session_duration_seconds": 60,
        "geoip": {"src": {"is_tor": True, "is_vpn": False}},
        "threat_intel": {"src": {"abuse_score": 80, "is_known_attacker": True}},
        "mitre_techniques": [{"technique_id": "T1110"}],
        "rule_groups": ["sshd"],
        "asset_criticality": "high",
        "protocol": "ssh",
    }
    
    result = classifier.predict(test_alert)
    assert result["attack_type"] in ["benign", "suspicious", "malicious"]
    assert 0 <= result["confidence"] <= 1


def test_anomaly_detector_trains() -> None:
    """Test anomaly detector baseline training."""
    detector = AnomalyDetector()
    normal_alerts = [
        {
            "rule_level": i % 7,
            "src_port": 22 + i,
            "session_event_count": i,
            "session_duration_seconds": i * 10,
            "geoip": {"src": {"is_tor": False, "is_vpn": False}},
            "threat_intel": {"src": {"abuse_score": i * 5, "is_known_attacker": False}},
            "mitre_techniques": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        for i in range(50)
    ]
    
    results = detector.train_baseline(normal_alerts)
    assert results["status"] == "trained"
    assert results["samples"] > 0


def test_anomaly_detector_scores() -> None:
    """Test anomaly detector scoring."""
    detector = AnomalyDetector()
    
    # Train on normal data
    normal_alerts = [
        {
            "rule_level": 2,
            "src_port": 22,
            "session_event_count": 1,
            "session_duration_seconds": 10,
            "geoip": {"src": {"is_tor": False, "is_vpn": False}},
            "threat_intel": {"src": {"abuse_score": 5, "is_known_attacker": False}},
            "mitre_techniques": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        for _ in range(50)
    ]
    detector.train_baseline(normal_alerts)
    
    # Test normal alert
    normal_test = {
        "rule_level": 2,
        "src_port": 22,
        "session_event_count": 1,
        "session_duration_seconds": 10,
        "geoip": {"src": {"is_tor": False, "is_vpn": False}},
        "threat_intel": {"src": {"abuse_score": 5, "is_known_attacker": False}},
        "mitre_techniques": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    result = detector.score(normal_test)
    assert 0 <= result["anomaly_score"] <= 1
    assert isinstance(result["is_anomaly"], bool)
    assert isinstance(result["anomaly_features"], list)


def test_detection_aggregator_combines_results(sample_alerts: list[dict]) -> None:
    """Test detection aggregator combining results."""
    aggregator = DetectionAggregator()
    
    alert = sample_alerts[1]  # The malicious alert
    
    # Mock rule matches
    from detection.rule_engine import MatchedRule
    rule_matches = [
        MatchedRule(
            rule_id="ssh-brute-force-001",
            rule_name="SSH Brute Force",
            severity="high",
            description="Detected SSH brute force",
            rule_level=8,
        ),
    ]
    
    ml_result = {
        "attack_type": "malicious",
        "confidence": 0.95,
        "class_probabilities": {"benign": 0.01, "suspicious": 0.04, "malicious": 0.95},
    }
    
    anomaly_result = {
        "anomaly_score": 0.85,
        "is_anomaly": True,
        "anomaly_features": ["threat_intel_abuse_score", "session_event_count"],
    }
    
    detection = aggregator.aggregate(alert, rule_matches, ml_result, anomaly_result)
    
    assert detection.alert_id == "alert-002"
    assert detection.attack_type == "malicious"
    assert detection.confidence_score > 0.5
    assert "ssh-brute-force-001" in detection.matched_rules
    assert "rule_engine" in detection.detection_sources
    assert "ml_classifier" in detection.detection_sources
    assert detection.threat_intel_severity == "high"


def test_detection_aggregator_threat_severity() -> None:
    """Test threat severity mapping."""
    aggregator = DetectionAggregator()
    
    assert aggregator._threat_severity_from_abuse_score(5) == "none"
    assert aggregator._threat_severity_from_abuse_score(25) == "low"
    assert aggregator._threat_severity_from_abuse_score(50) == "medium"
    assert aggregator._threat_severity_from_abuse_score(75) == "high"


@pytest.mark.benchmark
def test_detection_performance_under_100ms(sample_alerts: list[dict]) -> None:
    """Test that full detection pipeline completes under 100ms."""
    import time
    
    engine = RuleEngine("detection/sigma_rules")
    aggregator = DetectionAggregator()
    
    alert = sample_alerts[1]
    
    start = time.time()
    
    # Run detection pipeline
    rule_matches = engine.evaluate(alert)
    detection = aggregator.aggregate(alert, rule_matches)
    
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms < 100, f"Detection took {elapsed_ms:.1f}ms, exceeds 100ms target"
