"""Converts DetectionEvent + Decision into the API alert dict shape."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from detection.suricata_bridge import DetectionEvent
from pipeline.decision_engine import Decision


def build_alert(
    event: DetectionEvent,
    decision: Decision,
    ai_explanation: str,
    abuse_score: int = 0,
) -> dict[str, Any]:
    # Include flow_duration_us + 4 random bytes to prevent ID collision when two
    # flows from the same IP complete within the same millisecond timestamp.
    entropy = os.urandom(4).hex()
    alert_id = hashlib.sha1(
        f"{event.src_ip}:{event.timestamp}:{event.flow_duration_us}:{entropy}".encode()
    ).hexdigest()[:16]

    attack_type = event.prediction

    return {
        "id":               alert_id,
        "timestamp":        event.timestamp,
        "attack_type":      attack_type,
        "src_ip":           event.src_ip,
        "dst_ip":           event.dst_ip,
        "dst_port":         event.dst_port,
        "proto":            event.proto,
        "agent_name":       event.src_ip,
        "agent_id":         "",  # not available from Suricata flows; set by Wazuh bridge
        "rule_level":       (
            12 if decision.risk_score >= 81 else
            9  if decision.risk_score >= 61 else
            6  if decision.risk_score >= 31 else
            3
        ),
        "rule_description": event.suricata_alert or f"Suricata flow analysis — {attack_type} pattern",
        "status":           "active",
        "auto_blocked":     decision.auto_blocked,
        "confidence":       round(event.stage1_attack_prob, 4),
        "threat_level":     decision.threat_level,
        "risk_score":       decision.risk_score,
        "ai_explanation":   ai_explanation,
        "cic_confidence":   round(event.stage1_attack_prob, 4),
        "unsw_confidence":  round(event.unsw_attack_prob, 4) if event.unsw_attack_prob else None,
        "abuse_score":      abuse_score,
        "action_history":   [],
        "ingested_at":      event.timestamp,
    }
