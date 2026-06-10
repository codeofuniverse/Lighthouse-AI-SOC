"""Sigma rule engine for alert detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MatchedRule:
    """Result of a rule match."""

    rule_id: str
    rule_name: str
    severity: str
    description: str
    rule_level: int


class RuleEngine:
    """Load and evaluate Sigma-style detection rules against enriched alerts."""

    _compiled_patterns: dict[str, re.Pattern] = {}

    def __init__(self, rules_dir: str = "detection/sigma_rules") -> None:
        self.rules_dir = Path(rules_dir)
        self.rules: list[dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load all YAML rules from the rules directory."""
        if not self.rules_dir.exists():
            logger.warning("Rules directory not found: %s", self.rules_dir)
            return

        for rule_file in self.rules_dir.glob("*.yml"):
            try:
                with open(rule_file) as f:
                    rule = yaml.safe_load(f)
                    if rule:
                        self.rules.append(rule)
                        self._precompile_rule_patterns(rule)
                        logger.info("Loaded rule: %s", rule.get("title"))
            except Exception as exc:
                logger.error("Failed to load rule %s: %s", rule_file, exc)

    def _precompile_rule_patterns(self, rule: dict[str, Any]) -> None:
        """Walk rule detection dict and pre-compile any regex condition strings."""
        detection = rule.get("detection", {})
        for value in detection.values():
            if isinstance(value, str) and (value.startswith(".*") or value.endswith(".*")):
                if value not in RuleEngine._compiled_patterns:
                    try:
                        RuleEngine._compiled_patterns[value] = re.compile(
                            value.replace("\\.", "."), re.IGNORECASE
                        )
                    except re.error:
                        pass

    def _normalize_alert_view(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Normalize alert structure for rule evaluation.

        Supports both flat and nested threat_intel structures.
        """
        alert_view = dict(alert)
        threat_intel = alert.get("threat_intel", {})
        if isinstance(threat_intel, dict):
            if "abuse_score" in threat_intel or "is_known_attacker" in threat_intel:
                alert_view["threat_intel"] = threat_intel
            elif "src" in threat_intel and isinstance(threat_intel["src"], dict):
                alert_view["threat_intel"] = threat_intel["src"]
        return alert_view

    def _evaluate_condition(self, condition: Any, alert: dict[str, Any]) -> bool:
        """Evaluate a single condition against an alert.

        Args:
            condition: Condition value (can be dict, list, str).
            alert: Enriched alert dict.

        Returns:
            True if condition matches.
        """
        if isinstance(condition, dict):
            # Multiple conditions, all must match
            for key, value in condition.items():
                if key == "rule_group":
                    # Sigma rules may use rule_group while alerts store rule_groups
                    alert_value = alert.get("rule_groups")
                    if alert_value is None:
                        # If rule_groups are missing, do not block the rule match
                        continue
                    if not self._compare_condition(alert_value, value):
                        return False
                    continue

                alert_value = alert
                # Support nested paths like "threat_intel.abuse_score"
                for part in key.split("."):
                    if isinstance(alert_value, dict):
                        if part not in alert_value and part in {
                            "abuse_score",
                            "is_known_attacker",
                            "last_reported",
                        }:
                            alert_value = alert_value.get("src", {}).get(part)
                        else:
                            alert_value = alert_value.get(part)
                    else:
                        return False

                if not self._compare_condition(alert_value, value):
                    return False
            return True
        elif isinstance(condition, list):
            # All conditions in the list must match (AND semantics)
            return all(self._evaluate_condition(c, alert) for c in condition)
        return True

    def _compare_condition(self, alert_value: Any, condition_value: Any) -> bool:
        """Compare an alert value with a condition value.

        Args:
            alert_value: Value from the alert.
            condition_value: Condition value (can include operators like >10, regex, etc).

        Returns:
            True if the comparison matches.
        """
        if condition_value is None:
            return alert_value is None

        # Handle list-valued alert fields (e.g., rule_groups)
        if isinstance(alert_value, list):
            if isinstance(condition_value, str):
                if "|" in condition_value:
                    values = condition_value.split("|")
                    return any(v in alert_value for v in values)
                return condition_value in alert_value
            if isinstance(condition_value, list):
                return any(v in alert_value for v in condition_value)

        # Handle string conditions with operators
        if isinstance(condition_value, str):
            value_lower = condition_value.strip().lower()
            if value_lower in {"true", "false"}:
                return bool(alert_value) == (value_lower == "true")

            # Comparison operators
            for op in (">=", "<=", ">", "<"):
                if condition_value.startswith(op):
                    try:
                        threshold = float(condition_value[len(op):].strip())
                        if not isinstance(alert_value, (int, float)):
                            return False
                        if op == ">=":
                            return alert_value >= threshold
                        if op == "<=":
                            return alert_value <= threshold
                        if op == ">":
                            return alert_value > threshold
                        if op == "<":
                            return alert_value < threshold
                    except (ValueError, TypeError):
                        return False

            # Regex pattern — use pre-compiled version when available
            if condition_value.startswith(".*") or condition_value.endswith(".*"):
                compiled = RuleEngine._compiled_patterns.get(condition_value)
                if compiled is None:
                    try:
                        compiled = re.compile(condition_value.replace("\\.", "."), re.IGNORECASE)
                    except re.error:
                        pass
                return compiled is not None and isinstance(alert_value, str) and compiled.search(alert_value) is not None

            # Exact match or in list
            if "|" in condition_value:
                values = condition_value.split("|")
                return alert_value in values

            # Simple equality
            return str(alert_value) == condition_value

        # Numeric comparison
        if isinstance(condition_value, (int, float)):
            try:
                return alert_value == condition_value
            except TypeError:
                return False

        # Boolean comparison
        if isinstance(condition_value, bool):
            return bool(alert_value) == condition_value

        return False

    def evaluate(self, alert: dict[str, Any]) -> list[MatchedRule]:
        """Evaluate all rules against an alert.

        Args:
            alert: Enriched alert dictionary.

        Returns:
            List of matched rules.
        """
        matched = []
        alert_view = self._normalize_alert_view(alert)

        for rule in self.rules:
            try:
                # Get detection conditions
                detection = rule.get("detection", {})
                if not detection:
                    continue

                # Keys that should not be treated as conditions
                reserved_keys = {"condition", "description", "mitre_technique", "mitre_tactic"}
                detection_keys = {k: v for k, v in detection.items() if k not in reserved_keys}

                # Extract condition or evaluate all keys as conditions
                condition = detection.get("condition")
                if condition is None:
                    # No explicit condition, evaluate all detection keys
                    if self._evaluate_condition(detection_keys, alert_view):
                        matched.append(
                            MatchedRule(
                                rule_id=rule.get("id", "unknown"),
                                rule_name=rule.get("title", "Unknown Rule"),
                                severity=rule.get("severity", "medium"),
                                description=rule.get("description", ""),
                                rule_level=rule.get("rule_level", 5),
                            )
                        )
                else:
                    # Explicit condition plus any extra detection keys
                    if detection_keys:
                        if isinstance(condition, list):
                            combined = [detection_keys] + condition
                        else:
                            combined = [detection_keys, condition]
                        condition_to_eval = combined
                    else:
                        condition_to_eval = condition

                    if self._evaluate_condition(condition_to_eval, alert_view):
                        matched.append(
                            MatchedRule(
                                rule_id=rule.get("id", "unknown"),
                                rule_name=rule.get("title", "Unknown Rule"),
                                severity=rule.get("severity", "medium"),
                                description=rule.get("description", ""),
                                rule_level=rule.get("rule_level", 5),
                            )
                        )
            except Exception as exc:
                logger.warning("Error evaluating rule %s: %s", rule.get("id"), exc)

        return matched
