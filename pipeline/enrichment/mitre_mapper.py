"""MITRE ATT&CK technique mapping from Wazuh rule groups."""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class MitreMapper:
    """Map Wazuh rule groups to MITRE ATT&CK techniques."""

    def __init__(self, yaml_path: str) -> None:
        """Initialize the MITRE mapper.

        Args:
            yaml_path: Path to MITRE rule mapping YAML file.
        """
        self.mapping: dict[str, list[dict[str, str]]] = {}
        self._load_mapping(yaml_path)

    def _load_mapping(self, yaml_path: str) -> None:
        """Load the MITRE mapping from a YAML file."""
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
                self.mapping = data
                logger.info("Loaded MITRE mapping from %s", yaml_path)
        except Exception as exc:
            logger.warning("Failed to load MITRE mapping: %s", exc)

    def map(self, rule_groups: list[str]) -> list[dict[str, str]]:
        """Map rule groups to MITRE techniques.

        Args:
            rule_groups: List of Wazuh rule group names.

        Returns:
            List of MITRE technique dictionaries with technique_id, technique_name, tactic.
        """
        techniques: dict[str, dict[str, str]] = {}
        for group in rule_groups:
            if group in self.mapping:
                for technique in self.mapping[group]:
                    key = technique.get("technique_id", "")
                    if key and key not in techniques:
                        techniques[key] = technique
        return list(techniques.values())
