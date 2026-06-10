"""Wazuh host-alert ingestion for the live FastAPI pipeline.

The existing wazuh_bridge.py is Kafka-coupled and the live FastAPI path never
reads from Kafka, so host alerts (SSH brute force, FIM, process exec) were
invisible to the SOC dashboard. This module tails the Wazuh manager's local
alerts file and yields normalized host alerts that feed the SAME enrichment +
scoring + storage path as network alerts.

It also keeps a short-lived per-IP "recently seen on host layer" set so the
network consumer can detect host+network correlation and raise the risk score.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Default Wazuh manager alerts file (JSON-lines). Override via env.
WAZUH_ALERTS_PATH = os.getenv("WAZUH_ALERTS_PATH", "/var/ossec/logs/alerts/alerts.json")

# Wazuh rule groups -> normalized attack type for the dashboard.
_GROUP_ATTACK_MAP: tuple[tuple[str, str], ...] = (
    ("authentication_failures", "Brute Force"),
    ("authentication_failed",   "Brute Force"),
    ("sshd",                    "Brute Force"),
    ("win_authentication_failed", "Brute Force"),
    ("syscheck",                "File Integrity"),
    ("rootcheck",               "Rootkit"),
    ("vulnerability-detector",  "Vulnerability"),
    ("web",                     "Web Attack"),
    ("attack",                  "Exploit"),
    ("intrusion_detection",     "Intrusion"),
)


@dataclass
class WazuhAlert:
    timestamp: str
    rule_level: int
    rule_description: str
    rule_groups: list[str]
    agent_id: str
    agent_name: str
    agent_ip: str
    src_ip: str
    attack_type: str
    raw: dict = field(default_factory=dict)


def _group_to_attack(groups: list[str], description: str) -> str:
    hay = " ".join(groups).lower() + " " + description.lower()
    for needle, atype in _GROUP_ATTACK_MAP:
        if needle in hay:
            return atype
    return "Host Alert"


def normalize(raw: dict[str, Any]) -> WazuhAlert | None:
    """Normalize a raw Wazuh alerts.json record. Returns None for unusable rows."""
    rule = raw.get("rule", {}) or {}
    agent = raw.get("agent", {}) or {}
    data = raw.get("data", {}) or {}
    level = int(rule.get("level", 0) or 0)
    if level <= 0:
        return None
    groups = rule.get("groups", []) or []
    description = str(rule.get("description", ""))
    src_ip = str(data.get("srcip") or data.get("src_ip") or "")
    return WazuhAlert(
        timestamp=str(raw.get("timestamp", "")),
        rule_level=level,
        rule_description=description,
        rule_groups=groups,
        agent_id=str(agent.get("id", "")),
        agent_name=str(agent.get("name", "")),
        agent_ip=str(agent.get("ip", "")),
        src_ip=src_ip,
        attack_type=_group_to_attack(groups, description),
        raw=raw,
    )


def tail_wazuh_alerts(path: str | Path | None = None,
                      poll_interval: float = 0.5) -> Iterator[WazuhAlert | None]:
    """Tail the Wazuh alerts JSON-lines file, yielding normalized alerts.

    Yields None on idle so the caller stays responsive. Waits for the file to
    appear (Wazuh may start after us). Degrades gracefully: any parse error on a
    line is skipped, not fatal.
    """
    p = Path(path or WAZUH_ALERTS_PATH)
    logger.info("Tailing Wazuh alerts from %s", p)
    waited = 0
    while not p.exists():
        if waited == 0:
            logger.warning("Wazuh alerts file not found at %s — host fusion idle until it appears", p)
        waited += 1
        time.sleep(5)
        yield None
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                yield None
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            alert = normalize(raw)
            if alert is not None:
                yield alert


class HostSeenTracker:
    """Short-TTL set of source IPs recently seen on the host (Wazuh) layer.

    Used by the network consumer to detect host+network correlation. In-memory,
    process-local, bounded by TTL eviction on lookup.
    """

    def __init__(self, ttl_s: float = 120.0) -> None:
        self.ttl_s = ttl_s
        self._seen: dict[str, float] = {}

    def mark(self, ip: str, now: float | None = None) -> None:
        if ip:
            self._seen[ip] = now if now is not None else time.time()

    def is_correlated(self, ip: str, now: float | None = None) -> bool:
        if not ip:
            return False
        now = now if now is not None else time.time()
        ts = self._seen.get(ip)
        if ts is None:
            return False
        if now - ts > self.ttl_s:
            self._seen.pop(ip, None)
            return False
        return True
