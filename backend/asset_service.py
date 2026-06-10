"""Asset inventory lookup — loads data/asset_db.json at startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/asset_db.json")
_agents: dict[str, dict[str, Any]] = {}
_ips: dict[str, dict[str, Any]] = {}


def load(path: str | Path = _DB_PATH) -> None:
    """Load the asset database from disk. Safe to call on startup."""
    global _agents, _ips
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _agents = data.get("agents", {})
        _ips    = data.get("ips", {})
        logger.info("Asset DB loaded: %d agents, %d IPs", len(_agents), len(_ips))
    except FileNotFoundError:
        logger.warning("Asset DB not found at %s — enrichment skipped", path)
    except Exception as exc:
        logger.warning("Asset DB load failed: %s", exc)


def lookup(agent_id: str | None = None, agent_ip: str | None = None) -> dict[str, Any]:
    """Return asset metadata for an agent_id or IP. Empty dict if no match."""
    if agent_id and agent_id in _agents:
        return dict(_agents[agent_id])
    if agent_ip and agent_ip in _ips:
        return dict(_ips[agent_ip])
    return {}
