"""SOAR response layer — block/isolate actions with dry-run mode."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import shlex

import httpx

logger = logging.getLogger(__name__)

_DRY_RUN      = os.getenv("SOAR_DRY_RUN", "1") == "1"
_WAZUH_HOST   = os.getenv("WAZUH_HOST", "localhost")
_WAZUH_PORT   = os.getenv("WAZUH_PORT", "55000")
_WAZUH_USER   = os.getenv("WAZUH_USERNAME", "wazuh")
_WAZUH_PASS   = os.getenv("WAZUH_PASSWORD", "wazuh")


class SoarEngine:

    def __init__(self, dry_run: bool = _DRY_RUN) -> None:
        self._dry_run = dry_run
        if dry_run:
            logger.info("SOAR running in DRY-RUN mode — no real actions taken")

    def _validate_ip(self, ip: str) -> bool:
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            logger.error("SOAR: invalid IP address %r — action aborted", ip)
            return False

    async def block_ip(self, ip: str) -> bool:
        if not self._validate_ip(ip):
            return False
        if self._dry_run:
            logger.info("[DRY-RUN] SOAR block_ip: %s", ip)
            return True
        try:
            safe_ip = shlex.quote(ip)
            proc = await asyncio.create_subprocess_exec(
                "iptables", "-I", "INPUT", "-s", safe_ip, "-j", "DROP",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("iptables block failed for %s: %s", ip, stderr.decode())
                return False
            logger.info("SOAR blocked IP: %s", ip)
            return True
        except Exception as exc:
            logger.error("SOAR block_ip error: %s", exc)
            return False

    async def unblock_ip(self, ip: str) -> bool:
        if not self._validate_ip(ip):
            return False
        if self._dry_run:
            logger.info("[DRY-RUN] SOAR unblock_ip: %s", ip)
            return True
        try:
            safe_ip = shlex.quote(ip)
            proc = await asyncio.create_subprocess_exec(
                "iptables", "-D", "INPUT", "-s", safe_ip, "-j", "DROP",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("iptables unblock failed for %s: %s", ip, stderr.decode())
                return False
            logger.info("SOAR unblocked IP: %s", ip)
            return True
        except Exception as exc:
            logger.error("SOAR unblock_ip error: %s", exc)
            return False

    async def isolate_agent(self, agent_id: str) -> bool:
        if self._dry_run:
            logger.info("[DRY-RUN] SOAR isolate_agent: %s", agent_id)
            return True
        try:
            url = f"https://{_WAZUH_HOST}:{_WAZUH_PORT}/active-response"
            payload = {
                "command":   "firewall-drop",
                "arguments": [],
                "agent_id":  agent_id,
            }
            async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
                resp = await client.put(url, json=payload, auth=(_WAZUH_USER, _WAZUH_PASS))
                resp.raise_for_status()
            logger.info("SOAR isolated agent: %s", agent_id)
            return True
        except Exception as exc:
            logger.error("SOAR isolate_agent error: %s", exc)
            return False


soar = SoarEngine()
