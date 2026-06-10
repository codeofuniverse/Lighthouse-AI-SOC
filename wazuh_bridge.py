"""Async bridge from the Wazuh REST API to Kafka."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from asyncio.subprocess import PIPE
from typing import Any, Protocol

import httpx
from confluent_kafka import Producer

from backend.config import load_settings
from backend.schemas.alert import Alert


logger = logging.getLogger(__name__)


class ProducerLike(Protocol):
    """Protocol for Kafka producer used in tests."""

    def produce(self, topic: str, key: bytes | None = None, value: bytes | None = None) -> None:  # pragma: no cover
        ...

    def poll(self, timeout: float) -> int:  # pragma: no cover
        ...

    def flush(self) -> int:  # pragma: no cover
        ...


class WazuhBridge:
    """Poll Wazuh alerts, normalize them, and publish them to Kafka."""

    def __init__(self, producer: "ProducerLike" | None = None) -> None:
        self._settings = load_settings()
        self._client: httpx.AsyncClient | None = None
        self._token: str | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_seen_alert_key: tuple[int, int | str] | None = None
        self._shutdown_event = asyncio.Event()
        self._raw_topic = self._settings.kafka_raw_topic
        self._alerts_container = self._settings.wazuh_alerts_container
        self._alerts_source_mode = os.getenv("WAZUH_ALERTS_SOURCE", "api").strip().lower()
        self._producer = producer or Producer(
            {
                "bootstrap.servers": self._settings.kafka_brokers,
                "client.id": "wazuh-bridge",
            }
        )

    async def start(self) -> None:
        """Start the background polling loop."""

        if self._running:
            logger.info("Wazuh bridge already running")
            return
        self._running = True
        self._shutdown_event.clear()
        timeout = httpx.Timeout(self._settings.request_timeout)
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=timeout,
            verify=self._settings.wazuh_verify_ssl,
        )
        self._task = asyncio.create_task(self._run(), name="wazuh-bridge-poller")
        logger.info("Wazuh bridge started against %s", self._settings.base_url)

    async def stop(self) -> None:
        """Stop the polling loop and close the HTTP client."""

        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("Wazuh bridge polling task cancelled")
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._token = None
        try:
            self._producer.flush()
        except Exception:
            logger.debug("Kafka producer flush failed", exc_info=True)
        logger.info("Wazuh bridge stopped")

    async def _run(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                await self._ensure_token()
                alerts = await self._fetch_alerts()
                for alert in alerts:
                    normalized = self._normalize_alert(alert)
                    if normalized is None:
                        continue
                    self._produce_alert(normalized)
                    logger.info("Produced alert id=%s", normalized["id"])
                backoff = 1.0
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self._settings.alert_poll_interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except (httpx.TransportError, httpx.ReadTimeout, httpx.ConnectError) as exc:
                logger.warning("Wazuh connection error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            except Exception:
                logger.exception("Unexpected Wazuh bridge failure")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

    async def _ensure_token(self) -> None:
        if self._client is None:
            raise RuntimeError("Bridge client is not initialized")
        if self._token:
            return
        response = await self._client.post(
            "/security/user/authenticate",
            auth=(self._settings.wazuh_username, self._settings.wazuh_password),
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["data"]["token"]
        logger.info("Authenticated with Wazuh API")

    async def _fetch_alerts(self) -> list[dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("Bridge client is not initialized")
        if self._alerts_source_mode == "container":
            return await self._fetch_alerts_from_container()

        try:
            if self._token is None:
                await self._ensure_token()
            headers = {"Authorization": f"Bearer {self._token}"}
            response = await self._client.get("/alerts", headers=headers)
            if response.status_code == httpx.codes.UNAUTHORIZED:
                logger.info("Wazuh token rejected, refreshing")
                self._token = None
                await self._ensure_token()
                headers = {"Authorization": f"Bearer {self._token}"}
                response = await self._client.get("/alerts", headers=headers)
            if response.status_code == httpx.codes.NOT_FOUND:
                logger.warning(
                    "Wazuh API /alerts returned 404; switching bridge to container-backed alerts from %s",
                    self._alerts_container,
                )
                self._alerts_source_mode = "container"
                return await self._fetch_alerts_from_container()
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", payload)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                alerts = data.get("affected_items")
                if isinstance(alerts, list):
                    return alerts
            return []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == httpx.codes.NOT_FOUND:
                logger.warning(
                    "Wazuh API /alerts not found; switching bridge to container-backed alerts from %s",
                    self._alerts_container,
                )
                self._alerts_source_mode = "container"
                return await self._fetch_alerts_from_container()
            raise

    async def _fetch_alerts_from_container(self) -> list[dict[str, Any]]:
        command = [
            "docker",
            "exec",
            self._alerts_container,
            "bash",
            "-lc",
            "tail -n 200 /var/ossec/logs/alerts/alerts.json",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.warning(
                "Failed to read alerts from container %s: %s",
                self._alerts_container,
                stderr.decode("utf-8", errors="ignore").strip(),
            )
            return []

        alerts: list[dict[str, Any]] = []
        for line in stdout.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON alert line from %s", self._alerts_container)
                continue
            if isinstance(payload, dict):
                alerts.append(payload)
        return alerts

    def _normalize_alert(self, raw_alert: dict[str, Any]) -> dict[str, Any] | None:
        alert_id = str(raw_alert.get("id") or raw_alert.get("_id") or raw_alert.get("rule", {}).get("id") or "")
        if not alert_id:
            logger.debug("Skipping alert without identifier")
            return None
        alert_key = self._alert_id_key(alert_id)
        if self._last_seen_alert_key is not None and alert_key <= self._last_seen_alert_key:
            logger.debug("Skipping duplicate or stale alert id=%s", alert_id)
            return None

        rule = raw_alert.get("rule", {}) if isinstance(raw_alert.get("rule", {}), dict) else {}
        agent = raw_alert.get("agent", {}) if isinstance(raw_alert.get("agent", {}), dict) else {}
        data = raw_alert.get("data", {}) if isinstance(raw_alert.get("data", {}), dict) else {}

        normalized = {
            "id": alert_id,
            "timestamp": raw_alert.get("timestamp") or raw_alert.get("@timestamp") or raw_alert.get("time") or "1970-01-01T00:00:00Z",
            "rule_level": int(rule.get("level") or raw_alert.get("rule_level") or 0),
            "rule_description": str(rule.get("description") or raw_alert.get("rule_description") or ""),
            "rule_groups": rule.get("groups") or raw_alert.get("rule_groups") or [],
            "agent_id": str(agent.get("id") or raw_alert.get("agent_id") or ""),
            "agent_name": str(agent.get("name") or raw_alert.get("agent_name") or ""),
            "agent_ip": str(agent.get("ip") or raw_alert.get("agent_ip") or ""),
            "src_ip": str(data.get("srcip") or data.get("src_ip") or raw_alert.get("src_ip") or ""),
            "dst_ip": str(data.get("dstip") or data.get("dst_ip") or raw_alert.get("dst_ip") or ""),
            "src_port": int(data.get("srcport") or data.get("src_port") or raw_alert.get("src_port") or 0),
            "protocol": str(data.get("protocol") or raw_alert.get("protocol") or ""),
        }

        model = Alert.model_validate(normalized)
        self._last_seen_alert_key = alert_key
        return model.model_dump(mode="python")

    def _produce_alert(self, alert: dict[str, Any]) -> None:
        """Produce a normalized alert to Kafka."""
        key = (alert.get("src_ip") or alert.get("agent_id") or "unknown").encode("utf-8")
        payload = json.dumps(alert, default=str).encode("utf-8")
        self._producer.produce(self._raw_topic, key=key, value=payload)
        self._producer.poll(0)

    @staticmethod
    def _alert_id_key(alert_id: str) -> tuple[int, int | str]:
        if alert_id.isdigit():
            return (0, int(alert_id))
        return (1, alert_id)


__all__ = ["WazuhBridge"]
