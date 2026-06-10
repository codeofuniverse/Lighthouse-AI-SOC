from __future__ import annotations

import asyncio

import pytest
import respx
from httpx import Response

import json

from wazuh_bridge import WazuhBridge


class StubProducer:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def produce(self, topic: str, key: bytes | None = None, value: bytes | None = None) -> None:
        if value:
            payload = json.loads(value.decode("utf-8"))
            self.messages.append(payload)

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_bridge_authenticates_and_normalizes_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAZUH_HOST", "wazuh-manager")
    monkeypatch.setenv("WAZUH_USERNAME", "wazuh")
    monkeypatch.setenv("WAZUH_PASSWORD", "secret")
    monkeypatch.setenv("WAZUH_VERIFY_SSL", "false")
    monkeypatch.setenv("WAZUH_ALERT_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("WAZUH_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("WAZUH_QUEUE_MAXSIZE", "10")
    monkeypatch.setenv("WAZUH_ALERTS_SOURCE", "api")

    producer = StubProducer()
    bridge = WazuhBridge(producer=producer)

    with respx.mock() as router:
        router.post("https://wazuh-manager:55000/security/user/authenticate").mock(
            return_value=Response(200, json={"data": {"token": "jwt-token"}, "error": 0})
        )
        router.get("https://wazuh-manager:55000/alerts").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1001",
                            "timestamp": "2026-05-14T10:00:00Z",
                            "rule": {"level": 7, "description": "Suspicious login", "groups": ["auth"]},
                            "agent": {"id": "001", "name": "agent-1", "ip": "10.0.0.5"},
                            "data": {"srcip": "10.1.1.1", "dstip": "10.1.1.2", "srcport": 22, "protocol": "tcp"},
                        }
                    ]
                },
            )
        )

        await bridge.start()
        await asyncio.sleep(0.5)
        await bridge.stop()

    assert producer.messages
    item = producer.messages[0]
    assert item["id"] == "1001"
    assert item["rule_level"] == 7
    assert item["rule_groups"] == ["auth"]
    assert item["src_ip"] == "10.1.1.1"


@pytest.mark.asyncio
async def test_bridge_refreshes_token_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAZUH_HOST", "wazuh-manager")
    monkeypatch.setenv("WAZUH_USERNAME", "wazuh")
    monkeypatch.setenv("WAZUH_PASSWORD", "secret")
    monkeypatch.setenv("WAZUH_VERIFY_SSL", "false")
    monkeypatch.setenv("WAZUH_ALERT_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("WAZUH_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("WAZUH_QUEUE_MAXSIZE", "10")
    monkeypatch.setenv("WAZUH_ALERTS_SOURCE", "api")

    producer = StubProducer()
    bridge = WazuhBridge(producer=producer)

    with respx.mock() as router:
        router.post("https://wazuh-manager:55000/security/user/authenticate").mock(
            side_effect=[
                Response(200, json={"data": {"token": "first-token"}, "error": 0}),
                Response(200, json={"data": {"token": "second-token"}, "error": 0}),
            ]
        )
        router.get("https://wazuh-manager:55000/alerts").mock(
            side_effect=[
                Response(401, json={"message": "Unauthorized", "error": 1}),
                Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "2001",
                                "timestamp": "2026-05-14T10:00:00Z",
                                "rule": {"level": 3, "description": "Token refresh alert", "groups": ["auth"]},
                                "agent": {"id": "001", "name": "agent-1", "ip": "10.0.0.5"},
                                "data": {"srcip": "10.1.1.1", "dstip": "10.1.1.2", "srcport": 22, "protocol": "tcp"},
                            }
                        ]
                    },
                ),
            ]
        )

        await bridge.start()
        await asyncio.sleep(0.5)
        await bridge.stop()

    assert producer.messages
    item = producer.messages[0]
    assert item["id"] == "2001"
    assert item["rule_level"] == 3


def test_bridge_skips_duplicate_alert_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAZUH_HOST", "wazuh-manager")
    monkeypatch.setenv("WAZUH_USERNAME", "wazuh")
    monkeypatch.setenv("WAZUH_PASSWORD", "secret")
    monkeypatch.setenv("WAZUH_VERIFY_SSL", "false")
    monkeypatch.setenv("WAZUH_ALERT_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("WAZUH_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("WAZUH_QUEUE_MAXSIZE", "10")

    bridge = WazuhBridge()
    first = bridge._normalize_alert(
        {
            "id": "3001",
            "timestamp": "2026-05-14T10:00:00Z",
            "rule": {"level": 1, "description": "First", "groups": ["auth"]},
            "agent": {"id": "001", "name": "agent-1", "ip": "10.0.0.5"},
            "data": {"srcip": "10.1.1.1", "dstip": "10.1.1.2", "srcport": 22, "protocol": "tcp"},
        }
    )
    duplicate = bridge._normalize_alert(
        {
            "id": "3001",
            "timestamp": "2026-05-14T10:00:01Z",
            "rule": {"level": 1, "description": "Duplicate", "groups": ["auth"]},
            "agent": {"id": "001", "name": "agent-1", "ip": "10.0.0.5"},
            "data": {"srcip": "10.1.1.1", "dstip": "10.1.1.2", "srcport": 22, "protocol": "tcp"},
        }
    )

    assert first is not None
    assert duplicate is None
