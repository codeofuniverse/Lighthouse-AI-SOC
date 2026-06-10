from datetime import datetime

from backend.schemas.alert import Alert


def test_alert_schema_parses_timestamp_and_ips() -> None:
    alert = Alert.model_validate(
        {
            "id": "123",
            "timestamp": "2026-05-14T10:00:00Z",
            "rule_level": 5,
            "rule_description": "Example",
            "rule_groups": "auth,sshd",
            "agent_id": "001",
            "agent_name": "agent-1",
            "agent_ip": "10.0.0.2",
            "src_ip": "192.168.1.5",
            "dst_ip": "192.168.1.10",
            "src_port": 443,
            "protocol": "tcp",
        }
    )

    assert alert.timestamp == datetime.fromisoformat("2026-05-14T10:00:00+00:00")
    assert alert.rule_groups == ["auth", "sshd"]
