"""Configuration helpers for the Wazuh bridge."""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    wazuh_host: str
    wazuh_port: int
    wazuh_username: str
    wazuh_password: str
    wazuh_verify_ssl: bool
    alert_poll_interval: float
    request_timeout: float
    queue_maxsize: int
    log_level: str
    kafka_brokers: str
    kafka_raw_topic: str
    wazuh_alerts_container: str

    @property
    def base_url(self) -> str:
        """Return the Wazuh API base URL."""

        scheme = "https"
        return f"{scheme}://{self.wazuh_host}:{self.wazuh_port}"


def load_settings() -> Settings:
    """Load a fresh settings object from the current environment."""

    return Settings(
        wazuh_host=os.getenv("WAZUH_HOST", "localhost"),
        wazuh_port=_get_int("WAZUH_PORT", 55000),
        wazuh_username=os.getenv("WAZUH_USERNAME", "wazuh"),
        wazuh_password=os.getenv("WAZUH_PASSWORD", "change-me"),
        wazuh_verify_ssl=_get_bool("WAZUH_VERIFY_SSL", False),
        alert_poll_interval=float(os.getenv("WAZUH_ALERT_POLL_INTERVAL", "5")),
        request_timeout=float(os.getenv("WAZUH_REQUEST_TIMEOUT", "30")),
        queue_maxsize=_get_int("WAZUH_QUEUE_MAXSIZE", 1000),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        kafka_brokers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
        kafka_raw_topic=os.getenv("KAFKA_RAW_TOPIC", "raw-alerts"),
        wazuh_alerts_container=os.getenv("WAZUH_ALERTS_CONTAINER", "wazuh-manager"),
    )

