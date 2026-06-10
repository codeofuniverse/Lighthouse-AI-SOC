"""Normalized alert schema for bridge output."""

from __future__ import annotations

from datetime import datetime
import ipaddress
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Alert(BaseModel):
    """Flat normalized alert contract emitted by the Wazuh bridge."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(..., min_length=1)
    timestamp: datetime
    rule_level: int = Field(..., ge=0)
    rule_description: str = Field(default="")
    rule_groups: list[str] = Field(default_factory=list)
    agent_id: str = Field(default="")
    agent_name: str = Field(default="")
    agent_ip: str = Field(default="")
    src_ip: str = Field(default="")
    dst_ip: str = Field(default="")
    src_port: int = Field(default=0, ge=0)
    protocol: str = Field(default="")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> datetime:
        """Parse Wazuh timestamps into an aware datetime when possible."""

        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            parsed = value.replace("Z", "+00:00")
            return datetime.fromisoformat(parsed)
        raise TypeError("timestamp must be a datetime, string, or UNIX epoch")

    @field_validator("agent_ip", "src_ip", "dst_ip", mode="before")
    @classmethod
    def validate_ip(cls, value: Any) -> str:
        """Accept empty strings or valid IPv4/IPv6 addresses."""

        if value in {None, ""}:
            return ""
        address = str(value)
        ipaddress.ip_address(address)
        return address

    @field_validator("rule_groups", mode="before")
    @classmethod
    def coerce_rule_groups(cls, value: Any) -> list[str]:
        """Normalize rule groups into a list of strings."""

        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(value)]
