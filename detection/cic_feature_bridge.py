"""Bridge: maps Wazuh enriched-alert fields to the 20 CIC 2017 flow features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Exact column order the retrained port-agnostic CIC joblib model expects.
# 'Destination Port' was dropped during retraining so the model generalises
# across HTTP (port 80) and HTTPS (port 443) flood attacks.
CIC_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "FIN Flag Count",
    "SYN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
]

# Average payload sizes used when actual byte counts are unknown.
_AVG_FWD_BYTES = 512.0   # typical HTTP/S request
_AVG_BWD_BYTES = 200.0   # typical short response (ack / redirect)
_MTU = 1500.0


class CicFeatureBridge:
    """Converts a Wazuh enriched-alert dict to a 20-feature CIC 2017 DataFrame.

    Only fields available in Wazuh session enrichment are used; everything
    else is estimated from protocol heuristics.
    """

    def transform(self, alert: dict[str, Any]) -> pd.DataFrame:
        """Return a single-row DataFrame with the 20 CIC features.

        Args:
            alert: Enriched Wazuh alert dict (output of the sessionizer enricher).

        Returns:
            DataFrame with one row and columns matching CIC_FEATURES in order.
        """
        dst_port = int(alert.get("dst_port") or 0)
        event_count = float(alert.get("session_event_count") or 1)
        duration_s = float(alert.get("session_duration_seconds") or 0)
        protocol = str(alert.get("protocol") or "").lower()

        # Flow Duration in microseconds
        duration_us = duration_s * 1_000_000

        # Backward packet estimate: ~30% of forward events
        bwd_count = event_count * 0.3

        # Byte estimates
        fwd_bytes = event_count * _AVG_FWD_BYTES
        bwd_bytes = bwd_count * _AVG_BWD_BYTES

        # Throughput
        safe_duration = max(duration_s, 0.001)
        flow_bytes_s = (fwd_bytes + bwd_bytes) / safe_duration
        flow_pkts_s = (event_count + bwd_count) / safe_duration

        # Inter-arrival time estimates (μs)
        safe_count = max(event_count, 1)
        iat_mean = duration_us / safe_count
        iat_std = iat_mean * 0.3
        fwd_iat_total = duration_us
        fwd_iat_mean = iat_mean
        bwd_iat_mean = iat_mean * 1.5

        # TCP flag heuristics
        is_http = dst_port in (80, 8080) or protocol in ("http",)
        is_https = dst_port == 443 or protocol in ("https", "tls", "ssl")
        is_ssh = dst_port == 22 or protocol == "ssh"

        fin_count = 0.0
        if is_ssh:
            fin_count = 1.0

        syn_count = float(min(event_count, 10))
        psh_count = event_count if (is_http or is_https) else event_count * 0.5
        ack_count = event_count * 2

        row = {
            "Flow Duration": duration_us,
            "Total Fwd Packets": event_count,
            "Total Backward Packets": bwd_count,
            "Total Length of Fwd Packets": fwd_bytes,
            "Total Length of Bwd Packets": bwd_bytes,
            "Fwd Packet Length Mean": _AVG_FWD_BYTES,
            "Bwd Packet Length Mean": _AVG_BWD_BYTES,
            "Bwd Packet Length Max": _MTU,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": iat_mean,
            "Flow IAT Std": iat_std,
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_iat_mean,
            "Bwd IAT Mean": bwd_iat_mean,
            "FIN Flag Count": fin_count,
            "SYN Flag Count": syn_count,
            "PSH Flag Count": psh_count,
            "ACK Flag Count": ack_count,
        }

        return pd.DataFrame([row], columns=CIC_FEATURES)
