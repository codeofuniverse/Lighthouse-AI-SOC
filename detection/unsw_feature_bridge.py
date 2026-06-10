"""Bridge: maps Suricata eve.json flow fields to UNSW-NB15 feature space.

The UNSW-NB15 model expects 18 features derived from flow-level statistics.
This bridge extracts them from a parsed Suricata eve.json flow event dict,
using the same field names the training script used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Exact column order the UNSW-NB15 joblib model expects.
UNSW_FEATURES = [
    "dur",        # flow duration (seconds)
    "spkts",      # fwd packet count
    "dpkts",      # bwd packet count
    "sbytes",     # fwd bytes
    "dbytes",     # bwd bytes
    "smeansz",    # mean fwd packet size
    "dmeansz",    # mean bwd packet size
    "rate",       # packets/s
    "sload",      # fwd bits/s
    "dload",      # bwd bits/s
    "sjit",       # fwd jitter (IAT std proxy)
    "djit",       # bwd jitter
    "sintpkt",    # fwd inter-packet arrival mean (ms)
    "dintpkt",    # bwd inter-packet arrival mean (ms)
    "synack",     # SYN flag presence (0/1)
    "ackdat",     # ACK flag presence (0/1)
    "ct_srv_src", # connections to same service from same src (estimated)
    "ct_dst_ltm", # connections to same dst recently (estimated)
]


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class UnswFeatureBridge:
    """Converts a Suricata eve.json flow event to an 18-feature UNSW-NB15 DataFrame."""

    def transform(self, ev: dict[str, Any]) -> pd.DataFrame | None:
        """Return single-row DataFrame with UNSW_FEATURES columns, or None if invalid."""
        flow = ev.get("flow")
        if not flow:
            return None

        spkts = float(flow.get("pkts_toserver", 0) or 0)
        dpkts = float(flow.get("pkts_toclient", 0) or 0)
        sbytes = float(flow.get("bytes_toserver", 0) or 0)
        dbytes = float(flow.get("bytes_toclient", 0) or 0)

        if spkts == 0 and dpkts == 0:
            return None

        # Flow duration in seconds
        start_str = flow.get("start", "")
        end_str   = flow.get("end", "")
        if start_str and end_str:
            t0 = _parse_ts(start_str)
            t1 = _parse_ts(end_str)
            dur = max((t1 - t0).total_seconds(), 0.0001)
        else:
            dur = max(float(flow.get("age", 1) or 1), 0.0001)

        safe_spkts = max(spkts, 1)
        safe_dpkts = max(dpkts, 1)

        smeansz = sbytes / safe_spkts
        dmeansz = dbytes / safe_dpkts
        rate    = (spkts + dpkts) / dur
        sload   = (sbytes * 8) / dur   # bits/s
        dload   = (dbytes * 8) / dur

        # Jitter: estimated as 30% of mean inter-packet time
        sintpkt = (dur * 1000) / safe_spkts   # ms
        dintpkt = (dur * 1000) / safe_dpkts
        sjit    = sintpkt * 0.3
        djit    = dintpkt * 0.3

        # TCP flags
        tcp = ev.get("tcp", {}) or {}
        synack = 1.0 if tcp.get("syn") else 0.0
        ackdat = 1.0 if tcp.get("ack") else 0.0

        # Session context estimates (UNSW connection-tracking features)
        # Without a real session tracker, estimate from packet counts
        ct_srv_src = min(float(spkts) / 10.0, 10.0)
        ct_dst_ltm = min(float(dpkts) / 10.0, 10.0)

        row = {
            "dur":        dur,
            "spkts":      spkts,
            "dpkts":      dpkts,
            "sbytes":     sbytes,
            "dbytes":     dbytes,
            "smeansz":    smeansz,
            "dmeansz":    dmeansz,
            "rate":       rate,
            "sload":      sload,
            "dload":      dload,
            "sjit":       sjit,
            "djit":       djit,
            "sintpkt":    sintpkt,
            "dintpkt":    dintpkt,
            "synack":     synack,
            "ackdat":     ackdat,
            "ct_srv_src": ct_srv_src,
            "ct_dst_ltm": ct_dst_ltm,
        }

        return pd.DataFrame([row], columns=UNSW_FEATURES)
