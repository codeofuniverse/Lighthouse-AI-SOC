"""Zeek conn.log / http.log -> model feature vectors.

Zeek is the richer second sensor (see PHASE 3 of the plan). Its conn.log exposes
the flow primitives Suricata flow records lack, which is exactly what the weak
UNSW classes (Shellcode/Worms/Recon/DoS) and CIC Web Attack need.

This module is the Zeek analogue of detection/flow_features.py::suricata_to_vectors
and reuses the SAME compute_cic / compute_unsw arithmetic, so a flow scored from a
Zeek record yields the same core vector as the equivalent Suricata flow — no
training-serving skew across sensors.

Field mapping (Zeek JSON conn.log -> primitives):
    duration            -> dur
    orig_pkts           -> fwd/spkts        resp_pkts        -> bwd/dpkts
    orig_ip_bytes       -> fwd/sbytes       resp_ip_bytes    -> bwd/dbytes
    id.resp_p           -> dst_port         proto / service  -> proto / service
    missed_bytes,history-> sloss/dloss      conn_state       -> TCP-flag heuristic
The remaining UNSW-28 columns:
    ct_*    : computed by a sliding window over recent conn records (UNSW Argus method)
    sttl/dttl/stcpb/dtcpb : require the custom Zeek policy (infra/zeek/local.zeek);
              absent -> 0.0 (model still runs; those features just carry no signal).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from detection.flow_features import (
    CIC_FLOW_FEATURES_V2,
    UNSW_FLOW_FEATURES,
    compute_cic,
    compute_unsw,
    _MIN_DUR,
)

# conn_state -> which TCP flags were almost certainly present. Mirrors the Zeek
# state machine; used only for the CIC flag features (UNSW ignores flags).
_STATE_FLAGS = {
    # state:  (fin, syn, rst, psh, ack, urg)
    "S0":  (0, 1, 0, 0, 0, 0),   # SYN, no reply
    "S1":  (0, 1, 0, 0, 1, 0),   # established, not terminated
    "SF":  (1, 1, 0, 1, 1, 0),   # normal establish + terminate
    "REJ": (0, 1, 1, 0, 0, 0),   # rejected
    "S2":  (0, 1, 0, 0, 1, 0),
    "S3":  (0, 1, 0, 0, 1, 0),
    "RSTO": (0, 1, 1, 0, 1, 0),
    "RSTR": (0, 1, 1, 0, 1, 0),
    "RSTOS0": (0, 1, 1, 0, 0, 0),
    "RSTRH": (0, 0, 1, 0, 1, 0),
    "SH":  (0, 1, 0, 0, 0, 0),
    "SHR": (0, 0, 0, 0, 1, 0),
    "OTH": (0, 0, 0, 0, 1, 0),
}

# The full UNSW-28 order the offline model expects (see scripts/train_kfold_unsw.py).
UNSW28_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "smeansz", "dmeansz", "rate",
    "sload", "dload", "sjit", "djit", "sintpkt", "dintpkt", "synack", "ackdat",
    "ct_srv_src", "ct_dst_ltm", "ct_srv_dst", "proto", "sttl", "stcpb", "sloss",
    "dloss", "service", "ct_src_dport_ltm", "ct_dst_src_ltm", "dtcpb",
]


def _flags_from_state(state: str) -> tuple[int, int, int, int, int, int]:
    return _STATE_FLAGS.get(str(state or "").upper(), (0, 0, 0, 0, 1, 0))


def _primitives(conn: dict) -> tuple[float, float, float, float, float, int] | None:
    """(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur, dst_port) or None."""
    fwd_pkts = float(conn.get("orig_pkts", 0) or 0)
    bwd_pkts = float(conn.get("resp_pkts", 0) or 0)
    if fwd_pkts == 0 and bwd_pkts == 0:
        return None
    fwd_bytes = float(conn.get("orig_ip_bytes", conn.get("orig_bytes", 0)) or 0)
    bwd_bytes = float(conn.get("resp_ip_bytes", conn.get("resp_bytes", 0)) or 0)
    dur = max(float(conn.get("duration", 0) or 0), _MIN_DUR)
    dst_port = int(conn.get("id.resp_p", conn.get("id_resp_p", 0)) or 0)
    return fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur, dst_port


def zeek_conn_to_vectors(conn: dict) -> tuple[list[float], list[float]] | None:
    """Return (cic_vec_v2, unsw_core_vec) from a Zeek conn.log record, or None.

    Uses the shared compute_* funcs so these are identical to the Suricata path
    for the same flow (this is what the skew test asserts).
    """
    prim = _primitives(conn)
    if prim is None:
        return None
    fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur, dst_port = prim
    fin, syn, rst, psh, ack, urg = _flags_from_state(conn.get("conn_state", ""))

    cic = compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                      fin, syn, rst, psh, ack, urg, dst_port=dst_port)
    unsw = compute_unsw(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)
    cic_vec = [float(cic[f]) for f in CIC_FLOW_FEATURES_V2]
    unsw_vec = [float(unsw[f]) for f in UNSW_FLOW_FEATURES]
    return cic_vec, unsw_vec


class CtWindow:
    """Sliding 100-connection window to recompute the UNSW ct_* counters live,
    the same way the original UNSW-NB15 Argus pipeline did (last-100-connections
    based). Keeps (ts, src, dst, dport, service) tuples and counts matches."""

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.buf: deque[tuple] = deque(maxlen=size)

    def update_and_counts(self, src: str, dst: str, dport: int, service: str) -> dict[str, int]:
        b = self.buf
        ct = {
            "ct_srv_src": sum(1 for r in b if r[1] == src and r[4] == service),
            "ct_srv_dst": sum(1 for r in b if r[2] == dst and r[4] == service),
            "ct_dst_ltm": sum(1 for r in b if r[2] == dst),
            "ct_src_dport_ltm": sum(1 for r in b if r[1] == src and r[3] == dport),
            "ct_dst_src_ltm": sum(1 for r in b if r[1] == src and r[2] == dst),
        }
        b.append((time.time(), src, dst, dport, service))
        return ct


def unsw28_from_zeek(conn: dict, freq_maps: dict[str, dict],
                     ct: CtWindow | None = None) -> list[float] | None:
    """Build the full 28-feature UNSW vector from a Zeek conn record + ct window.

    freq_maps: the proto/service frequency-encode maps stored in the UNSW model
    dict (scripts/train_kfold_unsw.py). ct: optional sliding-window counter source.
    Features Zeek-default can't supply (sttl/dttl/stcpb/dtcpb/jitter) default to 0
    unless the custom Zeek policy populated them on the record.
    """
    prim = _primitives(conn)
    if prim is None:
        return None
    fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur, dst_port = prim
    total_bytes = fwd_bytes + bwd_bytes

    src = str(conn.get("id.orig_h", conn.get("id_orig_h", "")))
    dst = str(conn.get("id.resp_h", conn.get("id_resp_h", "")))
    service = str(conn.get("service", "-") or "-")
    proto = str(conn.get("proto", "-") or "-")
    counts = ct.update_and_counts(src, dst, dst_port, service) if ct else {}

    def freq(col: str, val: str) -> float:
        return float(freq_maps.get(col, {}).get(val, 0.0))

    feat = {
        "dur": dur,
        "spkts": fwd_pkts, "dpkts": bwd_pkts,
        "sbytes": fwd_bytes, "dbytes": bwd_bytes,
        "smeansz": fwd_bytes / max(fwd_pkts, 1.0),
        "dmeansz": bwd_bytes / max(bwd_pkts, 1.0),
        "rate": (fwd_pkts + bwd_pkts) / dur,
        "sload": (fwd_bytes * 8) / dur, "dload": (bwd_bytes * 8) / dur,
        "sjit": float(conn.get("sjit", 0) or 0), "djit": float(conn.get("djit", 0) or 0),
        "sintpkt": dur / max(fwd_pkts, 1.0), "dintpkt": dur / max(bwd_pkts, 1.0),
        "synack": float(conn.get("synack", 0) or 0), "ackdat": float(conn.get("ackdat", 0) or 0),
        "ct_srv_src": counts.get("ct_srv_src", 0), "ct_dst_ltm": counts.get("ct_dst_ltm", 0),
        "ct_srv_dst": counts.get("ct_srv_dst", 0),
        "proto": freq("proto", proto),
        "sttl": float(conn.get("sttl", conn.get("orig_ttl", 0)) or 0),
        "stcpb": float(conn.get("stcpb", 0) or 0),
        "sloss": float(conn.get("missed_bytes", 0) or 0), "dloss": 0.0,
        "service": freq("service", service),
        "ct_src_dport_ltm": counts.get("ct_src_dport_ltm", 0),
        "ct_dst_src_ltm": counts.get("ct_dst_src_ltm", 0),
        "dtcpb": float(conn.get("dtcpb", 0) or 0),
    }
    return [float(feat[f]) for f in UNSW28_FEATURES]


def zeek_http_features(http: dict) -> dict[str, Any]:
    """Distil a Zeek http.log record (joined by uid) into risk-relevant fields —
    same shape as detection/suricata_bridge._http_features."""
    uri = str(http.get("uri", "") or "")
    method = str(http.get("method", "") or "")
    status = http.get("status_code")
    susp = any(t in uri.lower() for t in
               ("'", "\"", "<script", "select ", "union ", "../", "%27", "%3c"))
    return {
        "http_method": method,
        "url_len": len(uri),
        "http_status": int(status) if isinstance(status, int) else None,
        "url_suspicious": bool(susp),
    }
