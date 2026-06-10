"""Canonical flow-feature computation — THE SINGLE SOURCE OF TRUTH.

Imported by BOTH the training scripts and the live serving bridges so that every
feature is produced by the exact same arithmetic, whether the raw numbers come
from a CIC/UNSW CSV row (training) or a live Suricata eve.json flow record
(serving). This structurally eliminates training-serving skew.

Only features Suricata can faithfully reproduce are included. The original
pipeline fabricated inter-arrival times, jitter, per-packet maxima and
connection-tracking counters — those guesses are what made the deployed model
output BENIGN for every flow. They are deliberately gone.

The core formulas use ``np.maximum`` so the *same* function works element-wise
on pandas Series (training over millions of rows) and on plain floats (serving
one flow at a time). Identical code path ⇒ identical features ⇒ no skew.

Design references:
  - Arp et al., "Dos and Don'ts of Machine Learning in Computer Security,"
    USENIX Security 2022 — avoid lab-only / non-reproducible feature spaces.
  - Sculley et al., "Hidden Technical Debt in ML Systems," NeurIPS 2015 —
    eliminate training-serving skew via a single shared transform.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

# Canonical feature order — models/scalers depend on it. DO NOT REORDER.
_CORE_FEATURES = [
    "duration_s",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_pkt_mean",
    "bwd_pkt_mean",
    "flow_bytes_s",
    "flow_pkts_s",
    "down_up_ratio",
    "avg_pkt_size",
]
_FLAG_FEATURES = ["fin_flag", "syn_flag", "rst_flag", "psh_flag", "ack_flag", "urg_flag"]

CIC_FLOW_FEATURES  = _CORE_FEATURES + _FLAG_FEATURES   # 17 — CIC has TCP flag info
UNSW_FLOW_FEATURES = list(_CORE_FEATURES)              # 11 — UNSW lacks clean flags

# ── V2 (Web-Attack-aware) CIC feature set ────────────────────────────────────
# SHAP feature-discovery over ALL 69 CIC flow columns (see
# scripts/discover_cic_features.py + reports/feature_discovery/cic_feature_ranking.json)
# showed the dominant Web-Attack discriminators are inter-arrival-time features
# (Fwd IAT Max, Flow IAT Std, ...) that Suricata flow records CANNOT reproduce.
# The single highest-ranked discriminator that IS reproducible and is NOT already
# captured by the 17 features is "Destination Port" (SHAP rank #22) — it pins a
# flow to web ports (80/443/8080), the context that makes the rate/duration
# features separate HTTP attacks from benign HTTP. So V2 = the 17 features +
# dst_port. (No IAT/Active/Idle/Init_Win: not in eve.json flow events.)
CIC_FLOW_FEATURES_V2 = CIC_FLOW_FEATURES + ["dst_port"]   # 18

_MIN_DUR = 1e-6  # 1 microsecond floor to avoid divide-by-zero


# ── Core computation (works on scalars OR pandas Series via numpy broadcasting) ──
def _core(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur) -> dict[str, Any]:
    fwd_pkts  = np.maximum(fwd_pkts, 0.0)
    bwd_pkts  = np.maximum(bwd_pkts, 0.0)
    fwd_bytes = np.maximum(fwd_bytes, 0.0)
    bwd_bytes = np.maximum(bwd_bytes, 0.0)
    dur       = np.maximum(dur, _MIN_DUR)
    total_pkts  = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    return {
        "duration_s":    dur,
        "fwd_pkts":      fwd_pkts,
        "bwd_pkts":      bwd_pkts,
        "fwd_bytes":     fwd_bytes,
        "bwd_bytes":     bwd_bytes,
        "fwd_pkt_mean":  fwd_bytes / np.maximum(fwd_pkts, 1.0),
        "bwd_pkt_mean":  bwd_bytes / np.maximum(bwd_pkts, 1.0),
        "flow_bytes_s":  total_bytes / dur,
        "flow_pkts_s":   total_pkts / dur,
        "down_up_ratio": bwd_pkts / np.maximum(fwd_pkts, 1.0),
        "avg_pkt_size":  total_bytes / np.maximum(total_pkts, 1.0),
    }


def compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                fin, syn, rst, psh, ack, urg, dst_port=None) -> dict[str, Any]:
    feats = _core(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)
    feats["fin_flag"] = np.where(np.asarray(fin) != 0, 1.0, 0.0) if hasattr(fin, "__len__") else (1.0 if fin else 0.0)
    feats["syn_flag"] = np.where(np.asarray(syn) != 0, 1.0, 0.0) if hasattr(syn, "__len__") else (1.0 if syn else 0.0)
    feats["rst_flag"] = np.where(np.asarray(rst) != 0, 1.0, 0.0) if hasattr(rst, "__len__") else (1.0 if rst else 0.0)
    feats["psh_flag"] = np.where(np.asarray(psh) != 0, 1.0, 0.0) if hasattr(psh, "__len__") else (1.0 if psh else 0.0)
    feats["ack_flag"] = np.where(np.asarray(ack) != 0, 1.0, 0.0) if hasattr(ack, "__len__") else (1.0 if ack else 0.0)
    feats["urg_flag"] = np.where(np.asarray(urg) != 0, 1.0, 0.0) if hasattr(urg, "__len__") else (1.0 if urg else 0.0)
    # V2: destination port (web-context discriminator). Same arithmetic CSV & live.
    if dst_port is not None:
        feats["dst_port"] = np.asarray(dst_port, dtype=float) if hasattr(dst_port, "__len__") else float(dst_port)
    return feats


def compute_unsw(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur) -> dict[str, Any]:
    return _core(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)


# ── Serving adapter: Suricata eve.json flow event → ordered feature vectors ──
def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _suricata_duration_s(flow: dict) -> float:
    start, end = flow.get("start", ""), flow.get("end", "")
    if start and end:
        return max((_parse_ts(end) - _parse_ts(start)).total_seconds(), _MIN_DUR)
    return max(float(flow.get("age", 0) or 0), _MIN_DUR)


# TCP flag bitmask (Suricata tcp_flags hex string is the union seen in the flow)
_FIN, _SYN, _RST, _PSH, _ACK, _URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


def _suricata_flags(tcp: dict) -> tuple[bool, bool, bool, bool, bool, bool]:
    raw = str(tcp.get("tcp_flags", "") or "")
    try:
        bits = int(raw, 16) if raw else 0
    except ValueError:
        bits = 0
    # Prefer explicit booleans when present, else fall back to the bitmask
    fin = bool(tcp.get("fin")) or bool(bits & _FIN)
    syn = bool(tcp.get("syn")) or bool(bits & _SYN)
    rst = bool(tcp.get("rst")) or bool(bits & _RST)
    psh = bool(tcp.get("psh")) or bool(bits & _PSH)
    ack = bool(tcp.get("ack")) or bool(bits & _ACK)
    urg = bool(tcp.get("urg")) or bool(bits & _URG)
    return fin, syn, rst, psh, ack, urg


def suricata_to_vectors(ev: dict) -> tuple[list[float], list[float]] | None:
    """Return (cic_vector, unsw_vector) in canonical order, or None if not a usable flow.

    The CIC vector is the V2 set (CIC_FLOW_FEATURES_V2 — the 17 flow features plus
    dst_port). Callers using the legacy 17-feature model take the leading 17 slots,
    which are byte-for-byte identical to CIC_FLOW_FEATURES (dst_port is appended).
    Both vectors come from the same primitives via the shared compute_* funcs, so a
    flow scored by CIC and UNSW sees mutually consistent inputs.
    """
    flow = ev.get("flow")
    if not flow:
        return None
    fwd_pkts = float(flow.get("pkts_toserver", 0) or 0)
    bwd_pkts = float(flow.get("pkts_toclient", 0) or 0)
    if fwd_pkts == 0 and bwd_pkts == 0:
        return None
    fwd_bytes = float(flow.get("bytes_toserver", 0) or 0)
    bwd_bytes = float(flow.get("bytes_toclient", 0) or 0)
    dur = _suricata_duration_s(flow)
    fin, syn, rst, psh, ack, urg = _suricata_flags(ev.get("tcp", {}) or {})
    dst_port = float(ev.get("dest_port", 0) or 0)

    cic = compute_cic(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur,
                      fin, syn, rst, psh, ack, urg, dst_port=dst_port)
    unsw = compute_unsw(fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dur)
    cic_vec  = [float(cic[f])  for f in CIC_FLOW_FEATURES_V2]
    unsw_vec = [float(unsw[f]) for f in UNSW_FLOW_FEATURES]
    return cic_vec, unsw_vec


# ── Training adapters: pandas DataFrame → feature DataFrame (vectorized) ──────
def cic_features_from_df(df, version: str = "v1") -> "Any":
    """Vectorized CIC feature matrix from a CIC-IDS-2017 DataFrame (stripped cols).

    version="v1" -> the 17 CIC_FLOW_FEATURES (production model).
    version="v2" -> CIC_FLOW_FEATURES_V2 (adds dst_port from "Destination Port").
    """
    import pandas as pd
    dur_s = df["Flow Duration"].astype(float) / 1e6  # CICFlowMeter duration is microseconds
    dst_port = df["Destination Port"].astype(float) if "Destination Port" in df.columns else None
    feats = compute_cic(
        df["Total Fwd Packets"].astype(float),
        df["Total Backward Packets"].astype(float),
        df["Total Length of Fwd Packets"].astype(float),
        df["Total Length of Bwd Packets"].astype(float),
        dur_s,
        df["FIN Flag Count"].astype(float),
        df["SYN Flag Count"].astype(float),
        df["RST Flag Count"].astype(float),
        df["PSH Flag Count"].astype(float),
        df["ACK Flag Count"].astype(float),
        df["URG Flag Count"].astype(float),
        dst_port=dst_port,
    )
    cols = CIC_FLOW_FEATURES_V2 if version == "v2" else CIC_FLOW_FEATURES
    out = pd.DataFrame({f: feats[f] for f in cols})
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def unsw_features_from_df(df) -> "Any":
    """Vectorized UNSW feature matrix from a UNSW-NB15 DataFrame (stripped cols)."""
    import pandas as pd
    feats = compute_unsw(
        df["spkts"].astype(float),
        df["dpkts"].astype(float),
        df["sbytes"].astype(float),
        df["dbytes"].astype(float),
        df["dur"].astype(float),  # UNSW dur is already seconds
    )
    out = pd.DataFrame({f: feats[f] for f in UNSW_FLOW_FEATURES})
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
