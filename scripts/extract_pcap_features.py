"""Extract RICH per-flow features from the CIC-2017 Thursday PCAP (streaming).

The CIC MachineLearningCVE CSVs have no HTTP content, which capped Web-Attack
recall (SHAP showed the real signal is HTTP payload + inter-arrival timing). The
user supplied the real capture at data/models/raw/cic 2017/pcap/Thursday-WorkingHours.pcap
(~7.8 GB, pcapng). This script streams it with dpkt (never loading it whole;
~124k pkts/sec, ~20x faster than scapy), reconstructs per-flow stats, and pulls
the features the CSVs lack:

  * real HTTP request: method, URL length, suspicious-token flag (SQLi/XSS/trav)
  * inter-arrival timing: fwd IAT mean/std/max (the top SHAP discriminators)
  * the 18 V2 flow features (so v3 is a strict superset of the servable v2 set)

Each flow is labelled by joining its 5-tuple to the labelled Web-Attacks CSV
(GeneratedLabelledFlows). Output: data/models/raw/cic2017_pcap_rich.parquet.

These features split by live-reproducibility:
  * HTTP method/url/suspicious  -> reproducible from Zeek http.log (PHASE 3 sensor)
  * IAT mean/std/max            -> reproducible only with per-packet timing (Zeek
                                   custom policy / packet logging) — kept for the
                                   OFFLINE ceiling model, flagged in the report.

Usage:
    python scripts/extract_pcap_features.py [--max-packets N] [--web-only]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore")

PCAP = Path("data/models/raw/cic 2017/pcap/Thursday-WorkingHours.pcap")
LABEL_CSV = Path("data/models/raw/cic 2017/GeneratedLabelledFlows/TrafficLabelling/"
                 "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
OUT = Path("data/models/raw/cic2017_pcap_rich.parquet")
WEB_PORTS = {80, 8080, 8000, 443, 8443}

_SUSPICIOUS = (b"'", b"\"", b"<script", b"select ", b"union ", b"../", b"%27",
               b"%3c", b"or 1=1", b"--", b"/etc/passwd", b"alert(")
_METHODS = (b"GET ", b"POST", b"HEAD", b"PUT ", b"DELE", b"OPTI")


class Flow:
    __slots__ = ("fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "t0", "tlast",
                 "fwd_times", "flags", "method", "url_len", "suspicious", "dport")

    def __init__(self, dport: int) -> None:
        self.fwd_pkts = self.bwd_pkts = 0
        self.fwd_bytes = self.bwd_bytes = 0.0
        self.t0 = self.tlast = None
        self.fwd_times: list[float] = []
        self.flags = 0
        self.method = ""
        self.url_len = 0
        self.suspicious = 0
        self.dport = dport


def _parse_http(payload: bytes) -> tuple[str, int, int] | None:
    if not payload[:4] in _METHODS:
        return None
    try:
        line = payload.split(b"\r\n", 1)[0]
        parts = line.split(b" ")
        if len(parts) < 2:
            return None
        method = parts[0].decode("ascii", "replace")
        url = parts[1]
        low = url.lower() + payload[:200].lower()
        susp = int(any(tok in low for tok in _SUSPICIOUS))
        return method, len(url), susp
    except Exception:
        return None


def _iat_stats(times: list[float]) -> tuple[float, float, float]:
    if len(times) < 2:
        return 0.0, 0.0, 0.0
    d = np.diff(np.array(sorted(times)))
    return float(d.mean()), float(d.std()), float(d.max())


def extract(max_packets: int, web_only: bool) -> pd.DataFrame:
    # dpkt is ~20x faster than scapy for bulk streaming (124k vs 6k pkts/sec here).
    import dpkt
    import socket

    flows: dict[tuple, Flow] = {}
    n = 0
    fh = open(PCAP, "rb")
    pcap = dpkt.pcapng.Reader(fh)
    print(f"  streaming {PCAP.name} (dpkt) ...", flush=True)
    for ts, buf in pcap:
        n += 1
        if max_packets and n > max_packets:
            break
        if n % 2_000_000 == 0:
            print(f"    {n:,} packets, {len(flows):,} flows", flush=True)
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            continue
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue
        tcp = ip.data
        if not isinstance(tcp, dpkt.tcp.TCP):
            continue
        sp, dp = int(tcp.sport), int(tcp.dport)
        if web_only and dp not in WEB_PORTS and sp not in WEB_PORTS:
            continue
        src = socket.inet_ntoa(ip.src)
        dst = socket.inet_ntoa(ip.dst)
        if dp in WEB_PORTS or (sp not in WEB_PORTS and sp > dp):
            key = (src, sp, dst, dp, 6); fwd = True
        else:
            key = (dst, dp, src, sp, 6); fwd = False
        f = flows.get(key)
        if f is None:
            f = flows[key] = Flow(dp if fwd else sp)
        plen = len(tcp.data)
        if f.t0 is None:
            f.t0 = ts
        f.tlast = ts
        f.flags |= int(tcp.flags)
        if fwd:
            f.fwd_pkts += 1
            f.fwd_bytes += plen
            f.fwd_times.append(ts)
            if plen and not f.method:
                h = _parse_http(bytes(tcp.data))
                if h:
                    f.method, f.url_len, f.suspicious = h
        else:
            f.bwd_pkts += 1
            f.bwd_bytes += plen
    fh.close()
    print(f"  done: {n:,} packets, {len(flows):,} flows", flush=True)

    rows = []
    for (src, sport, dst, dport, proto), f in flows.items():
        if f.fwd_pkts == 0 and f.bwd_pkts == 0:
            continue
        dur = max((f.tlast - f.t0) if f.t0 else 0.0, 1e-6)
        iat_mean, iat_std, iat_max = _iat_stats(f.fwd_times)
        rows.append({
            "Source IP": src, "Source Port": sport,
            "Destination IP": dst, "Destination Port": dport,
            "fwd_pkts": f.fwd_pkts, "bwd_pkts": f.bwd_pkts,
            "fwd_bytes": f.fwd_bytes, "bwd_bytes": f.bwd_bytes,
            "duration_s": dur,
            "http_method": f.method, "url_len": f.url_len, "url_suspicious": f.suspicious,
            "fwd_iat_mean": iat_mean, "fwd_iat_std": iat_std, "fwd_iat_max": iat_max,
            "fin_flag": int(bool(f.flags & 0x01)), "syn_flag": int(bool(f.flags & 0x02)),
            "rst_flag": int(bool(f.flags & 0x04)), "psh_flag": int(bool(f.flags & 0x08)),
            "ack_flag": int(bool(f.flags & 0x10)), "urg_flag": int(bool(f.flags & 0x20)),
        })
    return pd.DataFrame(rows)


def _label(df: pd.DataFrame) -> pd.DataFrame:
    lab = pd.read_csv(LABEL_CSV, encoding="latin-1", low_memory=False)
    lab.columns = lab.columns.str.strip()
    lab["Label"] = lab["Label"].astype(str).str.strip()
    keys = ["Source IP", "Source Port", "Destination IP", "Destination Port"]
    for k in ("Source Port", "Destination Port"):
        lab[k] = pd.to_numeric(lab[k], errors="coerce").fillna(0).astype(int)
    lab_small = lab[keys + ["Label"]].drop_duplicates(subset=keys)
    merged = df.merge(lab_small, on=keys, how="left")
    merged["Label"] = merged["Label"].fillna("BENIGN")
    # collapse web-attack subtypes
    merged.loc[merged["Label"].str.contains("Web Attack", na=False), "LabelFine"] = \
        merged["Label"]
    merged["LabelFine"] = merged.get("LabelFine", merged["Label"]).fillna(merged["Label"])
    merged.loc[merged["Label"].str.contains("Web Attack", na=False), "Label"] = "Web Attack"
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-packets", type=int, default=0, help="0 = all (7.8GB)")
    ap.add_argument("--web-only", action="store_true",
                    help="only flows touching web ports (faster, Web-Attack focus)")
    args = ap.parse_args()

    print("=" * 70)
    print("  CIC-2017 Thursday PCAP -> rich per-flow features (streaming)")
    print("=" * 70)
    if not PCAP.exists():
        raise FileNotFoundError(f"{PCAP} not found")

    df = extract(args.max_packets, args.web_only)
    print(f"  extracted {len(df):,} flows; joining labels ...")
    df = _label(df)
    print(df["Label"].value_counts().to_string())
    print(f"  HTTP requests captured: {(df['http_method'] != '').sum():,}")
    print(f"  suspicious URLs: {int(df['url_suspicious'].sum()):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n  Saved {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(df):,} flows)")
    print("  Done.")


if __name__ == "__main__":
    main()
