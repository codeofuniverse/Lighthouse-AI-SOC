"""Per-source rate aggregation for volumetric attack detection.

Per-flow ML cannot detect DDoS / port scans / floods: a single SYN to a closed
port looks identical whether it comes from an attacker or a normal client. The
discriminating signal is the *rate across many flows from one source*, which no
single flow carries (Sommer & Paxson, "Outside the Closed World," IEEE S&P 2010).

This module maintains a sliding time window of recent flows per source IP and
emits a volumetric verdict (DDoS / PortScan / DoS) when behaviour crosses
thresholds modelled on Suricata/Snort threshold rules:

  PortScan : many unique destination ports from one source in the window
             (cf. Snort sfportscan, Suricata flowint port-scan rules)
  DDoS     : high flow rate of half-open (SYN, no completed handshake) flows
             (cf. SYN-flood detection — half-open ratio + rate)
  DoS      : sustained high byte rate to a single destination

In-memory, process-local, O(1) amortised per flow. Bounded memory: oldest IPs
are evicted when the table exceeds max_ips.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass(slots=True)
class RateVerdict:
    attack_type: str          # "DDoS" | "PortScan" | "DoS"
    confidence: float         # 0.0–1.0, scaled by how far over threshold
    detail: dict = field(default_factory=dict)


@dataclass(slots=True)
class _FlowRec:
    ts: float
    dst_ip: str
    dst_port: int
    half_open: bool           # SYN seen but no completed handshake (no ACK back)
    total_bytes: float


class RateAggregator:
    """Sliding-window per-source-IP volumetric attack detector."""

    def __init__(
        self,
        window_s: float = 10.0,
        max_ips: int = 10_000,
        max_flows_per_ip: int = 5_000,
        portscan_unique_ports: int = 15,
        flood_flows_per_sec: float = 20.0,
        flood_halfopen_ratio: float = 0.5,
        dos_bytes_per_sec: float = 1_000_000.0,
    ) -> None:
        self.window_s = window_s
        self.max_ips = max_ips
        self.max_flows_per_ip = max_flows_per_ip
        self.portscan_unique_ports = portscan_unique_ports
        self.flood_flows_per_sec = flood_flows_per_sec
        self.flood_halfopen_ratio = flood_halfopen_ratio
        self.dos_bytes_per_sec = dos_bytes_per_sec

        # OrderedDict for LRU eviction; src_ip -> deque[_FlowRec]
        self._table: OrderedDict[str, Deque[_FlowRec]] = OrderedDict()
        self._lock = threading.Lock()

    def observe(
        self,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        syn: bool,
        ack: bool,
        bwd_pkts: float,
        total_bytes: float,
        now: float | None = None,
    ) -> RateVerdict | None:
        """Record a flow from src_ip and return a volumetric verdict, or None.

        half_open is inferred as: a SYN was sent but the flow never reached an
        established state (no bytes/packets returned, or no ACK observed).
        This captures both SYN floods and SYN-scan probes.
        """
        if not src_ip:
            return None
        now = now if now is not None else time.time()
        half_open = bool(syn) and (bwd_pkts == 0 or not ack)

        with self._lock:
            dq = self._table.get(src_ip)
            if dq is None:
                dq = deque(maxlen=self.max_flows_per_ip)
                self._table[src_ip] = dq
                if len(self._table) > self.max_ips:
                    self._table.popitem(last=False)  # evict oldest IP
            else:
                self._table.move_to_end(src_ip)

            dq.append(_FlowRec(now, dst_ip, dst_port, half_open, total_bytes))

            # Prune flows outside the window
            cutoff = now - self.window_s
            while dq and dq[0].ts < cutoff:
                dq.popleft()

            return self._verdict(dq, now)

    def _verdict(self, dq: Deque[_FlowRec], now: float) -> RateVerdict | None:
        n = len(dq)
        if n < 3:
            return None  # not enough signal

        span = max(now - dq[0].ts, 0.5)  # avoid div-by-zero; min 0.5s
        flows_per_sec = n / span
        unique_ports = len({r.dst_port for r in dq})
        half_open_n = sum(1 for r in dq if r.half_open)
        half_open_ratio = half_open_n / n

        # Bytes/sec to the single busiest destination
        by_dst: dict[str, float] = {}
        for r in dq:
            by_dst[r.dst_ip] = by_dst.get(r.dst_ip, 0.0) + r.total_bytes
        top_dst_bytes = max(by_dst.values()) if by_dst else 0.0
        bytes_per_sec = top_dst_bytes / span

        detail = {
            "flows_per_sec": round(flows_per_sec, 1),
            "unique_dst_ports": unique_ports,
            "half_open_ratio": round(half_open_ratio, 2),
            "bytes_per_sec": int(bytes_per_sec),
            "window_flows": n,
        }

        # ── PortScan: many unique destination ports ──
        if unique_ports >= self.portscan_unique_ports and half_open_ratio >= 0.4:
            conf = min(1.0, unique_ports / (self.portscan_unique_ports * 3))
            return RateVerdict("PortScan", conf, detail)

        # ── DDoS / SYN flood: high rate of half-open flows ──
        if flows_per_sec >= self.flood_flows_per_sec and half_open_ratio >= self.flood_halfopen_ratio:
            conf = min(1.0, flows_per_sec / (self.flood_flows_per_sec * 5))
            return RateVerdict("DDoS", conf, detail)

        # ── DoS: sustained high byte rate to one destination ──
        if bytes_per_sec >= self.dos_bytes_per_sec:
            conf = min(1.0, bytes_per_sec / (self.dos_bytes_per_sec * 5))
            return RateVerdict("DoS", conf, detail)

        return None
