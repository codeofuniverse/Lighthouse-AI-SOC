"""Empirically calibrate rate-aggregator thresholds from captured Suricata flows.

Per the production plan (Phase 5): thresholds must come from MEASURED traffic, not
guesses. Capture a baseline of normal traffic and (separately) an attack run, then
run this script to compute the per-source flow-rate / unique-port / byte-rate
distributions for each. It prints the benign 99th-percentile and the attack
minimums, and recommends thresholds that sit in the separating margin between them.

Usage:
    # 1. Capture baseline (normal traffic) — let it run ~10 min:
    #    cp /var/log/suricata/eve.json baseline_eve.json   (or capture a window)
    # 2. Capture attack traffic separately:
    #    cp /var/log/suricata/eve.json attack_eve.json
    # 3. Analyze:
    python scripts/calibrate_thresholds.py --baseline baseline_eve.json --attack attack_eve.json

    # Or analyze a single file just to see distributions:
    python scripts/calibrate_thresholds.py --baseline eve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import CIC_FLOW_FEATURES, suricata_to_vectors  # noqa: E402

WINDOW_S = 10.0  # must match RateAggregator.window_s


def _parse_ts(ts: str) -> float:
    from datetime import datetime
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def analyze(path: Path) -> dict:
    """Compute per-source sliding-window stats from an eve.json flow capture."""
    # Per source IP: list of (ts, dst_port, half_open, total_bytes, dst_ip)
    flows: dict[str, list] = defaultdict(list)
    n_flows = 0
    for line in path.open("r", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or '"flow"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_type") != "flow":
            continue
        vec = suricata_to_vectors(ev)
        if vec is None:
            continue
        cic = dict(zip(CIC_FLOW_FEATURES, vec[0]))
        src = ev.get("src_ip", "")
        if not src:
            continue
        ts = _parse_ts(ev.get("timestamp", ""))
        half_open = cic["syn_flag"] > 0 and (cic["bwd_pkts"] == 0 or cic["ack_flag"] == 0)
        flows[src].append((ts, int(ev.get("dest_port", 0) or 0), half_open,
                           cic["fwd_bytes"] + cic["bwd_bytes"], ev.get("dest_ip", "")))
        n_flows += 1

    # For each source, slide a WINDOW_S window and record peak metrics
    peak_flows_per_sec: list[float] = []
    peak_unique_ports: list[int] = []
    peak_bytes_per_sec: list[float] = []
    for src, recs in flows.items():
        recs.sort(key=lambda r: r[0])
        for i in range(len(recs)):
            t0 = recs[i][0]
            window = [r for r in recs if t0 <= r[0] < t0 + WINDOW_S]
            if len(window) < 3:
                continue
            span = max(window[-1][0] - window[0][0], 0.5)
            peak_flows_per_sec.append(len(window) / span)
            peak_unique_ports.append(len({r[1] for r in window}))
            by_dst: dict[str, float] = defaultdict(float)
            for r in window:
                by_dst[r[4]] += r[3]
            peak_bytes_per_sec.append(max(by_dst.values()) / span)

    def pct(vals: list, p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(len(s) - 1, int(len(s) * p))]

    return {
        "n_flows": n_flows,
        "n_sources": len(flows),
        "flows_per_sec_p50": pct(peak_flows_per_sec, 0.50),
        "flows_per_sec_p99": pct(peak_flows_per_sec, 0.99),
        "flows_per_sec_max": max(peak_flows_per_sec) if peak_flows_per_sec else 0,
        "unique_ports_p99": pct(peak_unique_ports, 0.99),
        "unique_ports_max": max(peak_unique_ports) if peak_unique_ports else 0,
        "bytes_per_sec_p99": pct(peak_bytes_per_sec, 0.99),
        "bytes_per_sec_max": max(peak_bytes_per_sec) if peak_bytes_per_sec else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="eve.json of NORMAL traffic")
    ap.add_argument("--attack", help="eve.json of ATTACK traffic (optional)")
    args = ap.parse_args()

    print("=" * 64)
    print("  Rate-aggregator threshold calibration")
    print("=" * 64)

    base = analyze(Path(args.baseline))
    print(f"\nBASELINE ({args.baseline}): {base['n_flows']:,} flows, {base['n_sources']} sources")
    print(f"  flows/sec  p50={base['flows_per_sec_p50']:.1f}  p99={base['flows_per_sec_p99']:.1f}  max={base['flows_per_sec_max']:.1f}")
    print(f"  uniq ports p99={base['unique_ports_p99']}  max={base['unique_ports_max']}")
    print(f"  bytes/sec  p99={base['bytes_per_sec_p99']:,.0f}  max={base['bytes_per_sec_max']:,.0f}")

    if args.attack:
        atk = analyze(Path(args.attack))
        print(f"\nATTACK ({args.attack}): {atk['n_flows']:,} flows, {atk['n_sources']} sources")
        print(f"  flows/sec  p50={atk['flows_per_sec_p50']:.1f}  max={atk['flows_per_sec_max']:.1f}")
        print(f"  uniq ports max={atk['unique_ports_max']}")
        print(f"  bytes/sec  max={atk['bytes_per_sec_max']:,.0f}")

        print("\n--- RECOMMENDED THRESHOLDS (between benign-p99 and attack level) ---")
        # Place threshold above benign p99, below attack level
        flood = max(base["flows_per_sec_p99"] * 1.5, 15)
        ports = max(int(base["unique_ports_p99"] * 1.5) + 1, 10)
        dosb  = max(base["bytes_per_sec_p99"] * 1.5, 500_000)
        print(f"  flood_flows_per_sec   = {flood:.0f}   (benign p99={base['flows_per_sec_p99']:.1f})")
        print(f"  portscan_unique_ports = {ports}      (benign p99={base['unique_ports_p99']})")
        print(f"  dos_bytes_per_sec     = {dosb:,.0f}  (benign p99={base['bytes_per_sec_p99']:,.0f})")
        print("\n  Write these into RateAggregator.__init__ with a comment citing this run.")
    else:
        print("\n(no --attack file given — showing baseline distribution only)")
    print("=" * 64)


if __name__ == "__main__":
    main()
