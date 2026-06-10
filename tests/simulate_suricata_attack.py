"""Inject synthetic Suricata eve.json flow records matching CIC 2017 attack statistics.

Bypasses hping3 -> Suricata -> eve.json chain by writing realistic flow records
directly into the shared eve.json volume so the live detector classifies them.

Usage:
    # Inside detector container:
    docker exec lh-detector python tests/simulate_suricata_attack.py --attack ddos

    # Locally against the volume (Windows path):
    python tests/simulate_suricata_attack.py --attack all --eve data/suricata/eve.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone, timedelta

# CIC 2017 attack flow profiles — (mean, std) tuples
# CIC 2017 key discriminators:
#   DDoS:       very high Flow Packets/s (>1000), short duration, mostly fwd, SYN+ACK
#   DoS:        high bytes/s, longer duration, PSH+ACK
#   PortScan:   1 pkt per flow, many ports, SYN only
#   BruteForce: moderate pkt count, specific ports (22/21), FIN+PSH+ACK
#   Bot:        low rate, long duration, periodic beaconing
# Profiles calibrated against actual CIC 2017 scaler means/stds extracted
# from the trained model. Key insight: CIC flows have mean=7 packets but
# very high packets/s because flows are very short (milliseconds).
# Flow Duration is in MICROSECONDS in the training data.
PROFILES: dict[str, dict] = {
# Profiles calibrated against CIC 2017 scaler means extracted from trained model.
# Key findings from feature importance analysis:
#   1. Bwd Packet Length Mean (26.9%) — must be ~640 bytes (server RST/ACK responses)
#   2. Fwd Packet Length Mean (25.3%) — must be ~40 bytes (small SYN packets)
#   3. PSH Flag Count (11.2%)        — DDoS/DoS must have PSH=1
#   4. Flow Duration                 — must be in microseconds range (~23M us mean)
PROFILES: dict[str, dict] = {
    # DDoS: high pkt rate, large server responses, SYN+PSH+ACK
    # Bwd mean=640 (server responses), Fwd mean=40 (SYN packets)
    "ddos": dict(
        pkts_fwd=(80, 15), pkts_bwd=(80, 15),
        bytes_fwd=(3200, 600), bytes_bwd=(51200, 8000),
        duration_s=(23, 5), dst_port=80,
        syn=1, fin=0, psh=1, ack=1,
    ),
    "ddos-https": dict(
        pkts_fwd=(80, 15), pkts_bwd=(80, 15),
        bytes_fwd=(3200, 600), bytes_bwd=(51200, 8000),
        duration_s=(23, 5), dst_port=443,
        syn=1, fin=0, psh=1, ack=1,
    ),
    # DoS Hulk: very high bytes, PSH+ACK, long duration
    "dos": dict(
        pkts_fwd=(50, 10), pkts_bwd=(50, 10),
        bytes_fwd=(2000, 400), bytes_bwd=(32000, 6000),
        duration_s=(30, 10), dst_port=80,
        syn=0, fin=0, psh=1, ack=1,
    ),
    # PortScan: 1 fwd packet, 0 bwd, SYN only, very short
    "portscan": dict(
        pkts_fwd=(1, 0), pkts_bwd=(0, 0),
        bytes_fwd=(40, 5), bytes_bwd=(0, 0),
        duration_s=(0.5, 0.2), dst_port=0,
        syn=1, fin=0, psh=0, ack=0,
    ),
    # Brute Force: FIN+PSH+ACK, symmetric, port 22
    "bruteforce": dict(
        pkts_fwd=(10, 3), pkts_bwd=(10, 3),
        bytes_fwd=(400, 80), bytes_bwd=(6400, 1200),
        duration_s=(5, 2), dst_port=22,
        syn=1, fin=1, psh=1, ack=1,
    ),
    # Bot: low rate, PSH+ACK beaconing, long duration
    "bot": dict(
        pkts_fwd=(6, 2), pkts_bwd=(6, 2),
        bytes_fwd=(240, 60), bytes_bwd=(3840, 800),
        duration_s=(60, 20), dst_port=80,
        syn=1, fin=1, psh=1, ack=1,
    ),
    # Web Attack: HTTP with PSH+ACK
    "fuzzer": dict(
        pkts_fwd=(15, 5), pkts_bwd=(15, 5),
        bytes_fwd=(600, 150), bytes_bwd=(9600, 2000),
        duration_s=(10, 4), dst_port=80,
        syn=1, fin=1, psh=1, ack=1,
    ),
    # Exploits: small request, large response
    "exploit": dict(
        pkts_fwd=(5, 2), pkts_bwd=(20, 5),
        bytes_fwd=(200, 50), bytes_bwd=(12800, 3000),
        duration_s=(8, 3), dst_port=443,
        syn=1, fin=1, psh=1, ack=1,
    ),
    # Recon: slow scan, short flows
    "recon": dict(
        pkts_fwd=(2, 1), pkts_bwd=(1, 0),
        bytes_fwd=(80, 20), bytes_bwd=(640, 150),
        duration_s=(1, 0.5), dst_port=0,
        syn=1, fin=0, psh=0, ack=1,
    ),
    # Shellcode: tiny flows
    "shellcode": dict(
        pkts_fwd=(3, 1), pkts_bwd=(3, 1),
        bytes_fwd=(120, 30), bytes_bwd=(1920, 400),
        duration_s=(2, 0.8), dst_port=0,
        syn=1, fin=1, psh=1, ack=0,
    ),
    # Worm: SYN propagation
    "worm": dict(
        pkts_fwd=(2, 1), pkts_bwd=(1, 0),
        bytes_fwd=(80, 20), bytes_bwd=(640, 150),
        duration_s=(0.5, 0.2), dst_port=0,
        syn=1, fin=0, psh=0, ack=1,
    ),
}

SRC_IP = "172.28.0.20"
DST_IP = "172.28.0.10"


def _rand(mean: float, std: float, lo: float = 0.0) -> float:
    return max(lo, random.gauss(mean, std))


def _make_flow(profile: dict, flow_id: int) -> dict:
    now = datetime.now(timezone.utc)
    dur_s = _rand(*profile["duration_s"], lo=0.001)
    start = now - timedelta(seconds=dur_s)

    pkts_fwd  = max(1, int(_rand(*profile["pkts_fwd"])))
    pkts_bwd  = max(0, int(_rand(*profile["pkts_bwd"])))
    bytes_fwd = max(pkts_fwd * 40, int(_rand(*profile["bytes_fwd"])))
    bytes_bwd = max(0, int(_rand(*profile["bytes_bwd"])))

    dst_port = profile["dst_port"] if profile["dst_port"] != 0 else random.randint(1, 65535)

    flag_val = 0
    if profile["syn"]: flag_val |= 0x02
    if profile["fin"]: flag_val |= 0x01
    if profile["psh"]: flag_val |= 0x08
    if profile["ack"]: flag_val |= 0x10

    ts_fmt = "%Y-%m-%dT%H:%M:%S.%f+0000"
    return {
        "timestamp": now.strftime(ts_fmt),
        "flow_id": flow_id,
        "in_iface": "eth0",
        "event_type": "flow",
        "src_ip": SRC_IP,
        "src_port": random.randint(1024, 65535),
        "dest_ip": DST_IP,
        "dest_port": dst_port,
        "ip_v": 4,
        "proto": "TCP",
        "flow": {
            "pkts_toserver": pkts_fwd,
            "pkts_toclient": pkts_bwd,
            "bytes_toserver": bytes_fwd,
            "bytes_toclient": bytes_bwd,
            "start": start.strftime(ts_fmt),
            "end": now.strftime(ts_fmt),
            "age": int(dur_s),
            "state": "closed",
            "reason": "timeout",
            "alerted": False,
        },
        "tcp": {
            "tcp_flags": format(flag_val, "02x"),
            "tcp_flags_ts": format(flag_val & 0x0f, "02x"),
            "tcp_flags_tc": format((flag_val >> 4) & 0x0f, "02x"),
            "syn": bool(profile["syn"]),
            "fin": bool(profile["fin"]),
            "psh": bool(profile["psh"]),
            "ack": bool(profile["ack"]),
            "state": "closed",
        },
    }


def inject(eve_path: str, attack: str, count: int, delay: float) -> None:
    attacks = list(PROFILES.keys()) if attack == "all" else [attack]
    print(f"[*] Injecting into {eve_path}")
    print(f"[*] Attacks: {attacks}  count={count}  delay={delay}s\n")

    flow_id = int(time.time() * 1000) % (2 ** 48)

    with open(eve_path, "a", encoding="utf-8") as fh:
        for atk in attacks:
            profile = PROFILES[atk]
            print(f"[+] {atk.upper()} — {count} flows...")
            for i in range(count):
                record = _make_flow(profile, flow_id + i)
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                if delay > 0:
                    time.sleep(delay)
            flow_id += count
            print(f"    done")
            time.sleep(0.5)

    print("\n[*] Done — watch docker logs lh-detector for detections")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="ddos",
                    choices=list(PROFILES.keys()) + ["all"],
                    help="Attack type: " + ", ".join(PROFILES.keys()) + ", all")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--eve", default="/var/log/suricata/eve.json")
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()
    inject(args.eve, args.attack, args.count, args.delay)


if __name__ == "__main__":
    main()
