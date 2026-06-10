"""Live attack detector — reads Suricata flows and prints/logs ML detections.

Usage (inside Docker):
    python detection/live_detector.py

Usage (local, pointing at a Suricata eve.json):
    EVE_PATH=/var/log/suricata/eve.json python detection/live_detector.py

Output:
  - Coloured terminal output for every detected threat
  - JSON lines appended to LOG_PATH (default: logs/detections.json)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from detection.suricata_bridge import DetectionEvent, SuricataBridge

# ── Config from environment ───────────────────────────────────────
MODEL_PATH = os.getenv("CIC_MODEL_PATH", "data/models/cic2017_pipeline_smote.joblib")
EVE_PATH   = os.getenv("EVE_PATH",       "/var/log/suricata/eve.json")
LOG_PATH   = os.getenv("LOG_PATH",       "logs/detections.json")
LOG_BENIGN = os.getenv("LOG_BENIGN", "false").lower() == "true"

# ── ANSI colours ─────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

THREAT_COLOUR = {
    "DDoS":        RED,
    "DoS":         RED,
    "PortScan":    YELLOW,
    "Brute Force": YELLOW,
    "Web Attack":  YELLOW,
    "Heartbleed":  RED,
    "Bot":         CYAN,
    "Infiltration":CYAN,
    "BENIGN":      GREEN,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _colour(label: str) -> str:
    return THREAT_COLOUR.get(label, CYAN)


def _format_terminal(ev: DetectionEvent) -> str:
    col   = _colour(ev.prediction)
    arrow = "!!" if ev.is_threat else "  "
    dur_ms = ev.flow_duration_us / 1_000
    rule_tag = f"  [rule: {ev.suricata_alert}]" if ev.suricata_alert else ""
    prob_pct = f"{ev.stage1_attack_prob * 100:.0f}%"

    # UNSW removed (2026) — CIC is the sole ML detector.
    return (
        f"{col}{BOLD}{arrow} {ev.prediction:<14}{RESET}"
        f"  {ev.src_ip:<16} -> {ev.dst_ip}:{ev.dst_port}"
        f"  {ev.proto}/{ev.app_proto or '?'}"
        f"  dur={dur_ms:.1f}ms"
        f"  prob={prob_pct}"
        f"{rule_tag}"
    )


def _to_json_line(ev: DetectionEvent) -> str:
    return json.dumps({
        "timestamp":        ev.timestamp or datetime.now(timezone.utc).isoformat(),
        "src_ip":           ev.src_ip,
        "dst_ip":           ev.dst_ip,
        "dst_port":         ev.dst_port,
        "proto":            ev.proto,
        "app_proto":        ev.app_proto,
        "cic_prediction":   ev.prediction,
        "unsw_prediction":  ev.unsw_prediction,
        "models_agree":     ev.unsw_prediction in (None, ev.prediction),
        "is_threat":        ev.is_threat,
        "cic_attack_prob":  round(ev.stage1_attack_prob, 4),
        "unsw_attack_prob": round(ev.unsw_attack_prob, 4),
        "suricata_rule":    ev.suricata_alert,
        "flow_duration_us": ev.flow_duration_us,
    }, ensure_ascii=False)


def _open_log(path: str) -> object:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("a", encoding="utf-8")


def main() -> None:
    logger.info("Lighthouse live detector starting")
    logger.info("  Model   : %s", MODEL_PATH)
    logger.info("  eve.json: %s", EVE_PATH)
    logger.info("  Log     : %s", LOG_PATH)

    # Handle Ctrl-C gracefully
    def _shutdown(sig: int, _frame: object) -> None:
        print(f"\n{YELLOW}[*] Shutting down detector{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bridge = SuricataBridge(model_path=MODEL_PATH, eve_path=EVE_PATH)
    log_fh = _open_log(LOG_PATH)

    # Print header
    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  Lighthouse Live Detector  |  watching {EVE_PATH}{RESET}")
    print(f"{BOLD}{'─'*72}{RESET}\n")

    threat_count  = 0
    total_count   = 0
    attack_counts: dict[str, int] = {}

    for ev in bridge.stream():
        total_count += 1

        if ev.is_threat:
            threat_count += 1
            attack_counts[ev.prediction] = attack_counts.get(ev.prediction, 0) + 1

        # Terminal: always print threats; print benign only if LOG_BENIGN=true
        if ev.is_threat or LOG_BENIGN:
            print(_format_terminal(ev))

        # File: always log threats; log benign only if LOG_BENIGN=true
        if ev.is_threat or LOG_BENIGN:
            log_fh.write(_to_json_line(ev) + "\n")
            log_fh.flush()

        # Rolling stats every 100 flows
        if total_count % 100 == 0:
            pct = 100 * threat_count / total_count
            summary = "  ".join(f"{k}:{v}" for k, v in sorted(attack_counts.items()))
            logger.info(
                "Flows=%d  Threats=%d (%.1f%%)  %s",
                total_count, threat_count, pct, summary or "none yet",
            )


if __name__ == "__main__":
    main()
