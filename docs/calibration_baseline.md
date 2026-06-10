# Rate-Aggregator Threshold Calibration

This file records the empirical basis for the volumetric-detection thresholds in
`detection/rate_aggregator.py`. Per the production-readiness plan, thresholds are
**measured from captured traffic**, not guessed — so each value below should cite
a real baseline/attack capture from this deployment.

## Method

1. **Baseline capture** — record ~10–15 min of *normal* VM1 traffic (browsing, the
   dashboard's own WebSocket, Wazuh agent heartbeats, normal service traffic):
   ```bash
   # On VM1, snapshot a window of normal eve.json
   cp /var/log/suricata/eve.json baseline_eve.json
   ```
2. **Attack capture** — run real attacks from Kali, snapshot separately:
   ```bash
   # Kali:
   sudo nmap -sS -p 1-1024 --min-rate 5000 <VM1_TS_IP>
   sudo hping3 -S -p 80 --flood <VM1_TS_IP>     # ~10s then Ctrl+C
   # VM1:
   cp /var/log/suricata/eve.json attack_eve.json
   ```
3. **Analyze** — compute per-source sliding-window distributions and the
   recommended thresholds (benign p99 → attack margin):
   ```bash
   python scripts/calibrate_thresholds.py --baseline baseline_eve.json --attack attack_eve.json
   ```
4. **Apply** — write the recommended values into `RateAggregator.__init__`
   defaults and record the measured numbers in the table below.

## Threshold rationale

Each rate threshold is placed **above the benign 99th percentile** (so normal
traffic never trips it) and **below the observed attack level** (so attacks
always do). The gap between those two is the separating margin — a real,
data-driven boundary rather than an arbitrary number.

## Current values (defaults — replace with VM1-measured numbers)

| Threshold | Current default | Benign p99 (measured) | Attack level (measured) | Status |
|---|---|---|---|---|
| `flood_flows_per_sec` | 20 | _TBD — run calibrate_thresholds.py_ | _TBD_ | research default, awaiting calibration |
| `portscan_unique_ports` | 15 | _TBD_ | _TBD_ | research default |
| `flood_halfopen_ratio` | 0.5 | n/a (ratio, not rate) | ~1.0 for SYN floods | research default (Snort/Suricata) |
| `dos_bytes_per_sec` | 1,000,000 | _TBD_ | _TBD_ | research default |
| `window_s` | 10.0 | n/a | n/a | matches rate-aggregator window |

> The current defaults are research-informed (Snort sfportscan, Suricata threshold
> rules, SYN-flood entropy/CUSUM literature) and verified to separate synthetic
> attacks from benign in unit tests. The fields above marked _TBD_ should be filled
> from a real VM1 calibration run before production sign-off.

## Decision cutoffs

`pipeline/decision_engine.py` thresholds (validate against the risk-score
distribution from the captures; do **not** tighten beyond what the data supports):

| Action | Risk score | Notes |
|---|---|---|
| `auto_block` | ≥ 81 | critical — confirmed volumetric attack on a known/critical asset or sustained flood |
| `review` | ≥ 61 | suspicious |
| `alert` | ≥ 31 | low-severity |
| `log` | < 31 | dropped from dashboard |

**Validation rule:** after calibration, the benign baseline must produce **zero**
alerts (no benign flow scores ≥ 31), and real hping3/nmap must reliably reach
auto-block when sustained. If benign clusters above 31, raise the cutoff or the
rate thresholds — never lower them in a way that contradicts the measured margin.

## Verification log

_Record each calibration run here: date, baseline/attack flow counts, measured
percentiles, and the thresholds chosen._

- _YYYY-MM-DD — initial VM1 calibration — (fill in)_
