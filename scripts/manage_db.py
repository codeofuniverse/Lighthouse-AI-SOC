"""Lighthouse SQLite database management utility.

Commands:
  status    — show DB file size, row count, and breakdown by attack_type / threat_level
  search    — query alerts with optional filters (mirrors /api/alerts/search)
  vacuum    — reclaim disk space after bulk deletes
  prune     — manually delete oldest rows, keeping last N
  export    — dump all alerts to a JSON file
  clear     — delete ALL alerts (irreversible, prompts for confirmation)

Usage:
  python scripts/manage_db.py status
  python scripts/manage_db.py search --attack_type DDoS --limit 10
  python scripts/manage_db.py search --src_ip 192.168.1.5
  python scripts/manage_db.py search --threat_level 2 --since 2026-05-01
  python scripts/manage_db.py export --out alerts_backup.json
  python scripts/manage_db.py prune --keep 5000
  python scripts/manage_db.py vacuum
  python scripts/manage_db.py clear
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("data/lighthouse_alerts.db")


def _conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        print("  Start the backend at least once to create it.")
        sys.exit(1)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status(_args: argparse.Namespace) -> None:
    size_kb = DB_PATH.stat().st_size / 1024
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        print(f"\nLighthouse SQLite — {DB_PATH}")
        print(f"  File size : {size_kb:.1f} KB")
        print(f"  Total rows: {total:,}")

        print("\n  By attack_type:")
        for row in c.execute(
            "SELECT attack_type, COUNT(*) as n FROM alerts GROUP BY attack_type ORDER BY n DESC"
        ).fetchall():
            print(f"    {row['attack_type']:<20} {row['n']:>6,}")

        print("\n  By threat_level:")
        labels = {0: "Unknown", 1: "Suspicious", 2: "Critical"}
        for row in c.execute(
            "SELECT threat_level, COUNT(*) as n FROM alerts GROUP BY threat_level ORDER BY threat_level DESC"
        ).fetchall():
            label = labels.get(row["threat_level"], str(row["threat_level"]))
            print(f"    {label:<12} ({row['threat_level']})  {row['n']:>6,}")

        print("\n  By status:")
        for row in c.execute(
            "SELECT status, COUNT(*) as n FROM alerts GROUP BY status ORDER BY n DESC"
        ).fetchall():
            print(f"    {row['status']:<15} {row['n']:>6,}")

        oldest = c.execute("SELECT MIN(timestamp) FROM alerts").fetchone()[0]
        newest = c.execute("SELECT MAX(timestamp) FROM alerts").fetchone()[0]
        print(f"\n  Oldest alert: {oldest}")
        print(f"  Newest alert: {newest}")


def cmd_search(args: argparse.Namespace) -> None:
    clauses: list[str] = []
    params:  list      = []

    if args.src_ip:
        clauses.append("src_ip = ?");        params.append(args.src_ip)
    if args.attack_type:
        clauses.append("attack_type = ?");   params.append(args.attack_type)
    if args.threat_level is not None:
        clauses.append("threat_level = ?");  params.append(args.threat_level)
    if args.status:
        clauses.append("status = ?");        params.append(args.status)
    if args.since:
        clauses.append("timestamp >= ?");    params.append(args.since)
    if args.auto_blocked:
        clauses.append("auto_blocked = 1")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT data FROM alerts {where} ORDER BY timestamp DESC LIMIT ?"
    params.append(args.limit)

    with _conn() as c:
        rows = c.execute(sql, params).fetchall()

    if not rows:
        print("No alerts matched.")
        return

    print(f"\nFound {len(rows)} alert(s):\n")
    for r in rows:
        a = json.loads(r["data"])
        blocked = " [BLOCKED]" if a.get("auto_blocked") else ""
        print(f"  {a.get('timestamp','')[:19]}  "
              f"{'★' * a.get('threat_level', 0):<3}  "
              f"{a.get('attack_type','?'):<15}  "
              f"{a.get('src_ip','?'):<18}  "
              f"risk={a.get('risk_score', 0):>5.1f}  "
              f"conf={a.get('confidence', 0):.2f}"
              f"{blocked}")
        if hasattr(args, "verbose") and args.verbose:
            print(f"     ai: {a.get('ai_explanation','')[:120]}")


def cmd_export(args: argparse.Namespace) -> None:
    out = Path(args.out)
    with _conn() as c:
        rows = c.execute("SELECT data FROM alerts ORDER BY timestamp DESC").fetchall()
    alerts = [json.loads(r["data"]) for r in rows]
    out.write_text(json.dumps(alerts, indent=2))
    print(f"Exported {len(alerts):,} alerts to {out}  ({out.stat().st_size / 1024:.1f} KB)")


def cmd_prune(args: argparse.Namespace) -> None:
    keep = args.keep
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        excess = total - keep
        if excess <= 0:
            print(f"Nothing to prune — DB has {total:,} rows, keep={keep:,}")
            return
        c.execute("""
            DELETE FROM alerts WHERE id IN (
                SELECT id FROM alerts ORDER BY timestamp ASC LIMIT ?
            )
        """, (excess,))
        c.commit()
    print(f"Pruned {excess:,} oldest rows. DB now has {keep:,} rows.")


def cmd_vacuum(_args: argparse.Namespace) -> None:
    before = DB_PATH.stat().st_size / 1024
    with _conn() as c:
        c.execute("VACUUM")
    after = DB_PATH.stat().st_size / 1024
    print(f"VACUUM complete. {before:.1f} KB -> {after:.1f} KB (saved {before - after:.1f} KB)")


def cmd_clear(_args: argparse.Namespace) -> None:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    confirm = input(f"Delete ALL {total:,} alerts? This cannot be undone. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return
    with _conn() as c:
        c.execute("DELETE FROM alerts")
        c.commit()
    print(f"Deleted {total:,} alerts.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lighthouse SQLite database management utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status",  help="Show DB size, row counts, and breakdowns")
    sub.add_parser("vacuum",  help="Reclaim disk space (run after bulk deletes)")
    sub.add_parser("clear",   help="Delete ALL alerts (irreversible)")

    p_prune = sub.add_parser("prune", help="Delete oldest rows, keep last N")
    p_prune.add_argument("--keep", type=int, default=5000, help="Rows to keep (default 5000)")

    p_export = sub.add_parser("export", help="Dump all alerts to JSON")
    p_export.add_argument("--out", default="alerts_export.json", help="Output file path")

    p_search = sub.add_parser("search", help="Query alerts with optional filters")
    p_search.add_argument("--src_ip",       help="Filter by source IP")
    p_search.add_argument("--attack_type",  help="Filter by attack type (DDoS, DoS, ...)")
    p_search.add_argument("--threat_level", type=int, choices=[0, 1, 2])
    p_search.add_argument("--status",       choices=["active", "dismissed", "isolated"])
    p_search.add_argument("--since",        help="ISO timestamp lower bound (e.g. 2026-05-01)")
    p_search.add_argument("--auto_blocked", action="store_true", help="Only blocked alerts")
    p_search.add_argument("--limit",        type=int, default=20)
    p_search.add_argument("-v", "--verbose", action="store_true", help="Show AI explanation")

    args = parser.parse_args()
    {
        "status": cmd_status,
        "search": cmd_search,
        "export": cmd_export,
        "prune":  cmd_prune,
        "vacuum": cmd_vacuum,
        "clear":  cmd_clear,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
