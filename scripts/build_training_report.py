"""Assemble reports/training_validation_report.md from the artifacts produced by
the k-fold trainers and the explainability script.

Consumes:
    reports/training_validation_report_data/{cic,unsw}_folds.csv
    reports/training_validation_report_data/{cic,unsw}_eval.json
    reports/explainability/{cic,unsw}/*.png

Produces a single markdown report covering BOTH datasets with:
  - proportional-subset + disjointness summary
  - per-fold, per-class P/R/F1/support tables (+ Stage-1 AUC, log-loss)
  - the resampling decision per class (SMOTE vs class-weight) and which params drive it
  - 20% TEST vs DISJOINT VALIDATION metrics, side by side
  - before/after false-positive comparison per attack type (the FP fix)
  - embedded SHAP + LIME figures per attack type, with narrative

Usage:
    python scripts/build_training_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

DATA = Path("reports/training_validation_report_data")
EXPL = Path("reports/explainability")
OUT = Path("reports/training_validation_report.md")
REPORT_ROOT = Path("reports")


def _fold_table(folds: pd.DataFrame, dataset: str) -> str:
    df = folds[folds["dataset"] == dataset]
    lines = []
    for fold in sorted(df["fold"].unique()):
        f = df[df["fold"] == fold]
        auc = f["stage1_auc"].iloc[0]
        ll = f["log_loss"].iloc[0]
        lines.append(f"\n**Fold {fold}** — Stage-1 AUC `{auc:.4f}`, log-loss `{ll:.4f}`\n")
        lines.append("| Class | Precision | Recall | F1 | Support | Resampling |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, r in f.iterrows():
            lines.append(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                         f"{r['f1']:.3f} | {int(r['support']):,} | {r['resampling'] or '-'} |")
    return "\n".join(lines)


def _xfold_summary(folds: pd.DataFrame, dataset: str) -> str:
    df = folds[folds["dataset"] == dataset]
    g = df.groupby("class").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        rec_mean=("recall", "mean"), prec_mean=("precision", "mean"),
        support=("support", "mean"))
    lines = ["| Class | F1 mean±std | Recall mean | Precision mean | Avg support |",
             "|---|---:|---:|---:|---:|"]
    for cls, r in g.iterrows():
        std = 0.0 if pd.isna(r["f1_std"]) else r["f1_std"]
        lines.append(f"| {cls} | {r['f1_mean']:.3f} ± {std:.3f} | {r['rec_mean']:.3f} | "
                     f"{r['prec_mean']:.3f} | {r['support']:.0f} |")
    return "\n".join(lines)


def _fp_table(ev: dict, benign_name: str, families: list[str]) -> str:
    """Before/after FP comparison for both TEST and VALIDATION."""
    lines = ["| Attack family | TEST FP% before | TEST FP% after | VAL FP% before | VAL FP% after |",
             "|---|---:|---:|---:|---:|"]
    for fam in families + ["__ANY__"]:
        tb = ev["test"]["fp_before"][fam]["fp_rate"] * 100
        ta = ev["test"]["fp_after"][fam]["fp_rate"] * 100
        vb = ev["validation"]["fp_before"][fam]["fp_rate"] * 100
        va = ev["validation"]["fp_after"][fam]["fp_rate"] * 100
        name = "**ANY attack**" if fam == "__ANY__" else fam
        lines.append(f"| {name} | {tb:.2f} | {ta:.2f} | {vb:.2f} | {va:.2f} |")
    return "\n".join(lines)


def _rel(p: Path) -> str:
    return p.relative_to(REPORT_ROOT).as_posix()


def _embed_figs(subdir: str, patterns: list[tuple[str, str]],
                root: Path = EXPL) -> str:
    """patterns: list of (glob, caption-prefix). Each PNG is embedded once: the
    first pattern that matches it wins, so later broad globs don't re-list a file
    an earlier specific glob already captured (fixes the duplicate-figure bug)."""
    base = root / subdir
    seen: set[Path] = set()
    out = []
    for glob, caption in patterns:
        for png in sorted(base.glob(glob)):
            if png in seen:
                continue
            seen.add(png)
            out.append(f"\n**{caption}: `{png.stem}`**\n\n![{png.stem}]({_rel(png)})\n")
    return "".join(out)


def _section(dataset_key: str, title: str, benign_name: str, families: list[str],
             fp_note: str) -> str:
    folds = pd.read_csv(DATA / f"{dataset_key}_folds.csv")
    ev = json.loads((DATA / f"{dataset_key}_eval.json").read_text())
    ds = folds["dataset"].iloc[0]

    parts = [f"\n## {title}\n"]
    parts.append(f"- **20% hold-out TEST macro-F1:** `{ev['test']['macro_f1']:.4f}`")
    parts.append(f"- **DISJOINT VALIDATION macro-F1:** `{ev['validation']['macro_f1']:.4f}`  "
                 f"_(data never used in the 80/20 split)_")
    parts.append("\n### Resampling decision per class (the FP fix)\n")
    parts.append("Ultra-rare classes use **class-weighting instead of SMOTE** to avoid the "
                 "synthetic-minority blob that bleeds across the benign region and manufactures "
                 "false positives. Mid-size classes use **capped** SMOTE (≤5× real rows).\n")
    parts.append("| Class | Decision |\n|---|---|")
    for cls, dec in ev["resampling_decisions"].items():
        parts.append(f"| {cls} | `{dec}` |")

    parts.append("\n### False-positive rate — before vs after the fix\n")
    parts.append(fp_note + "\n")
    parts.append(_fp_table(ev, benign_name, families))

    parts.append("\n### Cross-fold summary (mean ± std over 5 folds)\n")
    parts.append(_xfold_summary(folds, ds))

    parts.append("\n### Per-fold breakdown\n")
    parts.append(_fold_table(folds, ds))

    parts.append(f"\n### Explainable AI — SHAP & LIME ({title})\n")
    parts.append("SHAP shows which flow features *globally* shape each attack; LIME explains "
                 "*individual* decisions, including a benign false-positive case study.\n")
    # Order matters: specific globs first so the broad family glob doesn't re-list
    # the stage-1 / subtype / FP figures (de-dup is enforced in _embed_figs).
    parts.append(_embed_figs(dataset_key, [
        (f"shap_{dataset_key}_stage1.png", "SHAP Stage-1 (BENIGN vs Attack)"),
        (f"shap_{dataset_key}_webattack_*.png", "SHAP Web-Attack subtype"),
        (f"shap_{dataset_key}_*.png", "SHAP per attack family"),
        (f"lime_{dataset_key}_benign_false_positive.png", "LIME benign false-positive case study"),
        (f"lime_{dataset_key}_*.png", "LIME local explanation"),
    ]))
    return "\n".join(parts)


FD_DIR = REPORT_ROOT / "feature_discovery"


def _feature_discovery_section() -> str:
    """CIC Web-Attack feature discovery: SHAP ranking + reproducibility tags."""
    rank_path = FD_DIR / "cic_feature_ranking.json"
    if not rank_path.exists():
        return ""
    rank = json.loads(rank_path.read_text())
    parts = ["\n## Feature discovery — why CIC Web Attack was weak\n"]
    parts.append("The CIC `MachineLearningCVE` CSVs carry **no HTTP application-layer columns** "
                 "(no URL/method/status). So Web Attack must be separated from benign HTTP using "
                 "flow columns alone. Rather than guess, SHAP ranked **all "
                 f"{rank['n_features_considered']} numeric flow columns** by their contribution "
                 "toward Web Attack. Top 15:\n")
    parts.append("| Rank | CIC column | mean&#124;SHAP&#124; | Suricata-reproducible? |")
    parts.append("|---:|---|---:|:---:|")
    for g in rank["global_ranking"][:15]:
        repro = "✅ yes" if g["reproducible"] else "❌ no (not in eve.json)"
        parts.append(f"| {g['rank']} | {g['feature']} | {g['mean_abs_shap']:.0f} | {repro} |")
    parts.append("\n**Key finding:** the strongest discriminators are **inter-arrival-time** "
                 "features (`Fwd IAT Max`, `Flow IAT Std`, …) that a live Suricata flow record "
                 "**cannot reproduce**. The best reproducible signal not already in the 17-feature "
                 "set is **`Destination Port`** — it pins a flow to web ports (80/443/8080), the "
                 "context that lets the rate/duration features separate HTTP attacks from benign "
                 "HTTP. So **V2 = the 17 features + `dst_port`**. This is the honest reproducible "
                 "ceiling: without per-packet timing (or PCAPs/HTTP content) Web-Attack recall is "
                 "fundamentally bounded.\n")
    parts.append(_embed_figs("cic", [
        ("shap_webattack_global_top20.png", "SHAP global Web-Attack ranking"),
        ("shap_webattack_subtype_*.png", "SHAP Web-Attack subtype"),
        ("lime_webattack_case.png", "LIME Web-Attack local explanation"),
    ], root=FD_DIR))
    return "\n".join(parts)


def _webattack_beforeafter_section() -> str:
    """v1 (17 feat) vs v2 (+dst_port) Web-Attack recall + benign FP, on disjoint val."""
    v1p, v2p = DATA / "cic_eval_v1.json", DATA / "cic_eval_v2.json"
    if not (v1p.exists() and v2p.exists()):
        return ""
    v1, v2 = json.loads(v1p.read_text()), json.loads(v2p.read_text())
    parts = ["\n## Web Attack — before (17 feat) vs after (+dst_port), disjoint validation\n"]
    parts.append("| Metric | v1 (17 features) | v2 (+ dst_port) |")
    parts.append("|---|---:|---:|")
    parts.append(f"| Web-Attack recall | {v1['validation']['recall']['Web Attack']*100:.2f}% | "
                 f"**{v2['validation']['recall']['Web Attack']*100:.2f}%** |")
    parts.append(f"| Benign→ANY-attack FP rate | "
                 f"{v1['validation']['fp_after']['__ANY__']['fp_rate']*100:.2f}% | "
                 f"**{v2['validation']['fp_after']['__ANY__']['fp_rate']*100:.2f}%** |")
    parts.append(f"| Overall macro-F1 | {v1['validation']['macro_f1']:.4f} | "
                 f"**{v2['validation']['macro_f1']:.4f}** |")
    parts.append("\nAdding the single reproducible `dst_port` feature roughly **tripled Web-Attack "
                 "recall** and **cut the benign false-positive rate ~70%** — the FP drop is the "
                 "bigger win, because web-context lets Stage-1 stop confusing benign HTTP with "
                 "attacks (the source of the DoS-bucket false positives).\n")
    return "\n".join(parts)


def _threshold_section() -> str:
    """Per-model Stage-1 threshold tuning on disjoint validation (report-only)."""
    cic_p, unsw_p = DATA / "cic_thresholds.json", DATA / "unsw_thresholds.json"
    if not (cic_p.exists() and unsw_p.exists()):
        return ""
    cic, unsw = json.loads(cic_p.read_text()), json.loads(unsw_p.read_text())
    parts = ["\n## Threshold tuning — Stage-1 cutoff vs false-positive budget\n"]
    parts.append("**Report-only — serving is unchanged.** The benign FP rate is governed mostly "
                 "by the Stage-1 attack-probability cutoff (default 0.5). Sweeping it on the "
                 f"**disjoint validation** set, we find the operating point that holds benign→"
                 f"ANY-attack FP ≤ **{cic['fp_budget']*100:.0f}%** while maximising recall. The "
                 "chosen cutoffs are stored in each model dict under a `thresholds` key for later "
                 "opt-in use; the live prediction path does not consult them yet.\n")
    parts.append("| Dataset | Chosen cutoff | Benign FP @ cutoff | Attack recall @ cutoff | Met budget |")
    parts.append("|---|---:|---:|---:|:---:|")
    for name, d in [("CIC-IDS-2017 (v2)", cic), ("UNSW-NB15", unsw)]:
        c = d["chosen"]
        parts.append(f"| {name} | {c['cutoff']:.2f} | {c['benign_fp_rate']*100:.2f}% | "
                     f"{c['attack_recall']*100:.2f}% | {'✅' if c['met_budget'] else '❌'} |")
    parts.append("\nThe UNSW row is the important one: raising the Stage-1 cutoff drops UNSW's "
                 "real-world benign FP from **28.7%** (at the default 0.5) to **≤1%**, trading off "
                 "some attack recall — an explicit, tunable operating point rather than a hidden "
                 "default. Tradeoff curves:\n")
    parts.append(_embed_figs("cic", [("thresholds_cic.png", "CIC Stage-1 threshold sweep")]))
    parts.append(_embed_figs("unsw", [("thresholds_unsw.png", "UNSW Stage-1 threshold sweep")]))
    return "\n".join(parts)


def _unsw_discovery_section() -> str:
    """UNSW 28-feature SHAP discovery: which unused columns lift the weak classes."""
    rank_path = FD_DIR / "unsw_feature_ranking.json"
    if not rank_path.exists():
        return ""
    rank = json.loads(rank_path.read_text())
    parts = ["\n## UNSW weak-class fix — SHAP feature discovery (DoS/Recon/Shellcode/Worms)\n"]
    parts.append(f"The 18-feature UNSW model ignored {rank['n_features_considered'] - 18} of the "
                 f"{rank['n_features_considered']} available columns. SHAP per weak class found the "
                 "top discriminators it was missing; adding them (→ **28 features**) is the fix:\n")
    parts.append("| Weak class | Top SHAP drivers it was missing |")
    parts.append("|---|---|")
    for cls in ("DoS", "Reconnaissance", "Shellcode", "Worms"):
        miss = [it["feature"] for it in rank["per_class_ranking"].get(cls, [])
                if not it["currently_used"]][:5]
        parts.append(f"| {cls} | {', '.join(f'`{m}`' for m in miss) or '—'} |")
    parts.append(f"\nFinal added columns: {', '.join(f'`{c}`' for c in rank['new_columns_for_weak_classes'])}.\n")

    up = DATA / "unsw_eval.json"
    if up.exists():
        ev = json.loads(up.read_text())
        parts.append("\n### Weak-class recall — 18-feature baseline vs 28-feature (disjoint validation)\n")
        parts.append("| Class | recall (28-feat) | FP rate (28-feat) |\n|---|---:|---:|")
        rec = ev["validation"].get("recall", {})
        for cls in ("DoS", "Reconnaissance", "Shellcode", "Worms"):
            fp = ev["validation"]["fp_after"].get(cls, {}).get("fp_rate", 0) * 100
            parts.append(f"| {cls} | {rec.get(cls, 0)*100:.1f}% | {fp:.2f}% |")
        parts.append("\nShellcode and Worms jump dramatically (≈95% / ≈89% recall). **Caveat:** the "
                     "28-feature model needs `proto/service/sttl/ct_*` that Suricata flow records "
                     "don't expose — it serves live only via the **Zeek** sensor (below). It is the "
                     "high-accuracy offline/benchmark model until Zeek is deployed.\n")
    parts.append(_embed_figs("unsw", [
        ("shap_unsw_DoS.png", "SHAP DoS"), ("shap_unsw_Reconnaissance.png", "SHAP Reconnaissance"),
        ("shap_unsw_Shellcode.png", "SHAP Shellcode"), ("shap_unsw_Worms.png", "SHAP Worms"),
    ], root=FD_DIR))
    return "\n".join(parts)


def _webattack_v3_section() -> str:
    """Web-Attack recall by feature tier from the PCAP-extracted rich features."""
    p = DATA / "cic_v3.json"
    if not p.exists():
        return ""
    res = json.loads(p.read_text())
    parts = ["\n## Web Attack — breaking the ceiling with the real PCAP (v1→v2→v3)\n"]
    parts.append("Phase 2's ~17% recall was a *feature* limit, not a fundamental one. With the "
                 "supplied Thursday PCAP we extracted the features the CSVs lacked — real HTTP "
                 "request (method/url/suspicious) and inter-arrival timing (IAT) — and trained a "
                 "focused BENIGN-vs-Web-Attack model. Recall/precision by feature tier on a "
                 "disjoint hold-out (2,000 real Web-Attack flows, 21,935 HTTP requests captured):\n")
    parts.append("| Feature tier | Recall | Precision | Benign FP | AUC | Live sensor |")
    parts.append("|---|---:|---:|---:|---:|---|")
    sensors = {"v2-equiv (flow only, Suricata)": "Suricata (today)",
               "v3-http (flow + HTTP, Zeek-servable)": "Zeek http.log",
               "v3-full (+ IAT, offline ceiling)": "Zeek + packet-timing policy"}
    for tier, r in res.items():
        parts.append(f"| {tier} | {r['test_recall']*100:.1f}% | {r['test_precision']*100:.1f}% | "
                     f"{r['benign_fp_rate']*100:.2f}% | {r['test_auc']:.3f} | {sensors.get(tier,'')} |")
    parts.append("\n**The IAT timing features are the precision lever:** flow+HTTP reaches 96.8% "
                 "recall but only ~28% precision (5.7% benign FP); adding the 3 fwd-IAT features "
                 "lifts precision to **97.8%** and crushes benign FP to **0.05%** (AUC ≈ 1.000). "
                 "SHAP on the rich model ranks `fwd_iat_std` / `fwd_iat_max` at the very top — the "
                 "exact features Phase 2 predicted but the CSVs couldn't provide. This is why the "
                 "Zeek sensor (which can emit per-packet timing) is the path to a high-accuracy, "
                 "low-FP Web-Attack detector live.\n")
    parts.append(_embed_figs("cic", [
        ("shap_v3_webattack.png", "SHAP — PCAP-rich Web-Attack drivers (IAT dominates)"),
    ], root=FD_DIR))
    return "\n".join(parts)


def _sensor_gap_section() -> str:
    """The per-feature sensor availability table (Suricata vs Zeek vs Zeek+policy)."""
    parts = ["\n## Sensor gap — why a second sensor (Zeek) is the real fix\n"]
    parts.append("The models are excellent *given the right features*; the bottleneck is that "
                 "**Suricata flow records are feature-poor**. Zeek's conn.log/http.log supply the "
                 "missing primitives, joined to Suricata by `community_id` (like the existing Wazuh "
                 "host+network correlation). Per-feature availability:\n")
    parts.append("| Feature group (model) | Suricata flow | Zeek conn.log | Zeek + custom policy |")
    parts.append("|---|:---:|:---:|:---:|")
    rows = [
        ("core flow: pkts/bytes/dur/rates (CIC+UNSW)", "✅", "✅", "✅"),
        ("dst_port (CIC v2)", "✅", "✅", "✅"),
        ("TCP flags (CIC)", "✅", "≈ via conn_state", "✅"),
        ("proto / service (UNSW)", "❌", "✅", "✅"),
        ("conn_state / sloss,dloss (UNSW)", "❌", "✅ (missed_bytes+history)", "✅"),
        ("ct_* connection counters (UNSW)", "❌", "✅ (sliding window)", "✅"),
        ("sttl/dttl, stcpb/dtcpb (UNSW)", "❌", "❌", "✅ (local.zeek)"),
        ("HTTP method/url/status (CIC Web Attack)", "≈ enrichment only", "✅ http.log", "✅"),
        ("inter-arrival timing / jitter (CIC+UNSW)", "❌", "❌", "✅ (packet policy)"),
    ]
    for r in rows:
        parts.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    parts.append("\n**Deployment tiers:** *Suricata-only* serves CIC v2 (18-feat) fully and UNSW "
                 "core only; *+Zeek default* unlocks ~24/28 UNSW features and real Web-Attack HTTP; "
                 "*+Zeek custom policy* ([infra/zeek/local.zeek](infra/zeek/local.zeek)) completes "
                 "the full 28 UNSW features and the IAT timing that gives Web Attack its precision. "
                 "The Zeek bridge ([detection/zeek_bridge.py](detection/zeek_bridge.py)) reuses the "
                 "same `compute_*` arithmetic — verified skew-free by "
                 "[tests/test_flow_features_skew.py](tests/test_flow_features_skew.py).\n")
    return "\n".join(parts)


def main() -> None:
    if not (DATA / "cic_eval.json").exists():
        raise FileNotFoundError("Run scripts/train_kfold_cic.py and train_kfold_unsw.py first.")

    header = [
        "# CIC-IDS-2017 + UNSW-NB15 — Training & Validation Report",
        "",
        "Stratified 5-fold training on a **proportional 10% subset** with a rare-class "
        "floor, evaluated on a **disjoint validation set** (data never used in the 80/20 "
        "split), with a class-weight-instead-of-SMOTE fix for the false-positive problem "
        "and full **SHAP + LIME** explainability per attack type.",
        "",
        "## How the data was built",
        "",
        "- **Proportional subset:** each class drawn at ~10% of its full count, EXCEPT classes "
        "below the 2,000-row floor (CIC Bot; UNSW Worms/Shellcode) which are kept near-full so "
        "5-fold CV and SHAP/LIME stay statistically valid. The 80/20 train/test split and "
        "per-fold sizes therefore scale proportionally, exactly as they would on the full "
        "~2.8M-row dataset.",
        "- **Disjoint validation:** CIC validation rows are drawn from the complement pool "
        "(rows not in the subset); UNSW validation is the official testing-set partition. In "
        "both cases validation never overlaps the data used to fit/tune the model — so the "
        "validation metrics are an honest out-of-sample read.",
        "- **No training-serving skew (CIC):** features come from the shared "
        "`detection/flow_features.py` module used by the live Suricata bridge.",
        "",
        "## The false-positive fix",
        "",
        "The old pipeline SMOTE'd every minority family to a flat 10,000 synthetic rows. For "
        "CIC Web Attack (as few as 21 real SQLi rows) and UNSW Worms (174 rows) this "
        "synthesises a dense blob that overlaps benign traffic, pushing the decision boundary "
        "into the benign region and manufacturing false positives. The fix:",
        "",
        "1. classes with **< 500 real rows** → **no SMOTE**, use inverse-frequency "
        "**class weights**;",
        "2. classes in **[500, target)** → **capped** SMOTE at `min(target, 5× real)`;",
        "3. honest early-stopping validation always carved from **real** rows.",
    ]

    cic = _section(
        "cic", "CIC-IDS-2017", "BENIGN",
        ["Bot", "Brute Force", "DDoS", "DoS", "PortScan", "Web Attack"],
        "FP% = share of benign flows predicted as that attack family. The headline 'huge "
        "false positives' is the **ANY attack** row against benign volume; the per-family "
        "rows isolate Web Attack.")
    unsw = _section(
        "unsw", "UNSW-NB15", "Normal",
        ["Generic", "Exploits", "DoS", "Reconnaissance", "Shellcode", "Worms"],
        "FP% = share of Normal flows predicted as that category. Note the large TEST↔"
        "VALIDATION gap — the disjoint validation set exposes optimism the in-distribution "
        "test hides. Worms/Shellcode/Reconnaissance FP rates drop after the fix.")

    discovery = _feature_discovery_section()
    webattack = _webattack_beforeafter_section()
    webattack_v3 = _webattack_v3_section()
    thresholds = _threshold_section()
    unsw_discovery = _unsw_discovery_section()
    sensor_gap = _sensor_gap_section()

    body = ("\n".join(header) + "\n" + discovery + webattack + webattack_v3
            + sensor_gap + unsw_discovery + thresholds + cic + "\n" + unsw + "\n")
    OUT.write_text(body, encoding="utf-8")
    print(f"  Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print("  Done.")


if __name__ == "__main__":
    main()
