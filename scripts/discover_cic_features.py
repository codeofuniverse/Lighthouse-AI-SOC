"""Explainable-AI feature discovery for CIC-IDS-2017 Web Attack.

The CIC MachineLearningCVE CSVs carry NO HTTP application-layer columns (no URL /
method / status) — only flow statistics. Web Attack therefore has to be separated
from benign HTTP using flow columns alone. Rather than guess which columns help,
this script lets SHAP + LIME *discover* them:

  1. Build the full numeric feature matrix from ALL ~70 CIC flow columns.
  2. Train a LightGBM "Web-Attack vs rest" model (plus a 7-class model) on them.
  3. SHAP (TreeExplainer) ranks every column by its mean|SHAP| contribution toward
     Web Attack, globally and per subtype (Brute Force / XSS / SQLi via LabelFine).
  4. LIME explains representative Web-Attack rows locally.
  5. Emit cic_feature_ranking.json: every column ranked, each tagged whether it is
     reproducible live from a Suricata eve.json flow record.

The reproducible-and-discriminative intersection becomes CIC_FLOW_FEATURES_V2.

Outputs:
    reports/feature_discovery/cic/*.png
    reports/feature_discovery/cic_feature_ranking.json

Usage:
    python scripts/discover_cic_features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

DARK_BG, PANEL_BG, GRID, ACCENT = "#0d1117", "#161b22", "#30363d", "#00ff88"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG, "savefig.facecolor": DARK_BG,
    "text.color": "white", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.edgecolor": GRID,
})

SUBSET = Path("data/models/raw/cic2017_subset_10pct.parquet")
OUT_DIR = Path("reports/feature_discovery/cic")
RANK_JSON = Path("reports/feature_discovery/cic_feature_ranking.json")

# Columns that are NOT model features (identity / target / leakage-prone)
DROP_COLS = {
    "Label", "LabelFine", "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Timestamp", "Fwd Header Length.1",
}

# ── Suricata eve.json reproducibility map ────────────────────────────────────
# A CIC column is "reproducible" only if it derives purely from the primitives a
# live Suricata flow record exposes: pkts_toserver/toclient, bytes_toserver/
# toclient, flow duration, TCP flag booleans, and dest_port. IAT/jitter/active/
# idle/bulk/window/segment internals are NOT in eve.json flow events.
_REPRODUCIBLE = {
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",  # bytes/pkts per dir
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",       # == fwd/bwd pkt mean
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",           # single-subflow == totals
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
}


def _reproducible(col: str) -> bool:
    return col.strip() in _REPRODUCIBLE


def _savefig(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(title, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    saved -> {path}")


def _numeric_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # drop zero-variance columns (uninformative)
    keep = [c for c in feat_cols if X[c].std() > 0]
    return X[keep], keep


def main() -> None:
    print("=" * 70)
    print("  CIC Web-Attack feature discovery (SHAP + LIME over ALL flow columns)")
    print("=" * 70)
    if not SUBSET.exists():
        raise FileNotFoundError(f"{SUBSET} missing — run scripts/sample_proportional.py first")

    df = pd.read_parquet(SUBSET)
    df.columns = df.columns.str.strip()
    X, feat_cols = _numeric_matrix(df)
    print(f"  Rows: {len(df):,}   numeric feature columns: {len(feat_cols)}")

    labels = df["Label"].values
    y_web = (labels == "Web Attack").astype(int)
    print(f"  Web Attack rows: {int(y_web.sum()):,}  (vs {len(y_web) - int(y_web.sum()):,} rest)")

    # ── Web-Attack-vs-rest LightGBM (class_weight, no SMOTE — discovery only) ──
    n_pos = int(y_web.sum())
    spw = (len(y_web) - n_pos) / max(n_pos, 1)
    clf = lgb.LGBMClassifier(
        n_estimators=400, max_depth=8, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbose=-1, scale_pos_weight=spw,
    )
    clf.fit(X.values, y_web, feature_name=feat_cols)

    # ── SHAP global ranking toward Web Attack ──
    print("  SHAP global ranking ...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(4000, len(X)), replace=False)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X.iloc[idx])
    sv_pos = sv[1] if isinstance(sv, list) else sv          # toward class 1 = Web Attack
    mean_abs = np.abs(sv_pos).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(10, 9))
    top = order[:20]
    plt.barh([feat_cols[i] for i in top][::-1], mean_abs[top][::-1], color=ACCENT)
    plt.xlabel("mean |SHAP| toward Web Attack")
    _savefig(OUT_DIR / "shap_webattack_global_top20.png",
             "Web Attack — top-20 discriminative CIC columns (SHAP)")

    # ── Per-subtype SHAP (Brute Force / XSS / SQLi) ──
    print("  SHAP per Web-Attack subtype ...")
    labels_fine = df["LabelFine"].values
    web_mask = labels == "Web Attack"
    ranking_subtypes: dict[str, list] = {}
    if web_mask.sum() > 0:
        sv_web = expl.shap_values(X[web_mask])
        sv_web_pos = sv_web[1] if isinstance(sv_web, list) else sv_web
        for sub in pd.unique(labels_fine[web_mask]):
            rows = labels_fine[web_mask] == sub
            if rows.sum() == 0:
                continue
            sub_mean = np.abs(sv_web_pos[rows]).mean(axis=0)
            sub_order = np.argsort(sub_mean)[::-1][:15]
            ranking_subtypes[str(sub)] = [
                {"feature": feat_cols[i], "mean_abs_shap": float(sub_mean[i]),
                 "reproducible": _reproducible(feat_cols[i])} for i in sub_order
            ]
            plt.figure(figsize=(9, 6))
            plt.barh([feat_cols[i] for i in sub_order][::-1], sub_mean[sub_order][::-1],
                     color="#f093fb")
            plt.xlabel("mean |SHAP| toward Web Attack")
            safe = "".join(c if c.isalnum() else "_" for c in str(sub)).strip("_")
            _savefig(OUT_DIR / f"shap_webattack_subtype_{safe}.png",
                     f"Web Attack subtype: {sub} — top columns (SHAP)")

    # ── LIME on a representative Web-Attack row ──
    print("  LIME local explanation (Web Attack) ...")
    web_idx = np.where(web_mask)[0]
    if len(web_idx):
        lime_expl = LimeTabularExplainer(
            X.values, feature_names=feat_cols, class_names=["rest", "Web Attack"],
            discretize_continuous=True, random_state=42)
        exp = lime_expl.explain_instance(
            X.values[web_idx[0]], clf.predict_proba, num_features=12, labels=[1])
        pairs = exp.as_list(label=1)[::-1]
        names, weights = [p[0] for p in pairs], [p[1] for p in pairs]
        colors = [ACCENT if w >= 0 else "#ff6b6b" for w in weights]
        plt.figure(figsize=(10, 6))
        plt.barh(names, weights, color=colors)
        plt.axvline(0, color=GRID, lw=1)
        plt.xlabel("LIME local weight (green supports Web Attack, red opposes)")
        _savefig(OUT_DIR / "lime_webattack_case.png",
                 "LIME — why this flow is classified Web Attack")

    # ── Emit ranking JSON ──
    global_ranking = [
        {"rank": r + 1, "feature": feat_cols[i], "mean_abs_shap": float(mean_abs[i]),
         "reproducible": _reproducible(feat_cols[i])}
        for r, i in enumerate(order)
    ]
    repro_winners = [g["feature"] for g in global_ranking if g["reproducible"]][:15]
    payload = {
        "n_features_considered": len(feat_cols),
        "global_ranking": global_ranking,
        "subtype_ranking": ranking_subtypes,
        "reproducible_winners_top15": repro_winners,
        "note": "reproducible=True means computable from a live Suricata eve.json flow "
                "record (pkts/bytes per direction, duration, TCP flags, dest_port).",
    }
    RANK_JSON.parent.mkdir(parents=True, exist_ok=True)
    RANK_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump({"model": clf, "features": feat_cols}, OUT_DIR / "discovery_model.joblib")

    print(f"\n  Top reproducible Web-Attack drivers: {repro_winners[:10]}")
    print(f"  Wrote {RANK_JSON}")
    print("  Done.")


if __name__ == "__main__":
    main()
