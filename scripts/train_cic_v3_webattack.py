"""CIC Web-Attack v3 — break the flow-only recall ceiling using PCAP-extracted
HTTP + timing features.

Phase 2 showed flow-only Web-Attack recall tops out ~17% (the real discriminators
are HTTP content + inter-arrival timing, absent from the CSVs). With the user's
PCAP we now have those features (scripts/extract_pcap_features.py ->
cic2017_pcap_rich.parquet). This trains a focused BENIGN-vs-Web-Attack model and
reports recall under three feature tiers so the live-reproducibility tradeoff is
explicit:

  v2-equiv  : the 18 flow features only (servable today via Suricata)         -> the ceiling
  v3-http   : 18 flow + HTTP method/url_len/suspicious (servable via Zeek http.log)
  v3-full   : v3-http + fwd IAT mean/std/max (offline / needs per-packet timing)

5-fold stratified CV + a disjoint hold-out (60/20/20 by flow), class-weighted (no
SMOTE — Web Attack is small and SMOTE is what caused the FP blob in the first place).
Writes recall/FP per tier to reports/training_validation_report_data/cic_v3.json
and a model per tier (the http tier is the one that can actually ship via Zeek).

Usage:
    python scripts/train_cic_v3_webattack.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore")

RICH = Path("data/models/raw/cic2017_pcap_rich.parquet")
OUT_DATA = Path("reports/training_validation_report_data")
OUT_MODEL_DIR = Path("data/models")

FLOW_18 = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
           "fwd_pkt_mean", "bwd_pkt_mean", "flow_bytes_s", "flow_pkts_s",
           "down_up_ratio", "avg_pkt_size", "fin_flag", "syn_flag", "rst_flag",
           "psh_flag", "ack_flag", "urg_flag", "dst_port"]
HTTP_FEATS = ["url_len", "url_suspicious", "http_get", "http_post"]
IAT_FEATS = ["fwd_iat_mean", "fwd_iat_std", "fwd_iat_max"]

TIERS = {
    "v2-equiv (flow only, Suricata)": FLOW_18,
    "v3-http (flow + HTTP, Zeek-servable)": FLOW_18 + HTTP_FEATS,
    "v3-full (+ IAT, offline ceiling)": FLOW_18 + HTTP_FEATS + IAT_FEATS,
}


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    total_pkts = (d["fwd_pkts"] + d["bwd_pkts"]).clip(lower=1)
    total_bytes = d["fwd_bytes"] + d["bwd_bytes"]
    dur = d["duration_s"].clip(lower=1e-6)
    d["fwd_pkt_mean"] = d["fwd_bytes"] / d["fwd_pkts"].clip(lower=1)
    d["bwd_pkt_mean"] = d["bwd_bytes"] / d["bwd_pkts"].clip(lower=1)
    d["flow_bytes_s"] = total_bytes / dur
    d["flow_pkts_s"] = total_pkts / dur
    d["down_up_ratio"] = d["bwd_pkts"] / d["fwd_pkts"].clip(lower=1)
    d["avg_pkt_size"] = total_bytes / total_pkts
    d["dst_port"] = d["Destination Port"].astype(float)
    m = d["http_method"].fillna("").astype(str)
    d["http_get"] = (m == "GET").astype(float)
    d["http_post"] = (m == "POST").astype(float)
    return d.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _evaluate(df: pd.DataFrame, feats: list[str], y: np.ndarray) -> dict:
    X = df[feats].values.astype(float)
    # 60/20/20 -> here train 80 / disjoint test 20, 5-fold inside train
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_recalls = []
    for tr, va in skf.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[tr])
        clf = lgb.LGBMClassifier(n_estimators=300, max_depth=8, num_leaves=63,
                                 learning_rate=0.05, class_weight="balanced",
                                 random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(sc.transform(Xtr[tr]), ytr[tr])
        pred = clf.predict(sc.transform(Xtr[va]))
        fold_recalls.append(recall_score(ytr[va], pred, pos_label=1, zero_division=0))
    # final on disjoint test
    sc = StandardScaler().fit(Xtr)
    clf = lgb.LGBMClassifier(n_estimators=400, max_depth=8, num_leaves=63,
                             learning_rate=0.05, class_weight="balanced",
                             random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]
    pred = (proba >= 0.5).astype(int)
    benign = yte == 0
    return {
        "cv_recall_mean": float(np.mean(fold_recalls)),
        "test_recall": float(recall_score(yte, pred, pos_label=1, zero_division=0)),
        "test_precision": float(precision_score(yte, pred, pos_label=1, zero_division=0)),
        "test_f1": float(f1_score(yte, pred, pos_label=1, zero_division=0)),
        "test_auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else 0.0,
        "benign_fp_rate": float((pred[benign] == 1).mean()) if benign.any() else 0.0,
        "n_webattack": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
    }, (sc, clf)


def main() -> None:
    print("=" * 70)
    print("  CIC Web-Attack v3 — PCAP HTTP+timing features, recall by tier")
    print("=" * 70)
    if not RICH.exists():
        raise FileNotFoundError(f"{RICH} missing — run scripts/extract_pcap_features.py --web-only")

    df = _derive(pd.read_parquet(RICH))
    y = (df["Label"] == "Web Attack").astype(int).values
    print(f"  flows: {len(df):,}   Web Attack: {int(y.sum()):,}   BENIGN: {int((y==0).sum()):,}")
    if y.sum() < 20:
        print("  [WARN] too few Web-Attack flows matched — check the 5-tuple label join.")

    results = {}
    servable_model = None
    for tier, feats in TIERS.items():
        present = [f for f in feats if f in df.columns]
        res, model = _evaluate(df, present, y)
        results[tier] = res
        print(f"\n  [{tier}]  ({len(present)} feats)")
        print(f"    CV recall={res['cv_recall_mean']*100:5.1f}%  test recall={res['test_recall']*100:5.1f}%"
              f"  precision={res['test_precision']*100:5.1f}%  AUC={res['test_auc']:.3f}"
              f"  benign FP={res['benign_fp_rate']*100:.2f}%")
        if tier.startswith("v3-http"):
            servable_model = (model, present)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "cic_v3.json").write_text(json.dumps(results, indent=2))
    if servable_model:
        (sc, clf), feats = servable_model
        joblib.dump({"scaler": sc, "model": clf, "features": feats,
                     "label": "BENIGN-vs-WebAttack", "tier": "v3-http (Zeek-servable)"},
                    OUT_MODEL_DIR / "cic2017_webattack_v3_http.joblib")
        print(f"\n  Saved servable HTTP-tier model -> {OUT_MODEL_DIR/'cic2017_webattack_v3_http.joblib'}")
    print(f"  Saved tier metrics -> {OUT_DATA/'cic_v3.json'}")
    print("  Done.")


if __name__ == "__main__":
    main()
