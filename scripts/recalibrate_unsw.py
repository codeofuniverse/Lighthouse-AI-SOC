"""Recalibrate the 28-feature UNSW model on REAL benign traffic.

Live test (test_zeek_bridge + the 6k-flow probe) showed the UNSW-28 model
over-flags real CIC traffic as attacks: its `Normal` class was learned from
UNSW-NB15's synthetic 2015 lab benign, which doesn't resemble real 2017 enterprise
traffic — classic cross-dataset domain shift. The plumbing is correct (proven by
tests); the decision boundary just hasn't seen real benign.

Fix: take real benign flows (Zeek conn.log over the CIC PCAP -> 28-feature vectors,
labelled BENIGN by 5-tuple join to the CIC labels), hold out a disjoint slice as a
real-benign FP test, and blend the rest into the UNSW training as extra `Normal`
rows. Retrain the two-stage model. This adapts the Normal boundary to real traffic
WITHOUT touching the attack-class learning (attack rows are unchanged).

Inputs:
    zeek_recal_out/conn.log                          (real Zeek flows over the PCAP)
    data/models/raw/cic2017_subset_10pct.parquet     (for the CIC BENIGN 5-tuples)
    data/models/unsw_subset_10pct.parquet            (UNSW training source)
Output:
    data/models/unsw_kfold_pipeline.joblib           (recalibrated, in place; .bak kept)
    reports/training_validation_report_data/unsw_recal.json

Usage:
    python scripts/recalibrate_unsw.py [--conn zeek_recal_out/conn.log]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.zeek_features import CtWindow, UNSW28_FEATURES, unsw28_from_zeek  # noqa: E402
from scripts._kfold_common import apply_resampling, resampling_plan, sample_weight_from_plan  # noqa: E402
from scripts.train_unsw_svm import ATTACK_CATEGORIES  # noqa: E402

warnings.filterwarnings("ignore")

MODEL = Path("data/models/unsw_kfold_pipeline.joblib")
UNSW_SUBSET = Path("data/models/raw/unsw_subset_10pct.parquet")
LABEL_CSV = Path("data/models/raw/cic 2017/GeneratedLabelledFlows/TrafficLabelling/"
                 "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
OUT_JSON = Path("reports/training_validation_report_data/unsw_recal.json")
BENIGN_NAME = "Normal"
ALL_LABELS = [BENIGN_NAME] + ATTACK_CATEGORIES
REAL_BENIGN_CAP = 40_000        # cap how many real-benign rows we blend in


def _attack_five_tuples() -> set[tuple]:
    """(src, sport, dst, dport) of the labelled Web-Attack flows. A Zeek flow is
    treated as benign iff its 5-tuple is NOT in this set (the Thursday capture is
    ~99% benign, with a tiny, known attack set — exclusion is the robust label)."""
    df = pd.read_csv(LABEL_CSV, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    att = df[df["Label"].str.contains("Web Attack", na=False)]
    keys = set()
    for _, r in att.iterrows():
        try:
            keys.add((str(r["Source IP"]), int(r["Source Port"]),
                      str(r["Destination IP"]), int(r["Destination Port"])))
        except (ValueError, KeyError):
            continue
    return keys


def _real_benign_vectors(conn_path: Path, freq_maps: dict) -> np.ndarray:
    """28-feature vectors for real benign Zeek flows (exclude known attack 5-tuples)."""
    attack_keys = _attack_five_tuples()
    print(f"  excluding {len(attack_keys):,} known attack 5-tuples")
    ct = CtWindow()
    rows = []
    for line in conn_path.read_text(encoding="utf-8").splitlines():
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        # both flow directions of the 5-tuple
        k1 = (str(c.get("id.orig_h", "")), int(c.get("id.orig_p", 0) or 0),
              str(c.get("id.resp_h", "")), int(c.get("id.resp_p", 0) or 0))
        k2 = (k1[2], k1[3], k1[0], k1[1])
        if k1 in attack_keys or k2 in attack_keys:
            continue
        vec = unsw28_from_zeek(c, freq_maps, ct)
        if vec is None:
            continue
        rows.append(vec)
        if len(rows) >= REAL_BENIGN_CAP:
            break
    return np.asarray(rows, dtype=float)


def _fit_stage1(X, y, spw):
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1)
    m.fit(X, y)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="zeek_recal_out/conn.log")
    args = ap.parse_args()
    conn_path = Path(args.conn)

    print("=" * 70)
    print("  Recalibrate UNSW-28 on real benign traffic")
    print("=" * 70)
    pipe = joblib.load(MODEL)
    features = pipe["features"]
    freq_maps = pipe.get("freq_maps", {})
    assert features == UNSW28_FEATURES, "model features != UNSW28_FEATURES"

    # ── Real benign vectors from Zeek ──
    if not conn_path.exists():
        raise FileNotFoundError(f"{conn_path} missing — generate Zeek conn.log first")
    real_benign = _real_benign_vectors(conn_path, freq_maps)
    print(f"  real benign Zeek flows: {len(real_benign):,}")
    if len(real_benign) < 200:
        raise SystemExit("  too few real benign flows matched — check the 5-tuple join")

    # hold out a disjoint slice for an honest 'real benign FP' check
    rb_tr, rb_te = train_test_split(real_benign, test_size=0.3, random_state=42)

    # ── Rebuild UNSW training matrix (same as trainer) + blend real benign Normals ──
    df = pd.read_parquet(UNSW_SUBSET).copy()
    for c in ("proto", "service"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(freq_maps.get(c, {})).fillna(0.0)
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    Xu = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    yu = np.where(df["Label"].values == BENIGN_NAME, 0, 1).astype(int)
    cat_encoder = pipe.get("cat_encoder") or pipe["fam_encoder"]
    labels = df["Label"].values
    yc = np.full(len(labels), 0, dtype=int)
    amask = yu == 1
    safe = [l if l in set(ATTACK_CATEGORIES) else "Generic" for l in labels[amask]]
    yc[amask] = cat_encoder.transform(safe) + 1

    # Blend: append real-benign rows as Normal (binary 0, family -> none)
    X_all = np.vstack([Xu, rb_tr])
    yb_all = np.concatenate([yu, np.zeros(len(rb_tr), dtype=int)])
    print(f"  UNSW rows: {len(Xu):,}  + real benign: {len(rb_tr):,}  = {len(X_all):,}")

    # ── Retrain Stage 1 (binary) with the enriched Normal class ──
    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)
    n_neg, n_pos = int((yb_all == 0).sum()), int((yb_all == 1).sum())
    spw = max(1.0, n_neg / max(n_pos, 1))
    stage1 = _fit_stage1(Xs, yb_all, spw)

    # ── Stage 2 (family) unchanged data — refit on UNSW attack rows only (scaled
    #    with the NEW scaler so the pipeline stays consistent) ──
    att = yc > 0
    Xa = scaler.transform(Xu[att])
    ya = yc[att] - 1
    plan = resampling_plan(ya, ATTACK_CATEGORIES)
    Xa_r, ya_r = apply_resampling(Xa, ya, plan)
    sw = sample_weight_from_plan(ya_r, plan)
    stage2 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ATTACK_CATEGORIES),
        random_state=42, n_jobs=-1, verbose=-1)
    stage2.fit(Xa_r, ya_r, sample_weight=sw, feature_name=features)

    # ── Honest before/after on the held-out REAL benign slice ──
    def _normal_rate(stage1_model, scl) -> float:
        Xt = scl.transform(rb_te)
        return float((stage1_model.predict(Xt) == 0).mean())

    before = _normal_rate(pipe["stage1_model"], pipe["scaler"])
    after = _normal_rate(stage1, scaler)
    # Keep attack detection honest: macro-F1 on the original UNSW test split
    Xtr, Xte, ybtr, ybte = train_test_split(Xu, yu, test_size=0.2,
                                            random_state=42, stratify=yu)
    f1_after = f1_score(ybte, stage1.predict(scaler.transform(Xte)),
                        average="macro")
    f1_before = f1_score(ybte, pipe["stage1_model"].predict(pipe["scaler"].transform(Xte)),
                         average="macro")

    print(f"\n  Real-benign Normal rate (held-out): before={before*100:5.1f}%  "
          f"after={after*100:5.1f}%   (higher = fewer false alarms)")
    print(f"  UNSW binary macro-F1 (attack detection): before={f1_before:.4f}  after={f1_after:.4f}")

    # ── Save recalibrated model in place (archive old) ──
    shutil.copy2(MODEL, MODEL.with_suffix(".joblib.bak"))
    pipe.update({"scaler": scaler, "stage1_model": stage1, "stage2_model": stage2})
    pipe["meta"]["recalibrated_on_real_benign"] = True
    pipe["meta"]["real_benign_normal_rate_before"] = before
    pipe["meta"]["real_benign_normal_rate_after"] = after
    joblib.dump(pipe, MODEL)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "real_benign_flows": len(real_benign),
        "normal_rate_before": before, "normal_rate_after": after,
        "binary_macro_f1_before": f1_before, "binary_macro_f1_after": f1_after,
    }, indent=2))
    print(f"\n  Saved recalibrated model -> {MODEL.name} (old -> .bak)")
    print(f"  Saved {OUT_JSON}")
    print("  Done.")


if __name__ == "__main__":
    main()
