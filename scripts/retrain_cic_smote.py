"""Retrain CIC-IDS-2017 two-stage pipeline on SURICATA-REPRODUCIBLE features.

This is the corrected training script. It fixes three problems from the old one:

  1. Training-serving skew: features now come from detection/flow_features.py —
     the same module the live Suricata bridge uses. No more fabricated IAT/jitter
     features. Train and serve compute identical inputs.
  2. Double imbalance correction: Stage 1 uses scale_pos_weight ALONE (no SMOTE);
     Stage 2 uses SMOTE ALONE (no class_weight). Never both on the same stage.
  3. Honest early stopping: the validation set for early stopping is carved from
     REAL (pre-SMOTE) data, never from synthetic samples.

Stage 1: XGBoost binary  (BENIGN vs Attack), scale_pos_weight for imbalance.
Stage 2: LightGBM multiclass (attack family), SMOTE to lift minority families.

Usage:
    python scripts/retrain_cic_smote.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import CIC_FLOW_FEATURES, cic_features_from_df  # noqa: E402

warnings.filterwarnings("ignore")

CSV_DIR   = Path("data/models/raw/cic 2017/MachineLearningCSV/MachineLearningCVE")
OUT_MODEL = Path("data/models/cic2017_pipeline_smote.joblib")

ATTACK_FAMILIES = ["Bot", "Brute Force", "DDoS", "DoS", "PortScan", "Web Attack"]
FINAL_LABELS    = ["BENIGN"] + ATTACK_FAMILIES

LABEL_MAP = {
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS", "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
    "Web Attack  Brute Force": "Web Attack", "Web Attack  XSS": "Web Attack",
    "Web Attack  Sql Injection": "Web Attack",
}
EXCLUDE_LABELS = {"Heartbleed", "Infiltration"}

BENIGN_CAP       = 300_000
SMOTE_TARGET     = 10_000   # minority family target after SMOTE
SMOTE_MIN_REAL   = 100      # skip SMOTE for families with fewer real rows
EARLY_STOP       = 30


def load_all(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(csv_dir.glob("*.csv")):
        print(f"  {f.name} ...", flush=True)
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()
        for k, v in LABEL_MAP.items():
            df.loc[df["Label"] == k, "Label"] = v
        df.loc[df["Label"].str.contains("Web Attack", na=False), "Label"] = "Web Attack"
        df = df[~df["Label"].isin(EXCLUDE_LABELS)].copy()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Label"] = combined["Label"].str.strip()
    return combined


def main() -> None:
    print("=" * 70)
    print("  CIC-IDS-2017 retrain on Suricata-reproducible features")
    print(f"  Features ({len(CIC_FLOW_FEATURES)}): {CIC_FLOW_FEATURES}")
    print("=" * 70)

    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV dir not found: {CSV_DIR}")

    print("\n=== Loading CSVs ===")
    df = load_all(CSV_DIR)
    print(f"  Total rows: {len(df):,}")
    print(df["Label"].value_counts().to_string())

    # ── Features (shared module — identical to serving) ──
    print("\n=== Computing features (shared flow_features module) ===")
    X = cic_features_from_df(df).values.astype(float)
    labels = df["Label"].values
    y_binary = np.where(labels == "BENIGN", 0, 1).astype(int)

    fam_encoder = LabelEncoder()
    fam_encoder.fit(ATTACK_FAMILIES)
    fam_encoder.classes_ = np.array(ATTACK_FAMILIES)

    attack_mask = y_binary == 1
    safe_attack = [
        lbl if lbl in set(ATTACK_FAMILIES)
        else next((a for a in ATTACK_FAMILIES if a.lower() in str(lbl).lower()), "DDoS")
        for lbl in labels[attack_mask]
    ]
    y_family = np.full(len(labels), -1, dtype=int)
    y_family[attack_mask] = fam_encoder.transform(safe_attack)

    y_combined = np.where(y_binary == 0, 0, y_family + 1).astype(int)

    # ── Stratified split BEFORE any resampling ──
    print("\n=== Stratified 80/20 split (before SMOTE) ===")
    X_tr, X_te, yc_tr, yc_te = train_test_split(
        X, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    yb_tr = np.where(yc_tr == 0, 0, 1)
    yb_te = np.where(yc_te == 0, 0, 1)
    yf_tr = np.where(yc_tr == 0, -1, yc_tr - 1)
    yf_te = np.where(yc_te == 0, -1, yc_te - 1)

    # Downsample BENIGN on TRAIN only
    rng = np.random.default_rng(42)
    benign_idx = np.where(yc_tr == 0)[0]
    attack_idx = np.where(yc_tr != 0)[0]
    if len(benign_idx) > BENIGN_CAP:
        benign_idx = rng.choice(benign_idx, BENIGN_CAP, replace=False)
    keep = np.concatenate([benign_idx, attack_idx]); rng.shuffle(keep)
    X_tr, yb_tr, yf_tr = X_tr[keep], yb_tr[keep], yf_tr[keep]
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # ── Scale (fit on train) ──
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Stage 1: XGBoost binary — scale_pos_weight ONLY, real validation ──
    print("\n=== Stage 1: XGBoost binary (scale_pos_weight, real val) ===")
    n_neg, n_pos = int((yb_tr == 0).sum()), int((yb_tr == 1).sum())
    spw = max(1.0, n_neg / max(n_pos, 1))
    print(f"  BENIGN={n_neg:,}  Attack={n_pos:,}  scale_pos_weight={spw:.2f}")

    X_s1_tr, X_s1_val, y_s1_tr, y_s1_val = train_test_split(
        X_tr_s, yb_tr, test_size=0.1, stratify=yb_tr, random_state=42
    )
    stage1 = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP, tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    stage1.fit(X_s1_tr, y_s1_tr, eval_set=[(X_s1_val, y_s1_val)], verbose=False)
    print(f"  Best iteration: {stage1.best_iteration}")

    # ── Stage 2: LightGBM multiclass — SMOTE ONLY, real validation ──
    print("\n=== Stage 2: LightGBM family (SMOTE, real val) ===")
    att = yb_tr == 1
    X_att, y_att = X_tr_s[att], yf_tr[att]

    # Carve REAL validation BEFORE SMOTE
    X_a_tr, X_a_val, y_a_tr, y_a_val = train_test_split(
        X_att, y_att, test_size=0.1, stratify=y_att, random_state=42
    )
    # SMOTE only the training portion
    counts = dict(zip(*np.unique(y_a_tr, return_counts=True)))
    strat = {c: SMOTE_TARGET for c, n in counts.items() if SMOTE_MIN_REAL <= n < SMOTE_TARGET}
    if strat:
        k = min(5, min(counts[c] for c in strat) - 1)
        X_a_tr, y_a_tr = SMOTE(sampling_strategy=strat, k_neighbors=max(k, 1), random_state=42).fit_resample(X_a_tr, y_a_tr)
        print(f"  SMOTE applied to {len(strat)} families (target {SMOTE_TARGET:,})")

    stage2 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ATTACK_FAMILIES),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    stage2.fit(
        X_a_tr, y_a_tr, eval_set=[(X_a_val, y_a_val)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        feature_name=CIC_FLOW_FEATURES,
    )

    # ── Evaluate on held-out test ──
    print("\n=== Evaluation (held-out 20% test) ===")
    s1 = stage1.predict(X_te_s)
    pred = np.where(s1 == 0, "BENIGN", "Unknown").astype(object)
    att_idx = np.where(s1 == 1)[0]
    if len(att_idx):
        fam = stage2.predict(pd.DataFrame(X_te_s[att_idx], columns=CIC_FLOW_FEATURES))
        for i, fi in zip(att_idx, fam):
            pred[i] = fam_encoder.inverse_transform([int(fi)])[0]

    true_bin = np.where(yb_te == 0, "BENIGN", "ATTACK")
    pred_bin = np.where(pred == "BENIGN", "BENIGN", "ATTACK")
    macro_f1 = f1_score(true_bin, pred_bin, average="macro")
    print(f"  Binary macro-F1: {macro_f1:.4f}")
    print("\n  Per-class detection:")
    classes = list(fam_encoder.classes_)
    for name in FINAL_LABELS:
        if name == "BENIGN":
            mask = yb_te == 0; correct = int((pred[mask] == "BENIGN").sum())
        else:
            ci = classes.index(name); mask = yf_te == ci
            correct = int((pred[mask] == name).sum())
        total = int(mask.sum())
        pct = 100 * correct / total if total else 0.0
        print(f"    {name:<14} {correct:>7}/{total:<7} ({pct:5.1f}%)")

    # ── Save ──
    pipeline = {
        "scaler": scaler, "stage1_model": stage1, "stage2_model": stage2,
        "fam_encoder": fam_encoder, "features": CIC_FLOW_FEATURES,
        "final_labels": FINAL_LABELS,
        "meta": {
            "dataset": "CIC-IDS-2017", "feature_set": "suricata_reproducible",
            "n_features": len(CIC_FLOW_FEATURES), "attack_families": ATTACK_FAMILIES,
            "test_macro_f1": float(macro_f1),
        },
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)
    print(f"\n  Saved {OUT_MODEL} ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
    print("  Done.")


if __name__ == "__main__":
    main()
