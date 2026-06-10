"""Train UNSW-NB15 two-stage pipeline on SURICATA-REPRODUCIBLE features.

Corrected script — same three fixes as the CIC retrain:
  1. Features from detection/flow_features.py (shared with serving) — no skew.
  2. Stage 1 uses scale_pos_weight only; Stage 2 uses SMOTE only.
  3. Early-stopping validation is real (pre-SMOTE) data.

Native UNSW categories (no forced CIC mapping). Three noisy low-precision
classes are merged into semantic parents:
  Fuzzers -> Generic,  Analysis -> Reconnaissance,  Backdoor -> Exploits

Usage:
    python scripts/train_unsw_nb15.py
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.flow_features import UNSW_FLOW_FEATURES, unsw_features_from_df  # noqa: E402

warnings.filterwarnings("ignore")

RAW_DIR   = Path("data/models/raw/unsbw15")
TRAIN_CSV = RAW_DIR / "Training and Testing Sets/UNSW_NB15_training-set.csv"
TEST_CSV  = RAW_DIR / "Training and Testing Sets/UNSW_NB15_testing-set.csv"
OUT_MODEL = Path("data/models/unsw_nb15_pipeline.joblib")

ATTACK_CATEGORIES = ["Generic", "Exploits", "DoS", "Reconnaissance", "Shellcode", "Worms"]
CATEGORY_MERGE = {"Fuzzers": "Generic", "Analysis": "Reconnaissance", "Backdoor": "Exploits"}
_RAW_CATEGORIES = {"Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                   "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"}

BENIGN_CAP     = 80_000
SMOTE_TARGET   = 8_000
SMOTE_MIN_REAL = 100
MAX_MULT       = 5      # never synthesise more than 5x real samples
EARLY_STOP     = 30


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    df.columns = df.columns.str.strip()
    if "attack_cat" in df.columns:
        df = df.rename(columns={"attack_cat": "Label"})
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: x if x in _RAW_CATEGORIES else "Generic")
    df["Label"] = df["Label"].replace(CATEGORY_MERGE)
    return df


def main() -> None:
    print("=" * 70)
    print("  UNSW-NB15 retrain on Suricata-reproducible features")
    print(f"  Features ({len(UNSW_FLOW_FEATURES)}): {UNSW_FLOW_FEATURES}")
    print("=" * 70)

    for p in (TRAIN_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    df_tr, df_te = _load(TRAIN_CSV), _load(TEST_CSV)
    print(f"  Train rows: {len(df_tr):,}  Test rows: {len(df_te):,}")
    print(df_tr["Label"].value_counts().to_string())

    cat_encoder = LabelEncoder()
    cat_encoder.fit(ATTACK_CATEGORIES)
    cat_encoder.classes_ = np.array(ATTACK_CATEGORIES)

    def prep(df):
        X = unsw_features_from_df(df).values.astype(float)
        yb = np.where(df["Label"].values == "Normal", 0, 1).astype(int)
        yc = np.full(len(df), -1, dtype=int)
        m = yb == 1
        safe = [l if l in set(ATTACK_CATEGORIES) else "Generic" for l in df["Label"].values[m]]
        yc[m] = cat_encoder.transform(safe)
        return X, yb, yc

    X_tr, yb_tr, yc_tr = prep(df_tr)
    X_te, yb_te, yc_te = prep(df_te)

    # Downsample Normal on train
    rng = np.random.default_rng(42)
    nidx = np.where(yb_tr == 0)[0]; aidx = np.where(yb_tr == 1)[0]
    if len(nidx) > BENIGN_CAP:
        nidx = rng.choice(nidx, BENIGN_CAP, replace=False)
    keep = np.concatenate([nidx, aidx]); rng.shuffle(keep)
    X_tr, yb_tr, yc_tr = X_tr[keep], yb_tr[keep], yc_tr[keep]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Stage 1: XGBoost binary — scale_pos_weight only, real val ──
    print("\n=== Stage 1: XGBoost binary ===")
    n_neg, n_pos = int((yb_tr == 0).sum()), int((yb_tr == 1).sum())
    spw = max(1.0, n_neg / max(n_pos, 1))
    print(f"  Normal={n_neg:,}  Attack={n_pos:,}  scale_pos_weight={spw:.2f}")
    X1t, X1v, y1t, y1v = train_test_split(X_tr_s, yb_tr, test_size=0.1, stratify=yb_tr, random_state=42)
    stage1 = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=EARLY_STOP, tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    stage1.fit(X1t, y1t, eval_set=[(X1v, y1v)], verbose=False)
    print(f"  Best iteration: {stage1.best_iteration}")

    # ── Stage 2: LightGBM multiclass — SMOTE only, real val ──
    print("\n=== Stage 2: LightGBM category ===")
    att = yb_tr == 1
    X_att, y_att = X_tr_s[att], yc_tr[att]
    X2t, X2v, y2t, y2v = train_test_split(X_att, y_att, test_size=0.1, stratify=y_att, random_state=42)
    counts = dict(zip(*np.unique(y2t, return_counts=True)))
    strat = {c: min(SMOTE_TARGET, n * MAX_MULT) for c, n in counts.items()
             if SMOTE_MIN_REAL <= n < min(SMOTE_TARGET, n * MAX_MULT)}
    if strat:
        k = min(5, min(counts[c] for c in strat) - 1)
        X2t, y2t = SMOTE(sampling_strategy=strat, k_neighbors=max(k, 1), random_state=42).fit_resample(X2t, y2t)
        print(f"  SMOTE applied to {len(strat)} categories")
    stage2 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ATTACK_CATEGORIES),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    stage2.fit(X2t, y2t, eval_set=[(X2v, y2v)],
               callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
               feature_name=UNSW_FLOW_FEATURES)

    # ── Evaluate ──
    print("\n=== Evaluation (held-out test set) ===")
    s1 = stage1.predict(X_te_s)
    pred = np.where(s1 == 0, "Normal", "Unknown").astype(object)
    ai = np.where(s1 == 1)[0]
    if len(ai):
        fam = stage2.predict(pd.DataFrame(X_te_s[ai], columns=UNSW_FLOW_FEATURES))
        for i, fi in zip(ai, fam):
            pred[i] = cat_encoder.inverse_transform([int(fi)])[0]
    true_bin = np.where(yb_te == 0, "Normal", "Attack")
    pred_bin = np.where(pred == "Normal", "Normal", "Attack")
    macro_f1 = f1_score(true_bin, pred_bin, average="macro")
    print(f"  Binary macro-F1: {macro_f1:.4f}")
    classes = list(cat_encoder.classes_)
    for cat in ["Normal"] + ATTACK_CATEGORIES:
        if cat == "Normal":
            mask = yb_te == 0; correct = int((pred[mask] == "Normal").sum())
        else:
            ci = classes.index(cat); mask = yc_te == ci
            correct = int((pred[mask] == cat).sum())
        total = int(mask.sum())
        pct = 100 * correct / total if total else 0.0
        print(f"    {cat:<16} {correct:>6}/{total:<6} ({pct:5.1f}%)")

    pipeline = {
        "scaler": scaler, "stage1_model": stage1, "stage2_model": stage2,
        "cat_encoder": cat_encoder, "fam_encoder": cat_encoder,
        "features": UNSW_FLOW_FEATURES, "attack_categories": ATTACK_CATEGORIES,
        "meta": {"dataset": "UNSW-NB15", "feature_set": "suricata_reproducible",
                 "n_features": len(UNSW_FLOW_FEATURES), "test_macro_f1": float(macro_f1)},
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)
    print(f"\n  Saved {OUT_MODEL} ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
    print("  Done.")


if __name__ == "__main__":
    main()
