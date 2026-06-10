"""UNSW-NB15 two-stage trainer with Stratified 5-Fold CV + the rare-class FP fix.

Mirrors scripts/train_kfold_cic.py but for UNSW-NB15:
  - features = the engineered UNSW FEATURES set from scripts/train_unsw_svm.py
  - 6 merged attack categories: Generic, Exploits, DoS, Reconnaissance, Shellcode, Worms
  - subset = data/models/raw/unsw_subset_10pct.parquet (from the training-set partition)
  - DISJOINT validation = data/models/raw/unsw_val.parquet (the official testing-set partition)

Stage 1 XGBoost binary + Stage 2 LightGBM multiclass so predict_proba is available
for SHAP/ROC (matches the existing unsw_nb15_pipeline.joblib dict schema). Saved to
a NEW filename so production unsw_nb15_pipeline.joblib is never overwritten.

Usage:
    python scripts/train_kfold_unsw.py
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
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._kfold_common import (  # noqa: E402
    SMOTE_TARGET, FoldRecorder, apply_resampling, fp_rate_by_class,
    resampling_plan, sample_weight_from_plan,
)
from scripts.train_unsw_svm import ATTACK_CATEGORIES, FEATURES as BASE_FEATURES  # noqa: E402

warnings.filterwarnings("ignore")

SUBSET    = Path("data/models/raw/unsw_subset_10pct.parquet")
VAL       = Path("data/models/raw/unsw_val.parquet")
OUT_MODEL = Path("data/models/unsw_kfold_pipeline.joblib")
OUT_DATA  = Path("reports/training_validation_report_data")
N_SPLITS  = 5
EARLY_STOP = 30
BENIGN_NAME = "Normal"
ALL_LABELS = [BENIGN_NAME] + ATTACK_CATEGORIES

# SHAP feature discovery (scripts/discover_unsw_features.py) found 10 columns the
# 18-feature model ignored that are the TOP discriminators for the weak classes:
#   DoS -> ct_srv_dst, proto, sttl, stcpb ; Recon -> sloss, dloss, proto, service ;
#   Shellcode -> service, sttl, ct_src_dport_ltm, proto, ct_dst_src_ltm ;
#   Worms -> sttl, service, dtcpb. Adding them expands the set to 28.
NEW_FEATURES = ["ct_srv_dst", "proto", "sttl", "stcpb", "sloss", "dloss",
                "service", "ct_src_dport_ltm", "ct_dst_src_ltm", "dtcpb"]
FEATURES = list(BASE_FEATURES) + NEW_FEATURES          # 28
# proto/service are strings -> frequency-encoded (deterministic, serving-safe).
CATEGORICAL = ["proto", "service"]
_FREQ_MAPS: dict[str, dict] = {}        # built on the training subset, stored in model


def _build_freq_maps(df: pd.DataFrame) -> None:
    """Frequency-encode categorical cols: value -> its count share on training data.
    Deterministic and reproducible at serving time from the stored maps."""
    for c in CATEGORICAL:
        if c in df.columns:
            vc = df[c].astype(str).value_counts(normalize=True)
            _FREQ_MAPS[c] = vc.to_dict()


def _apply_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    for c in CATEGORICAL:
        if c in df.columns:
            fmap = _FREQ_MAPS.get(c, {})
            df[c] = df[c].astype(str).map(fmap).fillna(0.0)
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)


def _prepare(df: pd.DataFrame, cat_encoder: LabelEncoder):
    df = df.copy()
    X = _apply_features(df)
    labels = df["Label"].values
    y_binary = np.where(labels == BENIGN_NAME, 0, 1).astype(int)
    attack_mask = y_binary == 1
    safe = [l if l in set(ATTACK_CATEGORIES) else "Generic" for l in labels[attack_mask]]
    y_cat = np.full(len(labels), -1, dtype=int)
    y_cat[attack_mask] = cat_encoder.transform(safe)
    y_combined = np.where(y_binary == 0, 0, y_cat + 1).astype(int)
    return X, y_binary, y_cat, y_combined, labels


def _fit_stage2(X_att, y_att, plan, *, use_class_weight: bool):
    sw = sample_weight_from_plan(y_att, plan) if use_class_weight else None
    stage2 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ATTACK_CATEGORIES),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    stage2.fit(X_att, y_att, sample_weight=sw, feature_name=FEATURES)
    return stage2


def _end_to_end_pred(stage1, stage2, cat_encoder, X_s) -> np.ndarray:
    s1 = stage1.predict(X_s)
    pred = np.where(s1 == 0, BENIGN_NAME, "Unknown").astype(object)
    att_idx = np.where(s1 == 1)[0]
    if len(att_idx):
        cat = stage2.predict(pd.DataFrame(X_s[att_idx], columns=FEATURES))
        for i, ci in zip(att_idx, cat):
            pred[i] = cat_encoder.inverse_transform([int(ci)])[0]
    return pred.astype(str)


def main() -> None:
    print("=" * 70)
    print("  UNSW-NB15 — Stratified 5-Fold trainer + rare-class FP fix")
    print("=" * 70)
    if not SUBSET.exists():
        raise FileNotFoundError(f"{SUBSET} missing — run scripts/sample_proportional.py first")

    df = pd.read_parquet(SUBSET)
    df_val = pd.read_parquet(VAL)
    print(f"  Subset rows: {len(df):,}   Disjoint val rows: {len(df_val):,}")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(df["Label"].value_counts().to_string())

    # Frequency-encode categoricals from the SUBSET (training source) only — the
    # disjoint validation set is then encoded with these same maps (no leakage).
    _build_freq_maps(df)

    cat_encoder = LabelEncoder()
    cat_encoder.classes_ = np.array(ATTACK_CATEGORIES)

    X, yb, yc_cat, yc, labels = _prepare(df, cat_encoder)

    Xtr, Xte, ybtr, ybte, yctr, ycte, ltr, lte = train_test_split(
        X, yb, yc, labels, test_size=0.2, random_state=42, stratify=yc)
    print(f"\n  Train: {Xtr.shape}  Test: {Xte.shape}")

    print(f"\n=== Stratified {N_SPLITS}-Fold CV (scaler+resampling inside fold) ===")
    rec = FoldRecorder()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(Xtr, yctr), 1):
        sc = StandardScaler().fit(Xtr[tr])
        Xtr_s, Xva_s = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
        yb_f, yc_f = ybtr[tr], yctr[tr]

        n_neg, n_pos = int((yb_f == 0).sum()), int((yb_f == 1).sum())
        spw = max(1.0, n_neg / max(n_pos, 1))
        s1 = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1)
        s1.fit(Xtr_s, yb_f)

        att = yb_f == 1
        Xa, ya = Xtr_s[att], (yc_f[att] - 1)
        plan = resampling_plan(ya, ATTACK_CATEGORIES)
        Xa_r, ya_r = apply_resampling(Xa, ya, plan)
        s2 = _fit_stage2(Xa_r, ya_r, plan, use_class_weight=True)

        e2e = _end_to_end_pred(s1, s2, cat_encoder, Xva_s)
        name_to_combined = {BENIGN_NAME: 0, **{f: i + 1 for i, f in enumerate(ATTACK_CATEGORIES)}}
        y_true_c = yctr[va]
        y_pred_c = np.array([name_to_combined.get(p, 0) for p in e2e])

        s1_proba = s1.predict_proba(Xva_s)[:, 1]
        auc = roc_auc_score(ybtr[va], s1_proba) if len(np.unique(ybtr[va])) > 1 else float("nan")
        ll = log_loss(ybtr[va], s1_proba, labels=[0, 1])
        rec.record("UNSW", fold, ALL_LABELS, y_true_c, y_pred_c, auc, ll, plan.decisions)
        fpr = fp_rate_by_class(np.array([ALL_LABELS[c] for c in y_true_c]),
                               e2e, BENIGN_NAME, ATTACK_CATEGORIES)
        print(f"  Fold {fold}: S1-AUC={auc:.4f} logloss={ll:.4f} "
              f"normal->attack FP rate={fpr['__ANY__']['fp_rate']*100:.2f}%  "
              f"decisions={plan.decisions}")

    print("\n=== Final model (full train split) ===")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    n_neg, n_pos = int((ybtr == 0).sum()), int((ybtr == 1).sum())
    spw = max(1.0, n_neg / max(n_pos, 1))
    Xs1_tr, Xs1_val, ys1_tr, ys1_val = train_test_split(
        Xtr_s, ybtr, test_size=0.1, stratify=ybtr, random_state=42)
    stage1 = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=EARLY_STOP, tree_method="hist",
        random_state=42, n_jobs=-1)
    stage1.fit(Xs1_tr, ys1_tr, eval_set=[(Xs1_val, ys1_val)], verbose=False)

    att = ybtr == 1
    Xa, ya = Xtr_s[att], (yctr[att] - 1)

    plan_fix = resampling_plan(ya, ATTACK_CATEGORIES)
    Xa_fix, ya_fix = apply_resampling(Xa, ya, plan_fix)
    stage2 = _fit_stage2(Xa_fix, ya_fix, plan_fix, use_class_weight=True)

    from imblearn.over_sampling import SMOTE
    counts = dict(zip(*np.unique(ya, return_counts=True)))
    flat = {c: SMOTE_TARGET for c, n in counts.items() if 100 <= n < SMOTE_TARGET}
    if flat:
        k = max(1, min(5, min(counts[c] for c in flat) - 1))
        Xa_old, ya_old = SMOTE(sampling_strategy=flat, k_neighbors=k, random_state=42).fit_resample(Xa, ya)
    else:
        Xa_old, ya_old = Xa, ya
    stage2_old = _fit_stage2(Xa_old, ya_old, plan_fix, use_class_weight=False)

    def _report(tag: str, X_s, y_true_labels) -> dict:
        e2e_new = _end_to_end_pred(stage1, stage2, cat_encoder, X_s)
        e2e_old = _end_to_end_pred(stage1, stage2_old, cat_encoder, X_s)
        fpr_new = fp_rate_by_class(y_true_labels, e2e_new, BENIGN_NAME, ATTACK_CATEGORIES)
        fpr_old = fp_rate_by_class(y_true_labels, e2e_old, BENIGN_NAME, ATTACK_CATEGORIES)
        tb = np.where(y_true_labels == BENIGN_NAME, BENIGN_NAME, "ATTACK")
        pb_new = np.where(e2e_new == BENIGN_NAME, BENIGN_NAME, "ATTACK")
        f1_new = f1_score(tb, pb_new, average="macro")
        print(f"\n  [{tag}] macro-F1={f1_new:.4f}")
        print(f"    normal->ANY-attack FP rate:  before-fix={fpr_old['__ANY__']['fp_rate']*100:6.2f}%"
              f"   after-fix={fpr_new['__ANY__']['fp_rate']*100:6.2f}%")
        recall = {}
        for name in ALL_LABELS:
            mask = y_true_labels == name
            tot = int(mask.sum())
            recall[name] = float((e2e_new[mask] == name).sum()) / tot if tot else 0.0
        for fam in ("Worms", "Shellcode", "Reconnaissance", "DoS"):
            print(f"    {fam:<16} FP {fpr_old[fam]['fp_rate']*100:5.2f}%->{fpr_new[fam]['fp_rate']*100:5.2f}%"
                  f"   recall={recall[fam]*100:5.1f}%")
        return {"tag": tag, "macro_f1": float(f1_new),
                "fp_before": fpr_old, "fp_after": fpr_new, "recall": recall}

    print("\n=== Evaluation: 20% hold-out TEST ===")
    test_eval = _report("TEST (20% hold-out)", Xte_s, lte)

    print("\n=== Evaluation: DISJOINT validation set (official testing-set partition) ===")
    Xv, _, _, _, lv = _prepare(df_val, cat_encoder)
    Xv_s = scaler.transform(Xv)
    val_eval = _report("VALIDATION (disjoint)", Xv_s, lv)

    pipeline = {
        "scaler": scaler, "stage1_model": stage1, "stage2_model": stage2,
        "cat_encoder": cat_encoder, "features": FEATURES,
        "attack_categories": ATTACK_CATEGORIES,
        "freq_maps": _FREQ_MAPS, "categorical": CATEGORICAL,   # for serving reproducibility
        "meta": {"dataset": "UNSW-NB15", "model_type": "XGBoost + LightGBM",
                 "n_features": len(FEATURES), "attack_categories": ATTACK_CATEGORIES,
                 "label_style": "native-merged-6class",
                 "feature_set": "shap_discovered_28",
                 "training": "stratified_5fold + capped_smote + class_weight",
                 "test_binary_macro_f1": test_eval["macro_f1"],
                 "val_binary_macro_f1": val_eval["macro_f1"]},
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    rec.to_frame().to_csv(OUT_DATA / "unsw_folds.csv", index=False)
    (OUT_DATA / "unsw_eval.json").write_text(json.dumps(
        {"test": test_eval, "validation": val_eval,
         "resampling_decisions": {ATTACK_CATEGORIES[c]: d for c, d in plan_fix.decisions.items()}},
        indent=2))
    print(f"\n  Saved model -> {OUT_MODEL} ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
    print(f"  Saved fold metrics -> {OUT_DATA/'unsw_folds.csv'}")
    print(f"  Saved eval data   -> {OUT_DATA/'unsw_eval.json'}")
    print("  Done.")


if __name__ == "__main__":
    main()
