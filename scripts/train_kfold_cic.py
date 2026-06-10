"""CIC-IDS-2017 two-stage trainer with Stratified 5-Fold CV + the Web-Attack FP fix.

Reads the proportional subset built by scripts/sample_proportional.py, runs a
leakage-free 5-fold CV (StandardScaler + resampling fit INSIDE each fold), applies
the class-weight-instead-of-SMOTE fix for ultra-rare classes, then trains a final
model and evaluates it on BOTH the 20% hold-out test split AND the fully disjoint
validation set.

The headline result is the per-attack false-positive rate BEFORE the fix (old flat
SMOTE-to-10k path) vs AFTER the fix (capped SMOTE + class-weight), written to
reports/training_validation_report_data/cic_*.json for the report builder.

Same model dict schema as scripts/retrain_cic_smote.py, saved to a NEW filename so
production data/models/cic2017_pipeline_smote.joblib is never overwritten.

Usage:
    python scripts/train_kfold_cic.py
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
from detection.flow_features import (  # noqa: E402
    CIC_FLOW_FEATURES, CIC_FLOW_FEATURES_V2, cic_features_from_df,
)
from scripts._kfold_common import (  # noqa: E402
    SMOTE_TARGET, FoldRecorder, apply_resampling, fp_rate_by_class,
    resampling_plan, sample_weight_from_plan,
)
from scripts.retrain_cic_smote import ATTACK_FAMILIES, FINAL_LABELS  # noqa: E402

warnings.filterwarnings("ignore")

SUBSET   = Path("data/models/raw/cic2017_subset_10pct.parquet")
VAL      = Path("data/models/raw/cic2017_val.parquet")
OUT_DATA  = Path("reports/training_validation_report_data")
N_SPLITS  = 5
EARLY_STOP = 30
BENIGN_NAME = "BENIGN"

# Feature version — switchable via --features. v2 adds dst_port (SHAP-discovered
# reproducible Web-Attack discriminator). v1 = legacy 17 features (for before/after).
FEATURE_VERSION = "v2"
FEATURES = CIC_FLOW_FEATURES_V2
OUT_MODEL = Path("data/models/cic2017_kfold_v2_pipeline.joblib")


def _prepare(df: pd.DataFrame, fam_encoder: LabelEncoder):
    X = cic_features_from_df(df, version=FEATURE_VERSION).values.astype(float)
    labels = df["Label"].values
    y_binary = np.where(labels == BENIGN_NAME, 0, 1).astype(int)
    attack_mask = y_binary == 1
    safe = [lbl if lbl in set(ATTACK_FAMILIES)
            else next((a for a in ATTACK_FAMILIES if a.lower() in str(lbl).lower()), "DDoS")
            for lbl in labels[attack_mask]]
    y_family = np.full(len(labels), -1, dtype=int)
    y_family[attack_mask] = fam_encoder.transform(safe)
    y_combined = np.where(y_binary == 0, 0, y_family + 1).astype(int)
    return X, y_binary, y_family, y_combined, labels


def _fit_stage2(X_att, y_att, plan, *, use_class_weight: bool):
    """Stage-2 LightGBM. SMOTE already applied to X_att/y_att by caller."""
    sw = sample_weight_from_plan(y_att, plan) if use_class_weight else None
    stage2 = lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, num_leaves=127, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        objective="multiclass", num_class=len(ATTACK_FAMILIES),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    stage2.fit(X_att, y_att, sample_weight=sw, feature_name=FEATURES)
    return stage2


def _end_to_end_pred(stage1, stage2, fam_encoder, X_s) -> np.ndarray:
    s1 = stage1.predict(X_s)
    pred = np.where(s1 == 0, BENIGN_NAME, "Unknown").astype(object)
    att_idx = np.where(s1 == 1)[0]
    if len(att_idx):
        fam = stage2.predict(pd.DataFrame(X_s[att_idx], columns=FEATURES))
        for i, fi in zip(att_idx, fam):
            pred[i] = fam_encoder.inverse_transform([int(fi)])[0]
    return pred.astype(str)


def _per_class_recall(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> dict:
    """Recall per final label (BENIGN + families) — surfaces the Web-Attack recall."""
    out = {}
    for name in FINAL_LABELS:
        mask = y_true_labels == name
        tot = int(mask.sum())
        out[name] = (float((y_pred_labels[mask] == name).sum()) / tot) if tot else 0.0
    return out


def main() -> None:
    import argparse
    global FEATURE_VERSION, FEATURES, OUT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["v1", "v2"], default=FEATURE_VERSION,
                    help="v2 (default) adds dst_port; v1 = legacy 17 features")
    args = ap.parse_args()
    FEATURE_VERSION = args.features
    FEATURES = CIC_FLOW_FEATURES_V2 if FEATURE_VERSION == "v2" else CIC_FLOW_FEATURES
    OUT_MODEL = Path(f"data/models/cic2017_kfold_{FEATURE_VERSION}_pipeline.joblib")

    print("=" * 70)
    print(f"  CIC-IDS-2017 — Stratified 5-Fold trainer + Web-Attack FP fix [{FEATURE_VERSION}]")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print("=" * 70)
    if not SUBSET.exists():
        raise FileNotFoundError(f"{SUBSET} missing — run scripts/sample_proportional.py first")

    df = pd.read_parquet(SUBSET)
    df_val = pd.read_parquet(VAL)
    print(f"  Subset rows: {len(df):,}   Disjoint val rows: {len(df_val):,}")
    print(df["Label"].value_counts().to_string())

    fam_encoder = LabelEncoder()
    fam_encoder.classes_ = np.array(ATTACK_FAMILIES)

    X, yb, yf, yc, labels = _prepare(df, fam_encoder)

    # ── Stratified 80/20 train/test on the subset ──
    Xtr, Xte, ybtr, ybte, yctr, ycte, ltr, lte = train_test_split(
        X, yb, yc, labels, test_size=0.2, random_state=42, stratify=yc)
    print(f"\n  Train: {Xtr.shape}  Test: {Xte.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5-fold CV (leakage-free): scaler + resampling fit INSIDE each fold
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n=== Stratified {N_SPLITS}-Fold CV (scaler+resampling inside fold) ===")
    rec = FoldRecorder()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(Xtr, yctr), 1):
        sc = StandardScaler().fit(Xtr[tr])
        Xtr_s, Xva_s = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
        yb_f, yc_f = ybtr[tr], yctr[tr]

        # Stage 1 — binary, scale_pos_weight
        n_neg, n_pos = int((yb_f == 0).sum()), int((yb_f == 1).sum())
        spw = max(1.0, n_neg / max(n_pos, 1))
        s1 = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1)
        s1.fit(Xtr_s, yb_f)

        # Stage 2 — family, FP-fix resampling on attack rows
        att = yb_f == 1
        Xa, ya = Xtr_s[att], (yc_f[att] - 1)
        plan = resampling_plan(ya, ATTACK_FAMILIES)
        Xa_r, ya_r = apply_resampling(Xa, ya, plan)
        s2 = _fit_stage2(Xa_r, ya_r, plan, use_class_weight=True)

        # Validate this fold (end-to-end) on the held-out fold slice
        e2e = _end_to_end_pred(s1, s2, fam_encoder, Xva_s)
        # map predicted/true to combined indices for metric recording
        name_to_combined = {BENIGN_NAME: 0, **{f: i + 1 for i, f in enumerate(ATTACK_FAMILIES)}}
        y_true_c = yctr[va]
        y_pred_c = np.array([name_to_combined.get(p, 0) for p in e2e])

        s1_proba = s1.predict_proba(Xva_s)[:, 1]
        auc = roc_auc_score(ybtr[va], s1_proba) if len(np.unique(ybtr[va])) > 1 else float("nan")
        ll = log_loss(ybtr[va], s1_proba, labels=[0, 1])
        rec.record("CIC", fold, FINAL_LABELS, y_true_c, y_pred_c, auc, ll, plan.decisions)
        fpr = fp_rate_by_class(np.array([FINAL_LABELS[c] for c in y_true_c]),
                               e2e, BENIGN_NAME, ATTACK_FAMILIES)
        print(f"  Fold {fold}: S1-AUC={auc:.4f} logloss={ll:.4f} "
              f"benign->attack FP rate={fpr['__ANY__']['fp_rate']*100:.2f}%  "
              f"decisions={plan.decisions}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final model on full train split (with the fix), + BEFORE/AFTER FP comparison
    # ─────────────────────────────────────────────────────────────────────────
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

    # AFTER (the fix): capped SMOTE + class-weight
    plan_fix = resampling_plan(ya, ATTACK_FAMILIES)
    Xa_fix, ya_fix = apply_resampling(Xa, ya, plan_fix)
    stage2 = _fit_stage2(Xa_fix, ya_fix, plan_fix, use_class_weight=True)

    # BEFORE (old behaviour): flat SMOTE-to-10k, no class weight — for comparison only
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
        e2e_new = _end_to_end_pred(stage1, stage2, fam_encoder, X_s)
        e2e_old = _end_to_end_pred(stage1, stage2_old, fam_encoder, X_s)
        fpr_new = fp_rate_by_class(y_true_labels, e2e_new, BENIGN_NAME, ATTACK_FAMILIES)
        fpr_old = fp_rate_by_class(y_true_labels, e2e_old, BENIGN_NAME, ATTACK_FAMILIES)
        tb = np.where(y_true_labels == BENIGN_NAME, BENIGN_NAME, "ATTACK")
        pb_new = np.where(e2e_new == BENIGN_NAME, BENIGN_NAME, "ATTACK")
        f1_new = f1_score(tb, pb_new, average="macro")
        print(f"\n  [{tag}] macro-F1={f1_new:.4f}")
        print(f"    benign->ANY-attack FP rate:  before-fix={fpr_old['__ANY__']['fp_rate']*100:6.2f}%"
              f"   after-fix={fpr_new['__ANY__']['fp_rate']*100:6.2f}%")
        print(f"    Web-Attack FP rate:          before-fix={fpr_old['Web Attack']['fp_rate']*100:6.2f}%"
              f"   after-fix={fpr_new['Web Attack']['fp_rate']*100:6.2f}%")
        recall = _per_class_recall(y_true_labels, e2e_new)
        print(f"    Web-Attack RECALL ({FEATURE_VERSION}): {recall['Web Attack']*100:6.2f}%")
        return {"tag": tag, "macro_f1": float(f1_new),
                "fp_before": fpr_old, "fp_after": fpr_new,
                "recall": recall, "feature_version": FEATURE_VERSION}

    print("\n=== Evaluation: 20% hold-out TEST ===")
    test_eval = _report("TEST (20% hold-out)", Xte_s, lte)

    print("\n=== Evaluation: DISJOINT validation set ===")
    Xv, _, _, _, lv = _prepare(df_val, fam_encoder)
    Xv_s = scaler.transform(Xv)
    val_eval = _report("VALIDATION (disjoint)", Xv_s, lv)

    # ── Save model (production-compatible schema, NEW filename) ──
    pipeline = {
        "scaler": scaler, "stage1_model": stage1, "stage2_model": stage2,
        "fam_encoder": fam_encoder, "features": FEATURES,
        "final_labels": FINAL_LABELS,
        "meta": {"dataset": "CIC-IDS-2017", "feature_set": "suricata_reproducible",
                 "n_features": len(FEATURES), "attack_families": ATTACK_FAMILIES,
                 "training": "stratified_5fold + capped_smote + class_weight",
                 "test_macro_f1": test_eval["macro_f1"],
                 "val_macro_f1": val_eval["macro_f1"]},
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)

    # ── Persist report data (version-suffixed so v1 baseline + v2 both survive) ──
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    eval_payload = json.dumps(
        {"test": test_eval, "validation": val_eval,
         "resampling_decisions": {ATTACK_FAMILIES[c]: d for c, d in plan_fix.decisions.items()}},
        indent=2)
    rec.to_frame().to_csv(OUT_DATA / f"cic_folds_{FEATURE_VERSION}.csv", index=False)
    (OUT_DATA / f"cic_eval_{FEATURE_VERSION}.json").write_text(eval_payload)
    # v2 is the canonical model -> also write the un-suffixed names the report reads
    if FEATURE_VERSION == "v2":
        rec.to_frame().to_csv(OUT_DATA / "cic_folds.csv", index=False)
        (OUT_DATA / "cic_eval.json").write_text(eval_payload)
    print(f"\n  Saved model -> {OUT_MODEL} ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
    print(f"  Saved fold metrics -> {OUT_DATA/f'cic_folds_{FEATURE_VERSION}.csv'}")
    print(f"  Saved eval data   -> {OUT_DATA/f'cic_eval_{FEATURE_VERSION}.json'}")
    print("  Done.")


if __name__ == "__main__":
    main()
