"""Train UNSW-NB15 two-stage pipeline: RandomForest binary (Stage 1) + RF multi-class (Stage 2).

CPU-only version using scikit-learn RandomForestClassifier.
Replaces a previously cuML-trained model that cannot be loaded without a GPU.

Class merges (same as other UNSW scripts):
  Fuzzers  -> Generic
  Analysis -> Reconnaissance
  Backdoor -> Exploits
Final 6 attack categories: Generic, Exploits, DoS, Reconnaissance, Shellcode, Worms

Usage:
    python scripts/train_unsw_rf_cpu.py [--trials N] [--cv K] [--jobs N]

    --trials  Optuna trials per stage (default: 20)
    --cv      CV folds (default: 5)
    --jobs    Parallel jobs for RF (-1 = all cores, default: -1)

Output: data/models/unsw_nb15_rf_pipeline.joblib
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR   = Path("data/models/raw/unsbw15")
TRAIN_CSV = RAW_DIR / "Training and Testing Sets/UNSW_NB15_training-set.csv"
TEST_CSV  = RAW_DIR / "Training and Testing Sets/UNSW_NB15_testing-set.csv"
OUT_MODEL = Path("data/models/unsw_nb15_rf_pipeline.joblib")

COL_ALIASES = {
    "smean": "smeansz", "dmean": "dmeansz",
    "sinpkt": "sintpkt", "dinpkt": "dintpkt",
    "Sload": "sload", "Dload": "dload",
    "Spkts": "spkts", "Dpkts": "dpkts",
    "Sjit": "sjit", "Djit": "djit",
    "Sintpkt": "sintpkt", "Dintpkt": "dintpkt",
    "ct_src_ ltm": "ct_src_ltm",
}

FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "smeansz", "dmeansz",
    "rate", "sload", "dload", "sjit", "djit", "sintpkt", "dintpkt",
    "synack", "ackdat", "ct_srv_src", "ct_dst_ltm",
]

ATTACK_CATEGORIES = [
    "Generic", "Exploits", "DoS", "Reconnaissance", "Shellcode", "Worms",
]

CATEGORY_MERGE: dict[str, str] = {
    "Fuzzers":  "Generic",
    "Analysis": "Reconnaissance",
    "Backdoor": "Exploits",
}

_RAW_CATEGORIES = {
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
}

BENIGN_CAP = 80_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_csv(path: Path) -> pd.DataFrame:
    print(f"  Loading {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COL_ALIASES)
    for col in ["attack_cat", "Attack_cat"]:
        if col in df.columns:
            df = df.rename(columns={col: "Label"})
            break
    if "Label" not in df.columns:
        raise ValueError(f"No attack_cat column in {path.name}")
    df["Label"] = df["Label"].astype(str).str.strip()
    unknown = set(df["Label"].unique()) - _RAW_CATEGORIES - {"Normal"}
    if unknown:
        print(f"\n  [WARN] Unknown categories mapped to Generic: {unknown}")
        df["Label"] = df["Label"].apply(lambda x: x if x in _RAW_CATEGORIES else "Generic")
    df["Label"] = df["Label"].replace(CATEGORY_MERGE)
    counts = df["Label"].value_counts().to_dict()
    print({k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])})
    return df


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [TRAIN_CSV, TEST_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected CSV not found: {path}\n"
                "Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in:\n"
                "  data/models/raw/unsbw15/Training and Testing Sets/"
            )
    df_train = _load_csv(TRAIN_CSV)
    df_test  = _load_csv(TEST_CSV)
    return df_train, df_test


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def _prepare(df: pd.DataFrame, cat_encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    labels   = df["Label"].values
    y_binary = np.where(labels == "Normal", 0, 1).astype(int)
    attack_mask = y_binary == 1
    y_category  = np.full(len(labels), -1, dtype=int)
    if attack_mask.any():
        attack_labels = labels[attack_mask]
        known = set(cat_encoder.classes_)
        safe = [l if l in known else "Generic" for l in attack_labels]
        y_category[attack_mask] = cat_encoder.transform(safe)
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    return X, y_binary, y_category


# ---------------------------------------------------------------------------
# RF factory
# ---------------------------------------------------------------------------
def _make_rf(params: dict[str, Any], n_jobs: int, num_class: int | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------
def _s1_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                  n_splits: int, n_jobs: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []
    for tr, va in kf.split(X, y):
        model = _make_rf(params, n_jobs=1)
        model.fit(X[tr], y[tr])
        scores.append(f1_score(y[va], model.predict(X[va]), average="macro"))
    return float(np.mean(scores))


def _s2_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                  n_splits: int, n_jobs: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    counts  = np.bincount(y, minlength=len(ATTACK_CATEGORIES))
    min_cnt = int(counts[counts > 0].min())
    n_cv    = min(n_splits, min_cnt)
    if n_cv < 2:
        return 0.0
    kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    scores: list[float] = []
    for tr, va in kf.split(X, y):
        model = _make_rf(params, n_jobs=1, num_class=len(ATTACK_CATEGORIES))
        model.fit(X[tr], y[tr])
        scores.append(f1_score(y[va], model.predict(X[va]), average="weighted"))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _eval(stage1: Any, stage2: Any, cat_encoder: LabelEncoder,
          scaler: StandardScaler, df_test: pd.DataFrame) -> float:
    X_raw, y_binary, _ = _prepare(df_test, cat_encoder)
    X = scaler.transform(X_raw)
    s1_pred = stage1.predict(X)
    pred_labels: list[str] = ["Normal" if p == 0 else "Unknown" for p in s1_pred]
    att_idx = np.where(s1_pred == 1)[0]
    if len(att_idx):
        cat_preds = stage2.predict(X[att_idx])
        for i, name in zip(att_idx, cat_encoder.inverse_transform(cat_preds)):
            pred_labels[i] = str(name)
    pred_arr = np.array(pred_labels)
    true_bin = np.where(y_binary == 0, "Normal", "Attack")
    pred_bin = np.where(pred_arr == "Normal", "Normal", "Attack")
    macro_f1 = f1_score(true_bin, pred_bin, average="macro")
    print(f"\n  Binary macro F1 (TEST): {macro_f1:.4f}")
    labels = df_test["Label"].values
    known  = set(cat_encoder.classes_)
    safe   = [l if l in known or l == "Normal" else "Generic" for l in labels]
    print(classification_report(safe, pred_labels, labels=["Normal"] + ATTACK_CATEGORIES, zero_division=0))
    return float(macro_f1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(df_train: pd.DataFrame, df_test: pd.DataFrame,
          n_trials: int, cv_folds: int, n_jobs: int) -> dict:
    cat_encoder = LabelEncoder()
    cat_encoder.fit(ATTACK_CATEGORIES)
    cat_encoder.classes_ = np.array(ATTACK_CATEGORIES)

    print("\n=== Preparing features ===")
    X_raw, y_binary, y_category = _prepare(df_train, cat_encoder)
    print(f"  Train shape: {X_raw.shape}")

    rng = np.random.default_rng(42)
    normal_idx = np.where(y_binary == 0)[0]
    attack_idx = np.where(y_binary == 1)[0]
    if len(normal_idx) > BENIGN_CAP:
        normal_idx = rng.choice(normal_idx, size=BENIGN_CAP, replace=False)
        print(f"  Downsampled Normal to {BENIGN_CAP:,}")
    keep = np.concatenate([normal_idx, attack_idx])
    rng.shuffle(keep)
    X_raw, y_binary, y_category = X_raw[keep], y_binary[keep], y_category[keep]
    print(f"  After downsample: {X_raw.shape}")

    print("\n=== Scaling features ===")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ---- Stage 1: binary RF ----
    print(f"\n=== Stage 1: Optuna RF binary — {n_trials} trials, {cv_folds}-fold CV ===")
    study1 = optuna.create_study(direction="maximize", study_name="stage1_rf_binary")
    with tqdm(total=n_trials, desc="Optuna Stage1 RF", unit="trial", ncols=80) as bar:
        def _cb1(study: optuna.Study, trial: optuna.Trial) -> None:
            bar.set_postfix({"best_macro_f1": f"{study.best_value:.4f}"})
            bar.update(1)
        study1.optimize(
            lambda t: _s1_objective(t, X_scaled, y_binary, cv_folds, n_jobs),
            n_trials=n_trials, callbacks=[_cb1], show_progress_bar=False,
        )
    best1 = study1.best_params
    print(f"\n  Best Stage 1 params : {best1}")
    print(f"  Best Stage 1 CV F1  : {study1.best_value:.4f}")

    print("  Fitting final Stage 1 RF on full training set...")
    stage1 = _make_rf(best1, n_jobs=n_jobs)
    stage1.fit(X_scaled, y_binary)

    # ---- Stage 2: multi-class RF ----
    att_mask = y_binary == 1
    X_att    = X_scaled[att_mask]
    y_att    = y_category[att_mask]

    print(f"\n=== Stage 2: Optuna RF 6-class — {n_trials} trials, {cv_folds}-fold CV ===")
    study2 = optuna.create_study(direction="maximize", study_name="stage2_rf_multiclass")
    with tqdm(total=n_trials, desc="Optuna Stage2 RF", unit="trial", ncols=80) as bar:
        def _cb2(study: optuna.Study, trial: optuna.Trial) -> None:
            bar.set_postfix({"best_weighted_f1": f"{study.best_value:.4f}"})
            bar.update(1)
        study2.optimize(
            lambda t: _s2_objective(t, X_att, y_att, cv_folds, n_jobs),
            n_trials=n_trials, callbacks=[_cb2], show_progress_bar=False,
        )
    best2 = study2.best_params
    print(f"\n  Best Stage 2 params : {best2}")
    print(f"  Best Stage 2 CV F1  : {study2.best_value:.4f}")

    print("  Fitting final Stage 2 RF on full attack set...")
    stage2 = _make_rf(best2, n_jobs=n_jobs, num_class=len(ATTACK_CATEGORIES))
    stage2.fit(X_att, y_att)

    print("\n=== Evaluating on held-out test set ===")
    macro_f1 = _eval(stage1, stage2, cat_encoder, scaler, df_test)

    return {
        "scaler":            scaler,
        "stage1_model":      stage1,
        "stage2_model":      stage2,
        "cat_encoder":       cat_encoder,
        "features":          FEATURES,
        "attack_categories": ATTACK_CATEGORIES,
        "meta": {
            "dataset":               "UNSW-NB15",
            "model_type":            "RandomForest-binary + RandomForest-multiclass",
            "n_features":            len(FEATURES),
            "attack_categories":     ATTACK_CATEGORIES,
            "label_style":           "native-merged-6class",
            "smote_applied":         False,
            "optuna_trials":         n_trials,
            "cv_folds":              cv_folds,
            "stage1_best_params":    best1,
            "stage2_best_params":    best2,
            "stage1_cv_macro_f1":    study1.best_value,
            "stage2_cv_weighted_f1": study2.best_value,
            "test_binary_macro_f1":  macro_f1,
            "gpu_used":              False,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNSW-NB15 CPU Random Forest pipeline")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cv",     type=int, default=5)
    parser.add_argument("--jobs",   type=int, default=-1, help="n_jobs for final RF fit (-1 = all cores)")
    args = parser.parse_args()

    print("=" * 65)
    print("  UNSW-NB15  RandomForest binary (Stage 1) + RF 6-class (Stage 2)")
    print("  CPU-only — scikit-learn RandomForestClassifier")
    print("  Fuzzers->Generic  Analysis->Recon  Backdoor->Exploits")
    print(f"  Optuna trials: {args.trials}   CV folds: {args.cv}   n_jobs: {args.jobs}")
    print("=" * 65)

    print("\n=== Loading CSVs ===")
    df_train, df_test = load_train_test()
    print(f"\n  Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}")
    print("  Train label distribution:")
    for label, cnt in df_train["Label"].value_counts().items():
        print(f"    {label:<20} {cnt:>8,}")

    pipeline = train(df_train, df_test, n_trials=args.trials, cv_folds=args.cv, n_jobs=args.jobs)

    print(f"\n  Saving to: {OUT_MODEL}")
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_MODEL)
    size_mb = OUT_MODEL.stat().st_size / 1_048_576
    print(f"  Saved ({size_mb:.1f} MB)")

    meta = pipeline["meta"]
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Stage 1 RF best params     : {meta['stage1_best_params']}")
    print(f"  Stage 1 RF CV macro-F1     : {meta['stage1_cv_macro_f1']:.4f}")
    print(f"  Stage 2 RF best params     : {meta['stage2_best_params']}")
    print(f"  Stage 2 RF CV weighted-F1  : {meta['stage2_cv_weighted_f1']:.4f}")
    print(f"  Test binary macro-F1       : {meta['test_binary_macro_f1']:.4f}")
    print("=" * 65)
    print("  Done. Run generate_model_reports.py to generate evaluation graphs.")


if __name__ == "__main__":
    main()
