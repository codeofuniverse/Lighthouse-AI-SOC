"""Train UNSW-NB15 two-stage pipeline: XGBoost binary (Stage 1) + XGBoost multi-class (Stage 2).

Optuna hyperparameter search with StratifiedKFold.
SMOTE applied inside every fold — validation fold always sees raw data.

Class merges (same as train_unsw_nb15.py):
  Fuzzers  -> Generic
  Analysis -> Reconnaissance
  Backdoor -> Exploits
Final 6 attack categories: Generic, Exploits, DoS, Reconnaissance, Shellcode, Worms

GPU notes:
    XGBoost uses CUDA when a GPU is detected. On Docker/Linux this gives a native
    GPU training path without needing cuML.

Usage:
    python scripts/train_unsw_rf.py [--trials N] [--cv K]

    --trials  Optuna trials per stage (default: 30)
    --cv      CV folds (default: 5)

Output: data/models/unsw_nb15_xgb_pipeline.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.gpu_utils import detect_gpu  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR   = Path("data/models/raw/unsbw15")
TRAIN_CSV = RAW_DIR / "Training and Testing Sets/UNSW_NB15_training-set.csv"
TEST_CSV  = RAW_DIR / "Training and Testing Sets/UNSW_NB15_testing-set.csv"
RAW_FILES = [RAW_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
RAW_COL_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]
OUT_MODEL = Path("data/models/unsw_nb15_xgb_pipeline.joblib")

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
_KNOWN_CATEGORIES = {"Normal"} | set(ATTACK_CATEGORIES)

BENIGN_CAP      = 80_000


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _make_xgb(params: dict[str, object], use_gpu: bool, num_class: int | None = None) -> XGBClassifier:
    xgb_params: dict[str, object] = {
        "n_estimators": params.get("n_estimators", 300),
        "max_depth": params.get("max_depth", 8),
        "learning_rate": params.get("learning_rate", 0.05),
        "subsample": params.get("subsample", 0.9),
        "colsample_bytree": params.get("colsample_bytree", 0.9),
        "min_child_weight": params.get("min_child_weight", 1.0),
        "gamma": params.get("gamma", 0.0),
        "reg_alpha": params.get("reg_alpha", 0.0),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "max_bin": 512,
        "objective": "binary:logistic" if num_class is None else "multi:softprob",
        "eval_metric": "logloss" if num_class is None else "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }
    if num_class is not None:
        xgb_params["num_class"] = num_class
    if use_gpu:
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cuda"
        xgb_params["sampling_method"] = "gradient_based"
    else:
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cpu"
    return XGBClassifier(**xgb_params)


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


def _load_raw_files() -> pd.DataFrame:
    frames = []
    available = [p for p in RAW_FILES if p.exists()]
    if not available:
        print("  [INFO] Raw files not found — using pre-split training set only.")
        return pd.DataFrame()
    for path in tqdm(available, desc="Loading raw files", unit="file", ncols=80):
        try:
            df = pd.read_csv(
                path, header=None, names=RAW_COL_NAMES,
                low_memory=False, encoding="utf-8", encoding_errors="replace",
            )
            df = df.rename(columns=COL_ALIASES)
            df.columns = df.columns.str.strip()
            df["Label"] = df["attack_cat"].astype(str).str.strip()
            df = df[df["Label"].notna() & (df["Label"] != "nan") & (df["Label"] != "0")]
            df["Label"] = df["Label"].replace("Backdoors", "Backdoor")
            unknown = set(df["Label"].unique()) - _RAW_CATEGORIES - {"Normal"}
            if unknown:
                df["Label"] = df["Label"].apply(lambda x: x if x in _RAW_CATEGORIES else "Generic")
            df["Label"] = df["Label"].replace(CATEGORY_MERGE)
            frames.append(df)
            print(f"  {path.name}: {len(df):,} attack rows")
        except Exception as exc:
            print(f"  [WARN] Failed to load {path.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Raw files total attack rows: {len(combined):,}")
    return combined


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
    print("\n=== Loading raw files for minority class supplementation ===")
    df_raw = _load_raw_files()
    if not df_raw.empty:
        df_raw["label"] = 1
        df_raw = df_raw[df_raw["Label"].isin(_KNOWN_CATEGORIES - {"Normal"})]
        train_counts = df_train["Label"].value_counts().to_dict()
        rng = np.random.default_rng(42)
        parts = []
        for cat in ATTACK_CATEGORIES:
            raw_cat = df_raw[df_raw["Label"] == cat]
            if raw_cat.empty:
                continue
            train_cnt = train_counts.get(cat, 0)
            cap = train_cnt * 3
            if len(raw_cat) > cap > 0:
                raw_cat = raw_cat.iloc[rng.choice(len(raw_cat), cap, replace=False)]
            parts.append(raw_cat)
            print(f"  {cat:<20} +{len(raw_cat):>6,} raw rows  (train had {train_cnt:,})")
        if parts:
            df_train = pd.concat([df_train, pd.concat(parts, ignore_index=True)], ignore_index=True)
            print(f"\n  Training set after supplementation: {len(df_train):,} rows")
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
        safe = [l if l in set(cat_encoder.classes_) else "Generic" for l in attack_labels]
        y_category[attack_mask] = cat_encoder.transform(safe)
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
    return X, y_binary, y_category


# ---------------------------------------------------------------------------
def _sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------
def _s1_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                  n_splits: int, use_gpu: bool) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
    }
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []
    for tr, va in kf.split(X, y):
        model = _make_xgb(params, use_gpu)
        model.fit(X[tr], y[tr], sample_weight=_sample_weights(y[tr]))
        scores.append(f1_score(y[va], model.predict(X[va]), average="macro"))
    return float(np.mean(scores))


def _s2_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                  n_splits: int, use_gpu: bool) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
    }
    counts  = np.bincount(y, minlength=len(ATTACK_CATEGORIES))
    min_cnt = int(counts[counts > 0].min())
    n_cv    = min(n_splits, min_cnt)
    if n_cv < 2:
        return 0.0
    kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    scores: list[float] = []
    for tr, va in kf.split(X, y):
        model = _make_xgb(params, use_gpu, num_class=len(ATTACK_CATEGORIES))
        model.fit(X[tr], y[tr], sample_weight=_sample_weights(y[tr]))
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
def train(df_train: pd.DataFrame, df_test: pd.DataFrame, n_trials: int, cv_folds: int) -> dict:
    cat_encoder = LabelEncoder()
    cat_encoder.fit(ATTACK_CATEGORIES)
    cat_encoder.classes_ = np.array(ATTACK_CATEGORIES)

    print("\n=== Preparing features ===")
    X_raw, y_binary, y_category = _prepare(df_train, cat_encoder)
    print(f"  Train shape: {X_raw.shape}")

    gpu_available, gpu_name = detect_gpu()
    use_gpu = gpu_available
    if use_gpu:
        print(f"  GPU: {gpu_name} — using XGBoost CUDA")
    else:
        print("  No GPU — using XGBoost CPU")

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
    if use_gpu:
        X_scaled = X_scaled.astype(np.float32)

    # ---- Stage 1: Optuna RF binary ----
    print(f"\n=== Stage 1: Optuna XGBoost binary — {n_trials} trials, {cv_folds}-fold CV ===")
    study1 = optuna.create_study(direction="maximize", study_name="stage1_rf_binary")
    with tqdm(total=n_trials, desc="Optuna Stage1 XGB", unit="trial", ncols=80, colour="yellow") as bar:
        def _cb1(study: optuna.Study, trial: optuna.Trial) -> None:
            bar.set_postfix({"best_macro_f1": f"{study.best_value:.4f}"})
            bar.update(1)
        study1.optimize(
            lambda t: _s1_objective(t, X_scaled, y_binary, cv_folds, use_gpu),
            n_trials=n_trials, callbacks=[_cb1], show_progress_bar=False,
        )
    best1 = study1.best_params
    print(f"\n  Best Stage 1 params : {best1}")
    print(f"  Best Stage 1 CV F1  : {study1.best_value:.4f}")

    print("  Fitting final Stage 1 XGBoost on full training set...")
    stage1 = _make_xgb(best1, use_gpu)
    stage1.fit(X_scaled, y_binary, sample_weight=_sample_weights(y_binary))

    # ---- Stage 2: Optuna RF multi-class ----
    att_mask = y_binary == 1
    X_att    = X_scaled[att_mask]
    y_att    = y_category[att_mask]

    print(f"\n=== Stage 2: Optuna XGBoost 6-class — {n_trials} trials, {cv_folds}-fold CV ===")
    study2 = optuna.create_study(direction="maximize", study_name="stage2_rf_multiclass")
    with tqdm(total=n_trials, desc="Optuna Stage2 XGB", unit="trial", ncols=80, colour="cyan") as bar:
        def _cb2(study: optuna.Study, trial: optuna.Trial) -> None:
            bar.set_postfix({"best_weighted_f1": f"{study.best_value:.4f}"})
            bar.update(1)
        study2.optimize(
            lambda t: _s2_objective(t, X_att, y_att, cv_folds, use_gpu),
            n_trials=n_trials, callbacks=[_cb2], show_progress_bar=False,
        )
    best2 = study2.best_params
    print(f"\n  Best Stage 2 params : {best2}")
    print(f"  Best Stage 2 CV F1  : {study2.best_value:.4f}")

    print("  Fitting final Stage 2 XGBoost on full attack set with class weights...")
    stage2 = _make_xgb(best2, use_gpu, num_class=len(ATTACK_CATEGORIES))
    stage2.fit(X_att, y_att, sample_weight=_sample_weights(y_att))

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
            "model_type":            "XGBoost-binary + XGBoost-multiclass",
            "n_features":            len(FEATURES),
            "attack_categories":     ATTACK_CATEGORIES,
            "label_style":           "native-merged-6class",
            "smote_applied":         False,
            "smote_per_fold":        False,
            "optuna_trials":         n_trials,
            "cv_folds":              cv_folds,
            "stage1_best_params":    best1,
            "stage2_best_params":    best2,
            "stage1_cv_macro_f1":    study1.best_value,
            "stage2_cv_weighted_f1": study2.best_value,
            "test_binary_macro_f1":  macro_f1,
            "gpu_used":              use_gpu,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNSW-NB15 XGBoost pipeline with Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per stage (default: 30)")
    parser.add_argument("--cv",     type=int, default=5,  help="CV folds (default: 5)")
    args = parser.parse_args()

    print("=" * 65)
    print("  UNSW-NB15  XGBoost binary (Stage 1) + XGBoost 6-class (Stage 2)")
    print("  Fuzzers->Generic  Analysis->Recon  Backdoor->Exploits")
    print(f"  Optuna trials: {args.trials}   CV folds: {args.cv}")
    print("=" * 65)
    print(f"  Train CSV : {TRAIN_CSV}")
    print(f"  Test CSV  : {TEST_CSV}")
    print(f"  Out model : {OUT_MODEL}")

    print("\n=== Loading CSVs ===")
    df_train, df_test = load_train_test()
    print(f"\n  Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}")
    print("  Train label distribution:")
    for label, cnt in df_train["Label"].value_counts().items():
        print(f"    {label:<20} {cnt:>8,}")

    pipeline = train(df_train, df_test, n_trials=args.trials, cv_folds=args.cv)

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
    print(f"  GPU (cuML)                 : {meta['gpu_used']}")
    print("=" * 65)
    print("  Done.")


if __name__ == "__main__":
    main()
