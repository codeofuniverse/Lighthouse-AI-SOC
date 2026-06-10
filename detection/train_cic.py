"""CIC-DDoS-2018 dataset training pipeline.

Loads the CIC-DDoS-2018 CSV files, engineers network-flow features,
trains both XGBoost and LightGBM models, runs ModelBenchmark per attack
type, and saves all artefacts to detection/models/.

Usage:
    python -m detection.train_cic [--sample-per-file N] [--model-dir PATH]

Dataset layout expected:
    data/CSV-01-12/01-12/*.csv
    data/CSV-03-11/03-11/*.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_cic")

# ---------------------------------------------------------------------------
# CIC Label → (class_id, attack_type) mapping
# class_id: 0=benign, 1=suspicious, 2=malicious
# attack_type: one of ATTACK_TYPES from ml_classifier
# ---------------------------------------------------------------------------
CIC_LABEL_MAP: dict[str, tuple[int, str]] = {
    # Benign
    "Benign":               (0, "benign"),
    "BENIGN":               (0, "benign"),
    # High-volume DDoS flooding → malicious / brute_force
    "DrDoS_DNS":            (2, "brute_force"),
    "DrDoS_NTP":            (2, "brute_force"),
    "DrDoS_SNMP":           (2, "brute_force"),
    "DrDoS_SSDP":           (2, "brute_force"),
    "DrDoS_UDP":            (2, "brute_force"),
    "DrDoS_NetBIOS":        (2, "brute_force"),
    "Syn":                  (2, "brute_force"),
    "SYN":                  (2, "brute_force"),
    # Application-layer amplification → malicious / lateral_movement
    "DrDoS_LDAP":           (2, "lateral_movement"),
    "DrDoS_MSSQL":          (2, "lateral_movement"),
    "LDAP":                 (2, "lateral_movement"),
    "MSSQL":                (2, "lateral_movement"),
    "NetBIOS":              (2, "lateral_movement"),
    "NETBIOS":              (2, "lateral_movement"),
    # Slow/covert floods → suspicious / port_scan
    "TFTP":                 (1, "port_scan"),
    "UDPLag":               (1, "port_scan"),
    "UDP-lag":              (1, "port_scan"),  # actual label in CSV
    "UDP-Lag":              (1, "port_scan"),
    "UDP":                  (1, "port_scan"),
    "Portmap":              (1, "port_scan"),
    "PORTMAP":              (1, "port_scan"),
}

# Suricata-Compatible Feature Columns (13 features)
# These are the only CIC features that can be directly extracted from Suricata's eve.json flow logs.
CIC_FEATURE_COLS = [
    " Destination Port",
    " Protocol",
    " Flow Duration",               # Suricata: flow.age
    " Total Fwd Packets",           # Suricata: flow.pkts_toserver
    " Total Backward Packets",      # Suricata: flow.pkts_toclient
    "Total Length of Fwd Packets",  # Suricata: flow.bytes_toserver
    " Total Length of Bwd Packets", # Suricata: flow.bytes_toclient
    " FIN Flag Count",              # Suricata: tcp.fin
    " SYN Flag Count",              # Suricata: tcp.syn
    " RST Flag Count",              # Suricata: tcp.rst
    " PSH Flag Count",              # Suricata: tcp.psh
    " ACK Flag Count",              # Suricata: tcp.ack
    " URG Flag Count",              # Suricata: tcp.urg
]


def _load_csv_sample(path: Path, n_rows: int, label_map: dict) -> pd.DataFrame | None:
    """Load a sample of rows from a CIC CSV file.

    Args:
        path: Path to the CSV file.
        n_rows: Max rows to sample from this file.
        label_map: Mapping from raw label string → (class_id, attack_type).

    Returns:
        DataFrame with feature columns + 'label' + 'attack_type', or None on error.
    """
    try:
        # Read only a sample using skiprows for memory efficiency on multi-GB files
        df = pd.read_csv(path, nrows=n_rows, low_memory=False)
        logger.info("  Loaded %d rows from %s", len(df), path.name)
    except Exception as exc:
        logger.error("  Failed to read %s: %s", path.name, exc)
        return None

    if " Label" not in df.columns and "Label" not in df.columns:
        logger.warning("  No Label column found in %s", path.name)
        return None

    label_col = " Label" if " Label" in df.columns else "Label"
    raw_labels = df[label_col].astype(str).str.strip()

    # Map labels
    class_ids = []
    attack_types = []
    for raw in raw_labels:
        mapped = label_map.get(raw)
        if mapped is None:
            # Try partial match
            for key, val in label_map.items():
                if key.lower() in raw.lower() or raw.lower() in key.lower():
                    mapped = val
                    break
        if mapped is None:
            mapped = (2, "brute_force")  # unknown → malicious
        class_ids.append(mapped[0])
        attack_types.append(mapped[1])

    df["label"] = class_ids
    df["attack_type"] = attack_types

    return df


def _get_available_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that are actually present in the DataFrame."""
    available = []
    for col in CIC_FEATURE_COLS:
        if col in df.columns:
            available.append(col)
        elif col.strip() in df.columns:
            available.append(col.strip())
    return available


def load_dataset(
    data_dirs: list[Path],
    sample_per_file: int = 50_000,
    label_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load and merge all CIC CSV files into a single feature matrix.

    Args:
        data_dirs: Directories containing CSV files.
        sample_per_file: Max rows to load per CSV file.
        label_map: Label mapping dict (defaults to CIC_LABEL_MAP).

    Returns:
        Tuple of (X, y, attack_type_labels, feature_names).
    """
    if label_map is None:
        label_map = CIC_LABEL_MAP

    all_dfs: list[pd.DataFrame] = []
    file_stats: list[dict] = []

    for data_dir in data_dirs:
        csv_files = list(data_dir.rglob("*.csv"))
        logger.info("Found %d CSV files in %s", len(csv_files), data_dir)
        for csv_file in sorted(csv_files):
            df = _load_csv_sample(csv_file, sample_per_file, label_map)
            if df is None or len(df) == 0:
                continue
            label_counts = df["attack_type"].value_counts().to_dict()
            file_stats.append({
                "file": csv_file.name,
                "rows": len(df),
                "labels": label_counts,
            })
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No data loaded from any CSV file.")

    logger.info("\nDataset summary:")
    for stat in file_stats:
        logger.info("  %-35s  %6d rows  %s", stat["file"], stat["rows"], stat["labels"])

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total combined rows: %d", len(combined))

    # Print class distribution
    class_dist = combined["label"].value_counts().sort_index()
    attack_dist = combined["attack_type"].value_counts().to_dict()
    logger.info("Class distribution: %s", class_dist.to_dict())
    logger.info("Attack type distribution: %s", attack_dist)

    # Determine available features
    feat_cols = _get_available_features(combined)
    logger.info("Using %d feature columns", len(feat_cols))

    X_df = combined[feat_cols].copy()

    # Clean: replace inf/nan with column medians
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    for col in X_df.columns:
        median = X_df[col].median()
        X_df[col] = X_df[col].fillna(median if not np.isnan(median) else 0.0)

    X = X_df.values.astype(np.float32)
    y = combined["label"].values.astype(int)
    attack_type_labels = combined["attack_type"].tolist()
    feature_names = feat_cols

    return X, y, attack_type_labels, feature_names


def train_cic(
    data_dirs: list[Path],
    model_dir: Path,
    sample_per_file: int = 50_000,
) -> dict[str, Any]:
    """Full training pipeline on CIC-DDoS-2018 dataset.

    Args:
        data_dirs: Directories containing CIC CSV files.
        model_dir: Where to save models and benchmark JSON.
        sample_per_file: Max rows per CSV file.

    Returns:
        Training results dict.
    """
    import pickle
    from datetime import date

    import lightgbm as lgb
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from typing import cast

    from detection.gpu_utils import detect_gpu, xgb_gpu_params, lgb_gpu_params, xgb_fit_with_fallback, lgb_fit_with_fallback
    from detection.ml_classifier import ModelBenchmark, ATTACK_TYPES

    use_gpu, gpu_name = detect_gpu()
    if use_gpu:
        logger.info("GPU detected: %s — XGBoost and LightGBM will use CUDA", gpu_name)
    else:
        logger.info("No GPU detected — training on CPU")

    model_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading CIC-DDoS-2018 dataset")
    logger.info("=" * 60)
    X, y, attack_type_labels, feature_names = load_dataset(
        data_dirs, sample_per_file=sample_per_file
    )
    logger.info("Dataset loaded: X=%s, classes=%s", X.shape, np.unique(y))

    # -----------------------------------------------------------------------
    # 2. Train/test split (stratified 80/20)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Stratified train/test split (80/20)")
    logger.info("=" * 60)
    attack_arr = np.array(attack_type_labels)

    X_train, X_test, y_train, y_test, atl_train, atl_test = train_test_split(
        X, y, attack_arr, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %d samples, Test: %d samples", len(y_train), len(y_test))

    # -----------------------------------------------------------------------
    # 3. Normalize
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Feature normalization")
    logger.info("=" * 60)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler for inference
    scaler_path = model_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", scaler_path)

    # Save feature names
    feat_path = model_dir / "cic_feature_names.json"
    with open(feat_path, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    logger.info("Saved feature names to %s", feat_path)

    # -----------------------------------------------------------------------
    # 4. SMOTE balancing (for final model training only — not CV)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: SMOTE class balancing (train-set only, for final models)")
    logger.info("=" * 60)
    class_counts = np.bincount(y_train, minlength=3)
    logger.info("Before SMOTE: %s", dict(enumerate(class_counts)))
    min_count = int(class_counts[class_counts > 0].min())

    if min_count >= 2:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
            X_bal, y_bal = smote.fit_resample(X_train, y_train)
            X_bal = np.asarray(X_bal, dtype=np.float32)
            y_bal = np.asarray(y_bal, dtype=int)
            logger.info("After SMOTE:  %s", dict(enumerate(np.bincount(y_bal, minlength=3))))
        except Exception as exc:
            logger.warning("SMOTE failed (%s), using original data", exc)
            X_bal, y_bal = X_train, y_train
    else:
        logger.warning("Insufficient samples for SMOTE, skipping")
        X_bal, y_bal = X_train, y_train

    # -----------------------------------------------------------------------
    # 5. Train XGBoost
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Training XGBoost")
    logger.info("=" * 60)
    xgb_params: dict[str, Any] = xgb_gpu_params({
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }, use_gpu)
    logger.info("XGBoost: %s", "GPU (cuda)" if use_gpu else "CPU")

    xgb_model = xgb_fit_with_fallback(
        xgb.XGBClassifier, xgb_params, use_gpu,
        X=X_bal, y=y_bal,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # -----------------------------------------------------------------------
    # 6. Train LightGBM
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training LightGBM")
    logger.info("=" * 60)
    lgb_params: dict[str, Any] = lgb_gpu_params({
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "multiclass",
        "num_class": 3,
        "min_data_in_leaf": 20,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }, use_gpu)
    logger.info("LightGBM: %s", "GPU" if use_gpu else "CPU")

    lgb_model = lgb_fit_with_fallback(
        lgb.LGBMClassifier, lgb_params, use_gpu,
        X=X_bal, y=y_bal,
        eval_set=[(X_test, y_test)],
    )

    # -----------------------------------------------------------------------
    # 7. Cross-validation (SMOTE applied inside each fold to avoid leakage)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: 5-Fold Cross-Validation (per-fold SMOTE)")
    logger.info("=" * 60)
    # CV runs on the raw train set — SMOTE is applied inside each fold via
    # imblearn Pipeline so validation folds are never contaminated by synthetic
    # neighbours generated from their own real samples.
    cv_n = min(5, int(np.bincount(y_train, minlength=3)[np.bincount(y_train, minlength=3) > 0].min()))
    cv_n = max(2, cv_n)
    cv = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=42)

    # Use a subset of the raw train set for CV runtime
    cv_max = min(len(y_train), 100_000)
    idx = np.random.default_rng(42).choice(len(y_train), cv_max, replace=False)
    X_cv, y_cv = X_train[idx], y_train[idx]

    # Determine safe k_neighbors for the CV subset
    cv_class_counts = np.bincount(y_cv, minlength=3)
    cv_min = int(cv_class_counts[cv_class_counts > 0].min())
    cv_k = min(5, max(1, cv_min - 1))

    xgb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=cv_k)),
        ("clf", xgb.XGBClassifier(**xgb_params)),
    ])
    lgb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=cv_k)),
        ("clf", lgb.LGBMClassifier(**lgb_params)),
    ])

    logger.info("Running %d-fold CV on %d raw samples (SMOTE per fold, k=%d)...", cv_n, len(y_cv), cv_k)
    xgb_cv_scores = cross_val_score(xgb_pipe, X_cv, y_cv, cv=cv, scoring="f1_weighted", n_jobs=1)
    lgb_cv_scores = cross_val_score(cast(Any, lgb_pipe), X_cv, y_cv, cv=cv, scoring="f1_weighted", n_jobs=1)

    logger.info("XGBoost  CV F1 (weighted): %.4f ± %.4f", xgb_cv_scores.mean(), xgb_cv_scores.std())
    logger.info("LightGBM CV F1 (weighted): %.4f ± %.4f", lgb_cv_scores.mean(), lgb_cv_scores.std())

    # -----------------------------------------------------------------------
    # 8. Test set evaluation
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Test Set Evaluation")
    logger.info("=" * 60)
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    class_names = {0: "benign", 1: "suspicious", 2: "malicious"}

    logger.info("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred, target_names=["benign", "suspicious", "malicious"], zero_division=0))

    logger.info("\nLightGBM Classification Report:")
    print(classification_report(y_test, lgb_pred, target_names=["benign", "suspicious", "malicious"], zero_division=0))

    # -----------------------------------------------------------------------
    # 9. ModelBenchmark per attack type
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Per-Attack-Type ModelBenchmark")
    logger.info("=" * 60)
    benchmark = ModelBenchmark.compare(
        xgb_model,
        lgb_model,
        X_test,
        y_test,
        list(atl_test),
    )

    # -----------------------------------------------------------------------
    # 10. Save models (versioned)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: Saving Models")
    logger.info("=" * 60)
    today = date.today().strftime("%Y%m%d")

    def _save_versioned(path_canonical: Path, save_fn: Any) -> None:
        stem, ext = path_canonical.stem, path_canonical.suffix
        versioned = path_canonical.parent / f"{stem}_{today}{ext}"
        save_fn(versioned)
        try:
            if path_canonical.exists() or path_canonical.is_symlink():
                path_canonical.unlink()
            path_canonical.symlink_to(versioned.name)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(str(versioned), str(path_canonical))
        logger.info("Saved %s → %s (canonical: %s)", versioned.name, versioned, path_canonical)

    _save_versioned(
        model_dir / "xgb_model.json",
        lambda p: xgb_model.get_booster().save_model(str(p)),
    )
    _save_versioned(
        model_dir / "lgbm_model.pkl",
        lambda p: pickle.dump(lgb_model, open(str(p), "wb")),
    )

    # Save benchmark JSON
    benchmark_path = model_dir / "model_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    logger.info("Saved model_benchmark.json to %s", benchmark_path)

    results = {
        "xgb_cv_f1": float(xgb_cv_scores.mean()),
        "xgb_cv_f1_std": float(xgb_cv_scores.std()),
        "lgb_cv_f1": float(lgb_cv_scores.mean()),
        "lgb_cv_f1_std": float(lgb_cv_scores.std()),
        "overall_winner": benchmark["overall_winner"],
        "samples_train": len(y_train),
        "samples_test": len(y_test),
        "feature_count": X.shape[1],
        "benchmark": benchmark,
    }

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("XGBoost  CV F1: %.4f ± %.4f", results["xgb_cv_f1"], results["xgb_cv_f1_std"])
    logger.info("LightGBM CV F1: %.4f ± %.4f", results["lgb_cv_f1"], results["lgb_cv_f1_std"])
    logger.info("Overall winner: %s", results["overall_winner"].upper())

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Train detection models on CIC-DDoS-2018 dataset."
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=50_000,
        help="Max rows to load per CSV file (default: 50000).",
    )
    parser.add_argument(
        "--model-dir",
        default="detection/models",
        help="Directory to save models (default: detection/models).",
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=[
            "data/CSV-01-12/01-12",
            "data/CSV-03-11/03-11",
        ],
        help="Directories containing CIC CSV files.",
    )
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dir]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        logger.error("Data directories not found: %s", missing)
        sys.exit(1)

    model_dir = Path(args.model_dir)

    results = train_cic(
        data_dirs=data_dirs,
        model_dir=model_dir,
        sample_per_file=args.sample_per_file,
    )

    print(f"\nFinal Results:")
    print(f"  XGBoost  CV F1: {results['xgb_cv_f1']:.4f} ± {results['xgb_cv_f1_std']:.4f}")
    print(f"  LightGBM CV F1: {results['lgb_cv_f1']:.4f} ± {results['lgb_cv_f1_std']:.4f}")
    print(f"  Overall Winner: {results['overall_winner'].upper()}")
    print(f"  Train samples:  {results['samples_train']:,}")
    print(f"  Test  samples:  {results['samples_test']:,}")
    print(f"  Features used:  {results['feature_count']}")


if __name__ == "__main__":
    main()
